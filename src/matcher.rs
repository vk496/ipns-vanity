//! Pattern matching against an encoded Ed25519 identifier.
//!
//! Two flavours of identifier are supported:
//!
//! * **IPNS name** (`k51qzi5uqu5d…`) — CIDv1 with the libp2p-key codec,
//!   identity multihash, base36-lower multibase. 62 ASCII chars.
//! * **Peer ID** (`12D3KooW…`) — base58btc of the identity multihash. 52 ASCII
//!   chars.
//!
//! Both encode the same Ed25519 public key, so a single search can target
//! either or both. Patterns are sorted into two pools by the caller: IPNS
//! patterns are matched against the IPNS name; peer-ID patterns are matched
//! against the peer-ID string. A keypair is a hit if **any** pattern in either
//! pool matches its target.
//!
//! Each target also has a fixed leading prefix that no keypair can change —
//! `"k51qzi5uqu5d"` for IPNS, `"12D3KooW"` for peer IDs — and we automatically
//! prepend them so the user only types the variable part.
//!
//! ## Character classes
//!
//! Prefix patterns may use regex-style character classes at *any* position to
//! enumerate which characters are acceptable there, e.g. `[ghj]abc` matches
//! names whose variable part starts with `gabc`, `habc`, or `jabc`, and
//! `[g-m]vk49[6a3]` allows any of `g`–`m` in slot 0 and any of `6`/`a`/`3` in
//! slot 7. This is GPU-friendly: each fully concrete variant becomes its own
//! CID range and the kernel OR-tests them.
//!
//! The expansion is a Cartesian product, so multiple classes can blow up
//! quickly — we cap the total variant count and reject anything beyond it.

use anyhow::{Result, anyhow};
use memchr::memmem::Finder;
use regex::bytes::Regex;

use crate::ipns::{IPNS_NAME_LEN, ipns_name};
use crate::peerid::{FIXED_PEER_PREFIX, PEER_ID_LEN, peer_id};

/// Every Ed25519 IPNS name starts with these 12 characters. The user's prefix
/// pattern is appended to this when searching.
pub const FIXED_PREFIX: &[u8] = b"k51qzi5uqu5d";

/// Which encoding a pattern was supplied for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Ipns,
    Peer,
}

impl Target {
    fn fixed_prefix(self) -> &'static [u8] {
        match self {
            Target::Ipns => FIXED_PREFIX,
            Target::Peer => FIXED_PEER_PREFIX,
        }
    }
    fn encoded_len(self) -> usize {
        match self {
            Target::Ipns => IPNS_NAME_LEN,
            Target::Peer => PEER_ID_LEN,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Target::Ipns => "ipns",
            Target::Peer => "peer-id",
        }
    }
    /// Which characters are valid in patterns for this target.
    fn is_valid_char(self, c: char) -> bool {
        match self {
            Target::Ipns => c.is_ascii_lowercase() || c.is_ascii_digit(),
            // base58btc alphabet — alphanumeric minus 0/O/I/l.
            Target::Peer => c.is_ascii_alphanumeric() && !matches!(c, '0' | 'O' | 'I' | 'l'),
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Mode {
    /// Match the user pattern immediately after the constant `k51qzi5uqu5d`
    /// (or `12D3KooW` for peer IDs) prefix.
    Prefix,
    /// Match the user pattern at the very end of the IPNS name / peer ID.
    Suffix,
    /// Match the user pattern anywhere inside the IPNS name / peer ID.
    Substring,
    /// Treat the pattern as a regular expression applied to the full name bytes.
    Regex,
}

/// Which identifier(s) to search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Scope {
    /// Search the IPNS name (`k51qzi5uqu5d…`) only.
    Ipns,
    /// Search the peer ID (`12D3KooW…`) only.
    Peerid,
    /// Search both — each pattern is auto-routed to whichever alphabet it fits.
    Both,
}

pub enum Matcher {
    /// Full-prefix variants, kept per target so the GPU can translate the IPNS
    /// ones to CID ranges and the peer-ID ones to multihash ranges (re-cast as
    /// CID ranges, see `gpu::run`).
    Prefix {
        ipns: Vec<Vec<u8>>,
        peer: Vec<Vec<u8>>,
    },
    /// Suffix needles per target (matched via `ends_with`).
    Suffix {
        ipns: Vec<Vec<u8>>,
        peer: Vec<Vec<u8>>,
    },
    /// Substring needles per target.
    Substring { ipns: Substrings, peer: Substrings },
    /// Compiled regex per target (each only set when at least one pattern was
    /// supplied for that target).
    Regex {
        ipns: Option<Box<Regex>>,
        peer: Option<Box<Regex>>,
    },
}

/// Substring needles, kept in two parallel forms: precomputed `Finder`s for the
/// CPU hot loop, and raw bytes for shipping to the GPU.
pub struct Substrings {
    finders: Vec<Finder<'static>>,
    pub needles: Vec<Vec<u8>>,
}

impl Substrings {
    pub fn is_empty(&self) -> bool {
        self.needles.is_empty()
    }
}

impl Matcher {
    /// Build a matcher from a single pattern list plus the `--target` scope.
    /// In `Scope::Both`, each pattern is auto-routed by its alphabet to the
    /// IPNS pool, the peer-id pool, or both. A pattern that fits neither
    /// alphabet, or that's unreachable under every target it fits, errors.
    pub fn new(mode: Mode, scope: Scope, patterns: &[String]) -> Result<Self> {
        if patterns.is_empty() {
            return Err(anyhow!("at least one pattern is required"));
        }
        let (ipns_pats, peer_pats) = partition_patterns(scope, patterns)?;
        let tolerate_partial = scope == Scope::Both;

        match mode {
            Mode::Prefix => {
                let ipns = build_prefix_or_skip(Target::Ipns, &ipns_pats, tolerate_partial)?;
                let peer = build_prefix_or_skip(Target::Peer, &peer_pats, tolerate_partial)?;
                if ipns.is_empty() && peer.is_empty() {
                    return Err(anyhow!(
                        "no pattern is reachable as a prefix for the selected target(s)"
                    ));
                }
                Ok(Self::Prefix { ipns, peer })
            }
            Mode::Suffix => Ok(Self::Suffix {
                ipns: build_suffixes(Target::Ipns, &ipns_pats)?,
                peer: build_suffixes(Target::Peer, &peer_pats)?,
            }),
            Mode::Substring => Ok(Self::Substring {
                ipns: build_substrings(Target::Ipns, &ipns_pats)?,
                peer: build_substrings(Target::Peer, &peer_pats)?,
            }),
            Mode::Regex => Ok(Self::Regex {
                ipns: build_regex(&ipns_pats)?,
                peer: build_regex(&peer_pats)?,
            }),
        }
    }

    #[inline]
    pub fn matches(&self, ipns: &[u8], peer: &[u8]) -> bool {
        match self {
            Self::Prefix { ipns: i, peer: p } => {
                i.iter().any(|x| ipns.starts_with(x.as_slice()))
                    || p.iter().any(|x| peer.starts_with(x.as_slice()))
            }
            Self::Suffix { ipns: i, peer: p } => {
                i.iter().any(|x| ipns.ends_with(x.as_slice()))
                    || p.iter().any(|x| peer.ends_with(x.as_slice()))
            }
            Self::Substring { ipns: i, peer: p } => {
                i.finders.iter().any(|f| f.find(ipns).is_some())
                    || p.finders.iter().any(|f| f.find(peer).is_some())
            }
            Self::Regex { ipns: i, peer: p } => {
                i.as_ref().is_some_and(|re| re.is_match(ipns))
                    || p.as_ref().is_some_and(|re| re.is_match(peer))
            }
        }
    }

    /// Returns `true` if this matcher needs peer-ID computation (any peer-ID
    /// pattern was supplied). Used by the search loops to avoid encoding the
    /// peer ID when nothing would consume it.
    pub fn needs_peer_id(&self) -> bool {
        match self {
            Self::Prefix { peer, .. } => !peer.is_empty(),
            Self::Suffix { peer, .. } => !peer.is_empty(),
            Self::Substring { peer, .. } => !peer.is_empty(),
            Self::Regex { peer, .. } => peer.is_some(),
        }
    }
}

/// Decide which target pool(s) each input pattern belongs to based on the
/// `--target` scope and the pattern's alphabet.
///
/// For `Scope::Ipns` / `Scope::Peerid` every pattern goes into that one pool
/// unchanged (alphabet validation happens later inside `build_*`).
///
/// For `Scope::Both` we look at each pattern's characters: lowercase letters
/// and digits typically fit *both* alphabets and the reachability check sorts
/// out which target it can actually hit. Uppercase letters route to peer-id
/// only. A pattern that uses characters outside *both* alphabets is rejected.
fn partition_patterns(scope: Scope, patterns: &[String]) -> Result<(Vec<String>, Vec<String>)> {
    let want_ipns = matches!(scope, Scope::Ipns | Scope::Both);
    let want_peer = matches!(scope, Scope::Peerid | Scope::Both);

    let mut ipns_pats = Vec::new();
    let mut peer_pats = Vec::new();
    for pat in patterns {
        let fits_ipns = want_ipns && pattern_class_fits(Target::Ipns, pat);
        let fits_peer = want_peer && pattern_class_fits(Target::Peer, pat);

        match scope {
            Scope::Ipns => ipns_pats.push(pat.clone()),
            Scope::Peerid => peer_pats.push(pat.clone()),
            Scope::Both => {
                if !fits_ipns && !fits_peer {
                    return Err(anyhow!(
                        "pattern '{pat}' uses characters outside both the IPNS base36 and \
                         the peer-id base58btc alphabets"
                    ));
                }
                if fits_ipns {
                    ipns_pats.push(pat.clone());
                }
                if fits_peer {
                    peer_pats.push(pat.clone());
                }
            }
        }
    }
    Ok((ipns_pats, peer_pats))
}

/// Permissive alphabet check used during `Scope::Both` routing. Character
/// classes (`[abc]`, `[a-z]`) are accepted regardless of what's inside, since
/// they expand later — the full alphabet check fires inside `build_*`.
fn pattern_class_fits(target: Target, pat: &str) -> bool {
    let mut in_class = false;
    for c in pat.chars() {
        if c == '[' {
            in_class = true;
            continue;
        }
        if c == ']' {
            in_class = false;
            continue;
        }
        if in_class || c == '-' {
            continue;
        }
        if !target.is_valid_char(c) {
            return false;
        }
    }
    true
}

/// `build_prefix_variants`, but in `Scope::Both` mode we tolerate a one-side
/// failure (and just print the reason) so the other side can still produce a
/// usable matcher.
fn build_prefix_or_skip(
    target: Target,
    patterns: &[String],
    tolerate: bool,
) -> Result<Vec<Vec<u8>>> {
    if patterns.is_empty() {
        return Ok(Vec::new());
    }
    match build_prefix_variants(target, patterns) {
        Ok(v) => Ok(v),
        Err(e) if tolerate => {
            eprintln!(
                "[ipns-vanity] note: skipping {} target: {}",
                target.name(),
                e
            );
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}

fn build_prefix_variants(target: Target, patterns: &[String]) -> Result<Vec<Vec<u8>>> {
    if patterns.is_empty() {
        return Ok(Vec::new());
    }
    let fixed = target.fixed_prefix();
    let fixed_str = std::str::from_utf8(fixed).unwrap();

    let mut all: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for pat in patterns {
        // Be forgiving: a user can paste the whole `k51qzi5uqu5d…` or
        // `12D3KooW…` prefix and we just strip the fixed part.
        let pat = pat.strip_prefix(fixed_str).unwrap_or(pat).to_string();
        for v in expand_pattern(&pat)? {
            all.insert(v);
        }
    }

    let mut full: Vec<Vec<u8>> = Vec::with_capacity(all.len());
    for v in &all {
        ensure_alphabet(target, v)?;
        let mut p = Vec::with_capacity(fixed.len() + v.len());
        p.extend_from_slice(fixed);
        p.extend_from_slice(v.as_bytes());
        full.push(p);
    }

    let mut reachable: Vec<Vec<u8>> = Vec::new();
    let mut last_err: Option<anyhow::Error> = None;
    for p in &full {
        match check_prefix_achievable(target, p) {
            Ok(()) => reachable.push(p.clone()),
            Err(e) => last_err = Some(e),
        }
    }
    if reachable.is_empty() {
        return Err(last_err.unwrap_or_else(|| anyhow!("no reachable prefix")));
    }
    if reachable.len() < full.len() {
        eprintln!(
            "[ipns-vanity] note: {} of {} {} prefix variants are not reachable and were dropped",
            full.len() - reachable.len(),
            full.len(),
            target.name(),
        );
    }
    Ok(reachable)
}

fn build_suffixes(target: Target, patterns: &[String]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(patterns.len());
    for pat in patterns {
        ensure_alphabet(target, pat)?;
        out.push(pat.as_bytes().to_vec());
    }
    Ok(out)
}

fn build_substrings(target: Target, patterns: &[String]) -> Result<Substrings> {
    let mut needles: Vec<Vec<u8>> = Vec::with_capacity(patterns.len());
    let mut finders: Vec<Finder<'static>> = Vec::with_capacity(patterns.len());
    for pat in patterns {
        ensure_alphabet(target, pat)?;
        let bytes = pat.as_bytes().to_vec();
        finders.push(Finder::new(bytes.as_slice()).into_owned());
        needles.push(bytes);
    }
    Ok(Substrings { finders, needles })
}

fn build_regex(patterns: &[String]) -> Result<Option<Box<Regex>>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let combined = if patterns.len() == 1 {
        patterns[0].clone()
    } else {
        patterns
            .iter()
            .map(|p| format!("(?:{p})"))
            .collect::<Vec<_>>()
            .join("|")
    };
    let re = Regex::new(&combined).map_err(|e| anyhow!("invalid regex: {e}"))?;
    Ok(Some(Box::new(re)))
}

/// Cap on the Cartesian product of prefix variants. A handful of classes is
/// fine; multiplying many can produce thousands of CID ranges, which slows the
/// kernel down per work-item.
const MAX_PREFIX_VARIANTS: usize = 1024;

/// Expand a prefix pattern containing zero or more `[...]` character classes
/// into the concrete list of prefix strings it represents.
///
/// Examples:
///
/// * `abc`           → `["abc"]`
/// * `[gh]abc`       → `["gabc", "habc"]`
/// * `[g-i0]abc`     → `["0abc", "gabc", "habc", "iabc"]`
/// * `[gh]a[12]`     → `["ga1", "ga2", "ha1", "ha2"]`
fn expand_pattern(pattern: &str) -> Result<Vec<String>> {
    let mut variants: Vec<String> = vec![String::new()];
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            let mut class = String::new();
            loop {
                match chars.next() {
                    Some(']') => break,
                    Some(ch) => class.push(ch),
                    None => return Err(anyhow!("unterminated '[' in pattern")),
                }
            }
            let class_chars = expand_char_class(&class)?;
            if class_chars.is_empty() {
                return Err(anyhow!("empty character class '[]'"));
            }
            let mut next = Vec::with_capacity(variants.len() * class_chars.len());
            for v in &variants {
                for &cc in &class_chars {
                    let mut s = v.clone();
                    s.push(cc);
                    next.push(s);
                }
            }
            if next.len() > MAX_PREFIX_VARIANTS {
                return Err(anyhow!(
                    "pattern expands to {} variants (limit {}); shrink one of the character classes",
                    next.len(),
                    MAX_PREFIX_VARIANTS,
                ));
            }
            variants = next;
        } else {
            for v in &mut variants {
                v.push(c);
            }
        }
    }
    Ok(variants)
}

fn expand_char_class(class: &str) -> Result<Vec<char>> {
    let bytes = class.as_bytes();
    let mut set = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if i + 2 < bytes.len() && bytes[i + 1] == b'-' {
            let hi = bytes[i + 2];
            if hi < c {
                return Err(anyhow!(
                    "invalid range '{}-{}' in character class",
                    c as char,
                    hi as char,
                ));
            }
            for x in c..=hi {
                set.insert(x as char);
            }
            i += 3;
        } else {
            set.insert(c as char);
            i += 1;
        }
    }
    Ok(set.into_iter().collect())
}

/// Verify the user's prefix is reachable by *some* Ed25519 identifier of the
/// chosen target.
///
/// The full set of names for the target is bounded by encoding the all-zero
/// and all-ones public keys. A prefix `p` is reachable iff there exists a
/// string `S` with `p` as its prefix and `min ≤ S ≤ max`; checking the
/// extremes `p || "<min-char>…"` and `p || "<max-char>…"` against the bounds
/// is enough. Padding chars depend on the alphabet (`0`/`z` for IPNS, `1`/`z`
/// for base58btc).
fn check_prefix_achievable(target: Target, full: &[u8]) -> Result<()> {
    let (min_buf, max_buf, lo_pad, hi_pad, n) = match target {
        Target::Ipns => (
            ipns_name(&[0u8; 32]).to_vec(),
            ipns_name(&[0xffu8; 32]).to_vec(),
            b'0',
            b'z',
            IPNS_NAME_LEN,
        ),
        Target::Peer => (
            peer_id(&[0u8; 32]).to_vec(),
            peer_id(&[0xffu8; 32]).to_vec(),
            b'1',
            b'z',
            PEER_ID_LEN,
        ),
    };

    let mut padded_lo = vec![lo_pad; n];
    let mut padded_hi = vec![hi_pad; n];
    let m = full.len().min(n);
    padded_lo[..m].copy_from_slice(&full[..m]);
    padded_hi[..m].copy_from_slice(&full[..m]);

    if padded_lo[..] > max_buf[..] || padded_hi[..] < min_buf[..] {
        let allowed_pos = full
            .iter()
            .zip(min_buf.iter().zip(max_buf.iter()))
            .position(|(c, (lo, hi))| !(*lo..=*hi).contains(c))
            .unwrap_or(0);
        let bad_char = full.get(allowed_pos).copied().unwrap_or(b'?') as char;
        return Err(anyhow!(
            "{} prefix '{}' is not achievable: at position {} the character \
             must be between '{}' and '{}' (saw '{}').\n\
             Bounds:\n  {}\n  {}",
            target.name(),
            std::str::from_utf8(full).unwrap_or("?"),
            allowed_pos,
            min_buf[allowed_pos] as char,
            max_buf[allowed_pos] as char,
            bad_char,
            std::str::from_utf8(&min_buf).unwrap_or("?"),
            std::str::from_utf8(&max_buf).unwrap_or("?"),
        ));
    }
    let _ = target.encoded_len(); // silence unused-method warning during build
    Ok(())
}

fn ensure_alphabet(target: Target, pattern: &str) -> Result<()> {
    if pattern.is_empty() {
        return Err(anyhow!("pattern must not be empty"));
    }
    for c in pattern.chars() {
        if !target.is_valid_char(c) {
            let allowed = match target {
                Target::Ipns => "base36 lowercase (0-9, a-z)",
                Target::Peer => "base58btc (1-9, A-Z, a-z, excluding 0/O/I/l)",
            };
            return Err(anyhow!(
                "{} pattern contains '{c}': {} only uses {}",
                target.name(),
                target.name(),
                allowed,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(pat: &str) -> Vec<String> {
        expand_pattern(pat).unwrap()
    }

    #[test]
    fn expand_plain_pattern() {
        assert_eq!(expanded("abc"), vec!["abc"]);
    }

    #[test]
    fn expand_char_class() {
        assert_eq!(expanded("[gh]abc"), vec!["gabc", "habc"]);
    }

    #[test]
    fn expand_range_and_dedups() {
        // Range g-i plus an explicit 'g' must not produce duplicates.
        assert_eq!(expanded("[gg-i]x"), vec!["gx", "hx", "ix"]);
    }

    #[test]
    fn expand_mixes_chars_and_ranges() {
        assert_eq!(expanded("[g-i0]x"), vec!["0x", "gx", "hx", "ix"]);
    }

    #[test]
    fn expand_class_at_any_position() {
        // Two classes, one in the middle and one at the end.
        assert_eq!(expanded("g[hi]4[26]"), vec!["gh42", "gh46", "gi42", "gi46"],);
    }

    #[test]
    fn expand_classes_compose_cartesian() {
        // 2 * 3 * 1 * 2 = 12 variants
        assert_eq!(expanded("[gh][abc]4[xy]").len(), 12);
    }

    #[test]
    fn expand_rejects_too_many_variants() {
        // [a-z][a-z][a-z][a-z] = 26^4 = 456_976, way past the cap.
        assert!(expand_pattern("[a-z][a-z][a-z][a-z]").is_err());
    }

    #[test]
    fn expand_rejects_unterminated_class() {
        assert!(expand_pattern("[gh").is_err());
    }

    #[test]
    fn expand_rejects_empty_class() {
        assert!(expand_pattern("[]abc").is_err());
    }

    #[test]
    fn expand_rejects_reversed_range() {
        assert!(expand_pattern("[z-a]").is_err());
    }

    fn mk(mode: Mode, scope: Scope, p: &[&str]) -> Matcher {
        let pats: Vec<String> = p.iter().map(|s| s.to_string()).collect();
        Matcher::new(mode, scope, &pats).unwrap()
    }
    fn ipns_mk(p: &[&str]) -> Matcher {
        mk(Mode::Prefix, Scope::Ipns, p)
    }
    fn peer_mk(p: &[&str]) -> Matcher {
        mk(Mode::Prefix, Scope::Peerid, p)
    }

    /// Build an IPNS-name buffer with `ipns` written at the start and the rest
    /// filled with a known-irrelevant character.
    fn ipns_buf(ipns: &str) -> [u8; IPNS_NAME_LEN] {
        let mut b = [b'_'; IPNS_NAME_LEN];
        b[..ipns.len()].copy_from_slice(ipns.as_bytes());
        b
    }
    const EMPTY_IPNS: [u8; IPNS_NAME_LEN] = [b'_'; IPNS_NAME_LEN];
    const EMPTY_PEER: [u8; PEER_ID_LEN] = [b'_'; PEER_ID_LEN];

    #[test]
    fn matcher_prefix_matches() {
        let m = ipns_mk(&["h2"]);
        assert!(m.matches(&ipns_buf("k51qzi5uqu5dh2"), &EMPTY_PEER));
        assert!(!m.matches(&ipns_buf("k51qzi5uqu5dh3"), &EMPTY_PEER));
    }

    #[test]
    fn matcher_prefix_char_class_matches_any() {
        let m = ipns_mk(&["[hij]2"]);
        for first in ['h', 'i', 'j'] {
            let s = format!("k51qzi5uqu5d{first}2");
            assert!(m.matches(&ipns_buf(&s), &EMPTY_PEER));
        }
    }

    #[test]
    fn matcher_prefix_rejects_unreachable() {
        let err = Matcher::new(Mode::Prefix, Scope::Ipns, &["1".to_string()])
            .err()
            .expect("must error");
        assert!(err.to_string().contains("not achievable"));
    }

    #[test]
    fn matcher_prefix_rejects_non_base36() {
        assert!(Matcher::new(Mode::Prefix, Scope::Ipns, &["g!".to_string()]).is_err());
        assert!(Matcher::new(Mode::Prefix, Scope::Ipns, &["gX".to_string()]).is_err());
    }

    #[test]
    fn matcher_substring_matches_anywhere() {
        let m = mk(Mode::Substring, Scope::Ipns, &["cafe"]);
        let mut name = [b'a'; IPNS_NAME_LEN];
        name[20..24].copy_from_slice(b"cafe");
        assert!(m.matches(&name, &EMPTY_PEER));
        name[20] = b'x';
        assert!(!m.matches(&name, &EMPTY_PEER));
    }

    #[test]
    fn matcher_regex_anchored() {
        let m = mk(Mode::Regex, Scope::Ipns, &["^k51.*42$"]);
        let mut name = [b'a'; IPNS_NAME_LEN];
        name[..3].copy_from_slice(b"k51");
        name[60..].copy_from_slice(b"42");
        assert!(m.matches(&name, &EMPTY_PEER));
    }

    #[test]
    fn matcher_suffix_matches_end() {
        let m = mk(Mode::Suffix, Scope::Ipns, &["abc"]);
        let mut name = [b'x'; IPNS_NAME_LEN];
        name[IPNS_NAME_LEN - 3..].copy_from_slice(b"abc");
        assert!(m.matches(&name, &EMPTY_PEER));
        // Same bytes but not at the end → no match.
        let mut name = [b'x'; IPNS_NAME_LEN];
        name[10..13].copy_from_slice(b"abc");
        assert!(!m.matches(&name, &EMPTY_PEER));
    }

    #[test]
    fn matcher_suffix_unions_multiple_needles() {
        let m = mk(Mode::Suffix, Scope::Ipns, &["end1", "end2"]);
        let mut a = [b'x'; IPNS_NAME_LEN];
        a[IPNS_NAME_LEN - 4..].copy_from_slice(b"end1");
        assert!(m.matches(&a, &EMPTY_PEER));
        let mut b = [b'x'; IPNS_NAME_LEN];
        b[IPNS_NAME_LEN - 4..].copy_from_slice(b"end2");
        assert!(m.matches(&b, &EMPTY_PEER));
        assert!(!m.matches(&[b'x'; IPNS_NAME_LEN], &EMPTY_PEER));
    }

    #[test]
    fn matcher_prefix_unions_multiple_patterns() {
        let m = ipns_mk(&["h2", "i3"]);
        assert!(m.matches(&ipns_buf("k51qzi5uqu5dh2"), &EMPTY_PEER));
        assert!(m.matches(&ipns_buf("k51qzi5uqu5di3"), &EMPTY_PEER));
        assert!(!m.matches(&ipns_buf("k51qzi5uqu5dh3"), &EMPTY_PEER));
    }

    #[test]
    fn matcher_substring_unions_multiple_needles() {
        let m = mk(Mode::Substring, Scope::Ipns, &["beef", "cafe"]);
        let mut a = [b'x'; IPNS_NAME_LEN];
        a[10..14].copy_from_slice(b"beef");
        assert!(m.matches(&a, &EMPTY_PEER));
        let mut b = [b'x'; IPNS_NAME_LEN];
        b[40..44].copy_from_slice(b"cafe");
        assert!(m.matches(&b, &EMPTY_PEER));
        assert!(!m.matches(&[b'x'; IPNS_NAME_LEN], &EMPTY_PEER));
    }

    #[test]
    fn matcher_regex_ors_multiple_patterns() {
        let m = mk(Mode::Regex, Scope::Ipns, &["^k51.*xx$", "^k51.*42$"]);
        let mut a = [b'q'; IPNS_NAME_LEN];
        a[..3].copy_from_slice(b"k51");
        a[60..].copy_from_slice(b"42");
        assert!(m.matches(&a, &EMPTY_PEER));
    }

    #[test]
    fn matcher_peer_prefix_matches() {
        // The reachable range at peer-id position 8 is restricted; sample a real
        // keypair, take its 9th-11th chars as a known-reachable 3-char prefix,
        // and check the matcher fires on it.
        use rand::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xb15);
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        let pk = ed25519_dalek::SigningKey::from_bytes(&seed)
            .verifying_key()
            .to_bytes();
        let pid = peer_id(&pk);
        let three_chars = std::str::from_utf8(&pid[8..11]).unwrap().to_string();
        let m = peer_mk(&[three_chars.as_str()]);
        assert!(m.matches(&EMPTY_IPNS, &pid));
    }

    #[test]
    fn scope_both_routes_lowercase_to_ipns_and_uppercase_to_peer() {
        // "h2" is base36 (fits IPNS) and reachable; "B" is uppercase (fits
        // peer-id alphabet) and lands in the 9..T reachable range. Together
        // under `Scope::Both` they should produce a matcher with one entry in
        // each pool.
        let m = Matcher::new(
            Mode::Prefix,
            Scope::Both,
            &["h2".to_string(), "B".to_string()],
        )
        .expect("both patterns should route to their respective targets");
        // The IPNS-only buffer (peer slot empty) should match the IPNS pattern.
        assert!(m.matches(&ipns_buf("k51qzi5uqu5dh2"), &EMPTY_PEER));
    }

    #[test]
    fn scope_both_rejects_unreachable_pattern_with_no_other_home() {
        // "axy" fits both alphabets but is unreachable for IPNS (`a` not in
        // g..m) AND for peer-id (`a` lowercase, not in 9..T). It must error.
        let err = Matcher::new(Mode::Prefix, Scope::Both, &["axy".to_string()])
            .err()
            .expect("must error when reachable for neither target");
        let _ = err; // message is implementation-defined
    }
}
