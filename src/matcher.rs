//! Pattern matching against an encoded IPNS name.
//!
//! Every Ed25519 IPNS name begins with the fixed 12-character string
//! `"k51qzi5uqu5d"` — the `k` multibase tag plus 11 base36 digits derived from
//! the constant CID prefix bytes. Those 12 characters are impossible to alter
//! by changing the keypair, so for prefix mode we automatically prepend them
//! to whatever the user typed.
//!
//! The 13th character is variable but its alphabet is restricted (only seven
//! base36 digits are actually reachable, because the 32-byte public-key range
//! only covers about 18% of one base36-digit slot at that position). We
//! validate that the user's prefix can be reached by *some* keypair and bail
//! out early with a friendly error if not.
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

/// Every Ed25519 IPNS name starts with these 12 characters. The user's prefix
/// pattern is appended to this when searching.
pub const FIXED_PREFIX: &[u8] = b"k51qzi5uqu5d";

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Mode {
    /// Match the user pattern immediately after the constant `k51qzi5uqu5d` prefix.
    Prefix,
    /// Match the user pattern anywhere inside the IPNS name.
    Substring,
    /// Treat the pattern as a regular expression applied to the full name bytes.
    Regex,
}

pub enum Matcher {
    /// One or more full-prefix variants (e.g. `[gh]abc` expands to two).
    Prefix(Vec<Vec<u8>>),
    Substring(Box<Finder<'static>>),
    Regex(Box<Regex>),
}

impl Matcher {
    pub fn new(mode: Mode, pattern: &str) -> Result<Self> {
        match mode {
            Mode::Prefix => {
                let variants = expand_pattern(pattern)?;
                let mut full: Vec<Vec<u8>> = Vec::with_capacity(variants.len());
                for v in &variants {
                    ensure_base36(v)?;
                    let mut p = Vec::with_capacity(FIXED_PREFIX.len() + v.len());
                    p.extend_from_slice(FIXED_PREFIX);
                    p.extend_from_slice(v.as_bytes());
                    full.push(p);
                }

                // Keep only the variants that some keypair can actually
                // produce. If nothing is reachable, surface the most useful
                // error.
                let mut reachable: Vec<Vec<u8>> = Vec::new();
                let mut last_err: Option<anyhow::Error> = None;
                for p in &full {
                    match check_prefix_achievable(p) {
                        Ok(()) => reachable.push(p.clone()),
                        Err(e) => last_err = Some(e),
                    }
                }
                if reachable.is_empty() {
                    return Err(last_err.unwrap_or_else(|| anyhow!("no reachable prefix")));
                }
                if reachable.len() < full.len() {
                    eprintln!(
                        "[ipns-vanity] note: {} of {} prefix variants are not reachable and were dropped",
                        full.len() - reachable.len(),
                        full.len(),
                    );
                }
                Ok(Self::Prefix(reachable))
            }
            Mode::Substring => {
                ensure_base36(pattern)?;
                Ok(Self::Substring(Box::new(
                    Finder::new(pattern.as_bytes()).into_owned(),
                )))
            }
            Mode::Regex => {
                let re = Regex::new(pattern).map_err(|e| anyhow!("invalid regex: {e}"))?;
                Ok(Self::Regex(Box::new(re)))
            }
        }
    }

    #[inline]
    pub fn matches(&self, name: &[u8]) -> bool {
        match self {
            Self::Prefix(ps) => ps.iter().any(|p| name.starts_with(p.as_slice())),
            Self::Substring(f) => f.find(name).is_some(),
            Self::Regex(re) => re.is_match(name),
        }
    }

    /// Returns the substring needle if this is a substring matcher.
    pub fn as_substring(&self) -> Option<&[u8]> {
        if let Self::Substring(f) = self {
            Some(f.needle())
        } else {
            None
        }
    }
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

/// Verify the user's prefix is reachable by *some* Ed25519 IPNS name.
///
/// The full set of names is `[ipns_name(&[0; 32]), ipns_name(&[0xff; 32])]`
/// (lexicographically — base36 byte order matches numeric order). A prefix
/// `p` is reachable iff there exists a string `S` with `p` as its prefix and
/// `min ≤ S ≤ max`; checking the extremes `p || "0…"` and `p || "z…"` against
/// the bounds is enough.
fn check_prefix_achievable(full: &[u8]) -> Result<()> {
    let name_min = ipns_name(&[0u8; 32]);
    let name_max = ipns_name(&[0xffu8; 32]);

    let mut padded_lo = [b'0'; IPNS_NAME_LEN];
    let mut padded_hi = [b'z'; IPNS_NAME_LEN];
    let n = full.len().min(IPNS_NAME_LEN);
    padded_lo[..n].copy_from_slice(&full[..n]);
    padded_hi[..n].copy_from_slice(&full[..n]);

    if padded_lo[..] > name_max[..] || padded_hi[..] < name_min[..] {
        let allowed_pos = full
            .iter()
            .zip(name_min.iter().zip(name_max.iter()))
            .position(|(c, (lo, hi))| !(*lo..=*hi).contains(c))
            .unwrap_or(0);
        let bad_char = full.get(allowed_pos).copied().unwrap_or(b'?') as char;
        return Err(anyhow!(
            "prefix '{}' is not achievable: at name position {} the character \
             must be between '{}' and '{}' (saw '{}').\n\
             Ed25519 IPNS names are bounded by:\n  {}\n  {}",
            std::str::from_utf8(full).unwrap_or("?"),
            allowed_pos,
            name_min[allowed_pos] as char,
            name_max[allowed_pos] as char,
            bad_char,
            std::str::from_utf8(&name_min).unwrap_or("?"),
            std::str::from_utf8(&name_max).unwrap_or("?"),
        ));
    }
    Ok(())
}

fn ensure_base36(pattern: &str) -> Result<()> {
    if pattern.is_empty() {
        return Err(anyhow!("pattern must not be empty"));
    }
    for c in pattern.chars() {
        if !c.is_ascii_lowercase() && !c.is_ascii_digit() {
            return Err(anyhow!(
                "pattern contains '{c}': IPNS names only use base36 lowercase (0-9, a-z)"
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

    #[test]
    fn matcher_prefix_matches() {
        // 'h' is interior to the reachable g..m range, so any second char is OK.
        let m = Matcher::new(Mode::Prefix, "h2").unwrap();
        let mut name = [b'_'; IPNS_NAME_LEN];
        name[..14].copy_from_slice(b"k51qzi5uqu5dh2");
        assert!(m.matches(&name));
        name[13] = b'3';
        assert!(!m.matches(&name));
    }

    #[test]
    fn matcher_prefix_char_class_matches_any() {
        let m = Matcher::new(Mode::Prefix, "[hij]2").unwrap();
        for first in [b'h', b'i', b'j'] {
            let mut name = [b'_'; IPNS_NAME_LEN];
            name[..14].copy_from_slice(b"k51qzi5uqu5dh2");
            name[12] = first;
            assert!(
                m.matches(&name),
                "should match first char {}",
                first as char
            );
        }
    }

    #[test]
    fn matcher_prefix_rejects_unreachable() {
        // The variable position can only be g..m, so '1' is impossible.
        let err = Matcher::new(Mode::Prefix, "1").err().expect("must error");
        assert!(err.to_string().contains("not achievable"));
    }

    #[test]
    fn matcher_prefix_rejects_non_base36() {
        assert!(Matcher::new(Mode::Prefix, "g!").is_err());
        assert!(Matcher::new(Mode::Prefix, "gX").is_err()); // uppercase
    }

    #[test]
    fn matcher_substring_matches_anywhere() {
        let m = Matcher::new(Mode::Substring, "cafe").unwrap();
        let mut name = [b'a'; IPNS_NAME_LEN];
        name[20..24].copy_from_slice(b"cafe");
        assert!(m.matches(&name));
        name[20] = b'x';
        assert!(!m.matches(&name));
    }

    #[test]
    fn matcher_regex_anchored() {
        let m = Matcher::new(Mode::Regex, r"^k51.*42$").unwrap();
        let mut name = [b'a'; IPNS_NAME_LEN];
        name[..3].copy_from_slice(b"k51");
        name[60..].copy_from_slice(b"42");
        assert!(m.matches(&name));
    }
}
