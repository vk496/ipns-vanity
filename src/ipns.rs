//! Convert an Ed25519 public key into an IPNS name.
//!
//! IPNS names for Ed25519 keys are CIDv1 with the libp2p-key codec, identity
//! multihash, and base36-lower multibase encoding. The 40-byte CID is laid out as:
//!
//! ```text
//!   [0x01, 0x72]              CIDv1, codec = libp2p-key
//!   [0x00, 0x24]              identity multihash, length 36
//!   [0x08, 0x01, 0x12, 0x20]  protobuf: KeyType=Ed25519, Data length=32
//!   [..32 bytes of public key..]
//! ```
//!
//! Encoding 40 bytes in base36 yields 61 characters; the multibase prefix `k`
//! brings the total IPNS name length to 62 ASCII bytes.

/// Fixed 8-byte prefix every Ed25519 IPNS CID begins with.
pub const CID_PREFIX: [u8; 8] = [0x01, 0x72, 0x00, 0x24, 0x08, 0x01, 0x12, 0x20];

/// Length of a base36-encoded Ed25519 IPNS name including the `k` multibase prefix.
pub const IPNS_NAME_LEN: usize = 62;

const BASE36: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Encode a 32-byte Ed25519 public key as an IPNS name (62 ASCII bytes).
pub fn ipns_name(pubkey: &[u8; 32]) -> [u8; IPNS_NAME_LEN] {
    let mut out = [0u8; IPNS_NAME_LEN];
    write_ipns_name(pubkey, &mut out);
    out
}

/// Same as `ipns_name`, but writes into a caller-provided buffer to avoid copies.
pub fn write_ipns_name(pubkey: &[u8; 32], out: &mut [u8; IPNS_NAME_LEN]) {
    // Pack the 40-byte CID (8 prefix bytes + 32 pubkey bytes) as ten big-endian
    // u32 limbs. The encoder treats this as one big integer.
    let mut limbs = [0u32; 10];
    limbs[0] = u32::from_be_bytes([CID_PREFIX[0], CID_PREFIX[1], CID_PREFIX[2], CID_PREFIX[3]]);
    limbs[1] = u32::from_be_bytes([CID_PREFIX[4], CID_PREFIX[5], CID_PREFIX[6], CID_PREFIX[7]]);
    for i in 0..8 {
        limbs[2 + i] = u32::from_be_bytes([
            pubkey[4 * i],
            pubkey[4 * i + 1],
            pubkey[4 * i + 2],
            pubkey[4 * i + 3],
        ]);
    }

    // Repeatedly divide the integer by 36 and collect remainders as base36 digits.
    // Digits appear least-significant first; we'll write them out in reverse.
    let mut digits = [0u8; IPNS_NAME_LEN - 1];
    let mut n = 0;
    loop {
        let mut rem: u64 = 0;
        let mut all_zero = true;
        for limb in &mut limbs {
            let cur = (rem << 32) | *limb as u64;
            *limb = (cur / 36) as u32;
            rem = cur % 36;
            if *limb != 0 {
                all_zero = false;
            }
        }
        digits[n] = BASE36[rem as usize];
        n += 1;
        if all_zero {
            break;
        }
    }

    out[0] = b'k';
    let pad = (IPNS_NAME_LEN - 1) - n;
    for i in 0..pad {
        out[1 + i] = b'0';
    }
    for i in 0..n {
        out[1 + pad + i] = digits[n - 1 - i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check our hand-rolled base36 encoder against the canonical `multibase` crate.
    fn ref_encode(pubkey: &[u8; 32]) -> String {
        let mut cid = [0u8; 40];
        cid[..8].copy_from_slice(&CID_PREFIX);
        cid[8..].copy_from_slice(pubkey);
        multibase::encode(multibase::Base::Base36Lower, cid)
    }

    #[test]
    fn matches_multibase_for_zero_key() {
        let pubkey = [0u8; 32];
        let actual = String::from_utf8(ipns_name(&pubkey).to_vec()).unwrap();
        assert_eq!(actual, ref_encode(&pubkey));
    }

    #[test]
    fn matches_multibase_for_max_key() {
        let pubkey = [0xffu8; 32];
        let actual = String::from_utf8(ipns_name(&pubkey).to_vec()).unwrap();
        assert_eq!(actual, ref_encode(&pubkey));
    }

    #[test]
    fn matches_multibase_for_random_keys() {
        use rand::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xbeef);
        for _ in 0..64 {
            let mut pubkey = [0u8; 32];
            rng.fill_bytes(&mut pubkey);
            let actual = String::from_utf8(ipns_name(&pubkey).to_vec()).unwrap();
            assert_eq!(actual, ref_encode(&pubkey));
        }
    }

    #[test]
    fn names_have_fixed_shape() {
        let pubkey = [0u8; 32];
        let name = ipns_name(&pubkey);
        assert_eq!(name.len(), IPNS_NAME_LEN);
        assert_eq!(name[0], b'k');
        // Every byte is base36-lower: digits or a..z.
        for &b in &name[1..] {
            assert!(b.is_ascii_digit() || b.is_ascii_lowercase());
        }
    }

    /// Real Ed25519 keypairs all encode to names beginning with `k51qzi5uqu5d`
    /// — that's the visible fingerprint of the constant CID prefix bytes.
    #[test]
    fn real_keys_share_fixed_prefix() {
        use rand::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        for _ in 0..16 {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            let pk = ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            let name = ipns_name(&pk);
            assert!(name.starts_with(b"k51qzi5uqu5d"));
        }
    }
}
