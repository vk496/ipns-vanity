//! Encode an Ed25519 public key as a libp2p **peer ID** (base58btc / b58mh).
//!
//! The peer ID is the [identity multihash] of the libp2p protobuf-wrapped
//! public key, encoded as base58btc:
//!
//! ```text
//!   multihash = [0x00, 0x24,                     identity multihash, length 36
//!                0x08, 0x01, 0x12, 0x20,         protobuf: KeyType=Ed25519, len=32
//!                <32-byte public key>]
//!   peer_id   = base58btc(multihash)             52 ASCII chars, "12D3KooW…"
//! ```
//!
//! For Ed25519 keys, the multihash always begins with `[0x00, 0x24, 0x08,
//! 0x01, 0x12, 0x20]`, which after base58btc encoding shows up as the fixed
//! 8-character prefix `12D3KooW`.
//!
//! [identity multihash]: https://github.com/multiformats/multihash

/// Fixed 6-byte multihash header that every Ed25519 peer ID shares.
pub const MH_PREFIX: [u8; 6] = [0x00, 0x24, 0x08, 0x01, 0x12, 0x20];

/// Length of a base58btc-encoded Ed25519 peer ID.
pub const PEER_ID_LEN: usize = 52;

/// Every Ed25519 peer ID starts with these 8 characters.
pub const FIXED_PEER_PREFIX: &[u8] = b"12D3KooW";

/// Build the 38-byte identity multihash of an Ed25519 public key.
pub fn multihash(pubkey: &[u8; 32]) -> [u8; 38] {
    let mut mh = [0u8; 38];
    mh[..6].copy_from_slice(&MH_PREFIX);
    mh[6..].copy_from_slice(pubkey);
    mh
}

/// Encode an Ed25519 public key as its libp2p peer ID (52 ASCII bytes).
pub fn peer_id(pubkey: &[u8; 32]) -> [u8; PEER_ID_LEN] {
    let mh = multihash(pubkey);
    let s = bs58::encode(mh).into_string();
    let bytes = s.as_bytes();
    debug_assert_eq!(bytes.len(), PEER_ID_LEN, "unexpected peer-id length");
    let mut out = [0u8; PEER_ID_LEN];
    out.copy_from_slice(&bytes[..PEER_ID_LEN.min(bytes.len())]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_id_starts_with_fixed_prefix() {
        use rand::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1234);
        for _ in 0..16 {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            let pk = ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            let pid = peer_id(&pk);
            assert!(pid.starts_with(FIXED_PEER_PREFIX), "{:?}", &pid[..]);
            // Every byte is base58btc (no `0`, `O`, `I`, `l`).
            for &b in &pid {
                assert!(b.is_ascii_alphanumeric() && !matches!(b, b'0' | b'O' | b'I' | b'l'));
            }
        }
    }

    #[test]
    fn known_zero_seed_peer_id_round_trip() {
        // The seed `[0; 32]` -> a specific Ed25519 pubkey -> a specific peer ID.
        // We don't hard-code the value, just that bs58 round-trips correctly.
        let pk = ed25519_dalek::SigningKey::from_bytes(&[0u8; 32])
            .verifying_key()
            .to_bytes();
        let pid = peer_id(&pk);
        let decoded = bs58::decode(&pid[..]).into_vec().expect("valid base58");
        assert_eq!(decoded, multihash(&pk));
    }
}
