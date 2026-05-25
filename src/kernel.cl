// IPNS vanity kernel.
//
// Each work-item derives an Ed25519 keypair from a seed (base_seed XOR a
// 64-bit nonce), encodes the corresponding IPNS CID, and tests it against a
// pattern. On a match, the seed and public key are atomically written to the
// result slot and a found flag is raised so other work-items can bail out.
//
// Layout of the math:
//   * Field elements are 8 little-endian uint32 limbs (radix 2^32), lazily
//     reduced. 2^256 = 38 (mod p), so any 33rd bit that spills off the top
//     gets multiplied by 38 and folded back into limb 0.
//   * Edwards points use extended twisted-Edwards coordinates (X:Y:Z:T) with
//     T = XY/Z. The HWCD-3 unified addition formula handles doubling and the
//     identity correctly, so scalar multiplication is a plain double-and-add.
//   * The base point B and the curve constant 2*d are passed in as
//     __constant buffers from the host (computed there once at startup).

// =============================================================================
// SHA-512
// =============================================================================

__constant ulong K512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define SIG0(x) (ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define SIG1(x) (ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define sig0(x) (ROTR64(x,  1) ^ ROTR64(x,  8) ^ ((x) >> 7))
#define sig1(x) (ROTR64(x, 19) ^ ROTR64(x, 61) ^ ((x) >> 6))
#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

// SHA-512 of a 32-byte input. Single-block message; padding is the constant
// 0x80, zeros, and the 128-bit length 256.
static void sha512_32(const uchar msg[32], uchar out[64]) {
    ulong W[80];
    for (int i = 0; i < 4; i++) {
        W[i] = ((ulong)msg[i*8+0] << 56) | ((ulong)msg[i*8+1] << 48)
             | ((ulong)msg[i*8+2] << 40) | ((ulong)msg[i*8+3] << 32)
             | ((ulong)msg[i*8+4] << 24) | ((ulong)msg[i*8+5] << 16)
             | ((ulong)msg[i*8+6] <<  8) | ((ulong)msg[i*8+7]);
    }
    W[4]  = 0x8000000000000000UL;          // padding bit
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256;                            // length in bits
    for (int i = 16; i < 80; i++) {
        W[i] = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];
    }

    ulong a = 0x6a09e667f3bcc908UL, b = 0xbb67ae8584caa73bUL,
          c = 0x3c6ef372fe94f82bUL, d = 0xa54ff53a5f1d36f1UL,
          e = 0x510e527fade682d1UL, f = 0x9b05688c2b3e6c1fUL,
          g = 0x1f83d9abfb41bd6bUL, h = 0x5be0cd19137e2179UL;

    for (int i = 0; i < 80; i++) {
        ulong t1 = h + SIG1(e) + CH(e, f, g) + K512[i] + W[i];
        ulong t2 = SIG0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ulong H[8] = {
        a + 0x6a09e667f3bcc908UL, b + 0xbb67ae8584caa73bUL,
        c + 0x3c6ef372fe94f82bUL, d + 0xa54ff53a5f1d36f1UL,
        e + 0x510e527fade682d1UL, f + 0x9b05688c2b3e6c1fUL,
        g + 0x1f83d9abfb41bd6bUL, h + 0x5be0cd19137e2179UL
    };
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            out[i*8+j] = (uchar)(H[i] >> (56 - j*8));
        }
    }
}

// =============================================================================
// Field arithmetic over GF(2^255 - 19)
// =============================================================================
//
// Field elements are 8 uint32 limbs, little-endian (limb[0] is least
// significant). Values are kept in the range [0, 2^256); reductions are lazy
// and rely on 2^256 ≡ 38 (mod p).

typedef uint fe[8];

static inline void fe_copy(fe r, const fe a) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}
static inline void fe_zero(fe r) { for (int i = 0; i < 8; i++) r[i] = 0; }
static inline void fe_one(fe r)  { r[0] = 1; for (int i = 1; i < 8; i++) r[i] = 0; }

static void fe_add(fe r, const fe a, const fe b) {
    ulong c = 0;
    for (int i = 0; i < 8; i++) {
        c += (ulong)a[i] + b[i];
        r[i] = (uint)c;
        c >>= 32;
    }
    // Final carry: fold the 2^256 bit back as 38.
    c *= 38;
    for (int i = 0; i < 8 && c; i++) {
        c += r[i];
        r[i] = (uint)c;
        c >>= 32;
    }
}

static void fe_sub(fe r, const fe a, const fe b) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        long v = (long)a[i] - (long)b[i] - borrow;
        r[i] = (uint)(v & 0xffffffffL);
        borrow = (v < 0) ? 1 : 0;
    }
    if (borrow) {
        // Add 2p = 2^256 - 38 to bring r back into range. The 2^256 part wraps
        // away naturally, leaving r -= 38 (with implicit borrow absorbed).
        ulong c = (ulong)r[0] + 0xffffffdaUL;
        r[0] = (uint)c;
        c >>= 32;
        for (int i = 1; i < 8; i++) {
            c += (ulong)r[i] + 0xffffffffUL;
            r[i] = (uint)c;
            c >>= 32;
        }
    }
}

// Schoolbook 8x8 multiplication, then reduce: result_low + 38 * result_high.
static void fe_mul(fe r, const fe a, const fe b) {
    ulong t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;

    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong prod = t[i+j] + (ulong)a[i] * b[j] + carry;
            t[i+j] = prod & 0xffffffffUL;
            carry  = prod >> 32;
        }
        t[i+8] = carry;
    }

    // Fold the high half: r[i] = t[i] + 38 * t[i+8].
    ulong c = 0;
    for (int i = 0; i < 8; i++) {
        c += t[i] + 38UL * t[i+8];
        r[i] = (uint)c;
        c >>= 32;
    }
    // Second pass for any residual overflow.
    c *= 38;
    for (int i = 0; i < 8 && c; i++) {
        c += r[i];
        r[i] = (uint)c;
        c >>= 32;
    }
}

static inline void fe_sq(fe r, const fe a) { fe_mul(r, a, a); }

// a^(p-2) using a standard 250-squaring-plus-mul addition chain.
static void fe_invert(fe out, const fe z) {
    fe t0, t1, t2, t3;
    int i;

    fe_sq(t0, z);                      // z^2
    fe_sq(t1, t0); fe_sq(t1, t1);      // z^8
    fe_mul(t1, z, t1);                 // z^9
    fe_mul(t0, t0, t1);                // z^11
    fe_sq(t2, t0);                     // z^22
    fe_mul(t1, t2, t1);                // 2^5 - 2^0
    fe_sq(t2, t1);                     for (i = 1; i <  5; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                // 2^10 - 2^0
    fe_sq(t2, t1);                     for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                // 2^20 - 2^0
    fe_sq(t3, t2);                     for (i = 1; i < 20; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                // 2^40 - 2^0
    fe_sq(t2, t2);                     for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                // 2^50 - 2^0
    fe_sq(t2, t1);                     for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                // 2^100 - 2^0
    fe_sq(t3, t2);                     for (i = 1; i < 100; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                // 2^200 - 2^0
    fe_sq(t2, t2);                     for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                // 2^250 - 2^0
    fe_sq(t1, t1); fe_sq(t1, t1); fe_sq(t1, t1); fe_sq(t1, t1); fe_sq(t1, t1);
    fe_mul(out, t1, t0);               // 2^255 - 21 = p - 2
}

// Reduce r to the canonical representative in [0, p) and serialize as 32
// little-endian bytes.
static void fe_to_bytes(uchar out[32], const fe in) {
    fe r;
    fe_copy(r, in);

    // Try subtracting p three times — that's enough for any input < 2^256.
    for (int pass = 0; pass < 3; pass++) {
        // p = 2^255 - 19. In limbs (LE): [0xffffffed, 0xffffffff*6, 0x7fffffff].
        long borrow = 0;
        fe tmp;
        long v = (long)r[0] - 0xffffffedL;
        tmp[0] = (uint)(v & 0xffffffffL);
        borrow = (v < 0) ? 1 : 0;
        for (int i = 1; i < 7; i++) {
            v = (long)r[i] - 0xffffffffL - borrow;
            tmp[i] = (uint)(v & 0xffffffffL);
            borrow = (v < 0) ? 1 : 0;
        }
        v = (long)r[7] - 0x7fffffffL - borrow;
        tmp[7] = (uint)(v & 0xffffffffL);
        if (v >= 0) {
            fe_copy(r, tmp);
        } else {
            break;
        }
    }

    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (uchar)(r[i]);
        out[i*4+1] = (uchar)(r[i] >> 8);
        out[i*4+2] = (uchar)(r[i] >> 16);
        out[i*4+3] = (uchar)(r[i] >> 24);
    }
}

// =============================================================================
// Edwards curve operations (extended coordinates)
// =============================================================================

typedef struct { fe X, Y, Z, T; } point;

// Mixed addition: r = p + q' where q' is the *affine* point precomputed as
// (y-x, y+x, 2d·x·y). Z₂ = 1 collapses `2·Z₁·Z₂` to `2·Z₁` (no multiply), and
// folding 2·d into the table eliminates the `C *= 2d` step. Net cost is
// 7 multiplications vs 9 for the general unified addition.
//
// Safe with p == r: every read of `p` happens before any write to `r`.
static void ed_madd(point* r, const point* p, const fe q_ymx, const fe q_ypx, const fe q_t2d) {
    fe A, B, C, D, E, F, G, H, t1;

    fe_sub(t1, p->Y, p->X);
    fe_mul(A, t1, q_ymx);
    fe_add(t1, p->Y, p->X);
    fe_mul(B, t1, q_ypx);
    fe_mul(C, p->T, q_t2d);
    fe_add(D, p->Z, p->Z);          // 2·Z₁ (Z₂ = 1)
    fe_sub(E, B, A);
    fe_sub(F, D, C);
    fe_add(G, D, C);
    fe_add(H, B, A);
    fe_mul(r->X, E, F);
    fe_mul(r->Y, G, H);
    fe_mul(r->T, E, H);
    fe_mul(r->Z, F, G);
}

// Compute `scalar · B` using a 4-bit windowed precomputed table of affine
// points.
//
// `base_table` holds 64 × 15 entries; entry `(i, k-1)` is the affine triple
// (y−x, y+x, 2d·x·y) of `k · 16ⁱ · B`. Each loop iteration consumes 4 bits of
// the scalar and, when non-zero, performs one mixed addition — 64 iterations
// total, 7 muls per add. Compared to the previous 2-bit projective version
// (128 iterations × 9 muls) this is roughly 2.5× cheaper for scalar mult.
static void scalar_mult_base(
    const uchar scalar[32],
    point* result,
    __global const uint* base_table)
{
    fe_zero(result->X);
    fe_one (result->Y);
    fe_one (result->Z);
    fe_zero(result->T);

    for (int i = 0; i < 64; i++) {
        int v = (scalar[i >> 1] >> ((i & 1) << 2)) & 0xf;
        if (v != 0) {
            __global const uint* entry = base_table + (i * 15 + (v - 1)) * 24;
            fe ymx, ypx, t2d;
            for (int j = 0; j < 8; j++) ymx[j] = entry[0  + j];
            for (int j = 0; j < 8; j++) ypx[j] = entry[8  + j];
            for (int j = 0; j < 8; j++) t2d[j] = entry[16 + j];
            ed_madd(result, result, ymx, ypx, t2d);
        }
    }
}

// RFC 8032 point compression: encode y, set the high bit of the last byte
// to the parity of x.
static void point_compress(uchar out[32], const point* p) {
    fe zinv, x, y;
    fe_invert(zinv, p->Z);
    fe_mul(x, p->X, zinv);
    fe_mul(y, p->Y, zinv);
    fe_to_bytes(out, y);
    uchar x_bytes[32];
    fe_to_bytes(x_bytes, x);
    out[31] |= (x_bytes[0] & 1) << 7;
}

// =============================================================================
// Ed25519 key derivation
// =============================================================================
//
// From an Ed25519 seed: hash with SHA-512, clamp the lower 32 bytes, then
// compute scalar * B and serialize.

static void ed25519_pubkey(
    const uchar seed[32],
    uchar pubkey[32],
    __global const uint* base_table)
{
    uchar h[64];
    sha512_32(seed, h);

    uchar scalar[32];
    for (int i = 0; i < 32; i++) scalar[i] = h[i];
    scalar[0]  &= 248;
    scalar[31] &= 127;
    scalar[31] |=  64;

    point P;
    scalar_mult_base(scalar, &P, base_table);
    point_compress(pubkey, &P);
}

// =============================================================================
// Pattern matching
// =============================================================================
//
// For prefix mode, the host precomputes the CID byte-range [cid_lo, cid_hi)
// corresponding to all IPNS names that start with the user's prefix. This
// lets us skip base36 encoding entirely.
//
// For substring mode, we encode the 40-byte CID in base36 (61 characters) and
// run a naive substring search.

// Build the 40-byte CID for a given Ed25519 public key.
static void build_cid(uchar cid[40], const uchar pubkey[32]) {
    cid[0] = 0x01; cid[1] = 0x72; cid[2] = 0x00; cid[3] = 0x24;
    cid[4] = 0x08; cid[5] = 0x01; cid[6] = 0x12; cid[7] = 0x20;
    for (int i = 0; i < 32; i++) cid[8 + i] = pubkey[i];
}

// Compare a private 40-byte big-endian buffer to one in __global memory.
static int cid_cmp_g(const uchar a[40], __global const uchar* b) {
    for (int i = 0; i < 40; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return  1;
    }
    return 0;
}

// Encode CID as 61 base36 lowercase characters (no multibase prefix).
static void cid_to_base36(uchar out[61], const uchar cid[40]) {
    const uchar alphabet[36] = {
        '0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j',
        'k','l','m','n','o','p','q','r','s','t',
        'u','v','w','x','y','z'
    };

    // Process 40 bytes as 10 big-endian uint32 limbs.
    uint limbs[10];
    for (int i = 0; i < 10; i++) {
        limbs[i] = ((uint)cid[i*4+0] << 24)
                 | ((uint)cid[i*4+1] << 16)
                 | ((uint)cid[i*4+2] <<  8)
                 | ((uint)cid[i*4+3]);
    }
    uchar digits[61];
    int n = 0;
    while (n < 61) {
        ulong rem = 0;
        int all_zero = 1;
        for (int i = 0; i < 10; i++) {
            ulong cur = (rem << 32) | limbs[i];
            limbs[i] = (uint)(cur / 36);
            rem = cur % 36;
            if (limbs[i] != 0) all_zero = 0;
        }
        digits[n++] = alphabet[rem];
        if (all_zero) break;
    }
    int pad = 61 - n;
    for (int i = 0; i < pad; i++) out[i] = '0';
    for (int i = 0; i < n; i++) out[pad + i] = digits[n - 1 - i];
}

// =============================================================================
// Main kernel
// =============================================================================

// Mode codes match `gpu::Mode` on the host side.
#define MODE_PREFIX    0u
#define MODE_SUBSTRING 1u

__kernel void search(
    __global const uchar* base_seed,        // 32 bytes
    ulong  nonce_offset,                    // unique per dispatch
    uint   mode,                            // MODE_PREFIX or MODE_SUBSTRING
    __global const uchar* cid_lo,           // n_ranges × 40 bytes (prefix mode)
    __global const uchar* cid_hi,           // n_ranges × 40 bytes (prefix mode)
    uint   n_ranges,                        // number of (lo, hi) pairs
    __global const uchar* needle,           // substring needle
    uint   needle_len,                      // 0 if none
    __global const uint*  base_table,       // 64×15 affine table entries (3 fe each)
    __global volatile uint* found_flag,
    __global uchar* result_seed,            // 32 bytes
    __global uchar* result_pubkey)          // 32 bytes
{
    // Early-out if another work-item already claimed the win. A plain volatile
    // load is enough — at worst we waste work, never produce a wrong answer.
    if (*found_flag != 0u) return;

    ulong gid = (ulong)get_global_id(0);
    ulong nonce = nonce_offset + gid;

    uchar seed[32];
    for (int i = 0; i < 32; i++) seed[i] = base_seed[i];
    for (int i = 0; i < 8; i++) seed[i] ^= (uchar)(nonce >> (i * 8));

    uchar pubkey[32];
    ed25519_pubkey(seed, pubkey, base_table);

    uchar cid[40];
    build_cid(cid, pubkey);

    int hit = 0;
    if (mode == MODE_PREFIX) {
        // Test the CID against each (lo, hi) range in turn; any hit wins.
        for (uint r = 0; r < n_ranges; r++) {
            uint base = r * 40u;
            if (cid_cmp_g(cid, cid_lo + base) >= 0 &&
                cid_cmp_g(cid, cid_hi + base) <  0) {
                hit = 1;
                break;
            }
        }
    } else { // MODE_SUBSTRING
        uchar name62[62];
        name62[0] = 'k';
        cid_to_base36(name62 + 1, cid);
        if (needle_len > 0 && needle_len <= 62) {
            for (uint start = 0; start + needle_len <= 62; start++) {
                int match = 1;
                for (uint k = 0; k < needle_len; k++) {
                    if (name62[start + k] != needle[k]) { match = 0; break; }
                }
                if (match) { hit = 1; break; }
            }
        }
    }

    if (hit) {
        if (atomic_xchg(found_flag, 1u) == 0u) {
            for (int i = 0; i < 32; i++) result_seed[i]   = seed[i];
            for (int i = 0; i < 32; i++) result_pubkey[i] = pubkey[i];
        }
    }
}
