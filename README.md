# ipns-vanity

Find Ed25519 keypairs whose [IPNS](https://docs.ipfs.tech/concepts/ipns/) name **or** [libp2p peer ID](https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md) starts with — or contains — a string you choose. Runs on the GPU through OpenCL with a multi-threaded CPU fallback, and prints a ready-to-paste `ipfs key import` command for every match.

```
$ ipns-vanity -b gpu '[g-m]vk4'
[ipns-vanity] backend: Gpu
[ipns-vanity] pattern: [g-m]vk4 (Prefix)
[gpu] using NVIDIA Corporation / NVIDIA GeForce RTX 3060
[gpu] auto-calibrating batch size (budget ~5s)
[gpu]   batch=    262144  dispatch=  138ms  rate=  1.90 M/s
[gpu] chose batch=262144 (~1.90 M/s)

[match #1] elapsed   0.1s
  name:    k51qzi5uqu5divk4r98fa7ssnhtk1re616zt1p7phl6wvie50z805upplrc6ns
  seed:    a4528f27188ebf3524ca8cfa17ad75291e19e2383ede1918be77ec5bb117e5fc
  pubkey:  6c12bcb640a5cc6819642d7ac0db440e8d3c431dfd4bcef3e5321211761aebb8

  import into IPFS (libp2p-protobuf format):
    echo 08011240a4528f27...1aebb8 | xxd -r -p | ipfs key import <NAME> -
```

## What it does

An Ed25519 key has two common public identifiers, both derived from the same 32-byte public key:

| Identifier | Format                          | Example                                                        | Fixed prefix     |
|------------|---------------------------------|----------------------------------------------------------------|------------------|
| IPNS name  | base36 multibase of CIDv1       | `k51qzi5uqu5d…` (62 chars)                                     | `k51qzi5uqu5d`   |
| Peer ID    | base58btc of identity multihash | `12D3KooW…` (52 chars)                                         | `12D3KooW`       |

`ipns-vanity` brute-forces random seeds, derives their public keys, encodes them as IPNS names **and/or** peer IDs, and prints any that match your patterns. By default only the IPNS name is searched; choose with `--target ipns | peerid | both`.

## Install

You need:

- A recent **Rust toolchain** (edition 2024, so ≥ 1.85).
- The **OpenCL ICD loader** (`libOpenCL.so.1`). On Debian / Ubuntu: `sudo apt install ocl-icd-libopencl1`.
- For GPU mode, a working OpenCL driver — NVIDIA, AMD, or Intel.

Build:

```sh
git clone https://github.com/vk496/ipns-vanity
cd ipns-vanity
cargo build --release
```

The release binary is at `target/release/ipns-vanity`.

> The `build.rs` automatically symlinks `libOpenCL.so` into `OUT_DIR` if your distro ships only the versioned `libOpenCL.so.1`, so you don't usually need the `-dev` package.

## Quick start

```sh
# Auto: tries GPU, falls back to CPU if no OpenCL GPU is available.
ipns-vanity gvk4

# Force the backend.
ipns-vanity -b gpu gvk4
ipns-vanity -b cpu gvk4

# Substring instead of prefix.
ipns-vanity -m substring beef

# Regex (CPU only).
ipns-vanity -m regex '^k51qzi5uqu5dh.*42$'

# Several matches before stopping (default is 3).
ipns-vanity -n 5 gvk4

# Multiple patterns — any match wins, which multiplies your odds.
ipns-vanity gvk4 hvi6 jvk2
ipns-vanity -m substring beef cafe d00d

# Vanity peer ID instead of (or in addition to) IPNS.
ipns-vanity -t peerid Satoshi               # peer ID starts with 12D3KooWSatoshi
ipns-vanity -t both gvk4 Bh                 # any IPNS- or peer-prefix hit wins
ipns-vanity -t both Foo Bar                 # patterns auto-routed by alphabet
```

## Pattern modes

| Mode (`-m`)  | What it matches                                              | Backend            |
|--------------|--------------------------------------------------------------|--------------------|
| `prefix`     | Right after the constant `k51qzi5uqu5d`                      | CPU + GPU          |
| `substring`  | Anywhere inside the 62-character name                        | CPU + GPU          |
| `regex`      | A regex against the full name (incl. the `k51qzi5uqu5d`)     | CPU only           |

IPNS patterns use base36 lowercase (digits and `a`–`z`). Peer-ID patterns use base58btc (digits, upper- and lower-case letters minus `0`, `O`, `I`, `l`). Regex is unrestricted but only runs on CPU.

### Multiple patterns

You can pass any number of patterns and a match against **any** of them counts as a hit. The kernel tests them all in parallel, so widening the search this way is essentially free and multiplies your effective odds.

```sh
ipns-vanity gvk4 hvi6 mvk2           # any of the three IPNS prefixes
ipns-vanity -m substring beef cafe    # either substring anywhere
ipns-vanity -m regex 'aa$' '42$'      # OR-combined into one regex
```

### Selecting the target

`--target / -t` picks which identifier(s) the patterns apply to:

| Target          | Behaviour                                                                                                |
|-----------------|----------------------------------------------------------------------------------------------------------|
| `ipns` (default)| Patterns match the base36 IPNS name (`k51qzi5uqu5d…`).                                                  |
| `peerid`        | Patterns match the base58btc peer ID (`12D3KooW…`).                                                     |
| `both`          | Each pattern is **auto-routed** by its alphabet — lowercase / digits go to IPNS, uppercase letters and base58 digits to peer-id. Patterns that fit both alphabets are tried as both; ones that fit neither are rejected. |

```sh
ipns-vanity hvk                    # IPNS only (default)
ipns-vanity -t peerid Sat Eve Bob  # peer-id only, three prefixes
ipns-vanity -t both hvk Bh         # IPNS hvk OR peer-id Bh
```

A peer ID's identity-multihash bytes are literally a sub-range of the CID bytes (`cid[2..40]`), so the GPU kernel does the comparison via its existing range comparator with no extra encoder work — adding `--target both` is essentially free on the GPU.

Caveats:
- **GPU substring + peer-id**: not yet supported (no base58 encoder in the kernel). Use `--backend cpu` for that combo.
- The fixed prefix for peer IDs is `12D3KooW` and the 9th character is restricted to `9`–`T` (analogous to the IPNS `g`–`m` restriction). Unreachable patterns are detected and reported up-front.

### The reachable-alphabet caveat

The character right after the fixed prefix is variable but its alphabet is restricted, because the 32-byte public-key range only covers a fraction of one base-36/58 digit slot at that position:

- **IPNS** (`k51qzi5uqu5d` + …): only `g` through `m` are reachable.
- **Peer ID** (`12D3KooW` + …): only `9` through `T` are reachable.

`ipns-vanity` checks this up front:

```
$ ipns-vanity 1abc
Error: ipns prefix 'k51qzi5uqu5d1abc' is not achievable: at position 12
the character must be between 'g' and 'm' (saw '1').
…

$ ipns-vanity --peer-id zzz
Error: peer-id prefix '12D3KooWzzz' is not achievable: at position 8
the character must be between '9' and 'T' (saw 'z').
…
```

### Character classes (prefix mode)

You can widen the search across multiple acceptable characters at *any* position in the prefix:

```sh
ipns-vanity '[gh]abc'        # gabc | habc
ipns-vanity '[g-m]42'        # g42 | h42 | … | m42
ipns-vanity '[g-m]vk4[6a3]'  # 21 combinations: g..m × 6/a/3
```

The host expands the pattern into a Cartesian product, drops any unreachable variants (e.g. anything starting outside `g..m`), then ships the resulting list of CID byte-ranges to the kernel. The kernel OR-tests them, so widening rarely costs anything measurable — and on the search side it directly multiplies your odds of a hit. There's a 1024-variant cap to catch runaway patterns.

## Output

Each match shows both identifiers plus the seed/pubkey hex and a paste-ready import command:

```
[match #1] elapsed   0.2s
  ipns:    k51qzi5uqu5dgvqvp9bm7k2l6ih0zj6r93x0nmvr90i5i86ycseij1r6szsv5f
  peer:    12D3KooWBhpApNFqmiBEKv1QYuJTioqqzbNCE9mSuxjnjrKTJiYa
  seed:    8b4f1a1b68dcc993be5b68083c4dfd15f76ae9997cb7ce1073377bce77a858f5
  pubkey:  1c09b5395e59a64b9507b9a994d2c050202bfa60c78d55d1f8d4f77553f79fb3

  import into IPFS (libp2p-protobuf format):
    echo 080112408b4f1a1b…f79fb3 | xxd -r -p | ipfs key import <NAME> -
```

## Importing the key into IPFS

For every match, `ipns-vanity` prints the libp2p `PrivateKey` protobuf in hex (this is the canonical `libp2p-protobuf-cleartext` format that `ipfs key import` accepts on stdin). Copy-paste the line:

```sh
echo 08011240<seed_hex><pubkey_hex> | xxd -r -p | ipfs key import my-vanity-name -
```

The bytes are:

```
0x08, 0x01,    field 1 (KeyType varint)         = Ed25519
0x12, 0x40,    field 2 (Data length-delimited)  = 64 bytes
<seed[32]>     32-byte Ed25519 seed
<pubkey[32]>   32-byte Ed25519 public key
```

Verify it landed:

```sh
$ ipfs key list -l
k51qzi5uqu5d…   self
k51qzi5uqu5d…   my-vanity-name
```

## Performance

Measured throughput for the **full Ed25519 + IPNS encode + match** pipeline (no early-exit shortcuts):

| Backend                                  | Rate            |
|------------------------------------------|-----------------|
| CPU (16 threads, AMD Ryzen-class)        | ~360 K keys/s   |
| GPU (NVIDIA RTX 3060, OpenCL)            | ~14.5 M keys/s  |

GPU mode auto-tunes the work-group size at startup (3–5 s sweep) to find the best throughput / latency trade-off for your device. Disable with `--no-auto-batch` to use `--gpu-batch` directly.

Rough probabilities of a single match for an `N`-character pattern past the fixed prefix:

| `N` | Trials needed (≈)  | GPU @ 14.5 M/s      | CPU @ 360 K/s       |
|-----|--------------------|---------------------|---------------------|
| 3   | 4·10³              | < 0.01 s            | 0.01 s              |
| 4   | 1·10⁵              | < 0.01 s            | 0.3 s               |
| 5   | 5·10⁶              | 0.3 s               | 14 s                |
| 6   | 2·10⁸              | 14 s                | 9 min               |
| 7   | 7·10⁹              | 8 min               | 5.5 h               |
| 8   | 2.5·10¹¹           | 4.8 h               | 8 days              |
| 9   | 9·10¹²             | 7 days              | 290 days            |

(For prefix mode position 12, replace `36` with `7` because only `g`–`m` are reachable.)

## CLI reference

```
Usage: ipns-vanity [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  Pattern to search for. Base36 lowercase (0-9, a-z) for prefix
             and substring modes; arbitrary regex for regex mode. Prefix
             mode also accepts a leading [abc] or [a-z] character class.

Options:
  -m, --mode <MODE>            Match mode (prefix | substring | regex) [default: prefix]
  -b, --backend <BACKEND>      Compute backend (auto | cpu | gpu)         [default: auto]
  -t, --threads <THREADS>      CPU thread count, 0 = all cores            [default: 0]
  -n, --count <COUNT>          Stop after this many matches                [default: 1]
      --gpu-batch <GPU_BATCH>  GPU work items per dispatch                 [default: 1048576]
      --no-auto-batch          Skip the startup batch-size benchmark
  -h, --help                   Print help
```

## Tests

```sh
cargo test
```

The encoder is cross-checked against the canonical [`multibase`](https://crates.io/crates/multibase) crate so the output is guaranteed to match anything else in the IPFS ecosystem.

## How it works (briefly)

- **CPU**: each worker thread has its own ChaCha20 RNG; it loops over `SigningKey::from_bytes` → `to_bytes` → IPNS encode → match. Counter updates are batched to keep the shared atomic out of the hot loop.
- **GPU**: the OpenCL kernel does the full Ed25519 (SHA-512 + scalar mult on a base-point) + IPNS encode + match per work-item. Prefix mode is translated host-side into one or more 40-byte CID ranges so the kernel only does a byte comparison instead of running its base36 encoder. Every hit reported by the GPU is re-derived on CPU before being shown — a buggy or unstable kernel can never produce a wrong public key.
- **Fast scalar multiplication**: the host pre-computes a 4-bit windowed table of base-point multiples in *affine* form (64 positions × 15 multiples ≈ 90 KiB), with each entry stored as `(y−x, y+x, 2d·x·y)`. The kernel processes the scalar four bits at a time — 64 iterations of one mixed addition each (7 multiplications instead of the 9 you'd need for a general Edwards add). Two cumulative wins: a textbook double-and-add would do 256 doublings + ~128 additions; we do at most 64 additions and each is cheaper. On the RTX 3060 this brings throughput from ~2 M/s up to ~14.5 M/s.
- **Auto-batch**: the GPU backend dispatches a handful of work-group sizes at startup with match-detection disabled (substring mode + empty needle) so the timings reflect raw throughput, not early-exit shortcuts. It then picks the smallest batch that's within 5 % of the best throughput, balancing speed and Ctrl+C responsiveness.

## License

[GPLv3](LICENSE).
