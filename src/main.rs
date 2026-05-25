mod cpu;
mod gpu;
mod ipns;
mod matcher;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use crossbeam_channel::unbounded;

use crate::cpu::Match;
use crate::matcher::{Matcher, Mode};

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Backend {
    /// Try GPU first; fall back to CPU if no OpenCL GPU is available.
    Auto,
    /// Force the CPU backend.
    Cpu,
    /// Force the OpenCL GPU backend.
    Gpu,
}

#[derive(Parser, Debug)]
#[command(
    name = "ipns-vanity",
    about = "Find Ed25519 keys whose IPNS name matches a pattern.",
    long_about = "Search Ed25519 keypairs whose libp2p IPNS name matches a pattern.\n\n\
        Every Ed25519 IPNS name begins with the fixed 12-character string\n\
        'k51qzi5uqu5d' (the multibase tag plus the constant CID prefix bytes).\n\
        In --mode prefix the user pattern is matched immediately *after* those\n\
        12 characters; substring and regex modes match anywhere in the full name.\n\n\
        Prefix patterns may start with a regex-style character class to enumerate\n\
        acceptable first characters, e.g.\n  \
          ipns-vanity '[ghj]abc'      # gabc | habc | jabc\n  \
          ipns-vanity '[g-m]42'       # g42 | h42 | ... | m42\n\
        This is especially useful in prefix mode since only the 7 characters\n\
        g..m are reachable as the first variable position."
)]
struct Args {
    /// One or more patterns to search for; a match against any of them counts
    /// as a hit. Base36 lowercase (0-9, a-z) for prefix and substring modes;
    /// arbitrary regexes for regex mode. Prefix mode also accepts `[abc]` or
    /// `[a-z]` character classes at any position inside a pattern.
    #[arg(required = true, num_args = 1..)]
    patterns: Vec<String>,

    /// Match mode.
    #[arg(short, long, value_enum, default_value_t = Mode::Prefix)]
    mode: Mode,

    /// Compute backend.
    #[arg(short, long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,

    /// Number of CPU threads (0 = use all available cores).
    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    /// Stop after finding this many matches.
    #[arg(short = 'n', long, default_value_t = 3)]
    count: usize,

    /// GPU global work size per dispatch (larger = better throughput, but each
    /// dispatch takes longer before reacting to stop signals). Ignored when
    /// `--no-auto-batch` is *not* set, since auto-calibration picks a value.
    #[arg(long, default_value_t = 1 << 20)]
    gpu_batch: usize,

    /// Skip the 3–5 s startup benchmark that picks the best GPU batch size,
    /// and use `--gpu-batch` directly.
    #[arg(long)]
    no_auto_batch: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let matcher = Arc::new(Matcher::new(args.mode, &args.patterns)?);
    let stop = Arc::new(AtomicBool::new(false));
    let counter = Arc::new(AtomicU64::new(0));
    let (tx, rx) = unbounded::<Match>();

    {
        let stop = stop.clone();
        ctrlc::set_handler(move || {
            eprintln!("\n[ipns-vanity] caught Ctrl+C, shutting down...");
            stop.store(true, Ordering::Relaxed);
        })?;
    }

    // Resolve the backend choice. For Auto, try GPU first.
    let chosen = match args.backend {
        Backend::Cpu => ChosenBackend::Cpu,
        Backend::Gpu => ChosenBackend::Gpu,
        Backend::Auto => {
            // Regex mode is CPU-only.
            if matches!(args.mode, Mode::Regex) {
                eprintln!("[ipns-vanity] regex mode runs on CPU");
                ChosenBackend::Cpu
            } else if gpu::pick_device(gpu::Backend::Gpu).is_ok() {
                ChosenBackend::Gpu
            } else {
                eprintln!("[ipns-vanity] no OpenCL GPU found, falling back to CPU");
                ChosenBackend::Cpu
            }
        }
    };
    eprintln!("[ipns-vanity] backend: {chosen:?}");
    eprintln!(
        "[ipns-vanity] pattern{}: {} ({:?})",
        if args.patterns.len() == 1 { "" } else { "s" },
        args.patterns.join(" | "),
        args.mode,
    );

    // Stats reporter thread.
    let stats_handle = {
        let stop = stop.clone();
        let counter = counter.clone();
        std::thread::Builder::new()
            .name("stats".into())
            .spawn(move || stats_loop(counter, stop))?
    };

    // Worker thread runs whichever backend was chosen. The GPU path needs a
    // larger stack than Rust's default 2 MiB because the NVIDIA OpenCL runtime
    // puts the kernel's verifier and several large per-call buffers on the
    // calling thread's stack.
    let worker_handle = {
        let matcher = matcher.clone();
        let stop = stop.clone();
        let counter = counter.clone();
        let tx = tx.clone();
        let threads = args.threads;
        let batch = args.gpu_batch;
        let auto_batch = !args.no_auto_batch;
        std::thread::Builder::new()
            .name("search".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || -> Result<()> {
                match chosen {
                    ChosenBackend::Cpu => {
                        cpu::run(matcher, stop, counter, tx, threads);
                        Ok(())
                    }
                    ChosenBackend::Gpu => gpu::run(
                        matcher,
                        stop,
                        counter,
                        tx,
                        gpu::Backend::Gpu,
                        batch,
                        auto_batch,
                    ),
                }
            })?
    };
    drop(tx); // so rx closes when all senders are dropped

    let start = Instant::now();
    let mut found = 0usize;
    for m in rx.iter() {
        found += 1;
        report_match(&m, found, start.elapsed());
        if found >= args.count {
            stop.store(true, Ordering::Relaxed);
            break;
        }
    }

    stop.store(true, Ordering::Relaxed);
    if let Ok(Err(err)) = worker_handle.join() {
        eprintln!("\n[ipns-vanity] worker error: {err}");
    }
    let _ = stats_handle.join();

    if found == 0 {
        eprintln!("\n[ipns-vanity] no matches found");
    } else {
        eprintln!("\n[ipns-vanity] done: {found}/{} matches", args.count);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ChosenBackend {
    Cpu,
    Gpu,
}

fn report_match(m: &Match, idx: usize, elapsed: Duration) {
    let name = std::str::from_utf8(&m.name).unwrap_or("<non-utf8>");
    let protobuf_hex = libp2p_privkey_hex(&m.seed, &m.pubkey);
    println!("\n[match #{idx}] elapsed {:>5.1}s", elapsed.as_secs_f64());
    println!("  name:    {name}");
    println!("  seed:    {}", hex::encode(m.seed));
    println!("  pubkey:  {}", hex::encode(m.pubkey));
    println!();
    println!("  import into IPFS (libp2p-protobuf format):");
    println!("    echo {protobuf_hex} | xxd -r -p | ipfs key import <NAME> -");
}

/// Build the libp2p `PrivateKey` protobuf message for an Ed25519 keypair and
/// hex-encode it. The byte layout is:
///
///   0x08, 0x01,         field 1 (KeyType varint) = Ed25519
///   0x12, 0x40,         field 2 (Data length-delim) = 64 bytes
///   <seed[32]>          private seed
///   <pubkey[32]>        public key
///
/// This is exactly what `ipfs key import` consumes when given binary stdin
/// (the default `libp2p-protobuf-cleartext` format).
fn libp2p_privkey_hex(seed: &[u8; 32], pubkey: &[u8; 32]) -> String {
    let mut buf = [0u8; 68];
    buf[..4].copy_from_slice(&[0x08, 0x01, 0x12, 0x40]);
    buf[4..36].copy_from_slice(seed);
    buf[36..].copy_from_slice(pubkey);
    hex::encode(buf)
}

fn stats_loop(counter: Arc<AtomicU64>, stop: Arc<AtomicBool>) {
    let start = Instant::now();
    let mut last = (start, 0u64);
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(1));
        let now = Instant::now();
        let total = counter.load(Ordering::Relaxed);
        let dt = now.duration_since(last.0).as_secs_f64().max(1e-9);
        let inst_rate = (total - last.1) as f64 / dt;
        let avg_rate = total as f64 / now.duration_since(start).as_secs_f64().max(1e-9);
        eprint!(
            "\r[{:>5.0}s] tried={} rate={} avg={}     ",
            now.duration_since(start).as_secs_f64(),
            humanize(total as f64),
            humanize_rate(inst_rate),
            humanize_rate(avg_rate),
        );
        use std::io::Write;
        let _ = std::io::stderr().flush();
        last = (now, total);
    }
}

fn humanize(n: f64) -> String {
    const U: [&str; 5] = ["", "K", "M", "G", "T"];
    let mut x = n;
    let mut i = 0;
    while x >= 1000.0 && i + 1 < U.len() {
        x /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{x:.0}")
    } else {
        format!("{x:.2}{}", U[i])
    }
}

fn humanize_rate(n: f64) -> String {
    format!("{}/s", humanize(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libp2p_privkey_layout() {
        let seed = [0x11u8; 32];
        let pubkey = [0x22u8; 32];
        let hex = libp2p_privkey_hex(&seed, &pubkey);
        // 4 header bytes + 32 seed + 32 pubkey = 68 bytes = 136 hex chars.
        assert_eq!(hex.len(), 136);
        // Header: 08=KeyType field tag, 01=Ed25519, 12=Data field tag, 40=length 64.
        assert!(hex.starts_with("08011240"));
        // Followed by the seed (32 bytes of 0x11) and the pubkey (32 bytes of 0x22).
        assert_eq!(&hex[8..8 + 64], &"11".repeat(32));
        assert_eq!(&hex[8 + 64..], &"22".repeat(32));
    }

    #[test]
    fn libp2p_privkey_round_trips_via_ed25519_dalek() {
        // Generating from a known seed and round-tripping through the protobuf
        // layout we emit should reproduce that seed and the matching pubkey.
        let seed = [
            0xa1, 0x9c, 0x35, 0x6d, 0x42, 0xb0, 0xc4, 0x77, 0x2f, 0x1d, 0x65, 0x99, 0xee, 0xaa,
            0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0x00, 0x0f,
            0x1f, 0x2f, 0x3f, 0x4f,
        ];
        let pubkey = ed25519_dalek::SigningKey::from_bytes(&seed)
            .verifying_key()
            .to_bytes();
        let bytes = hex::decode(libp2p_privkey_hex(&seed, &pubkey)).unwrap();
        assert_eq!(&bytes[..4], &[0x08, 0x01, 0x12, 0x40]);
        assert_eq!(&bytes[4..36], &seed[..]);
        assert_eq!(&bytes[36..], &pubkey[..]);
    }

    #[test]
    fn humanize_picks_unit() {
        assert_eq!(humanize(900.0), "900");
        assert_eq!(humanize(1500.0), "1.50K");
        assert_eq!(humanize(2_500_000.0), "2.50M");
    }
}
