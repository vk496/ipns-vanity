//! OpenCL backend.
//!
//! The kernel does the heavy lifting (Ed25519 derivation, IPNS encoding,
//! pattern matching). The host side:
//!   * picks the requested device,
//!   * computes the curve constants (`B` in extended coordinates and `2*d`),
//!   * translates a user prefix into a CID byte range so the kernel can do a
//!     range comparison instead of base36 encoding,
//!   * pumps the dispatch loop, refreshing the seed nonce each iteration,
//!   * verifies every reported hit on the CPU before forwarding it — so a
//!     buggy or numerically unstable GPU result never reaches the user.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use crossbeam_channel::Sender;
use ed25519_dalek::SigningKey;
use ocl::{
    Buffer, Context, Device, DeviceType, Kernel, Platform, ProQue, Program, Queue, SpatialDims,
    flags,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::cpu::Match;
use crate::ipns::{IPNS_NAME_LEN, ipns_name};
use crate::matcher::{FIXED_PREFIX, Matcher};

const KERNEL_SRC: &str = include_str!("kernel.cl");

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Gpu,
}

pub fn pick_device(_backend: Backend) -> Result<(Platform, Device)> {
    // `Platform::list()` can panic OR hang depending on the driver state
    // (e.g. NVIDIA ICD registered but the kernel module not loaded). Run it
    // on a worker thread with a short deadline so the auto-fallback logic
    // never blocks the program.
    let (tx, rx) = std::sync::mpsc::channel::<Result<Vec<Platform>>>();
    std::thread::spawn(move || {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(Platform::list))
            .map_err(|_| anyhow!("OpenCL platform query panicked (driver unavailable?)"));
        let _ = tx.send(res);
    });
    let platforms = match rx.recv_timeout(std::time::Duration::from_secs(3)) {
        Ok(r) => r?,
        Err(_) => return Err(anyhow!("OpenCL platform query timed out")),
    };
    for platform in platforms {
        if let Ok(devs) = Device::list(platform, Some(DeviceType::GPU))
            && let Some(dev) = devs.into_iter().next()
        {
            return Ok((platform, dev));
        }
    }
    Err(anyhow!("no OpenCL GPU device found"))
}

pub fn describe_device(device: &Device) -> String {
    let name = device.name().unwrap_or_else(|_| "unknown".into());
    let vendor = device.vendor().unwrap_or_else(|_| "unknown".into());
    format!("{vendor} / {name}")
}

/// Run the OpenCL search loop until `stop` is set.
///
/// When `auto_batch` is true, the function spends 3–5 seconds at startup
/// sweeping a range of work-group sizes to find the throughput sweet spot
/// for the current device; otherwise it uses `global_size` as given.
pub fn run(
    matcher: Arc<Matcher>,
    stop: Arc<AtomicBool>,
    counter: Arc<AtomicU64>,
    tx: Sender<Match>,
    backend: Backend,
    global_size: usize,
    auto_batch: bool,
) -> Result<()> {
    let (platform, device) = pick_device(backend)?;
    eprintln!("[gpu] using {}", describe_device(&device));

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        // Force OpenCL 1.2 semantics. Without this, NVIDIA's compiler infers
        // `__generic` for struct-member access while still defaulting function
        // pointer parameters to `__private`, which makes every call into the
        // field arithmetic look like a cross-address-space cast.
        .cmplr_opt("-cl-std=CL1.2")
        .build(&context)
        .map_err(|e| anyhow!("kernel build failed:\n{e}"))?;

    let pro_que = ProQue::new(context.clone(), queue.clone(), program, Some(global_size));

    // Precompute the 4-bit windowed affine table (~90 KiB) the kernel reads
    // from. Building it eats one inversion per entry (960 total) which takes
    // a tens-of-ms at startup — negligible compared to the search itself.
    let base_table_limbs = curve_constants();

    // mode 0 = prefix (one or more CID ranges OR'd),
    // mode 1 = substring (one or more needles OR'd).
    //
    // Peer-ID prefixes are turned into CID ranges too: a peer-ID multihash is
    // exactly `cid[2..40]`, so we keep the kernel's existing range comparator
    // by prefixing the multihash range with `[0x01, 0x72]` (the constant CIDv1
    // bytes).
    //
    // The kernel receives concatenated needles in `needle_data` plus a
    // parallel `needle_lens` array; for prefix mode those buffers stay empty.
    // Peer-ID substrings/regex aren't supported on the GPU yet — they require
    // base58btc encoding inside the kernel — so we fall back to CPU.
    let (mode_code, cid_lo, cid_hi, needle_data, needle_lens, n_ranges) = match &*matcher {
        Matcher::Prefix { ipns, peer } => {
            let mut lo_all = Vec::with_capacity((ipns.len() + peer.len()) * 40);
            let mut hi_all = Vec::with_capacity((ipns.len() + peer.len()) * 40);
            for p in ipns {
                let (lo, hi) = prefix_to_cid_range(p)?;
                lo_all.extend_from_slice(&lo);
                hi_all.extend_from_slice(&hi);
            }
            for p in peer {
                let (lo, hi) = peer_prefix_to_cid_range(p)?;
                lo_all.extend_from_slice(&lo);
                hi_all.extend_from_slice(&hi);
            }
            (
                0u32,
                lo_all,
                hi_all,
                Vec::new(),
                Vec::new(),
                (ipns.len() + peer.len()) as u32,
            )
        }
        Matcher::Substring { ipns, peer } => {
            if !peer.is_empty() {
                bail!(
                    "peer-id substring patterns aren't supported by the GPU backend; \
                     rerun with --backend cpu"
                );
            }
            let mut data = Vec::new();
            let mut lens: Vec<u32> = Vec::with_capacity(ipns.needles.len());
            for n in &ipns.needles {
                if n.is_empty() || n.len() > 62 {
                    bail!("each substring needle must be between 1 and 62 base36 chars");
                }
                data.extend_from_slice(n);
                lens.push(n.len() as u32);
            }
            (
                1u32,
                vec![0u8; 40],
                vec![0u8; 40],
                data,
                lens,
                ipns.needles.len() as u32,
            )
        }
        Matcher::Regex { .. } => {
            bail!("regex mode is not supported by the GPU backend; rerun with --backend cpu");
        }
    };

    // ---- Persistent buffers ----------------------------------------------

    let base_seed_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(32)
        .copy_host_slice(&[0u8; 32])
        .build()?;
    let cid_lo_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(cid_lo.len())
        .copy_host_slice(&cid_lo)
        .build()?;
    let cid_hi_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(cid_hi.len())
        .copy_host_slice(&cid_hi)
        .build()?;
    // Needle buffers: OpenCL needs at least 1 byte each.
    let needle_data_storage: Vec<u8> = if needle_data.is_empty() {
        vec![0u8]
    } else {
        needle_data.clone()
    };
    let needle_lens_storage: Vec<u32> = if needle_lens.is_empty() {
        vec![0u32]
    } else {
        needle_lens.clone()
    };
    let needle_data_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(needle_data_storage.len())
        .copy_host_slice(&needle_data_storage)
        .build()?;
    let needle_lens_buf = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(needle_lens_storage.len())
        .copy_host_slice(&needle_lens_storage)
        .build()?;
    let n_needles = needle_lens.len() as u32;
    let base_table_buf = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(base_table_limbs.len())
        .copy_host_slice(&base_table_limbs)
        .build()?;
    let found_flag_buf = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(1)
        .copy_host_slice(&[0u32])
        .build()?;
    let result_seed_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(32)
        .build()?;
    let result_pubkey_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(32)
        .build()?;

    let mut kernel = pro_que
        .kernel_builder("search")
        .arg(&base_seed_buf) // 0
        .arg(0u64) // 1  nonce_offset
        .arg(mode_code) // 2  mode
        .arg(&cid_lo_buf) // 3  cid_lo (concat of n_ranges × 40 bytes)
        .arg(&cid_hi_buf) // 4  cid_hi
        .arg(n_ranges) // 5  n_ranges
        .arg(&needle_data_buf) // 6  concatenated substring needles
        .arg(&needle_lens_buf) // 7  per-needle lengths
        .arg(n_needles) // 8  number of needles
        .arg(&base_table_buf) // 9  4-bit windowed affine table
        .arg(&found_flag_buf) // 10
        .arg(&result_seed_buf) // 11
        .arg(&result_pubkey_buf) // 12
        .global_work_size(global_size)
        .build()?;

    let mut rng = ChaCha20Rng::from_entropy();
    let mut base_seed = [0u8; 32];

    // One dispatch: refresh the seed, run the kernel, and forward any match.
    // Returns the kernel time so the caller can compute throughput.
    let mut do_dispatch = |kernel: &mut Kernel, batch: usize| -> Result<Duration> {
        rng.fill_bytes(&mut base_seed);
        base_seed_buf.write(&base_seed[..]).enq()?;
        let zero = [0u32; 1];
        found_flag_buf.write(&zero[..]).enq()?;

        let t = Instant::now();
        unsafe {
            kernel.enq()?;
        }
        queue.finish()?;
        let dt = t.elapsed();

        counter.fetch_add(batch as u64, Ordering::Relaxed);

        let mut found_buf = [0u32; 1];
        found_flag_buf.read(&mut found_buf[..]).enq()?;
        if found_buf[0] != 0 {
            let mut result_seed = [0u8; 32];
            let mut result_pubkey = [0u8; 32];
            result_seed_buf.read(&mut result_seed[..]).enq()?;
            result_pubkey_buf.read(&mut result_pubkey[..]).enq()?;

            // Re-derive on CPU and re-check the match — guards against kernel
            // arithmetic bugs ever surfacing to the user.
            let cpu_pubkey = SigningKey::from_bytes(&result_seed)
                .verifying_key()
                .to_bytes();
            if cpu_pubkey == result_pubkey {
                let mut name_arr = [0u8; IPNS_NAME_LEN];
                name_arr.copy_from_slice(&ipns_name(&cpu_pubkey));
                let peer = crate::peerid::peer_id(&cpu_pubkey);
                if matcher.matches(&name_arr, &peer) {
                    let _ = tx.send(Match {
                        seed: result_seed,
                        pubkey: cpu_pubkey,
                        name: name_arr,
                    });
                }
            } else {
                eprintln!(
                    "[gpu] kernel produced inconsistent public key; ignoring (seed={})",
                    hex::encode(result_seed)
                );
            }
        }
        Ok(dt)
    };

    let mut batch = global_size;
    if auto_batch {
        // Force the kernel into a "no possible match" configuration so the
        // calibration dispatches all run the full Ed25519 + encode pipeline
        // for every work item. Without this the first found match would let
        // later work items bail early and skew the timing.
        kernel.set_arg(2, 1u32)?; // mode = substring
        kernel.set_arg(8, 0u32)?; // n_needles = 0 → no hit
        batch = calibrate(&mut kernel, &mut do_dispatch, batch)?;
        kernel.set_arg(2, mode_code)?;
        kernel.set_arg(8, n_needles)?;
    }
    kernel.set_default_global_work_size(SpatialDims::One(batch));

    while !stop.load(Ordering::Relaxed) {
        do_dispatch(&mut kernel, batch)?;
    }
    Ok(())
}

/// Sweep a handful of work-group sizes and return the one that gave the best
/// throughput. The total time budget is roughly 3–5 seconds: we test in
/// increasing order, stop once a single dispatch takes more than ~1.5 s, and
/// keep the candidate with the highest keys/second.
fn calibrate<F>(kernel: &mut Kernel, do_dispatch: &mut F, start_hint: usize) -> Result<usize>
where
    F: FnMut(&mut Kernel, usize) -> Result<Duration>,
{
    let mut candidates: Vec<usize> = (16..=23).map(|e| 1usize << e).collect(); // 64K..8M
    if !candidates.contains(&start_hint) {
        candidates.push(start_hint);
        candidates.sort_unstable();
    }
    let total_budget = Duration::from_secs(5);
    let start = Instant::now();

    eprintln!(
        "[gpu] auto-calibrating batch size (budget ~{:.0}s)",
        total_budget.as_secs_f64()
    );

    // Warm-up dispatch: absorbs JIT-compile / first-launch costs so the first
    // real measurement isn't an outlier.
    let warmup = candidates[0];
    kernel.set_default_global_work_size(SpatialDims::One(warmup));
    let _ = do_dispatch(kernel, warmup);

    let mut samples: Vec<(usize, f64)> = Vec::new();
    for batch in candidates {
        if start.elapsed() >= total_budget {
            break;
        }
        kernel.set_default_global_work_size(SpatialDims::One(batch));
        let dt = match do_dispatch(kernel, batch) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[gpu]   batch={batch:>10} failed: {e}");
                break;
            }
        };
        let rate = batch as f64 / dt.as_secs_f64();
        eprintln!(
            "[gpu]   batch={batch:>10}  dispatch={:>5.0}ms  rate={:>6.2} M/s",
            dt.as_millis() as f64,
            rate / 1e6,
        );
        samples.push((batch, rate));
        if dt > Duration::from_millis(1500) {
            break; // larger sizes will only hurt responsiveness
        }
    }

    if samples.is_empty() {
        return Err(anyhow!("calibration failed"));
    }
    // Throughput plateaus quickly — among samples within 5 % of the best, pick
    // the smallest, which keeps each dispatch short and Ctrl+C responsive.
    let peak = samples.iter().map(|(_, r)| *r).fold(0f64, f64::max);
    let (chosen, rate) = samples
        .into_iter()
        .find(|(_, r)| *r >= 0.95 * peak)
        .unwrap();
    eprintln!("[gpu] chose batch={chosen} (~{:.2} M/s)", rate / 1e6);
    Ok(chosen)
}

// =============================================================================
// Constants for the kernel
// =============================================================================

/// Pack a 32-byte little-endian field element into eight LE u32 limbs.
fn pack_fe(bytes: &[u8; 32]) -> [u32; 8] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]);
    }
    limbs
}

/// Build the precomputed base-point table the kernel needs.
///
/// The kernel no longer takes 2·d as a runtime constant — it's already folded
/// into every table entry — so this function only returns the table itself.
/// See [`windowed_table`] for the layout.
fn curve_constants() -> Vec<u32> {
    // Ed25519 generator. RFC 8032 §5.1.5 gives the coordinates as big-endian
    // hex; we flip to little-endian so they match the kernel's limb layout.
    let bx_le = reverse_hex("216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A");
    let by_le = reverse_hex("6666666666666666666666666666666666666666666666666666666666666658");
    let d_le = reverse_hex("52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3");

    let bx_l = pack_fe(&bx_le);
    let by_l = pack_fe(&by_le);
    let d_l = pack_fe(&d_le);
    let d2_l = fe::add(&d_l, &d_l);

    windowed_table(&bx_l, &by_l, &d2_l)
}

fn reverse_hex(s: &str) -> [u8; 32] {
    let bytes = hex::decode(s).expect("valid hex");
    assert_eq!(bytes.len(), 32);
    let mut out = [0u8; 32];
    for (i, b) in bytes.iter().rev().enumerate() {
        out[i] = *b;
    }
    out
}

/// Translate an IPNS-name prefix (the bytes that must appear at the start of
/// the encoded name) into a 40-byte big-endian CID range.
///
/// The kernel can then do a single byte-by-byte range comparison instead of
/// running its base36 encoder on every candidate.
fn prefix_to_cid_range(prefix_bytes: &[u8]) -> Result<([u8; 40], [u8; 40])> {
    // The first byte must be the 'k' multibase tag.
    if prefix_bytes.is_empty() || prefix_bytes[0] != b'k' {
        bail!("internal: prefix must start with 'k'");
    }
    // The 11 characters after 'k' must be the constant CID prefix digits.
    let fixed_after_k = &FIXED_PREFIX[1..];
    if prefix_bytes.len() < FIXED_PREFIX.len()
        || &prefix_bytes[1..FIXED_PREFIX.len()] != fixed_after_k
    {
        bail!(
            "internal: prefix does not start with the fixed '{}' bytes",
            std::str::from_utf8(FIXED_PREFIX).unwrap()
        );
    }

    // The IPNS name has 1 multibase char + 61 base36 digits, so its variable
    // portion is up to 61 digits long. The prefix uses `prefix_digits` of
    // those, leaving `tail` zero-padding digits.
    let total_digits = IPNS_NAME_LEN - 1; // 61
    let prefix_digits = prefix_bytes.len() - 1; // chars after 'k'
    if prefix_digits > total_digits {
        bail!("prefix is longer than an IPNS name");
    }
    let tail = total_digits - prefix_digits;

    // Decode the prefix base36 digits as a big integer (lower bound's high digits).
    let mut value = [0u32; 13]; // up to 40 bytes + a couple of guard limbs
    for &c in &prefix_bytes[1..] {
        let digit = base36_decode(c)? as u64;
        // value = value * 36 + digit
        let mut carry = digit;
        for limb in value.iter_mut() {
            let v = (*limb as u64) * 36 + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            bail!("prefix value overflowed the CID");
        }
    }
    // Multiply by 36^tail to push the prefix into the high digits.
    let multiplier = pow36_limbs(tail);
    let lo_full = mul_limbs(&value, &multiplier);

    // Upper bound = lower + 36^tail.
    let one_at_tail = pow36_limbs(tail);
    let hi_full = add_limbs(&lo_full, &one_at_tail);

    // Pack as 40-byte big-endian for the kernel.
    //
    // Note: `lo` can fall slightly below the absolute minimum valid CID for
    // short user patterns (e.g. a single character), because
    //   decode(prefix) · 36^tail  ≤  CID_min
    // and the inequality is strict whenever `CID_min mod 36^tail ≠ 0`. That's
    // harmless: every real CID is ≥ CID_min and the upper-bound test
    // `cid < hi` is what actually selects the prefix.
    let lo_be = limbs_to_be40(&lo_full)?;
    let hi_be = limbs_to_be40(&hi_full)?;
    Ok((lo_be, hi_be))
}

fn base36_decode(c: u8) -> Result<u32> {
    match c {
        b'0'..=b'9' => Ok((c - b'0') as u32),
        b'a'..=b'z' => Ok(10 + (c - b'a') as u32),
        _ => Err(anyhow!("non-base36 char in pattern: '{}'", c as char)),
    }
}

const BASE58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

fn base58_decode(c: u8) -> Result<u32> {
    let idx = BASE58_ALPHABET
        .iter()
        .position(|&b| b == c)
        .ok_or_else(|| anyhow!("non-base58btc char in peer-id pattern: '{}'", c as char))?;
    Ok(idx as u32)
}

/// Translate a peer-ID prefix into a 40-byte big-endian CID range so the
/// kernel's existing range comparator can match it.
///
/// A peer ID is `base58btc(identity_multihash(protobuf_pubkey))` — 52 ASCII
/// chars. The multihash itself is exactly `cid[2..40]` (the CIDv1 prefix
/// `[0x01, 0x72]` lives at `cid[0..2]`). So we decode the user's peer-ID
/// prefix into a 38-byte multihash byte range and then prepend `[0x01, 0x72]`
/// to get a 40-byte CID range. The kernel does the comparison exactly as it
/// does for IPNS prefixes.
fn peer_prefix_to_cid_range(prefix_bytes: &[u8]) -> Result<([u8; 40], [u8; 40])> {
    use crate::peerid::{FIXED_PEER_PREFIX, PEER_ID_LEN};

    if prefix_bytes.len() < FIXED_PEER_PREFIX.len()
        || &prefix_bytes[..FIXED_PEER_PREFIX.len()] != FIXED_PEER_PREFIX
    {
        bail!(
            "internal: peer-id prefix does not start with the fixed '{}' bytes",
            std::str::from_utf8(FIXED_PEER_PREFIX).unwrap()
        );
    }
    if prefix_bytes.len() > PEER_ID_LEN {
        bail!("peer-id prefix is longer than a peer ID ({})", PEER_ID_LEN);
    }

    // base58btc treats each leading '1' as one leading 0x00 byte in the
    // decoded output. Count those and drop them from the digit stream we'll
    // multiply through.
    let leading_ones = prefix_bytes.iter().take_while(|&&c| c == b'1').count();
    let digit_bytes = &prefix_bytes[leading_ones..];

    let mh_len = 38usize;
    let prefix_value_bytes = leading_ones;
    let prefix_digit_count = digit_bytes.len();
    let total_digit_slots = mh_len - prefix_value_bytes; // bytes available below the leading zeros

    // Decode the (non-leading-zero) prefix base58 digits as a big integer.
    let mut value = [0u32; 13];
    for &c in digit_bytes {
        let digit = base58_decode(c)? as u64;
        let mut carry = digit;
        for limb in value.iter_mut() {
            let v = (*limb as u64) * 58 + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            bail!("peer-id prefix value overflowed the multihash");
        }
    }

    // Bound the variable tail as a base58 integer: 58^(total_digit_slots
    // worth of digits - prefix_digit_count digits already consumed).
    //
    // Working in base58 over `mh_len - leading_ones` bytes gives a value range
    // of `[0, 256^(mh_len - leading_ones))`. The user pinned the most-
    // significant `prefix_digit_count` base58 digits, so the tail spans
    // `[0, 58^(D - prefix_digit_count))` where D is the total digit count for
    // a value of that byte width.
    //
    // We compute D by counting how many base58 digits a full
    // `256^(mh_len - leading_ones)`-magnitude value needs — this is `ceil`,
    // which for our sizes (37 bytes ⇒ 51 digits) is one more than
    // `floor(log_58(256^bytes))`.
    let d = base58_digits_for_bytes(total_digit_slots);
    if prefix_digit_count > d {
        bail!("peer-id prefix is longer than a peer ID");
    }
    let tail_digits = d - prefix_digit_count;

    // value · 58^tail_digits is the lower bound of the multihash value (less
    // its leading zero bytes); +1·58^tail_digits gives the exclusive upper.
    let multiplier = pow58_limbs(tail_digits);
    let lo_full = mul_limbs(&value, &multiplier);
    let hi_full = add_limbs(&lo_full, &multiplier);

    // Pack into a 38-byte big-endian buffer (the multihash value below the
    // leading zeros). Limb 0 is the least-significant 32 bits.
    let mut mh = [0u8; 40]; // 2-byte CIDv1 header + 38-byte multihash
    mh[0] = 0x01;
    mh[1] = 0x72;
    pack_be_into(&mut mh[2 + leading_ones..40], &lo_full)?;

    let mut hi = [0u8; 40];
    hi[0] = 0x01;
    hi[1] = 0x72;
    // For the upper bound the leading-zero prefix can be exceeded by exactly
    // one (carry into the byte above), so pack into a 1-byte-longer slot if
    // there's room. We just pack into the same window and rely on add_limbs
    // not overflowing past mh_len bits in practice.
    pack_be_into(&mut hi[2 + leading_ones..40], &hi_full)?;

    Ok((mh, hi))
}

/// Number of base58 digits needed to represent any value in `[0, 256^bytes)`.
/// `ceil(bytes · log_58(256))`.
fn base58_digits_for_bytes(bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    // log_58(256) ≈ 1.365658 — close enough for the modest sizes we hit.
    let log_ratio = (256f64).ln() / (58f64).ln();
    (bytes as f64 * log_ratio).ceil() as usize
}

fn pow58_limbs(n: usize) -> [u32; 13] {
    let mut out = [0u32; 13];
    out[0] = 1;
    for _ in 0..n {
        let mut carry = 0u64;
        for limb in out.iter_mut() {
            let v = (*limb as u64) * 58 + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        debug_assert_eq!(carry, 0, "pow58 overflow");
    }
    out
}

/// Pack the value held in `limbs` (little-endian 32-bit) into the given
/// big-endian byte slice, error if it doesn't fit.
fn pack_be_into(dst: &mut [u8], limbs: &[u32; 13]) -> Result<()> {
    let cap = dst.len();
    // Check the high limbs are zero past the slot.
    for (i, &limb) in limbs.iter().enumerate() {
        for byte_idx in 0..4 {
            let global_byte = i * 4 + byte_idx;
            if global_byte >= cap && ((limb >> (8 * byte_idx)) & 0xff) != 0 {
                bail!("peer-id prefix value doesn't fit in 38 bytes");
            }
        }
    }
    // Write big-endian.
    for (byte_pos, slot) in dst.iter_mut().enumerate() {
        let global_byte = cap - 1 - byte_pos;
        let limb = limbs[global_byte / 4];
        *slot = ((limb >> (8 * (global_byte % 4))) & 0xff) as u8;
    }
    Ok(())
}

/// 36^n as a little-endian u32 limb array (13 limbs ≈ 416 bits — plenty).
fn pow36_limbs(n: usize) -> [u32; 13] {
    let mut out = [0u32; 13];
    out[0] = 1;
    for _ in 0..n {
        let mut carry = 0u64;
        for limb in out.iter_mut() {
            let v = (*limb as u64) * 36 + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        debug_assert_eq!(carry, 0, "pow36 overflow");
    }
    out
}

fn mul_limbs(a: &[u32; 13], b: &[u32; 13]) -> [u32; 13] {
    // Truncated schoolbook multiplication into a 13-limb result. Higher limbs
    // would mean the user's prefix is too long; we check for that elsewhere.
    let mut out = [0u64; 26];
    for i in 0..13 {
        for j in 0..13 {
            out[i + j] += (a[i] as u64) * (b[j] as u64);
            // Propagate immediate carries to avoid u64 overflow.
            let mut k = i + j;
            while out[k] >> 32 != 0 && k + 1 < 26 {
                out[k + 1] += out[k] >> 32;
                out[k] &= 0xffffffff;
                k += 1;
            }
        }
    }
    let mut packed = [0u32; 13];
    for i in 0..13 {
        packed[i] = out[i] as u32;
    }
    packed
}

fn add_limbs(a: &[u32; 13], b: &[u32; 13]) -> [u32; 13] {
    let mut out = [0u32; 13];
    let mut carry = 0u64;
    for i in 0..13 {
        let s = (a[i] as u64) + (b[i] as u64) + carry;
        out[i] = s as u32;
        carry = s >> 32;
    }
    out
}

fn limbs_to_be40(limbs: &[u32; 13]) -> Result<[u8; 40]> {
    for &limb in &limbs[10..] {
        if limb != 0 {
            bail!("internal: CID value exceeds 40 bytes");
        }
    }
    let mut out = [0u8; 40];
    for i in 0..10 {
        let be = limbs[9 - i].to_be_bytes();
        out[i * 4..(i + 1) * 4].copy_from_slice(&be);
    }
    Ok(out)
}

// =============================================================================
// Minimal host-side field arithmetic mod p = 2^255 - 19.
// Matches the GPU kernel's representation (8 LE uint32 limbs), so the constants
// we upload are byte-for-byte what the kernel expects.
// =============================================================================

mod fe {
    pub type Fe = [u32; 8];

    pub fn add(a: &Fe, b: &Fe) -> Fe {
        let mut r = [0u32; 8];
        let mut c: u64 = 0;
        for (i, slot) in r.iter_mut().enumerate() {
            c += a[i] as u64 + b[i] as u64;
            *slot = c as u32;
            c >>= 32;
        }
        propagate_overflow(&mut r, c);
        r
    }

    pub fn sub(a: &Fe, b: &Fe) -> Fe {
        let mut r = [0u32; 8];
        let mut borrow: i64 = 0;
        for (i, slot) in r.iter_mut().enumerate() {
            let v = a[i] as i64 - b[i] as i64 - borrow;
            *slot = v as u32;
            borrow = if v < 0 { 1 } else { 0 };
        }
        if borrow != 0 {
            // Add 2p = 2^256 - 38; the 2^256 wraps off, leaving r -= 38.
            let mut c: u64 = r[0] as u64 + 0xffffffda;
            r[0] = c as u32;
            c >>= 32;
            for slot in &mut r[1..] {
                c += *slot as u64 + 0xffffffff;
                *slot = c as u32;
                c >>= 32;
            }
        }
        r
    }

    pub fn mul(a: &Fe, b: &Fe) -> Fe {
        let mut t = [0u64; 16];
        for i in 0..8 {
            let mut carry: u64 = 0;
            for j in 0..8 {
                let prod = t[i + j] + (a[i] as u64) * (b[j] as u64) + carry;
                t[i + j] = prod & 0xffffffff;
                carry = prod >> 32;
            }
            t[i + 8] = carry;
        }
        let mut r = [0u32; 8];
        let mut c: u64 = 0;
        for (i, slot) in r.iter_mut().enumerate() {
            c += t[i] + 38 * t[i + 8];
            *slot = c as u32;
            c >>= 32;
        }
        propagate_overflow(&mut r, c);
        r
    }

    /// Fold any 2²⁵⁶ overflow back into the low limbs by multiplying by 38.
    fn propagate_overflow(r: &mut Fe, mut c: u64) {
        c *= 38;
        for slot in r.iter_mut() {
            if c == 0 {
                break;
            }
            c += *slot as u64;
            *slot = c as u32;
            c >>= 32;
        }
    }

    pub fn sq(a: &Fe) -> Fe {
        mul(a, a)
    }

    /// `z^(p-2) mod p`, i.e. the modular inverse. Same addition chain the
    /// kernel uses, so the host- and device-side answers agree byte-for-byte.
    pub fn invert(z: &Fe) -> Fe {
        let t0 = sq(z); // z^2
        let mut t1 = sq(&sq(&t0)); // z^8
        t1 = mul(z, &t1); // z^9
        let t0_z11 = mul(&t0, &t1); // z^11
        let t2 = sq(&t0_z11); // z^22
        t1 = mul(&t2, &t1); // 2^5 - 2^0

        let mut t = sq(&t1);
        for _ in 1..5 {
            t = sq(&t);
        }
        t1 = mul(&t, &t1); // 2^10 - 2^0

        t = sq(&t1);
        for _ in 1..10 {
            t = sq(&t);
        }
        let mut t2 = mul(&t, &t1); // 2^20 - 2^0

        t = sq(&t2);
        for _ in 1..20 {
            t = sq(&t);
        }
        t2 = mul(&t, &t2); // 2^40 - 2^0

        t = sq(&t2);
        for _ in 1..10 {
            t = sq(&t);
        }
        t1 = mul(&t, &t1); // 2^50 - 2^0

        t = sq(&t1);
        for _ in 1..50 {
            t = sq(&t);
        }
        t2 = mul(&t, &t1); // 2^100 - 2^0

        t = sq(&t2);
        for _ in 1..100 {
            t = sq(&t);
        }
        t2 = mul(&t, &t2); // 2^200 - 2^0

        t = sq(&t2);
        for _ in 1..50 {
            t = sq(&t);
        }
        t1 = mul(&t, &t1); // 2^250 - 2^0

        for _ in 0..5 {
            t1 = sq(&t1);
        }
        mul(&t1, &t0_z11) // 2^255 - 21 = p - 2
    }

    /// Reduce a field element to its canonical representative in `[0, p)`.
    /// Useful when packing the precomputed table so the bytes we upload are
    /// well-defined regardless of accumulated lazy reductions.
    pub fn canonical(a: &Fe) -> Fe {
        const P: Fe = [
            0xffffffed, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
            0x7fffffff,
        ];
        let mut r = *a;
        for _ in 0..2 {
            let mut tmp = [0u32; 8];
            let mut borrow: i64 = 0;
            for (i, slot) in tmp.iter_mut().enumerate() {
                let v = r[i] as i64 - P[i] as i64 - borrow;
                *slot = v as u32;
                borrow = if v < 0 { 1 } else { 0 };
            }
            if borrow == 0 {
                r = tmp;
            } else {
                break;
            }
        }
        r
    }
}

/// One Edwards-curve point in extended coordinates: `[X, Y, Z, T]`. `T = XY/Z`.
type Point = [fe::Fe; 4];

/// HWCD-3 unified addition. Mirrors the kernel's `ed_add` byte-for-byte; we
/// reuse it on the host to precompute the base-point doubling table.
fn ed_add(p: &Point, q: &Point, d2: &fe::Fe) -> Point {
    let t1 = fe::sub(&p[1], &p[0]);
    let t2 = fe::sub(&q[1], &q[0]);
    let a = fe::mul(&t1, &t2);
    let t1 = fe::add(&p[1], &p[0]);
    let t2 = fe::add(&q[1], &q[0]);
    let b = fe::mul(&t1, &t2);
    let c = fe::mul(&fe::mul(&p[3], &q[3]), d2);
    let d = fe::add(&fe::mul(&p[2], &q[2]), &fe::mul(&p[2], &q[2])); // 2·Z1·Z2
    let e = fe::sub(&b, &a);
    let f = fe::sub(&d, &c);
    let g = fe::add(&d, &c);
    let h = fe::add(&b, &a);
    [
        fe::mul(&e, &f),
        fe::mul(&g, &h),
        fe::mul(&f, &g),
        fe::mul(&e, &h),
    ]
}

/// Precompute a 4-bit window table over the base point in **affine** form.
///
/// For each of the 64 window positions, we store the 15 affine multiples
/// `k · Pᵢ` for `k ∈ {1..15}` and `Pᵢ = 16^i · B`, packed as the three field
/// elements `(y − x, y + x, 2d · x · y)`. Two compounding wins over the old
/// 2-bit projective table:
///
/// 1. Twice as many bits per iteration → 64 iterations instead of 128.
///    Each iteration is the warp's bottleneck (one `ed_add`), so halving the
///    iteration count nearly halves scalar-multiplication cost.
/// 2. Mixed (affine + projective) addition uses 7 multiplications instead of
///    9: `2 · Z1 · Z2` collapses to `2 · Z1` since `Z2 = 1`, and the `2·d`
///    factor is folded into the table at build time.
///
/// Layout: `64 positions × 15 multiples × 24 limbs = 23 040 u32` (= 90 KiB).
/// Larger than the old table, but L2 caches it easily on any modern GPU.
fn windowed_table(bx: &fe::Fe, by: &fe::Fe, d2: &fe::Fe) -> Vec<u32> {
    let bt = fe::mul(bx, by);
    let one = {
        let mut f = [0u32; 8];
        f[0] = 1;
        f
    };
    // `p` walks through 16^i · B as i increments.
    let mut p: Point = [*bx, *by, one, bt];

    let mut out = Vec::with_capacity(64 * 15 * 24);
    let mut scratch = [Point::default(); 15];
    for _ in 0..64 {
        // Compute 1·P, 2·P, …, 15·P in projective form.
        scratch[0] = p;
        for k in 1..15 {
            scratch[k] = ed_add(&scratch[k - 1], &p, d2);
        }
        // Convert each to affine and pack as (y-x, y+x, 2d·x·y).
        for m in &scratch {
            let z_inv = fe::invert(&m[2]);
            let x = fe::mul(&m[0], &z_inv);
            let y = fe::mul(&m[1], &z_inv);
            let ymx = fe::canonical(&fe::sub(&y, &x));
            let ypx = fe::canonical(&fe::add(&y, &x));
            let t2d = fe::canonical(&fe::mul(&fe::mul(&x, &y), d2));
            out.extend_from_slice(&ymx);
            out.extend_from_slice(&ypx);
            out.extend_from_slice(&t2d);
        }
        // Advance: next P = 16 · current P, four doublings.
        for _ in 0..4 {
            p = ed_add(&p, &p, d2);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_to_range_roundtrip() {
        // Re-derive the full prefix bytes the matcher would build internally,
        // feed them to prefix_to_cid_range, then check the lower bound CID
        // re-encodes to a name with that exact prefix.
        let user_pat = "hello";
        let mut full = Vec::new();
        full.extend_from_slice(FIXED_PREFIX);
        full.extend_from_slice(user_pat.as_bytes());

        let (lo, hi) = prefix_to_cid_range(&full).unwrap();
        assert!(lo < hi);

        let mut pk = [0u8; 32];
        pk.copy_from_slice(&lo[8..]);
        let name = crate::ipns::ipns_name(&pk);
        for (i, &c) in full.iter().enumerate() {
            assert_eq!(name[i], c, "prefix mismatch at byte {i}");
        }
    }
}
