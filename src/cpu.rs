//! Multi-threaded CPU search using `ed25519-dalek` and `rayon`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_channel::Sender;
use ed25519_dalek::SigningKey;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::ipns::{IPNS_NAME_LEN, write_ipns_name};
use crate::matcher::Matcher;

/// One vanity match produced by either backend.
#[derive(Clone)]
pub struct Match {
    pub seed: [u8; 32],
    pub pubkey: [u8; 32],
    pub name: [u8; IPNS_NAME_LEN],
}

/// Run the CPU search until `stop` is set.
///
/// Each worker thread keeps its own ChaCha20 RNG seeded from OS entropy, so the
/// work is fully independent — no shared state on the hot path.
pub fn run(
    matcher: Arc<Matcher>,
    stop: Arc<AtomicBool>,
    counter: Arc<AtomicU64>,
    tx: Sender<Match>,
    threads: usize,
) {
    let n = if threads == 0 {
        num_cpus::get()
    } else {
        threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("ipns-cpu-{i}"))
        .build()
        .expect("rayon pool");

    pool.install(|| {
        (0..n)
            .into_par_iter()
            .for_each(|_| worker(&matcher, &stop, &counter, &tx));
    });
}

fn worker(matcher: &Matcher, stop: &AtomicBool, counter: &AtomicU64, tx: &Sender<Match>) {
    let mut rng = ChaCha20Rng::from_entropy();
    let mut seed = [0u8; 32];
    let mut name = [0u8; IPNS_NAME_LEN];

    // Tally locally and flush in batches to keep the shared atomic out of
    // the hot loop.
    const FLUSH_EVERY: u64 = 4096;
    let mut local = 0u64;

    while !stop.load(Ordering::Relaxed) {
        rng.fill_bytes(&mut seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let pubkey = signing_key.verifying_key().to_bytes();

        write_ipns_name(&pubkey, &mut name);

        if matcher.matches(&name) {
            let _ = tx.send(Match { seed, pubkey, name });
        }

        local += 1;
        if local >= FLUSH_EVERY {
            counter.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
    }
    counter.fetch_add(local, Ordering::Relaxed);
}
