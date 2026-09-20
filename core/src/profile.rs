//! Where a sweep's time goes, when built with `--features profile`.
//!
//! The instrumentation lives here so a run can be timed without editing the
//! kernels.

/// The entry points worth timing separately.
pub const NAMES: [&str; 8] = [
    "ig_all",
    "expected_guesses_all",
    "two_turn_scores",
    "distinct_partitions",
    "signature_keys",
    "same_split",
    "filter_candidates",
    "table build",
];

#[cfg(feature = "profile")]
mod on {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static NANOS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    pub static CALLS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

    /// Adds its lifetime to a slot when dropped.
    pub struct Span(usize, Instant);

    impl Span {
        pub fn new(slot: usize) -> Span {
            Span(slot, Instant::now())
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            NANOS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CALLS[self.0].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `(seconds, calls)` per entry point.
    pub fn totals() -> Vec<(f64, u64)> {
        (0..8)
            .map(|i| {
                (
                    NANOS[i].load(Ordering::Relaxed) as f64 / 1e9,
                    CALLS[i].load(Ordering::Relaxed),
                )
            })
            .collect()
    }
}

#[cfg(not(feature = "profile"))]
mod on {
    pub struct Span;
    impl Span {
        #[inline(always)]
        pub fn new(_slot: usize) -> Span {
            Span
        }
    }
    pub fn totals() -> Vec<(f64, u64)> {
        Vec::new()
    }
}

pub use on::{totals, Span};
