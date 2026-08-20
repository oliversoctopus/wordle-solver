//! One float reduction, shared with the Python.
//!
//! `np.sum` adds pairwise rather than left to right, which makes it both the
//! more accurate of the two -- error grows as log n rather than n -- and the
//! harder to reproduce. Reproducing it is what keeps the two implementations
//! bit-identical wherever they reduce a vector, so this is a transcription of
//! numpy's `pairwise_sum_DOUBLE` rather than something merely equivalent to it.
//!
//! `kernels.rs` is the exception and stays naive: the numba kernels accumulate
//! in an explicit left-to-right loop, so a naive sum is what matches there.
//!
//! numpy's BLAS dot products are a second exception, and one nothing can fix.
//! `analysis/common.py`'s `weights @ lengths` reduces in whatever order the
//! shipped BLAS chooses, which is not written down and varies by build.

/// Naive under eight elements, eight independent accumulators up to 128, and
/// recursive halving on a multiple of eight above that.
pub fn pairwise_of<F: Fn(usize) -> f64>(n: usize, at: F) -> f64 {
    fold(&at, 0, n)
}

/// The same, over a slice.
pub fn pairwise(values: &[f64]) -> f64 {
    pairwise_of(values.len(), |i| values[i])
}

fn fold<F: Fn(usize) -> f64>(at: &F, start: usize, n: usize) -> f64 {
    if n < 8 {
        return (0..n).fold(0.0, |total, i| total + at(start + i));
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        for (j, cell) in r.iter_mut().enumerate() {
            *cell = at(start + j);
        }
        let mut i = 8;
        while i + 8 <= n {
            for (j, cell) in r.iter_mut().enumerate() {
                *cell += at(start + i + j);
            }
            i += 8;
        }
        let mut total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            total += at(start + i);
            i += 1;
        }
        return total;
    }
    let half = n / 2 - (n / 2) % 8;
    fold(at, start, half) + fold(at, start + half, n - half)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Left to right is the thing this is not, and the gap is what it buys.
    #[test]
    fn beats_a_naive_sum_on_a_vector_it_should() {
        // 1 followed by n copies of eps/2: naive loses every one of them, and
        // the pairwise tree keeps them by adding them to each other first
        let n = 4096;
        let mut values = vec![f64::EPSILON / 2.0; n + 1];
        values[0] = 1.0;
        let naive = values.iter().fold(0.0, |a, &b| a + b);
        let truth = 1.0 + (n as f64) * (f64::EPSILON / 2.0);

        assert_eq!(naive, 1.0, "a naive sum was expected to lose them all");
        assert!((pairwise(&values) - truth).abs() * 100.0 < (naive - truth).abs());
    }

    #[test]
    fn the_indexed_form_is_the_slice_form() {
        for n in [0, 1, 7, 8, 9, 128, 129, 1000] {
            let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            assert_eq!(pairwise(&values), pairwise_of(n, |i| values[i]));
        }
    }
}
