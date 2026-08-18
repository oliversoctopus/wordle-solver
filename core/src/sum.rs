//! Pairwise summation, matching what `numpy.sum` does.
//!
//! Adding 12,947 floats left to right accumulates error proportional to the
//! count; splitting the range in half and recursing makes it proportional to
//! its logarithm. numpy has done the latter since 1.9, so a naive loop here
//! disagrees with the Python by ~1e-13 relative -- small, but it is avoidable
//! noise in every parity check downstream, so this reproduces numpy's shape
//! exactly rather than merely being accurate.
//!
//! Only for the places the Python reaches for `np.sum`. The kernels accumulate
//! into their 243 buckets one candidate at a time and must keep doing exactly
//! that, in that order; see `kernels.py`.

/// numpy's block size, below which it sums with eight accumulators.
const BLOCK: usize = 128;

/// The same value `numpy.sum` produces for the same slice.
pub fn pairwise(values: &[f64]) -> f64 {
    let n = values.len();

    if n < 8 {
        let mut total = 0.0;
        for &v in values {
            total += v;
        }
        return total;
    }

    if n <= BLOCK {
        // eight independent chains, then a balanced tree over them
        let mut r = [0.0f64; 8];
        r[..8].copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += values[i + j];
            }
            i += 8;
        }
        let mut total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            total += values[i];
            i += 1;
        }
        return total;
    }

    // split on a multiple of eight, so each half takes the path above
    let mut half = n / 2;
    half -= half % 8;
    pairwise(&values[..half]) + pairwise(&values[half..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agrees_with_a_naive_sum_on_easy_input() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(pairwise(&values), 499_500.0);
    }

    #[test]
    fn beats_a_naive_sum_where_it_should() {
        // a large value followed by many small ones: adding left to right loses
        // the tail, and pairwise keeps most of it
        let mut values = vec![1e16];
        values.extend(std::iter::repeat_n(1.0, 10_000));
        let naive = values.iter().fold(0.0f64, |a, b| a + b);
        let exact = 1e16 + 10_000.0;
        assert!((pairwise(&values) - exact).abs() <= (naive - exact).abs());
    }

    #[test]
    fn handles_every_length_around_the_block_boundaries() {
        for n in [0, 1, 7, 8, 9, 127, 128, 129, 256, 1000] {
            let values: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();
            let naive: f64 = values.iter().sum();
            assert!((pairwise(&values) - naive).abs() < 1e-12, "n = {n}");
        }
    }
}
