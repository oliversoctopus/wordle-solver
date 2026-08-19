//! Pairwise summation.
//!
//! Adding 12,947 floats left to right accumulates error proportional to the
//! count; splitting the range in half and recursing makes it proportional to
//! its logarithm. That is the whole reason this is here -- it is the better
//! way to add up a slice, and it is what every total in the core uses.
//!
//! It happens to be what `numpy.sum` does too, so the port and the Python it
//! was ported from tend to agree to the last bit. That is a convenience when
//! reading a parity report, not a requirement: the closest pair of scores among
//! all 12,947 guesses at the opening differs by 1.1e-9 against an ulp of 4e-16,
//! and `exact::SPLIT_TIE_TOL` calls anything within 1e-9 the same guess. Do not
//! contort anything here to preserve the coincidence.

/// Below this many values, sum with eight independent chains instead of
/// recursing further. Matches numpy's block size.
const BLOCK: usize = 128;

/// The sum of `values`, accumulated pairwise.
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
