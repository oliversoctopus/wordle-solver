//! The same three scores, read out of an answer-major copy of the table.
//!
//! `kernels` fixes a guess and walks the candidates, which `feedback[guess,
//! answer]` makes a row: contiguous for a whole candidate set, and a cache line
//! fetched per candidate for a scattered handful. That is the bill at every
//! mid-sized position, and an answer-major copy is what pays it off.
//!
//! Reading that way alone is not enough, because the reduce is per guess and
//! walking guesses in the inner loop leaves all of them accumulating at once,
//! which puts the accumulator out of L1 and costs more than the reads save.
//! These kernels dodge the trade instead of taking it. For a block of G guesses
//! they read the G x m tile out of the answer-major table -- contiguous, G
//! bytes per candidate -- transposing it into a small buffer on the way in.
//! Each guess's feedback vector is then a contiguous row of that tile, so the
//! reduce is the one in `kernels`, over one accumulator that stays in L1.
//!
//! Ported from `fused_kernels.py`, band and block rule included. Everything
//! here falls back to `kernels` when there is no transposed table or the
//! position sits outside the band, and the accumulation order is the shipped
//! order throughout, so a score is bit-identical rather than merely close.
//!
//! Native only. A tab holds a word list and colors pairs on the spot, so there
//! is no table for it to turn over and a second 168 MB is not something it
//! could be lent; under WASM these entry points forward to `kernels` and none
//! of the layout below reaches the module.

/// Candidate counts the fused path is used for. Below the floor a position is
/// too cheap for the dispatch to be worth taking; above the ceiling the tile
/// stops fitting in cache and the shipped kernels win.
pub const FUSED_MIN: usize = 13;
pub const FUSED_MAX: usize = 600;

/// Guesses per tile. Wider tiles amortize the transpose but have to stay in
/// cache, so the block is cut back as the candidate set grows.
#[cfg(not(target_arch = "wasm32"))]
const TILE_BYTES: usize = 32 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const MIN_BLOCK: usize = 16;
#[cfg(not(target_arch = "wasm32"))]
const MAX_BLOCK: usize = 256;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{expected_guesses_all, ig_all, two_turn_scores};

#[cfg(target_arch = "wasm32")]
pub use shipped::{expected_guesses_all, ig_all, two_turn_scores};

/// Whether a solver builds the second table when it is not told either way.
/// `WORDLE_FUSED=1` turns it on for every solver in the process.
#[cfg(not(target_arch = "wasm32"))]
pub fn enabled_by_default() -> bool {
    std::env::var("WORDLE_FUSED").as_deref() == Ok("1")
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{FUSED_MAX, FUSED_MIN, MAX_BLOCK, MIN_BLOCK, TILE_BYTES};
    use crate::estimator::Form;
    use crate::feedback::{N_PATTERNS, WIN_PATTERN};
    use crate::kernels::{self, all_rows, Buckets, RESCAN_MAX};
    use crate::source::{Source, Transposed};
    use rayon::prelude::*;

    /// Guesses per tile, or `None` to leave this position to `kernels`.
    pub(super) fn plan(n_pool: usize, m: usize) -> Option<usize> {
        if !(FUSED_MIN..=FUSED_MAX).contains(&m) || n_pool < MIN_BLOCK {
            return None;
        }
        let block = (TILE_BYTES / m).clamp(MIN_BLOCK, MAX_BLOCK);
        // a short pool must still reach every thread, or this runs on one core
        let units = 4 * rayon::current_num_threads().max(1);
        Some(block.min(n_pool.div_ceil(units)).max(MIN_BLOCK))
    }

    /// Runs `fill` over one block of guesses at a time, each worker holding its
    /// own tile and accumulators for as long as rayon leaves it the job.
    fn tiled<A, I, F>(pool: &[usize], block: usize, init: I, fill: F) -> Vec<f64>
    where
        A: Send,
        I: Fn() -> A + Sync + Send,
        F: Fn(&mut A, &[usize], &mut [f64]) + Sync + Send,
    {
        let mut out = vec![0.0; pool.len()];
        pool.par_chunks(block)
            .zip(out.par_chunks_mut(block))
            .for_each_init(init, |state, (part, slot)| fill(state, part, slot));
        out
    }

    /// Fills `tile` with the block's feedback, one contiguous row per guess.
    ///
    /// The read runs down a candidate's row of the transposed table, where the
    /// block's guesses sit next to each other; the write strides. That is the
    /// transpose, and doing it here is what leaves the reduce reading a row.
    #[inline]
    fn fill_tile(tt: &Transposed, tile: &mut [u8], candidates: &[usize], part: &[usize]) {
        let m = candidates.len();
        for (i, &c) in candidates.iter().enumerate() {
            for (j, &guess) in part.iter().enumerate() {
                tile[j * m + i] = tt.at(c, guess);
            }
        }
    }

    /// `kernels::ig_all`, over the transposed table where that is faster.
    pub fn ig_all(
        source: &impl Source,
        tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        let all;
        let rows: &[usize] = match pool {
            Some(p) => p,
            None => {
                all = all_rows(source.n_guesses());
                &all
            }
        };
        let m = candidates.len();
        let (Some(tt), Some(block)) = (tt, plan(rows.len(), m)) else {
            return kernels::ig_all(source, candidates, weights, pool);
        };
        let total: f64 = weights.iter().sum();

        tiled(
            rows,
            block,
            || (vec![0u8; block * m], [0.0f64; N_PATTERNS]),
            |(tile, mass), part, out| {
                fill_tile(tt, tile, candidates, part);
                for j in 0..part.len() {
                    let row = &tile[j * m..j * m + m];
                    for i in 0..m {
                        mass[row[i] as usize] += weights[i];
                    }

                    let mut h = 0.0;
                    if m < RESCAN_MAX {
                        for &r in row {
                            let r = r as usize;
                            if mass[r] > 0.0 {
                                let p = mass[r] / total;
                                h -= p * p.log2();
                                mass[r] = 0.0;
                            }
                        }
                    } else {
                        for value in mass.iter_mut() {
                            if *value > 0.0 {
                                let p = *value / total;
                                h -= p * p.log2();
                                *value = 0.0;
                            }
                        }
                    }
                    out[j] = h;
                }
            },
        )
    }

    /// `kernels::expected_guesses_all`, over the transposed table where faster.
    pub fn expected_guesses_all(
        source: &impl Source,
        tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        params: &[f64],
        form: &Form,
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        let all;
        let rows: &[usize] = match pool {
            Some(p) => p,
            None => {
                all = all_rows(source.n_guesses());
                &all
            }
        };
        let m = candidates.len();
        let (Some(tt), Some(block)) = (tt, plan(rows.len(), m)) else {
            return kernels::expected_guesses_all(source, candidates, weights, params, form, pool);
        };
        let total: f64 = weights.iter().sum();
        let wlw = kernels::w_log_w(weights);

        tiled(
            rows,
            block,
            || (vec![0u8; block * m], Buckets::new()),
            |(tile, buckets), part, out| {
                fill_tile(tt, tile, candidates, part);
                for j in 0..part.len() {
                    let row = &tile[j * m..j * m + m];
                    for i in 0..m {
                        buckets.add(row[i] as usize, weights[i], wlw[i]);
                    }

                    let mut expected = 1.0;
                    if m < RESCAN_MAX {
                        for &r in row {
                            expected += buckets.take(r as usize, total, params, form);
                        }
                    } else {
                        for r in 0..N_PATTERNS {
                            expected += buckets.take(r, total, params, form);
                        }
                    }
                    out[j] = expected;
                }
            },
        )
    }

    /// `kernels::two_turn_scores`, over the transposed table where faster.
    pub fn two_turn_scores(
        source: &impl Source,
        tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        penalty: f64,
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        let all;
        let rows: &[usize] = match pool {
            Some(p) => p,
            None => {
                all = all_rows(source.n_guesses());
                &all
            }
        };
        let m = candidates.len();
        let (Some(tt), Some(block)) = (tt, plan(rows.len(), m)) else {
            return kernels::two_turn_scores(source, candidates, weights, penalty, pool);
        };
        let total: f64 = weights.iter().sum();

        tiled(
            rows,
            block,
            || {
                (
                    vec![0u8; block * m],
                    [0.0f64; N_PATTERNS],
                    [0.0f64; N_PATTERNS],
                )
            },
            |(tile, mass, heaviest), part, out| {
                fill_tile(tt, tile, candidates, part);
                for j in 0..part.len() {
                    let row = &tile[j * m..j * m + m];
                    for i in 0..m {
                        let r = row[i] as usize;
                        let w = weights[i];
                        mass[r] += w;
                        if w > heaviest[r] {
                            heaviest[r] = w;
                        }
                    }

                    // a bucket of zero mass has a zero maximum too, so mass
                    // alone says whether this bucket has already been taken
                    let mut mass_sum = 0.0;
                    let mut max_sum = 0.0;
                    let mut take = |r: usize, mass: &mut [f64; N_PATTERNS]| {
                        if mass[r] > 0.0 {
                            if r != WIN_PATTERN as usize {
                                mass_sum += mass[r];
                                max_sum += heaviest[r];
                            }
                            mass[r] = 0.0;
                            heaviest[r] = 0.0;
                        }
                    };
                    if m < RESCAN_MAX {
                        for &r in row {
                            take(r as usize, mass);
                        }
                    } else {
                        for r in 0..N_PATTERNS {
                            take(r, mass);
                        }
                    }
                    out[j] = 1.0 + (mass_sum + (mass_sum - max_sum) * penalty) / total;
                }
            },
        )
    }
}

#[cfg(target_arch = "wasm32")]
mod shipped {
    use crate::estimator::Form;
    use crate::kernels;
    use crate::source::{Source, Transposed};

    pub fn ig_all(
        source: &impl Source,
        _tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        kernels::ig_all(source, candidates, weights, pool)
    }

    pub fn expected_guesses_all(
        source: &impl Source,
        _tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        params: &[f64],
        form: &Form,
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        kernels::expected_guesses_all(source, candidates, weights, params, form, pool)
    }

    pub fn two_turn_scores(
        source: &impl Source,
        _tt: Option<&Transposed>,
        candidates: &[usize],
        weights: &[f64],
        penalty: f64,
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        kernels::two_turn_scores(source, candidates, weights, penalty, pool)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::native::plan;
    use super::*;
    use crate::estimator::get_form;
    use crate::kernels;
    use crate::source::{Source, Table, Transposed};

    /// A table with no Wordle in it, only a pattern per pair that both layouts
    /// have to agree about.
    fn fixture(n: usize) -> (Table, Vec<f64>) {
        let mut data = vec![0u8; n * n];
        for guess in 0..n {
            for answer in 0..n {
                data[guess * n + answer] = ((guess * 7 + answer * 13 + guess * answer) % 243) as u8;
            }
        }
        let weights = (0..n).map(|i| 1.0 + (i % 11) as f64 / 7.0).collect();
        (Table::from_raw(data, n), weights)
    }

    /// Scattered, the way a real candidate set is. The two sizes sit either
    /// side of `RESCAN_MAX`, where the reduce runs a different loop.
    fn position(m: usize) -> Vec<usize> {
        (0..m).map(|i| i * 199 % 200).collect()
    }

    #[test]
    fn the_two_layouts_score_entropy_alike() {
        let (table, weights) = fixture(200);
        let tt = Transposed::build(&table);
        for m in [20, 150] {
            let candidates = position(m);
            let w: Vec<f64> = candidates.iter().map(|&c| weights[c]).collect();
            let shipped = kernels::ig_all(&table, &candidates, &w, None);
            let fused = ig_all(&table, Some(&tt), &candidates, &w, None);
            assert_eq!(shipped, fused, "entropy differs at {m} candidates");
        }
    }

    #[test]
    fn the_two_layouts_score_expected_guesses_alike() {
        let (table, weights) = fixture(200);
        let tt = Transposed::build(&table);
        let form = get_form("loglinear").expect("a form to score through");
        let params = vec![0.6, 0.4];
        for m in [20, 150] {
            let candidates = position(m);
            let w: Vec<f64> = candidates.iter().map(|&c| weights[c]).collect();
            let shipped =
                kernels::expected_guesses_all(&table, &candidates, &w, &params, &form, None);
            let fused =
                expected_guesses_all(&table, Some(&tt), &candidates, &w, &params, &form, None);
            assert_eq!(shipped, fused, "expected guesses differ at {m} candidates");
        }
    }

    #[test]
    fn the_two_layouts_score_two_turn_values_alike() {
        let (table, weights) = fixture(200);
        let tt = Transposed::build(&table);
        for m in [20, 150] {
            let candidates = position(m);
            let w: Vec<f64> = candidates.iter().map(|&c| weights[c]).collect();
            let shipped = kernels::two_turn_scores(&table, &candidates, &w, 5.0, None);
            let fused = two_turn_scores(&table, Some(&tt), &candidates, &w, 5.0, None);
            assert_eq!(shipped, fused, "two-turn values differ at {m} candidates");
        }
    }

    #[test]
    fn a_position_outside_the_band_goes_to_the_shipped_kernels() {
        assert_eq!(plan(12_947, FUSED_MIN - 1), None);
        assert_eq!(plan(12_947, FUSED_MAX + 1), None);
        assert!(plan(12_947, FUSED_MIN).is_some());
        assert!(plan(12_947, FUSED_MAX).is_some());
        // a pool too short to fill one tile is not worth the transpose
        assert_eq!(plan(MIN_BLOCK - 1, 100), None);
    }

    #[test]
    fn the_transpose_turns_the_table_over() {
        let (table, _) = fixture(64);
        let tt = Transposed::build(&table);
        for guess in 0..64 {
            for answer in 0..64 {
                assert_eq!(tt.at(answer, guess), table.score(guess, answer));
            }
        }
    }
}
