//! Scoring every guess against a candidate set.
//!
//! Each scorer buckets the candidates by the pattern a guess would produce
//! and reduces the buckets to one number: bits gained, expected guesses
//! through `f`, or the exact two-turn value. Ported from `kernels.py`,
//! arithmetic order included, so a score is the same float. Nothing here
//! reduces through `sum::pairwise`: a bucket is filled by scattering, and the
//! weight total is accumulated left to right to match the buckets exactly.
//! Every kernel takes a `pool` of guess indices, all of them when `None`,
//! and hard mode passes the candidate set as the pool.

use crate::estimator::{f_eval_x, Form};
use crate::feedback::{N_PATTERNS, WIN_PATTERN};
use crate::profile::Span;
use crate::source::Source;

/// Maps `f` over `0..n`, across cores natively.
///
/// The split helpers walk every one of the 12,947 guesses, and `kernels.py`
/// declares all of them `parallel=True`. `work` is the pairs the whole call
/// touches; below the threshold it runs straight through, because rayon's
/// adaptive splitting costs more than a handful of byte comparisons is worth.
#[cfg(not(target_arch = "wasm32"))]
fn map_guesses<T, F>(n: usize, work: usize, f: F) -> Vec<T>
where
    T: Send + Default + Clone,
    F: Fn(usize) -> T + Sync + Send,
{
    use rayon::prelude::*;

    if work < PARALLEL_MIN_WORK || n < 2 {
        return (0..n).map(f).collect();
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(threads).max(1);
    let mut out = vec![T::default(); n];
    out.par_chunks_mut(chunk).enumerate().for_each(|(b, slot)| {
        let base = b * chunk;
        for (i, cell) in slot.iter_mut().enumerate() {
            *cell = f(base + i);
        }
    });
    out
}

#[cfg(target_arch = "wasm32")]
fn map_guesses<T, F>(n: usize, _work: usize, f: F) -> Vec<T>
where
    T: Default + Clone,
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}

/// Least (guess, candidate) pairs worth splitting across cores.
///
/// The kernels are called constantly on small candidate sets -- the exact
/// search asks for a few hundred representatives against a handful of
/// candidates, tens of thousands of times over a sweep -- and below this the
/// coordination costs more than the work it divides.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_MIN_WORK: usize = 1 << 12;

/// Runs `fill` over the pool, writing one score per entry.
///
/// Natively this spreads the pool across rayon's standing thread pool when
/// there is enough work to pay for it, as the numba kernels do with `prange`;
/// each chunk keeps accumulators of its own. Under WASM it runs
/// straight through, threads there needing SharedArrayBuffer and headers a
/// static host does not set.
#[cfg(not(target_arch = "wasm32"))]
fn scatter<F>(pool: &[usize], work: usize, fill: F) -> Vec<f64>
where
    F: Fn(&[usize], &mut [f64]) + Sync,
{
    use rayon::prelude::*;

    let mut out = vec![0.0; pool.len()];
    if work < PARALLEL_MIN_WORK || pool.len() < 2 {
        fill(pool, &mut out);
        return out;
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = pool.len().div_ceil(threads).max(1);
    pool.par_chunks(chunk)
        .zip(out.par_chunks_mut(chunk))
        .for_each(|(part, slot)| fill(part, slot));
    out
}

#[cfg(target_arch = "wasm32")]
fn scatter<F>(pool: &[usize], _work: usize, fill: F) -> Vec<f64>
where
    F: Fn(&[usize], &mut [f64]) + Sync,
{
    let mut out = vec![0.0; pool.len()];
    fill(pool, &mut out);
    out
}

/// Below this many candidates the reduce walks the candidates a second time
/// rather than sweeping all 243 patterns, taking each bucket the first time it
/// is reached and zeroing it there. Above it the second walk stops paying for
/// itself, since it re-reads the table where the sweep only touches
/// accumulators already in cache.
pub const RESCAN_MAX: usize = 128;

/// Every guess index, built once and lent out thereafter.
///
/// `pool = None` means "score every word", which is most calls, and building
/// 12,947 indices each time is pure waste -- `kernels.py` caches the same array
/// in `_ALL_ROWS` for the same reason. The length check covers the one case a
/// single process could hold two word lists; it has never happened here.
pub(crate) fn all_rows(n: usize) -> std::borrow::Cow<'static, [usize]> {
    static CACHE: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let cached = CACHE.get_or_init(|| (0..n).collect());
    if cached.len() == n {
        std::borrow::Cow::Borrowed(cached)
    } else {
        std::borrow::Cow::Owned((0..n).collect())
    }
}

/// `w * log2(w)` per candidate, the guess-independent half of a bucket's
/// entropy. Zero weights contribute nothing, as `0 log 0 = 0`.
pub fn w_log_w(weights: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .map(|&w| if w > 0.0 { w * w.log2() } else { 0.0 })
        .collect()
}

/// Expected information gain, in bits, of every pooled guess.
pub fn ig_all(
    source: &impl Source,
    candidates: &[usize],
    weights: &[f64],
    pool: Option<&[usize]>,
) -> Vec<f64> {
    let _span = Span::new(0);
    let all;
    let pool: &[usize] = match pool {
        Some(p) => p,
        None => {
            all = all_rows(source.n_guesses());
            &all
        }
    };
    let m = candidates.len();
    let total: f64 = weights.iter().sum();

    scatter(pool, pool.len().saturating_mul(m), |part, out| {
        let mut mass = [0.0f64; N_PATTERNS];
        for (slot, &guess) in part.iter().enumerate() {
            for i in 0..m {
                mass[source.score(guess, candidates[i]) as usize] += weights[i];
            }

            let mut h = 0.0;
            if m < RESCAN_MAX {
                for &c in candidates {
                    let r = source.score(guess, c) as usize;
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
            out[slot] = h;
        }
    })
}

/// Guesses still needed from one bucket. One and two candidates are worth
/// exactly 1 and `2 - p_max`, so the estimator is only consulted above that.
#[inline]
pub(crate) fn bucket_guesses(
    mass: f64,
    count: f64,
    wlw: f64,
    heaviest: f64,
    params: &[f64],
    form: &Form,
) -> f64 {
    if count == 1.0 {
        return 1.0;
    }
    if count == 2.0 {
        // guess the likelier one; if wrong, the other is now known
        return 2.0 - heaviest / mass;
    }
    f_eval_x(mass.log2() - wlw / mass, params, form)
}

/// Estimated total guesses to finish, for every pooled guess:
///
/// ```text
/// E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)
/// ```
///
/// The `GGGGG` bucket contributes nothing, so a guess that could itself be the
/// answer comes out ahead of an equally informative one that could not.
pub fn expected_guesses_all(
    source: &impl Source,
    candidates: &[usize],
    weights: &[f64],
    params: &[f64],
    form: &Form,
    pool: Option<&[usize]>,
) -> Vec<f64> {
    let _span = Span::new(1);
    let all;
    let pool: &[usize] = match pool {
        Some(p) => p,
        None => {
            all = all_rows(source.n_guesses());
            &all
        }
    };
    let m = candidates.len();
    let total: f64 = weights.iter().sum();
    let wlw = w_log_w(weights);

    scatter(pool, pool.len().saturating_mul(m), |part, out| {
        let mut buckets = Buckets::new();
        for (slot, &guess) in part.iter().enumerate() {
            for i in 0..m {
                let r = source.score(guess, candidates[i]) as usize;
                buckets.add(r, weights[i], wlw[i]);
            }

            let mut expected = 1.0;
            if m < RESCAN_MAX {
                for &c in candidates {
                    let r = source.score(guess, c) as usize;
                    expected += buckets.take(r, total, params, form);
                }
            } else {
                for r in 0..N_PATTERNS {
                    expected += buckets.take(r, total, params, form);
                }
            }
            out[slot] = expected;
        }
    })
}

/// The four 243-slot accumulators the expected-guess reduce needs: a bucket's
/// running mass, its candidate count, its total of `w log2 w`, and its heaviest
/// single weight.
pub(crate) struct Buckets {
    mass: [f64; N_PATTERNS],
    count: [f64; N_PATTERNS],
    wlw: [f64; N_PATTERNS],
    heaviest: [f64; N_PATTERNS],
}

impl Buckets {
    /// One candidate's contribution to the bucket its pattern names. Shared
    /// with `fused`, so both layouts accumulate in the same order.
    #[inline]
    pub(crate) fn add(&mut self, r: usize, w: f64, wlw: f64) {
        self.mass[r] += w;
        self.count[r] += 1.0;
        self.wlw[r] += wlw;
        if w > self.heaviest[r] {
            self.heaviest[r] = w;
        }
    }

    pub(crate) fn new() -> Buckets {
        Buckets {
            mass: [0.0; N_PATTERNS],
            count: [0.0; N_PATTERNS],
            wlw: [0.0; N_PATTERNS],
            heaviest: [0.0; N_PATTERNS],
        }
    }

    /// This bucket's contribution, clearing it for the next guess. Zeroing here
    /// is what makes the second walk over the candidates correct as well as
    /// cheap: it stops a bucket being counted twice when a later candidate
    /// lands in it.
    #[inline]
    pub(crate) fn take(&mut self, r: usize, total: f64, params: &[f64], form: &Form) -> f64 {
        if self.count[r] <= 0.0 {
            return 0.0;
        }
        let contribution = if r == WIN_PATTERN as usize {
            0.0
        } else {
            self.mass[r] / total
                * bucket_guesses(
                    self.mass[r],
                    self.count[r],
                    self.wlw[r],
                    self.heaviest[r],
                    params,
                    form,
                )
        };
        self.mass[r] = 0.0;
        self.count[r] = 0.0;
        self.wlw[r] = 0.0;
        self.heaviest[r] = 0.0;
        contribution
    }
}

/// Exact expected score of every pooled guess when two turns remain.
///
/// With one turn left the best available in a bucket is its heaviest member,
/// worth `1 + (1 - p_max) * penalty`, so two-turn play needs no estimator:
///
/// ```text
/// V(g, 2) = 1 + (A + (A - B) * penalty) / total
/// ```
///
/// with `A` the mass and `B` the sum of bucket maxima over the non-winning
/// buckets. Called with `penalty = min(loss_penalty, 1)` this is also a valid
/// lower bound on the value at any larger number of turns, and the exact
/// search prunes with it.
pub fn two_turn_scores(
    source: &impl Source,
    candidates: &[usize],
    weights: &[f64],
    penalty: f64,
    pool: Option<&[usize]>,
) -> Vec<f64> {
    let _span = Span::new(2);
    let all;
    let pool: &[usize] = match pool {
        Some(p) => p,
        None => {
            all = all_rows(source.n_guesses());
            &all
        }
    };
    let m = candidates.len();
    let total: f64 = weights.iter().sum();

    scatter(pool, pool.len().saturating_mul(m), |part, out| {
        let mut mass = [0.0f64; N_PATTERNS];
        let mut heaviest = [0.0f64; N_PATTERNS];
        for (slot, &guess) in part.iter().enumerate() {
            for i in 0..m {
                let r = source.score(guess, candidates[i]) as usize;
                let w = weights[i];
                mass[r] += w;
                if w > heaviest[r] {
                    heaviest[r] = w;
                }
            }

            let mut mass_sum = 0.0;
            let mut max_sum = 0.0;
            // a bucket of zero mass has a zero maximum too, so mass alone says
            // whether this bucket has already been taken
            if m < RESCAN_MAX {
                for &c in candidates {
                    let r = source.score(guess, c) as usize;
                    if mass[r] > 0.0 {
                        if r != WIN_PATTERN as usize {
                            mass_sum += mass[r];
                            max_sum += heaviest[r];
                        }
                        mass[r] = 0.0;
                        heaviest[r] = 0.0;
                    }
                }
            } else {
                for r in 0..N_PATTERNS {
                    if mass[r] > 0.0 {
                        if r != WIN_PATTERN as usize {
                            mass_sum += mass[r];
                            max_sum += heaviest[r];
                        }
                        mass[r] = 0.0;
                        heaviest[r] = 0.0;
                    }
                }
            }
            out[slot] = 1.0 + (mass_sum + (mass_sum - max_sum) * penalty) / total;
        }
    })
}

// ------------------------------------------------------------------- splits

/// At most this many candidates for a packed signature to be the partition
/// itself: eight bits each, in the 56 available.
pub const MAX_PACKED: usize = 7;

const FNV_OFFSET: i64 = -3750763034362895579; // 14695981039346656037, wrapped
const FNV_PRIME: i64 = 1099511628211;

/// Packs each guess's feedback over the candidates into one integer. Guesses
/// sharing a key split the candidates identically. Callers must keep to
/// `MAX_PACKED` candidates or fewer.
pub fn signature_keys(source: &impl Source, candidates: &[usize]) -> Vec<i64> {
    let _span = Span::new(4);
    map_guesses(
        source.n_guesses(),
        source.n_guesses() * candidates.len(),
        |guess| {
            let mut key = 0i64;
            for &c in candidates {
                key = (key << 8) | source.score(guess, c) as i64;
            }
            key
        },
    )
}

/// FNV-1a over each guess's feedback vector, for grouping equal splits.
fn partition_hash(source: &impl Source, candidates: &[usize]) -> Vec<i64> {
    map_guesses(
        source.n_guesses(),
        source.n_guesses() * candidates.len(),
        |guess| {
            let mut h = FNV_OFFSET;
            for &c in candidates {
                h = (h ^ source.score(guess, c) as i64).wrapping_mul(FNV_PRIME);
            }
            h
        },
    )
}

/// `(representatives, the head of each guess's group)`, both out of one sort.
///
/// This is `np.unique(keys, return_index=True, return_inverse=True)`: the
/// representatives in ascending key order, and for every guess the
/// representative of its group. The order matters -- it decides which word
/// stands for a split, and so the order the search tries them in. Both come out
/// of the one sort rather than a map keyed on the hash, which would have to be
/// built and then looked up once per guess on every call.
fn group_by_key(keys: &[i64]) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_unstable_by_key(|&i| (keys[i], i));

    let mut first = Vec::new();
    let mut heads = vec![0usize; keys.len()];
    let mut previous: Option<i64> = None;
    let mut head = 0;
    for &i in &order {
        if previous != Some(keys[i]) {
            first.push(i);
            head = i;
            previous = Some(keys[i]);
        }
        heads[i] = head;
    }
    (first, heads)
}

/// One representative guess per distinct split of the candidate set.
///
/// Guesses are grouped by a 64-bit hash of their feedback vector, then each is
/// checked against its group's representative and promoted to one of its own if
/// it really does split differently. The result is exact rather than
/// probabilistic: a collision can only cost a redundant representative, never
/// lose a distinct split.
pub fn distinct_partitions(source: &impl Source, candidates: &[usize]) -> Vec<usize> {
    let _span = Span::new(3);
    let keys = partition_hash(source, candidates);
    let (first, heads) = group_by_key(&keys);

    // the recheck walks every guess, so it splits across cores like the rest
    let differs = map_guesses(keys.len(), keys.len() * candidates.len(), |guess| {
        let head = heads[guess];
        head != guess
            && candidates
                .iter()
                .any(|&c| source.score(guess, c) != source.score(head, c))
    });

    let mut out = first;
    out.extend((0..keys.len()).filter(|&guess| differs[guess]));
    out
}

/// The words interchangeable with `guess` here: same feedback against every
/// candidate, so the same buckets for the rest of the game.
///
/// `guess` itself is in the result. Only a candidate can come back all green,
/// and only against itself, so such a group holds either one possible answer or
/// any number of words that cannot be the answer.
pub fn same_split(
    source: &impl Source,
    candidates: &[usize],
    guess: usize,
    pool: Option<&[usize]>,
) -> Vec<usize> {
    let _span = Span::new(5);
    let all;
    let pool: &[usize] = match pool {
        Some(p) => p,
        None => {
            all = all_rows(source.n_guesses());
            &all
        }
    };
    // the seventh of the numba kernels, and parallel there too
    let keep = map_guesses(pool.len(), pool.len() * candidates.len(), |slot| {
        let other = pool[slot];
        candidates
            .iter()
            .all(|&c| source.score(other, c) == source.score(guess, c))
    });
    (0..pool.len())
        .filter(|&slot| keep[slot])
        .map(|slot| pool[slot])
        .collect()
}

/// Candidates still consistent with seeing `pattern` after `guess`.
pub fn filter_candidates(
    source: &impl Source,
    guess: usize,
    pattern: u8,
    candidates: &[usize],
) -> Vec<usize> {
    let _span = Span::new(6);
    candidates
        .iter()
        .copied()
        .filter(|&c| source.score(guess, c) == pattern)
        .collect()
}
