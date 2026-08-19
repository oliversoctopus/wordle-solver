//! Scoring every guess against a candidate set.
//!
//! All three scorers share one shape: for each guessable word, bucket the
//! candidates by the feedback pattern they would produce, then reduce those 243
//! buckets to a single number. Entropy reduces them to bits gained, the
//! expected guess count reduces them through the fitted `f`, and the two-turn
//! value reduces them to the exact score when two guesses remain.
//!
//! Ported from `kernels.py`. Not from its arithmetic order, which is not worth
//! preserving: the closest pair of scores among all 12,947 guesses at the
//! opening differs by 1.1e-9 against an ulp of 4e-16, so nothing the solver
//! decides turns on the last few bits of a total.
//!
//! Every kernel takes a `pool` of guess indices, standing in for all of them
//! when `None`. That parameter is what keeps a deeper search affordable, and it
//! is also how hard mode works: the pool becomes the candidate set itself.

use crate::estimator::{f_eval_x, Form};
use crate::feedback::{N_PATTERNS, WIN_PATTERN};
use crate::source::Source;

/// The candidate mass, accumulated one at a time in candidate order.
///
/// Deliberately not `sum::pairwise`, which is more accurate and wrong here: the
/// buckets are filled one candidate at a time, so a guess that drops every
/// candidate into one bucket only gets `mass / total` to come out at exactly
/// 1.0 -- and so scores exactly zero bits, as a guess that learns nothing must
/// -- if the total was accumulated the same way. Two of the 12,947 guesses do
/// that on a five-candidate set.
#[inline]
fn bucketwise_total(values: &[f64]) -> f64 {
    let mut total = 0.0;
    for &v in values {
        total += v;
    }
    total
}

/// Below this many candidates the reduce walks the candidates a second time
/// rather than sweeping all 243 patterns, taking each bucket the first time it
/// is reached and zeroing it there. Above it the second walk stops paying for
/// itself, since it re-reads the table where the sweep only touches
/// accumulators already in cache.
pub const RESCAN_MAX: usize = 128;

/// Resolves `pool = None` to every guess.
fn pool_or_all(source: &impl Source, pool: Option<&[usize]>) -> Vec<usize> {
    match pool {
        Some(p) => p.to_vec(),
        None => (0..source.n_guesses()).collect(),
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
    let pool = pool_or_all(source, pool);
    let m = candidates.len();
    let total = bucketwise_total(weights);
    let mut out = vec![0.0; pool.len()];
    let mut mass = [0.0f64; N_PATTERNS];

    for (slot, &guess) in pool.iter().enumerate() {
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
            for slot in mass.iter_mut() {
                if *slot > 0.0 {
                    let p = *slot / total;
                    h -= p * p.log2();
                    *slot = 0.0;
                }
            }
        }
        out[slot] = h;
    }
    out
}

/// Guesses still needed from one bucket. One and two candidates are worth
/// exactly 1 and `2 - p_max`, so the estimator is only consulted above that.
#[inline]
fn bucket_guesses(
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
/// The `GGGGG` bucket contributes nothing, which is what makes a guess that
/// could itself be the answer come out ahead of an equally informative one that
/// could not.
pub fn expected_guesses_all(
    source: &impl Source,
    candidates: &[usize],
    weights: &[f64],
    params: &[f64],
    form: &Form,
    pool: Option<&[usize]>,
) -> Vec<f64> {
    let pool = pool_or_all(source, pool);
    let m = candidates.len();
    let total = bucketwise_total(weights);
    let wlw = w_log_w(weights);
    let mut out = vec![0.0; pool.len()];

    let mut buckets = Buckets::new();

    for (slot, &guess) in pool.iter().enumerate() {
        for i in 0..m {
            let r = source.score(guess, candidates[i]) as usize;
            let w = weights[i];
            buckets.mass[r] += w;
            buckets.count[r] += 1.0;
            buckets.wlw[r] += wlw[i];
            if w > buckets.heaviest[r] {
                buckets.heaviest[r] = w;
            }
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
    out
}

/// The four 243-slot accumulators the expected-guess reduce needs: a bucket's
/// running mass, its candidate count, its total of `w log2 w`, and its heaviest
/// single weight.
struct Buckets {
    mass: [f64; N_PATTERNS],
    count: [f64; N_PATTERNS],
    wlw: [f64; N_PATTERNS],
    heaviest: [f64; N_PATTERNS],
}

impl Buckets {
    fn new() -> Buckets {
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
    fn take(&mut self, r: usize, total: f64, params: &[f64], form: &Form) -> f64 {
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
/// lower bound on the value at any larger number of turns, which is what the
/// exact search prunes with.
pub fn two_turn_scores(
    source: &impl Source,
    candidates: &[usize],
    weights: &[f64],
    penalty: f64,
    pool: Option<&[usize]>,
) -> Vec<f64> {
    let pool = pool_or_all(source, pool);
    let m = candidates.len();
    let total = bucketwise_total(weights);
    let mut out = vec![0.0; pool.len()];

    let mut mass = [0.0f64; N_PATTERNS];
    let mut heaviest = [0.0f64; N_PATTERNS];

    for (slot, &guess) in pool.iter().enumerate() {
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
    out
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
    (0..source.n_guesses())
        .map(|guess| {
            let mut key = 0i64;
            for &c in candidates {
                key = (key << 8) | source.score(guess, c) as i64;
            }
            key
        })
        .collect()
}

/// FNV-1a over each guess's feedback vector, for grouping equal splits.
fn partition_hash(source: &impl Source, candidates: &[usize]) -> Vec<i64> {
    (0..source.n_guesses())
        .map(|guess| {
            let mut h = FNV_OFFSET;
            for &c in candidates {
                h = (h ^ source.score(guess, c) as i64).wrapping_mul(FNV_PRIME);
            }
            h
        })
        .collect()
}

/// `numpy.unique(keys, return_index=True)`: for each distinct key in ascending
/// order, the lowest index carrying it. The order matters -- it decides which
/// word represents a split, and so the order the search tries them in.
fn unique_first(keys: &[i64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&i| (keys[i], i));
    let mut first = Vec::new();
    let mut previous: Option<i64> = None;
    for &i in &order {
        if previous != Some(keys[i]) {
            first.push(i);
            previous = Some(keys[i]);
        }
    }
    first
}

/// One representative guess per distinct split of the candidate set.
///
/// Guesses are grouped by a 64-bit hash of their feedback vector, then each is
/// checked against its group's representative and promoted to one of its own if
/// it really does split differently. The result is exact rather than
/// probabilistic: a collision can only cost a redundant representative, never
/// lose a distinct split.
pub fn distinct_partitions(source: &impl Source, candidates: &[usize]) -> Vec<usize> {
    let keys = partition_hash(source, candidates);
    let first = unique_first(&keys);

    // head of each guess's group, by key
    let mut head_of_key = std::collections::HashMap::with_capacity(first.len());
    for &i in &first {
        head_of_key.insert(keys[i], i);
    }

    let mut extra = Vec::new();
    for guess in 0..keys.len() {
        let head = head_of_key[&keys[guess]];
        if head != guess
            && candidates
                .iter()
                .any(|&c| source.score(guess, c) != source.score(head, c))
        {
            extra.push(guess);
        }
    }

    let mut out = first;
    out.extend(extra);
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
    pool_or_all(source, pool)
        .into_iter()
        .filter(|&other| {
            candidates
                .iter()
                .all(|&c| source.score(other, c) == source.score(guess, c))
        })
        .collect()
}

/// Candidates still consistent with seeing `pattern` after `guess`.
pub fn filter_candidates(
    source: &impl Source,
    guess: usize,
    pattern: u8,
    candidates: &[usize],
) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&c| source.score(guess, c) == pattern)
        .collect()
}
