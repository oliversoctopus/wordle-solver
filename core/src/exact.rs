//! Exact endgame search, ported from `exact.py`.
//!
//! Once the candidate set is small the value function is computed outright:
//!
//! ```text
//! V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
//! V(S, 0) = loss_penalty
//! ```
//!
//! with `t` the guesses left. README *The exact endgame* has the rest.

use crate::fused;
use crate::kernels::{distinct_partitions, signature_keys, MAX_PACKED};
use crate::memo::Memo;
use crate::source::{Columns, Source, Transposed};
use crate::sum::pairwise_of;

pub const MAX_TURNS: usize = 6;

/// How many splits a node may try, best-bound first. README *The exact
/// endgame* has what the cap costs.
pub const DEFAULT_MAX_BRANCH: usize = 50;

/// How many splits the node being *ranked* may try. Only that node ranks, and
/// its runners-up sit near the cap in a way its winner does not.
pub const DEFAULT_ROOT_BRANCH: usize = 150;

/// That cap, or whatever `WORDLE_ROOT_BRANCH` narrows it to. `exact.py` reads
/// the same variable, so an experiment that turns the wider ranking off gets
/// both implementations at once and parity still holds. There is no environment
/// under WASM, which is the one build that should never be narrowed anyway.
pub fn root_branch_default() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("WORDLE_ROOT_BRANCH") {
        Err(_) => DEFAULT_ROOT_BRANCH,
        Ok(setting) => parse_root_branch(&setting).unwrap_or_else(|| {
            eprintln!("WORDLE_ROOT_BRANCH={setting:?} is not a cap; using {DEFAULT_ROOT_BRANCH}");
            DEFAULT_ROOT_BRANCH
        }),
    })
}

/// `None` for anything that is not a positive count. `exact.py` reads the same
/// variable through `int`, which raises on the same input rather than warning.
fn parse_root_branch(setting: &str) -> Option<usize> {
    setting.trim().parse().ok().filter(|&n| n > 0)
}

pub const DEFAULT_MAX_NODES: usize = 2000;
pub const DEFAULT_MEMO_LIMIT: usize = 500_000;

/// Above this many candidates a search is not worth tabulating: the scan it
/// would save is the only one a wide position does.
pub const COLUMNS_MAX: usize = 256;

/// Two guesses that split the candidates identically score identically -- the
/// same buckets filled in the same order -- so this gates the feedback-vector
/// comparison at equality rather than at a tolerance.
pub const SPLIT_TIE_TOL: f64 = 0.0;

/// The branch cap orders on the bound rounded to this, so two bounds within a
/// float's noise of each other tie and the lower word index decides. `exact.py`
/// rounds the same way, so the two agree without agreeing bitwise.
pub const BRANCH_TIE_TOL: f64 = 1e-11;

/// The search visited more nodes than allowed; the caller falls back to the
/// heuristic policy.
#[derive(Debug)]
pub struct BudgetExceeded {
    pub nodes: usize,
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "more than {} nodes", self.nodes)
    }
}

impl std::error::Error for BudgetExceeded {}

pub type Solved = (i64, f64);

pub struct ExactEndgame<'a, S: Source> {
    source: &'a S,
    weights: &'a [f64],
    /// The answer-major table, where one is held. Lent to the scans that run
    /// against `source`; a search off `Columns` has a smaller one already.
    fused: Option<&'a Transposed>,
    pub loss_penalty: f64,
    /// Capping at 1 is what makes V non-decreasing in t: running out then costs
    /// exactly one guess, so no deeper value undercuts the two-turn one.
    pub bound_penalty: f64,
    pub max_nodes: usize,
    pub max_branch: Option<usize>,
    pub root_branch: Option<usize>,
    pub hard: bool,
    memo: Memo<(Vec<usize>, usize), Solved>,
    pub nodes: usize,
    pub searches: usize,
}

impl<'a, S: Source> ExactEndgame<'a, S> {
    pub fn new(source: &'a S, weights: &'a [f64], loss_penalty: f64) -> ExactEndgame<'a, S> {
        ExactEndgame {
            source,
            weights,
            fused: None,
            loss_penalty,
            bound_penalty: loss_penalty.min(1.0),
            max_nodes: DEFAULT_MAX_NODES,
            max_branch: Some(DEFAULT_MAX_BRANCH),
            root_branch: Some(root_branch_default()),
            hard: false,
            memo: Memo::new(DEFAULT_MEMO_LIMIT),
            nodes: 0,
            searches: 0,
        }
    }

    /// `None` asks for the proof, at both caps: the ranked node must search
    /// every split too, or its list is still the capped one.
    pub fn with_branch(mut self, max_branch: Option<usize>) -> Self {
        self.max_branch = max_branch;
        if max_branch.is_none() {
            self.root_branch = None;
        }
        self
    }

    pub fn with_hard(mut self, hard: bool) -> Self {
        self.hard = hard;
        self
    }

    pub fn with_fused(mut self, fused: Option<&'a Transposed>) -> Self {
        self.fused = fused;
        self
    }

    /// `(guess, expected score)` under optimal play within the cap.
    pub fn solve(
        &mut self,
        candidates: &[usize],
        turns_left: usize,
    ) -> Result<Solved, BudgetExceeded> {
        self.searches += 1;
        self.nodes = 0;
        let turns = turns_left.min(MAX_TURNS);
        match self.columns(candidates, turns) {
            Some(held) => self.solve_inner(&held, None, candidates, turns),
            None => self.solve_inner(self.source, self.fused, candidates, turns),
        }
    }

    /// The k best splits of this position, best first, each score as true as
    /// `solve`'s. The k-th best value stands in for the incumbent, so a guess is
    /// still abandoned the moment it cannot reach the list.
    pub fn top(
        &mut self,
        candidates: &[usize],
        turns_left: usize,
        k: usize,
    ) -> Result<Vec<Solved>, BudgetExceeded> {
        self.searches += 1;
        self.nodes = 0;
        let turns = turns_left.min(MAX_TURNS);
        let k = k.max(1);
        match self.columns(candidates, turns) {
            Some(held) => self.top_inner(&held, None, candidates, turns, k),
            None => self.top_inner(self.source, self.fused, candidates, turns, k),
        }
    }

    /// The table to answer this search's scans from, if it will pay for itself.
    /// A search that recurses rescans the pool once per node; one that returns
    /// from a closed form scans once and would only be paying to tabulate.
    fn columns(&self, candidates: &[usize], turns: usize) -> Option<Columns> {
        let m = candidates.len();
        (turns >= 3 && (3..=COLUMNS_MAX).contains(&m))
            .then(|| Columns::build(self.source, candidates))
    }

    /// Mass, heaviest member's index within the slice, and that member's share.
    ///
    fn shape(&self, candidates: &[usize]) -> (f64, usize, f64) {
        let total = pairwise_of(candidates.len(), |i| self.weights[candidates[i]]);
        // numpy's argmax takes the first maximum on ties
        let mut heaviest = 0;
        for (i, &c) in candidates.iter().enumerate() {
            if self.weights[c] > self.weights[candidates[heaviest]] {
                heaviest = i;
            }
        }
        (total, heaviest, self.weights[candidates[heaviest]] / total)
    }

    fn solve_inner<F: Source>(
        &mut self,
        fb: &F,
        tt: Option<&Transposed>,
        candidates: &[usize],
        turns: usize,
    ) -> Result<Solved, BudgetExceeded> {
        let m = candidates.len();
        assert!(
            m > 0,
            "no candidates left; the feedback given is inconsistent"
        );
        let (total, heaviest, p_max) = self.shape(candidates);

        if m == 1 {
            return Ok((candidates[0] as i64, 1.0));
        }
        if turns <= 1 {
            // nothing to learn from the feedback; play the likeliest candidate
            return Ok((
                candidates[heaviest] as i64,
                1.0 + (1.0 - p_max) * self.loss_penalty,
            ));
        }
        if m == 2 {
            // guess the likelier one; if wrong the other is known
            return Ok((candidates[heaviest] as i64, 2.0 - p_max));
        }

        let key = (candidates.to_vec(), turns);
        if let Some(cached) = self.memo.get(&key) {
            return Ok(cached);
        }

        // the same two-turn position is reached along many different guess
        // orders -- four calls in five are repeats -- so even the closed form
        // is memoized, and runs once per distinct position
        if turns == 2 {
            let weights = self.slice_weights(candidates);
            let pool = if self.hard { Some(candidates) } else { None };
            let scores =
                fused::two_turn_scores(fb, tt, candidates, &weights, self.loss_penalty, pool);
            let best = argmin(&scores);
            let guess = if self.hard { candidates[best] } else { best };
            let result = (guess as i64, scores[best]);
            self.memo.put(key, result);
            return Ok(result);
        }

        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(BudgetExceeded {
                nodes: self.max_nodes,
            });
        }

        let (representatives, bounds) = self.bounded_splits(fb, tt, candidates);
        let order = self.branch_order(&bounds, self.max_branch);

        let mut best_value = f64::INFINITY;
        let mut best_guess: i64 = -1;
        for j in order {
            if bounds[j] >= best_value - 1e-12 {
                break; // sorted, so nothing left can beat the incumbent
            }
            let guess = representatives[j];
            let value = self.evaluate(fb, tt, guess, candidates, total, turns, best_value)?;
            if value < best_value {
                best_value = value;
                best_guess = guess as i64;
            }
        }

        let result = (best_guess, best_value);
        self.memo.put(key, result);
        Ok(result)
    }

    fn top_inner<F: Source>(
        &mut self,
        fb: &F,
        tt: Option<&Transposed>,
        candidates: &[usize],
        turns: usize,
        k: usize,
    ) -> Result<Vec<Solved>, BudgetExceeded> {
        let m = candidates.len();
        assert!(
            m > 0,
            "no candidates left; the feedback given is inconsistent"
        );
        let (total, _, _) = self.shape(candidates);

        if m == 1 {
            return Ok(vec![(candidates[0] as i64, 1.0)]);
        }
        if turns <= 1 {
            // every word that cannot be the answer loses alike, so the ranking
            // here is the candidates by weight and nothing else
            let order = self.by_weight_descending(candidates);
            return Ok(order
                .into_iter()
                .take(k)
                .map(|i| {
                    (
                        candidates[i] as i64,
                        1.0 + (1.0 - self.weights[candidates[i]] / total) * self.loss_penalty,
                    )
                })
                .collect());
        }
        if m == 2 {
            let order = self.by_weight_descending(candidates);
            return Ok(order
                .into_iter()
                .map(|i| {
                    (
                        candidates[i] as i64,
                        2.0 - self.weights[candidates[i]] / total,
                    )
                })
                .collect());
        }
        if turns == 2 {
            // the closed form scores every guess rather than the distinct
            // splits, so the ranking is deduped on the way out
            let weights = self.slice_weights(candidates);
            let pool = if self.hard { Some(candidates) } else { None };
            let scores =
                fused::two_turn_scores(fb, tt, candidates, &weights, self.loss_penalty, pool);
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut out: Vec<Solved> = Vec::new();
            for j in order {
                let guess = if self.hard { candidates[j] } else { j };
                if self.duplicates(fb, guess, scores[j], &out, candidates) {
                    continue;
                }
                out.push((guess as i64, scores[j]));
                if out.len() == k {
                    break;
                }
            }
            return Ok(out);
        }

        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(BudgetExceeded {
                nodes: self.max_nodes,
            });
        }

        let (representatives, bounds) = self.bounded_splits(fb, tt, candidates);
        let order = self.branch_order(&bounds, self.root_branch);

        let mut best: Vec<(f64, usize)> = Vec::new(); // best first, never longer than k
        for j in order {
            let cutoff = if best.len() == k {
                best[best.len() - 1].0
            } else {
                f64::INFINITY
            };
            if bounds[j] >= cutoff - 1e-12 {
                break;
            }
            let guess = representatives[j];
            let value = self.evaluate(fb, tt, guess, candidates, total, turns, cutoff)?;
            if value < cutoff {
                best.push((value, guess));
                best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                best.truncate(k);
            }
        }

        // the winner was found under a cutoff no tighter than solve's, so it is
        // the same word solve would return and worth leaving in the memo
        self.memo
            .put((candidates.to_vec(), turns), (best[0].1 as i64, best[0].0));
        Ok(best.into_iter().map(|(v, g)| (g as i64, v)).collect())
    }

    /// Whether `guess` splits the candidates as something already listed does,
    /// and so would be that guess again under another spelling.
    fn duplicates<F: Source>(
        &self,
        fb: &F,
        guess: usize,
        score: f64,
        listed: &[Solved],
        candidates: &[usize],
    ) -> bool {
        listed.iter().any(|&(other, value)| {
            (score - value).abs() <= SPLIT_TIE_TOL
                && candidates
                    .iter()
                    .all(|&c| fb.score(guess, c) == fb.score(other as usize, c))
        })
    }

    /// Exact value of one guess, abandoned early if it cannot beat `cutoff`.
    /// Starts from the per-bucket lower bounds and replaces them with exact
    /// values one bucket at a time.
    fn evaluate<F: Source>(
        &mut self,
        fb: &F,
        tt: Option<&Transposed>,
        guess: usize,
        candidates: &[usize],
        total: f64,
        turns: usize,
        cutoff: f64,
    ) -> Result<f64, BudgetExceeded> {
        // Patterns once, then the distinct ones in ascending order, which is
        // what np.unique gives the Python.  A map keyed on the pattern would
        // allocate per bucket and this is the search's hot recursion; the sets
        // here are small enough that a sort and a scan beat hashing.
        let scored: Vec<u8> = candidates
            .iter()
            .map(|&c| fb.score(guess, c))
            .collect();
        let mut patterns = scored.clone();
        patterns.sort_unstable();
        patterns.dedup();

        let mut buckets: Vec<(f64, f64, Vec<usize>)> = Vec::new();
        let mut running = 1.0;
        for pattern in patterns {
            if pattern == crate::feedback::WIN_PATTERN {
                continue;
            }
            let members: Vec<usize> = candidates
                .iter()
                .zip(&scored)
                .filter(|(_, &p)| p == pattern)
                .map(|(&c, _)| c)
                .collect();
            let (mass, _, sub_p_max) = self.shape(&members);
            let share = mass / total;
            let size = members.len();
            if size == 1 {
                running += share; // certain win next turn
                continue;
            }
            if size == 2 {
                running += share * (2.0 - sub_p_max);
                continue;
            }
            let bound = 1.0 + (1.0 - sub_p_max) * self.bound_penalty;
            running += share * bound;
            buckets.push((share, bound, members));
        }

        if running >= cutoff {
            return Ok(f64::INFINITY);
        }
        // biggest contributors first: they tighten the total fastest
        buckets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (share, bound, members) in buckets {
            let (_, value) = self.solve_inner(fb, tt, &members, turns - 1)?;
            running += share * (value - bound);
            if running >= cutoff {
                return Ok(f64::INFINITY);
            }
        }
        Ok(running)
    }

    /// `(representatives, two-turn bounds)` -- one guess per distinct split,
    /// each with the lower bound the search orders and prunes by.
    fn bounded_splits<F: Source>(
        &self,
        fb: &F,
        tt: Option<&Transposed>,
        candidates: &[usize],
    ) -> (Vec<usize>, Vec<f64>) {
        let representatives = if self.hard {
            // nothing for the dedupe to merge: only a candidate itself can come
            // back all-green, so no two of them split S the same way
            candidates.to_vec()
        } else {
            self.distinct_partitions(fb, candidates)
        };
        let weights = self.slice_weights(candidates);
        let bounds = fused::two_turn_scores(
            fb,
            tt,
            candidates,
            &weights,
            self.bound_penalty,
            Some(&representatives),
        );
        (representatives, bounds)
    }

    fn distinct_partitions<F: Source>(&self, fb: &F, candidates: &[usize]) -> Vec<usize> {
        if candidates.len() <= MAX_PACKED {
            // the packed key *is* the split, so equal keys need no checking
            let keys = signature_keys(fb, candidates);
            let mut order: Vec<usize> = (0..keys.len()).collect();
            order.sort_by_key(|&i| (keys[i], i));
            let mut first = Vec::new();
            let mut previous: Option<i64> = None;
            for i in order {
                if previous != Some(keys[i]) {
                    first.push(i);
                    previous = Some(keys[i]);
                }
            }
            first
        } else {
            distinct_partitions(fb, candidates)
        }
    }

    /// Guesses in the order the search should try them, capped.
    ///
    /// The bounds tie in bulk -- a position can hold a thousand splits at one
    /// value -- so which of them the cap keeps rests entirely on the tie-break,
    /// and it has to be one `exact.py` can spell too. Word index is that, over
    /// the bound rounded to `BRANCH_TIE_TOL`: `np.argsort(kind="stable")` on
    /// the same key on the far side.
    ///
    /// Rounded and not floored because the ties sit on whole values like 2.0,
    /// which a floor would run a bucket edge through; spelled `floor(x + 0.5)`
    /// because numpy rounds a half to even where Rust rounds it away from zero,
    /// and floor is IEEE in both.
    fn branch_order(&self, bounds: &[f64], cap: Option<usize>) -> Vec<usize> {
        let mut order: Vec<usize> = (0..bounds.len()).collect();
        let key = |i: usize| (bounds[i] / BRANCH_TIE_TOL + 0.5).floor();
        order.sort_by(|&a, &b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if let Some(cap) = cap {
            order.truncate(cap);
        }
        order
    }

    fn by_weight_descending(&self, candidates: &[usize]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        // stable, so equal weights leave the winner the one solve names
        order.sort_by(|&a, &b| {
            self.weights[candidates[b]]
                .partial_cmp(&self.weights[candidates[a]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order
    }

    fn slice_weights(&self, candidates: &[usize]) -> Vec<f64> {
        candidates.iter().map(|&c| self.weights[c]).collect()
    }

    pub fn clear_memo(&mut self) {
        self.memo.clear();
    }

    /// Exchanges the cache for another, so a caller that outlives the search
    /// can keep what it found. See `solver::Memos`.
    pub fn swap_memo(&mut self, other: &mut Memo<(Vec<usize>, usize), Solved>) {
        std::mem::swap(&mut self.memo, other);
    }
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v < values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{parse_root_branch, DEFAULT_ROOT_BRANCH};

    #[test]
    fn the_root_cap_takes_a_count_and_nothing_else() {
        assert_eq!(parse_root_branch("50"), Some(50));
        assert_eq!(parse_root_branch(" 50\n"), Some(50));
        for bad in ["", "0", "-1", "50 splits", "many"] {
            assert_eq!(parse_root_branch(bad), None, "{bad:?} read as a cap");
        }
        assert_eq!(DEFAULT_ROOT_BRANCH, 150);
    }
}
