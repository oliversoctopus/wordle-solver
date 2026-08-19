//! Exact endgame search.
//!
//! Once the candidate set is small the value function can be computed outright
//! instead of estimated:
//!
//! ```text
//! V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
//! V(S, 0) = loss_penalty
//! ```
//!
//! with `t` the guesses left including this one. Four things keep it tractable:
//! partition dedupe, closed-form leaves at one or two turns, branch and bound
//! on the two-turn value, and a cap on how many splits a node may try at all.
//!
//! The cap is what makes the returned value an upper bound on the true one
//! rather than the true one -- branch and bound finds the best guess early and
//! then spends nearly all its time proving nothing beats it, so the cap buys
//! the answer and skips the proof. Ported from `exact.py`.

use std::collections::HashMap;

use crate::kernels::{distinct_partitions, signature_keys, two_turn_scores, MAX_PACKED};
use crate::memo::Memo;
use crate::source::Source;

pub const MAX_TURNS: usize = 6;

/// How many splits a node may try, best-bound first. Over 1,275 real positions
/// a cap of 25 or 50 picks the uncapped-optimal guess every time.
pub const DEFAULT_MAX_BRANCH: usize = 50;
pub const DEFAULT_MAX_NODES: usize = 2000;
pub const DEFAULT_MEMO_LIMIT: usize = 500_000;

/// Two guesses that split the candidates identically score identically, so this
/// only decides when their feedback vectors are worth comparing.
pub const SPLIT_TIE_TOL: f64 = 1e-9;

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

type Solved = (i64, f64);

pub struct ExactEndgame<'a, S: Source> {
    source: &'a S,
    weights: &'a [f64],
    pub loss_penalty: f64,
    /// Capping at 1 is what makes V non-decreasing in t: running out then costs
    /// exactly one guess, so no deeper value undercuts the two-turn one.
    pub bound_penalty: f64,
    pub max_nodes: usize,
    pub max_branch: Option<usize>,
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
            loss_penalty,
            bound_penalty: loss_penalty.min(1.0),
            max_nodes: DEFAULT_MAX_NODES,
            max_branch: Some(DEFAULT_MAX_BRANCH),
            hard: false,
            memo: Memo::new(DEFAULT_MEMO_LIMIT),
            nodes: 0,
            searches: 0,
        }
    }

    pub fn with_branch(mut self, max_branch: Option<usize>) -> Self {
        self.max_branch = max_branch;
        self
    }

    pub fn with_hard(mut self, hard: bool) -> Self {
        self.hard = hard;
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
        self.solve_inner(candidates, turns_left.min(MAX_TURNS))
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
        self.top_inner(candidates, turns_left.min(MAX_TURNS), k.max(1))
    }

    /// Mass, heaviest member's index within the slice, and that member's share.
    ///
    fn shape(&self, candidates: &[usize]) -> (f64, usize, f64) {
        let masses: Vec<f64> = candidates.iter().map(|&c| self.weights[c]).collect();
        let total = crate::sum::pairwise(&masses);
        // numpy's argmax takes the first maximum on ties
        let mut heaviest = 0;
        for (i, &c) in candidates.iter().enumerate() {
            if self.weights[c] > self.weights[candidates[heaviest]] {
                heaviest = i;
            }
        }
        (total, heaviest, self.weights[candidates[heaviest]] / total)
    }

    fn solve_inner(
        &mut self,
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
                two_turn_scores(self.source, candidates, &weights, self.loss_penalty, pool);
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

        let (representatives, bounds) = self.bounded_splits(candidates);
        let order = self.branch_order(&bounds);

        let mut best_value = f64::INFINITY;
        let mut best_guess: i64 = -1;
        for j in order {
            if bounds[j] >= best_value - 1e-12 {
                break; // sorted, so nothing left can beat the incumbent
            }
            let guess = representatives[j];
            let value = self.evaluate(guess, candidates, total, turns, best_value)?;
            if value < best_value {
                best_value = value;
                best_guess = guess as i64;
            }
        }

        let result = (best_guess, best_value);
        self.memo.put(key, result);
        Ok(result)
    }

    fn top_inner(
        &mut self,
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
                two_turn_scores(self.source, candidates, &weights, self.loss_penalty, pool);
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut out: Vec<Solved> = Vec::new();
            for j in order {
                let guess = if self.hard { candidates[j] } else { j };
                if self.duplicates(guess, scores[j], &out, candidates) {
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

        let (representatives, bounds) = self.bounded_splits(candidates);
        let order = self.branch_order(&bounds);

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
            let value = self.evaluate(guess, candidates, total, turns, cutoff)?;
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
    fn duplicates(
        &self,
        guess: usize,
        score: f64,
        listed: &[Solved],
        candidates: &[usize],
    ) -> bool {
        listed.iter().any(|&(other, value)| {
            (score - value).abs() <= SPLIT_TIE_TOL
                && candidates
                    .iter()
                    .all(|&c| self.source.score(guess, c) == self.source.score(other as usize, c))
        })
    }

    /// Exact value of one guess, abandoned early if it cannot beat `cutoff`.
    /// Starts from the per-bucket lower bounds and replaces them with exact
    /// values one bucket at a time.
    fn evaluate(
        &mut self,
        guess: usize,
        candidates: &[usize],
        total: f64,
        turns: usize,
        cutoff: f64,
    ) -> Result<f64, BudgetExceeded> {
        // group by pattern, visiting patterns in ascending order as np.unique does
        let mut by_pattern: HashMap<u8, Vec<usize>> = HashMap::new();
        for &c in candidates {
            by_pattern
                .entry(self.source.score(guess, c))
                .or_default()
                .push(c);
        }
        let mut patterns: Vec<u8> = by_pattern.keys().copied().collect();
        patterns.sort_unstable();

        let mut buckets: Vec<(f64, f64, Vec<usize>)> = Vec::new();
        let mut running = 1.0;
        for pattern in patterns {
            if pattern == crate::feedback::WIN_PATTERN {
                continue;
            }
            let members = &by_pattern[&pattern];
            let (mass, _, sub_p_max) = self.shape(members);
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
            buckets.push((share, bound, members.clone()));
        }

        if running >= cutoff {
            return Ok(f64::INFINITY);
        }
        // biggest contributors first: they tighten the total fastest
        buckets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (share, bound, members) in buckets {
            let (_, value) = self.solve_inner(&members, turns - 1)?;
            running += share * (value - bound);
            if running >= cutoff {
                return Ok(f64::INFINITY);
            }
        }
        Ok(running)
    }

    /// `(representatives, two-turn bounds)` -- one guess per distinct split,
    /// each with the lower bound the search orders and prunes by.
    fn bounded_splits(&self, candidates: &[usize]) -> (Vec<usize>, Vec<f64>) {
        let representatives = if self.hard {
            // nothing for the dedupe to merge: only a candidate itself can come
            // back all-green, so no two of them split S the same way
            candidates.to_vec()
        } else {
            self.distinct_partitions(candidates)
        };
        let weights = self.slice_weights(candidates);
        let bounds = two_turn_scores(
            self.source,
            candidates,
            &weights,
            self.bound_penalty,
            Some(&representatives),
        );
        (representatives, bounds)
    }

    fn distinct_partitions(&self, candidates: &[usize]) -> Vec<usize> {
        if candidates.len() <= MAX_PACKED {
            // the packed key *is* the split, so equal keys need no checking
            let keys = signature_keys(self.source, candidates);
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
            distinct_partitions(self.source, candidates)
        }
    }

    /// Guesses in the order the search should try them, capped.
    fn branch_order(&self, bounds: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..bounds.len()).collect();
        order.sort_by(|&a, &b| {
            bounds[a]
                .partial_cmp(&bounds[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if let Some(cap) = self.max_branch {
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
