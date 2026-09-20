//! Choosing a guess: the dispatch that decides which valuation answers.
//!
//! Four things can settle a position: a closed form at one or two candidates,
//! the exact search where it reaches, the heaviest candidate on the last turn,
//! and otherwise the heuristic scorer. Ported from `solver.py`, the fused
//! answer-major dispatch included: lend the solver a `Transposed` and
//! mid-sized positions go to `fused`, with identical scores.

use crate::estimator::{effective_size, Form};
use crate::exact::{BudgetExceeded, ExactEndgame, Solved, MAX_TURNS, SPLIT_TIE_TOL};
use crate::fused;
use crate::kernels;
use crate::memo::Memo;
use crate::source::{Source, Transposed};
use crate::sum::{pairwise, pairwise_of};

pub const DEFAULT_EXACT_MAX: usize = 60;
pub const DEFAULT_LOOKAHEAD_BRANCH: usize = 10;
pub const DEFAULT_LOSS_PENALTY: f64 = 5.0;
pub const DEFAULT_MEMO_LIMIT: usize = 200_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Maximize expected information gain, and nothing else.
    Entropy,
    /// Minimize the estimated total guesses the game will take.
    Expected,
}


/// How many positions a browser may keep. The native sweeps keep far more --
/// `DEFAULT_MEMO_LIMIT` -- but they are not sharing an address space with a
/// page, and what a limit costs is only ever time. See `memo`.
pub const BROWSER_MEMO_LIMIT: usize = 50_000;
pub const BROWSER_EXACT_MEMO_LIMIT: usize = 100_000;

/// The positions a solver has already worked out, held apart from the solver.
///
/// A `Solver` borrows its source and its weights, so it cannot outlive the
/// request that built it. What it caches has no such trouble: a position's
/// value is a pure function of the position, and stays true afterwards.
/// Swapping these in and out is what lets the same board arrive twice -- which
/// is the usual case, the page sending the whole board on every submit -- and
/// be solved once.
pub struct Memos {
    best: Memo<(Vec<usize>, usize), (i64, f64)>,
    lookahead: Memo<(Vec<usize>, usize, usize), (i64, f64)>,
    exact: Memo<(Vec<usize>, usize), Solved>,
}

impl Memos {
    pub fn new() -> Memos {
        Memos {
            best: Memo::new(BROWSER_MEMO_LIMIT),
            lookahead: Memo::new(BROWSER_MEMO_LIMIT),
            exact: Memo::new(BROWSER_EXACT_MEMO_LIMIT),
        }
    }
}

impl Default for Memos {
    fn default() -> Memos {
        Memos::new()
    }
}
pub struct Solver<'a, S: Source> {
    source: &'a S,
    weights: &'a [f64],
    params: Vec<f64>,
    form: Form,
    pub mode: Mode,
    pub hard: bool,
    pub use_exact: bool,
    pub exact_max: usize,
    pub lookahead: usize,
    pub lookahead_branch: usize,
    pub lookahead_exact_max: usize,
    /// Bits paid to a guess that could itself be the answer. Priced in bits, so
    /// it only means something against information gain.
    pub win_bonus: f64,
    /// An answer-major copy of the table, or `None` to score the shipped
    /// way. A speed knob and never a behavior one: the scores are identical.
    fused: Option<&'a Transposed>,
    exact: ExactEndgame<'a, S>,
    memo: Memo<(Vec<usize>, usize), (i64, f64)>,
    lookahead_memo: Memo<(Vec<usize>, usize, usize), (i64, f64)>,
    pub exact_hits: usize,
    pub exact_bailouts: usize,
}

impl<'a, S: Source> Solver<'a, S> {
    pub fn new(
        source: &'a S,
        weights: &'a [f64],
        params: Vec<f64>,
        form: Form,
        mode: Mode,
    ) -> Solver<'a, S> {
        Solver {
            source,
            weights,
            params,
            form,
            mode,
            hard: false,
            use_exact: true,
            exact_max: DEFAULT_EXACT_MAX,
            lookahead: 0,
            lookahead_branch: DEFAULT_LOOKAHEAD_BRANCH,
            lookahead_exact_max: DEFAULT_EXACT_MAX,
            win_bonus: 0.0,
            fused: None,
            exact: ExactEndgame::new(source, weights, DEFAULT_LOSS_PENALTY),
            memo: Memo::new(DEFAULT_MEMO_LIMIT),
            lookahead_memo: Memo::new(DEFAULT_MEMO_LIMIT),
            exact_hits: 0,
            exact_bailouts: 0,
        }
    }

    /// Exchanges the solver's caches for the ones given. Called twice around a
    /// request: once to lend them, once to take back what the request added.
    pub fn swap_memos(&mut self, memos: &mut Memos) {
        std::mem::swap(&mut self.memo, &mut memos.best);
        std::mem::swap(&mut self.lookahead_memo, &mut memos.lookahead);
        self.exact.swap_memo(&mut memos.exact);
    }

    /// Lends the solver an answer-major copy of the table, which `fused`
    /// dispatches mid-sized positions to. Off unless asked for: it is a second
    /// copy of the table, and it wins nothing at the widest positions or the
    /// smallest.
    pub fn with_fused(mut self, fused: Option<&'a Transposed>) -> Self {
        self.fused = fused;
        self.exact = self.exact.with_fused(fused);
        self
    }

    pub fn with_hard(mut self, hard: bool) -> Self {
        self.hard = hard;
        self.exact = self.exact.with_hard(hard);
        self
    }

    pub fn with_loss_penalty(mut self, penalty: f64) -> Self {
        self.exact = ExactEndgame::new(self.source, self.weights, penalty).with_hard(self.hard);
        self
    }

    pub fn with_exact_max(mut self, exact_max: usize) -> Self {
        self.exact_max = exact_max;
        self.lookahead_exact_max = exact_max;
        self
    }

    pub fn with_use_exact(mut self, use_exact: bool) -> Self {
        self.use_exact = use_exact;
        self
    }

    pub fn with_lookahead(mut self, plies: usize) -> Self {
        self.lookahead = plies;
        self
    }

    pub fn with_win_bonus(mut self, bonus: f64) -> Self {
        self.win_bonus = bonus;
        self
    }

    /// How many guessable words there are, and so how big the opening
    /// candidate set is.
    pub fn n_words(&self) -> usize {
        self.source.n_guesses()
    }

    /// Rows of the feedback table a guess may come from: every word, or in hard
    /// mode only the words that could still be the answer.
    pub fn guess_pool<'c>(&self, candidates: &'c [usize]) -> Option<&'c [usize]> {
        if self.hard {
            Some(candidates)
        } else {
            None
        }
    }

    /// The word a pooled score belongs to. Without a pool the score's own
    /// position is already the word index.
    fn row(pool: Option<&[usize]>, index: usize) -> usize {
        match pool {
            Some(p) => p[index],
            None => index,
        }
    }

    fn slice_weights(&self, candidates: &[usize]) -> Vec<f64> {
        candidates.iter().map(|&c| self.weights[c]).collect()
    }

    pub fn effective_size(&self, candidates: &[usize]) -> f64 {
        effective_size(&self.slice_weights(candidates))
    }

    /// Score for every guessable word, or, with `pool`, for the words it names
    /// and in its order. Higher is better in entropy mode, lower in expected.
    pub fn score_all(&self, candidates: &[usize], pool: Option<&[usize]>) -> Vec<f64> {
        let weights = self.slice_weights(candidates);
        match self.mode {
            Mode::Entropy => {
                let scores = fused::ig_all(self.source, self.fused, candidates, &weights, pool);
                if self.win_bonus != 0.0 {
                    self.priced_win(scores, candidates, &weights, pool)
                } else {
                    scores
                }
            }
            Mode::Expected => fused::expected_guesses_all(
                self.source,
                self.fused,
                candidates,
                &weights,
                &self.params,
                &self.form,
                pool,
            ),
        }
    }

    /// Pay each guess that could itself be the answer for the chance it is one.
    fn priced_win(
        &self,
        mut scores: Vec<f64>,
        candidates: &[usize],
        weights: &[f64],
        pool: Option<&[usize]>,
    ) -> Vec<f64> {
        let total = pairwise(weights);
        if total <= 0.0 {
            return scores;
        }
        for (i, &c) in candidates.iter().enumerate() {
            let paid = self.win_bonus * weights[i] / total;
            match pool {
                // candidates run ascending, so a search finds the pooled ones
                Some(p) => {
                    if let Ok(slot) = p.binary_search(&c) {
                        scores[slot] += paid;
                    }
                }
                None => scores[c] += paid,
            }
        }
        scores
    }

    /// The same score, for a single guess.
    pub fn score_one(&self, guess: usize, candidates: &[usize]) -> f64 {
        self.score_all(candidates, Some(&[guess]))[0]
    }

    /// Two turns or fewer costs one pass over the guesses, the same as scoring
    /// them heuristically, so it is always worth doing. Deeper than that, only
    /// small candidate sets are affordable.
    fn exact_applies(&self, m: usize, turns: usize, cap: Option<usize>) -> bool {
        // the exact value is a score in guesses, so it only composes with the
        // expected-guess objective; entropy mode stays a clean baseline
        if !self.use_exact || self.mode != Mode::Expected {
            return false;
        }
        turns <= 2 || m <= cap.unwrap_or(self.exact_max)
    }

    /// `(guess, score)`: bits gained in entropy mode, or estimated total
    /// guesses remaining in expected mode.
    ///
    /// `guesses_left` counts this turn; at 1 the guess must be a possible
    /// answer. It is part of the memo key rather than reduced to "is this the
    /// last turn", because the exact search's answer depends on how
    /// many turns are really left.
    pub fn best_guess(
        &mut self,
        candidates: &[usize],
        guesses_left: Option<usize>,
    ) -> (usize, f64) {
        let turns = guesses_left.unwrap_or(MAX_TURNS).min(MAX_TURNS);
        let key = (candidates.to_vec(), turns);
        if let Some(hit) = self.memo.get(&key) {
            return (hit.0 as usize, hit.1);
        }
        let result = self.best_guess_uncached(candidates, guesses_left);
        self.memo.put(key, (result.0 as i64, result.1));
        result
    }

    fn best_guess_uncached(
        &mut self,
        candidates: &[usize],
        guesses_left: Option<usize>,
    ) -> (usize, f64) {
        let m = candidates.len();
        assert!(
            m > 0,
            "no candidates left; the feedback given is inconsistent"
        );

        if m == 1 {
            return (
                candidates[0],
                if self.mode == Mode::Entropy { 0.0 } else { 1.0 },
            );
        }

        if m == 2 {
            // exact either way: guess the likelier one, and if wrong the other
            // is known
            let weights = self.slice_weights(candidates);
            let total = pairwise(&weights);
            let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
            let first = if p[0] >= p[1] { 0 } else { 1 };
            if self.mode == Mode::Entropy {
                let h = -pairwise_of(p.len(), |i| p[i] * p[i].log2());
                return (candidates[first], h);
            }
            return (candidates[first], kernels::two_candidates(p[first]));
        }

        let turns = guesses_left.unwrap_or(MAX_TURNS);
        if self.exact_applies(m, turns, None) {
            match self.exact.solve(candidates, turns) {
                Ok((guess, value)) => {
                    self.exact_hits += 1;
                    return (guess as usize, value);
                }
                Err(BudgetExceeded { .. }) => self.exact_bailouts += 1,
            }
        }

        if guesses_left == Some(1) {
            // last turn: a guess that cannot be the answer loses outright, so
            // take the likeliest candidate however uninformative it is
            let heaviest = self.heaviest(candidates);
            return (heaviest, self.score_one(heaviest, candidates));
        }

        if self.lookahead > 0 && self.mode == Mode::Expected {
            return self.lookahead_guess(candidates, turns, self.lookahead);
        }

        let pool = self.guess_pool(candidates);
        let scores = self.score_all(candidates, pool);
        let best = match self.mode {
            Mode::Expected => kernels::argmin(&scores),
            Mode::Entropy => kernels::argmax(&scores),
        };
        (Self::row(pool, best), scores[best])
    }

    fn heaviest(&self, candidates: &[usize]) -> usize {
        let mut best = candidates[0];
        for &c in candidates {
            if self.weights[c] > self.weights[best] {
                best = c;
            }
        }
        best
    }

    /// `[(mass, members), ...]` over the non-winning feedback buckets.
    fn partition(&self, guess: usize, candidates: &[usize]) -> Vec<(f64, Vec<usize>)> {
        // One pass for the patterns and one for the distinct ones, rather than
        // 243 vectors of which nearly all stay empty. The Python reaches the
        // same shape with a single argsort.
        let scored: Vec<u8> = candidates
            .iter()
            .map(|&c| self.source.score(guess, c))
            .collect();
        let mut patterns = scored.clone();
        patterns.sort_unstable();
        patterns.dedup();

        let mut out = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            if pattern == crate::feedback::WIN_PATTERN {
                continue; // winning now costs nothing further
            }
            let members: Vec<usize> = candidates
                .iter()
                .zip(&scored)
                .filter(|(_, &p)| p == pattern)
                .map(|(&c, _)| c)
                .collect();
            let mass = pairwise_of(members.len(), |i| self.weights[members[i]]);
            out.push((mass, members));
        }
        out
    }

    /// Value of a position under the branch-capped search.
    fn lookahead_value(&mut self, candidates: &[usize], turns: usize, depth: usize) -> f64 {
        let m = candidates.len();
        if m == 1 {
            return 1.0;
        }
        // the exact search gives a true value here, but the lookahead reaches
        // these sub-positions tens of thousands of times, so how far it is
        // allowed to reach *inside* the search is its own knob
        if self.exact_applies(m, turns, Some(self.lookahead_exact_max)) {
            match self.exact.solve(candidates, turns) {
                Ok((_, value)) => {
                    self.exact_hits += 1;
                    return value;
                }
                Err(BudgetExceeded { .. }) => self.exact_bailouts += 1,
            }
        }
        if depth == 0 || turns <= 1 {
            let pool = self.guess_pool(candidates);
            let scores = self.score_all(candidates, pool);
            return scores.iter().copied().fold(f64::INFINITY, f64::min);
        }
        self.lookahead_guess(candidates, turns, depth).1
    }

    /// `(guess, value)` from expanding the best `lookahead_branch` guesses and
    /// scoring each by what its buckets are really worth.
    ///
    /// Only the guess is pruned, never the feedback: dropping a bucket would
    /// leave the sum something other than an expectation. The cap therefore
    /// limits the fan-out at this node alone, and every level below pays it
    /// again.
    fn lookahead_guess(
        &mut self,
        candidates: &[usize],
        turns: usize,
        depth: usize,
    ) -> (usize, f64) {
        let key = (candidates.to_vec(), turns, depth);
        if let Some(hit) = self.lookahead_memo.get(&key) {
            return (hit.0 as usize, hit.1);
        }

        let pool = self.guess_pool(candidates);
        let scores = self.score_all(candidates, pool);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order.truncate(self.lookahead_branch);
        let total = pairwise_of(candidates.len(), |i| self.weights[candidates[i]]);

        let mut best_guess = Self::row(pool, order[0]);
        let mut best_value = f64::INFINITY;
        for &slot in &order {
            let guess = Self::row(pool, slot);
            let mut value = 1.0;
            for (mass, members) in self.partition(guess, candidates) {
                value += mass / total * self.lookahead_value(&members, turns - 1, depth - 1);
                if value >= best_value {
                    break; // the remaining buckets can only add to it
                }
            }
            if value < best_value {
                best_value = value;
                best_guess = guess;
            }
        }

        self.lookahead_memo
            .put(key, (best_guess as i64, best_value));
        (best_guess, best_value)
    }

    /// True if the guess for this position comes from somewhere other than the
    /// heuristic scorer, so its score is a true value rather than an estimate.
    pub fn solved_exactly(&self, candidates: &[usize], guesses_left: Option<usize>) -> bool {
        let turns = guesses_left.unwrap_or(MAX_TURNS);
        candidates.len() <= 2
            || guesses_left == Some(1)
            || self.exact_applies(candidates.len(), turns, None)
    }

    /// `[(guess, score), ...]` for the k best guesses, best first.
    ///
    /// One word per distinct split: the rest of a split's words give the same
    /// feedback against every candidate, so listing them would be listing the
    /// same guess spelled several ways, and `split_siblings` names them
    /// instead. Where the position is one the solver settles exactly the scores
    /// are true values, and the list never mixes those with estimates.
    pub fn top_guesses(
        &mut self,
        candidates: &[usize],
        k: usize,
        guesses_left: Option<usize>,
    ) -> Vec<(usize, f64)> {
        let k = k.max(1);
        if self.solved_exactly(candidates, guesses_left) {
            let turns = guesses_left.unwrap_or(MAX_TURNS);
            if let Ok(top) = self.exact.top(candidates, turns, k) {
                return top.into_iter().map(|(g, v)| (g as usize, v)).collect();
            }
        }

        let pool = self.guess_pool(candidates);
        let scores = self.score_all(candidates, pool);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        // stable, so a tie leaves the winner where argmin and argmax have it
        // and the ranking opens on the guess best_guess would play
        order.sort_by(|&a, &b| {
            let (x, y) = if self.mode == Mode::Entropy {
                (scores[b], scores[a])
            } else {
                (scores[a], scores[b])
            };
            x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept: Vec<(usize, f64, Vec<u8>)> = Vec::new();
        for i in order {
            let guess = Self::row(pool, i);
            let score = scores[i];
            let split: Vec<u8> = candidates
                .iter()
                .map(|&c| self.source.score(guess, c))
                .collect();
            let duplicate = kept
                .iter()
                .any(|(_, seen, other)| (score - seen).abs() <= SPLIT_TIE_TOL && *other == split);
            if duplicate {
                continue;
            }
            kept.push((guess, score, split));
            if kept.len() == k {
                break;
            }
        }
        kept.into_iter().map(|(g, s, _)| (g, s)).collect()
    }

    /// The prior probability that `guess` is itself the answer, here.
    pub fn win_probability(&self, candidates: &[usize], guess: usize) -> f64 {
        let total = pairwise_of(candidates.len(), |i| self.weights[candidates[i]]);
        if total <= 0.0 {
            return 0.0;
        }
        let mine: Vec<f64> = candidates
            .iter()
            .filter(|&&c| c == guess)
            .map(|&c| self.weights[c])
            .collect();
        pairwise(&mine) / total
    }

    /// Every other word that gives the same feedback as `guess` against every
    /// candidate, and so plays identically from here on.
    pub fn split_siblings(&self, guess: usize, candidates: &[usize]) -> Vec<usize> {
        kernels::same_split(self.source, candidates, guess, self.guess_pool(candidates))
            .into_iter()
            .filter(|&w| w != guess)
            .collect()
    }

    pub fn filter(&self, guess: usize, pattern: u8, candidates: &[usize]) -> Vec<usize> {
        kernels::filter_candidates(self.source, guess, pattern, candidates)
    }

    pub fn feedback(&self, guess: usize, answer: usize) -> u8 {
        self.source.score(guess, answer)
    }

    pub fn reset_memo(&mut self) {
        self.memo.clear();
        self.lookahead_memo.clear();
    }
}
