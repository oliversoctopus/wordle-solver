//! Playing games out, and the sweep over every possible answer.
//!
//! This is the piece that decides whether the port is worth anything for the
//! analysis scripts. `analysis/common.py`'s `every_answer` loops 12,947 answers
//! and calls into the solver once per *position*, so with Python driving that
//! loop the interpreter sits in the innermost place it can. Here the whole
//! sweep happens on this side of the boundary and Python calls once.
//!
//! The solver is deterministic, so a game's outcome is a pure function of the
//! answer: `every_answer` is the entire sample space rather than a draw from
//! it, and the mean it feeds is computed rather than estimated.

use crate::solver::Solver;
use crate::source::Source;

pub const MAX_GUESSES: usize = 6;

/// A position as it stood before a guess: its effective size and its count.
pub type State = (f64, usize);

/// Plays a full game against `answer`, returning the guesses and the state
/// before each.
///
/// `deadline` is the turn on which the game must be won, which restricts that
/// turn's guess to possible answers. `None` plays with no deadline at all, so
/// the recorded counts are unbiased estimates of the value function -- which is
/// what training rollouts want and what measured results do not.
pub fn play<S: Source>(
    solver: &mut Solver<S>,
    answer: usize,
    max_guesses: usize,
    deadline: Option<usize>,
) -> (Vec<usize>, Vec<State>) {
    let mut candidates: Vec<usize> = (0..solver.n_words()).collect();
    let mut guesses = Vec::new();
    let mut states = Vec::new();

    for turn in 1..=max_guesses {
        states.push((solver.effective_size(&candidates), candidates.len()));
        let guesses_left = deadline.map(|d| (d + 1).saturating_sub(turn).max(1));
        let (guess, _) = solver.best_guess(&candidates, guesses_left);
        guesses.push(guess);
        if guess == answer {
            return (guesses, states);
        }
        let pattern = solver.feedback(guess, answer);
        candidates = solver.filter(guess, pattern, &candidates);
    }
    (guesses, states)
}

/// Guesses taken with each word in turn as the answer.
///
/// A loss counts as one guess over the limit, the convention every measured
/// result in this repository uses.
pub fn every_answer<S: Source>(solver: &mut Solver<S>, max_guesses: usize) -> Vec<f64> {
    let n = solver.n_words();
    let mut out = vec![0.0; n];
    for (answer, slot) in out.iter_mut().enumerate() {
        let (guesses, _) = play(solver, answer, max_guesses, Some(max_guesses));
        let solved = guesses.last() == Some(&answer);
        *slot = if solved {
            guesses.len() as f64
        } else {
            (max_guesses + 1) as f64
        };
    }
    out
}

/// Prior-weighted mean and loss probability -- computed, not estimated.
pub fn exact_stats(lengths: &[f64], weights: &[f64], max_guesses: usize) -> (f64, f64) {
    let limit = max_guesses as f64;
    let mean: f64 = lengths
        .iter()
        .zip(weights)
        .map(|(&len, &w)| w * len)
        .sum::<f64>();
    let lost: f64 = lengths
        .iter()
        .zip(weights)
        .filter(|(&len, _)| len > limit)
        .map(|(_, &w)| w)
        .sum();
    (mean, lost)
}
