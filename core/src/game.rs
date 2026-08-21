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
use crate::sum::{pairwise, pairwise_of};

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
        *slot = length(solver, answer, max_guesses);
    }
    out
}

/// The same sweep, played a few games at a time.
///
/// The games are independent and a memo holds pure functions of a position, so
/// which thread plays which answer changes nothing about the answer: this
/// returns the same lengths as `every_answer`, in the same order.
///
/// The kernels already spread each scan across every core, and what this adds
/// is somewhere for a core to go while one game is between scans -- recursing
/// through the endgame, filtering a candidate set, walking the memo. Two or
/// three games in flight covers that; more does not, and measurably so. At one
/// chunk per core the sweep runs half as fast again as it does serially,
/// because sixteen games at sixteen different depths of the table share none of
/// their cache and each starts from a memo the others cannot help.
#[cfg(not(target_arch = "wasm32"))]
pub fn every_answer_across_cores<'a, S, F>(n: usize, max_guesses: usize, solver: F) -> Vec<f64>
where
    S: Source + 'a,
    F: Fn() -> Solver<'a, S> + Sync,
{
    use rayon::prelude::*;

    let chunk = n.div_ceil(SWEEP_CHUNKS).max(1);
    let mut out = vec![0.0; n];
    out.par_chunks_mut(chunk).enumerate().for_each(|(b, slot)| {
        let mut solver = solver();
        for (i, cell) in slot.iter_mut().enumerate() {
            *cell = length(&mut solver, b * chunk + i, max_guesses);
        }
    });
    out
}

/// Games in flight at once. Not a thread count -- the threads are the kernels'
/// -- but how many of them may be between scans before the cache gives out.
#[cfg(not(target_arch = "wasm32"))]
const SWEEP_CHUNKS: usize = 3;

/// Guesses taken against one answer, a loss counting as one over the limit.
fn length<S: Source>(solver: &mut Solver<S>, answer: usize, max_guesses: usize) -> f64 {
    let (guesses, _) = play(solver, answer, max_guesses, Some(max_guesses));
    if guesses.last() == Some(&answer) {
        guesses.len() as f64
    } else {
        (max_guesses + 1) as f64
    }
}

/// Prior-weighted mean and loss probability -- computed, not estimated.
pub fn exact_stats(lengths: &[f64], weights: &[f64], max_guesses: usize) -> (f64, f64) {
    let limit = max_guesses as f64;
    // the mean is the one reduction that cannot be matched: the Python spells
    // it `weights @ lengths`, and BLAS reduces in an order of its own choosing
    let mean = pairwise_of(lengths.len(), |i| weights[i] * lengths[i]);
    let lost: Vec<f64> = lengths
        .iter()
        .zip(weights)
        .filter(|(&len, _)| len > limit)
        .map(|(_, &w)| w)
        .collect();
    (mean, pairwise(&lost))
}
