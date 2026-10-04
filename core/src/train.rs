//! Policy iteration: fit `f`, act greedily on it, refit, until the fit settles.
//!
//! The loop is `train.policy_iteration` term for term, including the parts that
//! look incidental and are not: rollouts are played with no deadline so the
//! targets stay unbiased estimates of the value function, round 0 is seeded
//! with greedy entropy, and what comes back is the parameter set whose *policy*
//! measured best rather than the last one fitted.
//!
//! Reporting stays in Python. This returns the outcome and `estimator.save_params`
//! writes it, so the cache keeps one writer and one format.

use std::time::Instant;

use crate::estimator::{f_eval, Form};
use crate::fit::fit;
use crate::game::{play, SWEEP_CHUNKS};
use crate::rng::sample_answers;
use crate::solver::{Mode, Solver};
use crate::source::Source;
use crate::sum::pairwise;

/// `f` is only ever asked about buckets of three or more, since one and two are
/// closed forms, so it is only fitted on those.
pub const MIN_COUNT_FOR_FIT: usize = 3;
/// Training games run long enough to finish rather than to a deadline.
pub const MAX_ROLLOUT_GUESSES: usize = 12;
/// The real limit, which evaluation does apply.
pub const DEADLINE: usize = 6;

pub struct Options {
    pub n_games: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub seed: u64,
    pub patience: usize,
    pub holdout_games: Option<usize>,
    pub exact_max: usize,
    pub use_exact: bool,
    pub loss_penalty: f64,
    /// Games in flight at once, or `None` to take it from `exact_max`.
    pub chunks: Option<usize>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            n_games: 2000,
            max_iters: 8,
            tol: 1e-3,
            seed: 0,
            patience: 3,
            holdout_games: None,
            exact_max: 0,
            use_exact: true,
            loss_penalty: 5.0,
            chunks: None,
        }
    }
}

#[derive(Default)]
pub struct Rollouts {
    pub n_eff: Vec<f64>,
    pub counts: Vec<usize>,
    pub targets: Vec<f64>,
    pub lengths: Vec<usize>,
}

pub struct Round {
    pub label: String,
    pub mean_guesses: f64,
    pub policy_params: Option<Vec<f64>>,
    pub fitted_params: Vec<f64>,
    pub terms_used: usize,
    pub seconds: f64,
}

pub struct Holdout {
    pub games: usize,
    pub mean_solved: f64,
    pub mean_penalized: f64,
    pub solve_rate: f64,
    pub distribution: Vec<(usize, usize)>,
}

pub struct Outcome {
    pub params: Vec<f64>,
    pub terms_used: usize,
    pub iterations: usize,
    pub converged: bool,
    pub entropy_baseline: f64,
    pub best_train_mean: f64,
    pub mean_last_eval: f64,
    pub fit_rmse: f64,
    pub holdout: Holdout,
    pub opening: usize,
    pub opening_score: f64,
    pub history: Vec<Round>,
}

/// Play every answer and keep the (state, guesses-still-needed) samples.
pub fn collect_rollouts<S: Source>(solver: &mut Solver<S>, answers: &[usize]) -> Rollouts {
    let mut out = Rollouts::default();
    for &answer in answers {
        let (guesses, states) = play(solver, answer, MAX_ROLLOUT_GUESSES, None);
        let total = guesses.len();
        out.lengths.push(total);
        // states[0] is the whole word list, identical in every game and never an
        // argument to f, so it is not a fitting sample
        for (depth, &(size, count)) in states.iter().enumerate().skip(1) {
            out.n_eff.push(size);
            out.counts.push(count);
            out.targets.push((total - depth) as f64);
        }
    }
    out
}

/// The same rollouts, a few games in flight at once.
///
/// Chunks are collected in index order and each game's samples are appended
/// whole, so the fit sees the identical vector in the identical order that one
/// solver playing them in turn would have produced: this is a speed knob and
/// never a behavior one.
///
/// Unlike the sweep it is off by default, because training turns the endgame
/// search off and the search is what leaves a core anywhere to go; README
/// *Fitting in Rust* has the measurement.
#[cfg(not(target_arch = "wasm32"))]
pub fn collect_rollouts_across_cores<'a, S, F>(
    answers: &[usize],
    chunks: usize,
    build: F,
) -> Rollouts
where
    S: Source + 'a,
    F: Fn() -> Solver<'a, S> + Sync,
{
    use rayon::prelude::*;

    if chunks <= 1 {
        return collect_rollouts(&mut build(), answers);
    }
    let chunk = answers.len().div_ceil(chunks).max(1);
    let pieces: Vec<Rollouts> = answers
        .par_chunks(chunk)
        .map(|slice| collect_rollouts(&mut build(), slice))
        .collect();

    let mut out = Rollouts::default();
    for piece in pieces {
        out.n_eff.extend(piece.n_eff);
        out.counts.extend(piece.counts);
        out.targets.extend(piece.targets);
        out.lengths.extend(piece.lengths);
    }
    out
}

/// Guesses each answer took under the real limit, a loss counting as one over.
#[cfg(not(target_arch = "wasm32"))]
fn lengths_across_cores<'a, S, F>(
    answers: &[usize],
    max_guesses: usize,
    chunks: usize,
    build: F,
) -> Vec<usize>
where
    S: Source + 'a,
    F: Fn() -> Solver<'a, S> + Sync,
{
    use rayon::prelude::*;

    let chunk = answers.len().div_ceil(chunks.max(1)).max(1);
    answers
        .par_chunks(chunk)
        .map(|slice| {
            let mut solver = build();
            slice
                .iter()
                .map(|&answer| {
                    let (guesses, _) = play(&mut solver, answer, max_guesses, Some(max_guesses));
                    // play stops after max_guesses whether or not it has won
                    if guesses.last() == Some(&answer) {
                        guesses.len()
                    } else {
                        max_guesses + 1
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<usize>>>()
        .concat()
}

fn kept(rollouts: &Rollouts) -> (Vec<f64>, Vec<f64>) {
    let mut n_eff = Vec::new();
    let mut targets = Vec::new();
    for i in 0..rollouts.counts.len() {
        if rollouts.counts[i] >= MIN_COUNT_FOR_FIT {
            n_eff.push(rollouts.n_eff[i]);
            targets.push(rollouts.targets[i]);
        }
    }
    (n_eff, targets)
}

/// `n_max` is the largest effective size `f` must stay monotone up to: the full
/// word count, so a bucket wider than anything seen in training can never be
/// scored as cheaper than a narrower one.
pub fn fit_from_rollouts(rollouts: &Rollouts, form: &Form, n_max: f64) -> (Vec<f64>, usize) {
    let (n_eff, targets) = kept(rollouts);
    fit(&n_eff, &targets, form, n_max)
}

/// How far `f` sits from the guesses actually needed, on the states it decides.
pub fn fit_rmse(rollouts: &Rollouts, params: &[f64], form: &Form) -> f64 {
    let (n_eff, targets) = kept(rollouts);
    if n_eff.is_empty() {
        return f64::NAN;
    }
    let errors: Vec<f64> = n_eff
        .iter()
        .zip(&targets)
        .map(|(&n, &t)| {
            let gap = f_eval(n, params, form) - t;
            gap * gap
        })
        .collect();
    (pairwise(&errors) / errors.len() as f64).sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    pairwise(values) / values.len() as f64
}

/// Play under the real guess limit and report the distribution.
///
/// A failure counts as one guess over the limit, so two policies stay
/// comparable when one of them loses and the other does not.
pub fn evaluate<S: Source>(
    solver: &mut Solver<S>,
    weights: &[f64],
    n_games: usize,
    seed: u64,
    max_guesses: usize,
) -> Holdout {
    let answers = sample_answers(weights, n_games, seed);
    let mut lengths = Vec::with_capacity(n_games);
    for answer in answers {
        let (guesses, _) = play(solver, answer, max_guesses, Some(max_guesses));
        // play stops after max_guesses whether or not it has won
        let solved = guesses.last() == Some(&answer);
        lengths.push(if solved { guesses.len() } else { max_guesses + 1 });
    }

    summarize(&lengths, max_guesses)
}

/// The distribution a set of game lengths makes, failures included.
pub fn summarize(lengths: &[usize], max_guesses: usize) -> Holdout {
    let all: Vec<f64> = lengths.iter().map(|&l| l as f64).collect();
    let won: Vec<f64> = lengths
        .iter()
        .filter(|&&l| l <= max_guesses)
        .map(|&l| l as f64)
        .collect();
    let mut distribution: Vec<(usize, usize)> = Vec::new();
    for &length in lengths {
        match distribution.iter_mut().find(|(k, _)| *k == length) {
            Some((_, count)) => *count += 1,
            None => distribution.push((length, 1)),
        }
    }
    distribution.sort_by_key(|(k, _)| *k);

    Holdout {
        games: lengths.len(),
        mean_solved: mean(&won),
        mean_penalized: mean(&all),
        solve_rate: won.len() as f64 / lengths.len() as f64,
        distribution,
    }
}

/// Fit `f` for one prior combination.
///
/// Exact convergence is not guaranteed: the policy is greedy, so a small change
/// in `f` can flip a guess, and the evaluation is a finite sample. The loop
/// therefore also stops when the measured mean stops improving, and what comes
/// back is the parameter set whose policy measured best.
pub fn policy_iteration<S: Source>(
    source: &S,
    weights: &[f64],
    form: &Form,
    n_words: usize,
    options: &Options,
) -> Outcome {
    let answers = sample_answers(weights, options.n_games, options.seed);
    let chunks = options.chunks.unwrap_or(if options.exact_max == 0 {
        1
    } else {
        SWEEP_CHUNKS
    });
    let build = |params: Vec<f64>, mode: Mode| {
        Solver::new(source, weights, params, form.clone(), mode)
            .with_use_exact(options.use_exact)
            .with_exact_max(options.exact_max)
            .with_loss_penalty(options.loss_penalty)
    };

    let mut params: Option<Vec<f64>> = None;
    let mut history: Vec<Round> = Vec::new();
    let mut converged = false;
    let mut best_params: Option<Vec<f64>> = None;
    let mut best_mean = f64::INFINITY;
    let mut stale_rounds = 0usize;

    for iteration in 0..=options.max_iters {
        let (policy, mode, label) = match &params {
            None => (
                vec![0.0; form.codes.len()],
                Mode::Entropy,
                "entropy seed".to_string(),
            ),
            Some(current) => (current.clone(), Mode::Expected, format!("iter {iteration}")),
        };

        let started = Instant::now();
        let rollouts = collect_rollouts_across_cores(&answers, chunks, || build(policy.clone(), mode));
        let seconds = started.elapsed().as_secs_f64();

        let lengths: Vec<f64> = rollouts.lengths.iter().map(|&l| l as f64).collect();
        let mean_guesses = mean(&lengths);
        let (new_params, terms_used) = fit_from_rollouts(&rollouts, form, n_words as f64);
        history.push(Round {
            label,
            mean_guesses,
            policy_params: params.clone(),
            fitted_params: new_params.clone(),
            terms_used,
            seconds,
        });

        if let Some(current) = &params {
            // `current` is what produced these rollouts, so this is its measured value
            if mean_guesses < best_mean - 1e-12 {
                best_mean = mean_guesses;
                best_params = Some(current.clone());
                stale_rounds = 0;
            } else {
                stale_rounds += 1;
            }
            let delta = current
                .iter()
                .zip(&new_params)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if delta < options.tol {
                converged = true;
                break;
            }
        }
        if stale_rounds >= options.patience {
            break;
        }
        params = Some(new_params);
    }

    let entropy_baseline = history[0].mean_guesses;
    let mean_last_eval = history[history.len() - 1].mean_guesses;
    if best_params.is_none() {
        // only the seed round ran
        best_params = params.clone();
        best_mean = f64::NAN;
    }
    let params = best_params.unwrap_or_else(|| vec![0.0; form.codes.len()]);
    let terms_used = params.iter().filter(|p| **p != 0.0).count();

    let mut final_solver = build(params.clone(), Mode::Expected);
    let all: Vec<usize> = (0..n_words).collect();
    let (opening, opening_score) = final_solver.best_guess(&all, None);

    let rollouts =
        collect_rollouts_across_cores(&answers, chunks, || {
            build(params.clone(), Mode::Expected)
        });
    let rmse = fit_rmse(&rollouts, &params, form);

    let holdout_games = options
        .holdout_games
        .unwrap_or_else(|| (options.n_games / 2).max(200));
    let holdout_answers = sample_answers(weights, holdout_games, options.seed + 10_000);
    let holdout = summarize(
        &lengths_across_cores(&holdout_answers, DEADLINE, chunks, || {
            build(params.clone(), Mode::Expected)
        }),
        DEADLINE,
    );

    Outcome {
        params,
        terms_used,
        iterations: history.len(),
        converged,
        entropy_baseline,
        best_train_mean: best_mean,
        mean_last_eval,
        fit_rmse: rmse,
        holdout,
        opening,
        opening_score,
        history,
    }
}
