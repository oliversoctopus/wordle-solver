//! Fits `f` by policy iteration and writes the outcome as JSON.
//!
//! One object per prior on stdout, which `train.py` hands to
//! `estimator.save_params`; progress goes to stderr.
//!
//!     train <repo root> [--prior <slug>] [--form <name>] [--games <n>]
//!         [--seed <n>] [--max-iters <n>] [--tol <x>] [--patience <n>]
//!         [--holdout-games <n>] [--exact-max <n>] [--no-exact]
//!         [--loss-penalty <x>] [--chunks <n>]

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use wordle_core::estimator::{get_form, Form};
use wordle_core::json::{write_f64, Object};
use wordle_core::priors::{all_configs, build_weights, PriorConfig, ShapeTables};
use wordle_core::source::Table;
use wordle_core::train::{policy_iteration, Options, Outcome};
use wordle_core::WordleData;

/// `estimator.DEFAULT_FORM`.
const DEFAULT_FORM: &str = "saturating";
const MASK_LAST: usize = 0;

struct Args {
    root: String,
    prior: Option<String>,
    form: String,
    options: Options,
}

fn next_number<I: Iterator<Item = String>>(args: &mut I, flag: &str) -> Result<f64, String> {
    let text = args.next().ok_or_else(|| format!("{flag} wants a number"))?;
    text.parse()
        .map_err(|_| format!("{flag} got {text:?}, which is not a number"))
}

fn parse() -> Result<Args, String> {
    let mut out = Args {
        root: ".".to_string(),
        prior: None,
        form: DEFAULT_FORM.to_string(),
        options: Options::default(),
    };
    let mut args = env::args().skip(1);
    let mut seen_root = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prior" => out.prior = args.next(),
            "--form" => {
                out.form = args.next().ok_or("--form wants a name")?;
            }
            "--games" => out.options.n_games = next_number(&mut args, "--games")? as usize,
            "--seed" => out.options.seed = next_number(&mut args, "--seed")? as u64,
            "--max-iters" => {
                out.options.max_iters = next_number(&mut args, "--max-iters")? as usize
            }
            "--tol" => out.options.tol = next_number(&mut args, "--tol")?,
            "--patience" => out.options.patience = next_number(&mut args, "--patience")? as usize,
            "--holdout-games" => {
                out.options.holdout_games =
                    Some(next_number(&mut args, "--holdout-games")? as usize)
            }
            "--exact-max" => out.options.exact_max = next_number(&mut args, "--exact-max")? as usize,
            "--loss-penalty" => out.options.loss_penalty = next_number(&mut args, "--loss-penalty")?,
            "--chunks" => out.options.chunks = Some(next_number(&mut args, "--chunks")? as usize),
            "--no-exact" => out.options.use_exact = false,
            _ if !arg.starts_with("--") && !seen_root => {
                out.root = arg;
                seen_root = true;
            }
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    if get_form(&out.form).is_none() {
        return Err(format!("no form named {:?}", out.form));
    }
    Ok(out)
}

fn write_numbers(out: &mut String, values: &[f64]) {
    out.push('[');
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_f64(out, *value);
    }
    out.push(']');
}

fn write_outcome(out: &mut String, slug: &str, form: &Form, word: &str, outcome: &Outcome) {
    let mut object = Object::new(out);
    object
        .str("slug", slug)
        .str("form", form.name)
        .raw("params", |body| write_numbers(body, &outcome.params))
        .int("terms_used", outcome.terms_used as i64)
        .int("iterations", outcome.iterations as i64)
        .bool("converged", outcome.converged)
        .num("entropy_baseline_mean_guesses", outcome.entropy_baseline)
        .num("best_train_mean_guesses", outcome.best_train_mean)
        .num("mean_guesses_last_eval", outcome.mean_last_eval)
        .num("fit_rmse", outcome.fit_rmse)
        .str("opening_word", word)
        .num("opening_expected_guesses", outcome.opening_score)
        .raw("holdout", |body| {
            let mut holdout = Object::new(body);
            holdout
                .int("games", outcome.holdout.games as i64)
                .num("mean_guesses_solved", outcome.holdout.mean_solved)
                .num("mean_guesses_penalized", outcome.holdout.mean_penalized)
                .num("solve_rate", outcome.holdout.solve_rate)
                .raw("distribution", |dist| {
                    let mut counts = Object::new(dist);
                    for (length, count) in &outcome.holdout.distribution {
                        counts.int(&length.to_string(), *count as i64);
                    }
                    counts.end();
                });
            holdout.end();
        })
        .raw("history", |body| {
            body.push('[');
            for (i, round) in outcome.history.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                let mut entry = Object::new(body);
                entry.str("label", &round.label).num("mean_guesses", round.mean_guesses);
                match &round.policy_params {
                    None => {
                        entry.null("policy_params");
                    }
                    Some(values) => {
                        entry.raw("policy_params", |slot| write_numbers(slot, values));
                    }
                }
                entry
                    .raw("fitted_params", |slot| write_numbers(slot, &round.fitted_params))
                    .int("terms_used", round.terms_used as i64)
                    .num("seconds", round.seconds);
                entry.end();
            }
            body.push(']');
        });
    object.end();
}

fn run(args: &Args) -> Result<String, Box<dyn std::error::Error>> {
    let data = WordleData::load(&args.root, MASK_LAST)?;
    let tables = ShapeTables::load(&args.root)?;
    let form = get_form(&args.form).expect("checked while parsing");

    let configs: Vec<PriorConfig> = match &args.prior {
        None => all_configs(),
        Some(slug) => vec![all_configs()
            .into_iter()
            .find(|c| &c.slug() == slug)
            .ok_or_else(|| format!("no prior with slug {slug:?}"))?],
    };

    // the shipped table where there is one; building it is two seconds a run
    let path = format!("{}/data/derived/feedback_table.npy", args.root);
    let table = match Table::load(&path, data.n_words()) {
        Ok(table) => table,
        Err(_) => Table::build(&data.words),
    };

    let mut out = String::from("[");
    for (i, config) in configs.iter().enumerate() {
        let slug = config.slug();
        eprintln!("=== {slug} / {}", form.name);
        let weights = build_weights(&data, config, &tables)?;
        let started = Instant::now();
        let outcome = policy_iteration(&table, &weights, &form, data.n_words(), &args.options);
        for round in &outcome.history {
            eprintln!(
                "  {:>12}: mean {:.4} guesses  ({:.1}s)",
                round.label, round.mean_guesses, round.seconds
            );
        }
        eprintln!(
            "  {} rounds in {:.1}s, opening {} (E = {:.4})",
            outcome.iterations,
            started.elapsed().as_secs_f64(),
            data.words.word(outcome.opening),
            outcome.opening_score
        );
        if i > 0 {
            out.push(',');
        }
        write_outcome(
            &mut out,
            &slug,
            &form,
            data.words.word(outcome.opening),
            &outcome,
        );
    }
    out.push(']');
    Ok(out)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
