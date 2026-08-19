//! Assembles everything the static page needs that is not already a file.
//!
//! Two things: every fit in `estimator_cache/`, gathered into one object, and
//! the opening ranking for every set of switches the page can select. The
//! opening is the one position a browser cannot afford live -- 12,947 guesses
//! against 12,947 candidates is 167 million scorings -- and it is also a
//! constant, so it is computed here once.
//!
//! The *Shape of f* menu is why there are 180 of those rather than 20: a form
//! is a different `f`, which is what expected mode ranks by. Entropy consults
//! no `f` at all, so its half is 10 rankings written out under 9 names each.
//!
//!     build_site <repo root> <output directory>

use std::env;
use std::fs;
use std::process::ExitCode;

use wordle_core::estimator::{forms, load_fit, load_form_fit, Form};
use wordle_core::json::{write_f64, write_string};
use wordle_core::priors::{all_configs, build_weights, PriorConfig, ShapeTables};
use wordle_core::solver::{Mode, Solver};
use wordle_core::source::Table;
use wordle_core::WordleData;

/// How many guesses the page lists: the pick plus its alternatives.
const TOP: usize = 5;

/// The key the page's settings collapse to, and the key `openings.json` is
/// indexed by. Must match `settings_key` in the `api` module.
fn settings_key(config: &PriorConfig, mode: Mode, hard: bool, form: &str) -> String {
    format!(
        "{}|{}|{}|{}",
        config.slug(),
        if mode == Mode::Entropy {
            "entropy"
        } else {
            "expected"
        },
        if hard { "hard" } else { "free" },
        form
    )
}

/// One ranking into `openings.json`. Only the active form's line is printed:
/// there are 180 keys and the other 160 say the same handful of words.
fn write_opening(
    out: &mut String,
    first: &mut bool,
    key: &str,
    top: &[(usize, f64)],
    data: &WordleData,
    announce: bool,
) {
    if !*first {
        out.push(',');
    }
    *first = false;
    write_string(out, key);
    out.push_str(":[");
    for (i, &(guess, score)) in top.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"word\":");
        write_string(out, data.words.word(guess));
        out.push_str(",\"score\":");
        write_f64(out, score);
        out.push('}');
    }
    out.push(']');
    if announce {
        println!("  {:<56} {}", key, data.words.word(top[0].0));
    }
}

fn run(root: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let data = WordleData::load(root, 1)?;
    let tables = ShapeTables::load(root)?;
    let table = Table::build(&data.words);
    let everything: Vec<usize> = (0..data.n_words()).collect();

    let mut fits = String::from("{");
    let mut openings = String::from("{");
    let mut first_fit = true;
    let mut first_opening = true;

    let params_list = |out: &mut String, params: &[f64]| {
        out.push('[');
        for (i, value) in params.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_f64(out, *value);
        }
        out.push(']');
    };

    for config in all_configs() {
        let slug = config.slug();
        let (active, active_params) = load_fit(root, &slug)?;

        // Every fit the page's *Shape of f* menu can name, not just the active
        // one: choosing a form there is choosing a different f, which is what
        // expected mode ranks by.
        let mut archive: Vec<(Form, Vec<f64>)> = Vec::new();
        for form in forms() {
            match load_form_fit(root, &slug, form.name) {
                Ok(params) => archive.push((form, params)),
                // no archived fit for this pair; the menu will say so
                Err(_) => continue,
            }
        }

        if !first_fit {
            fits.push(',');
        }
        first_fit = false;
        write_string(&mut fits, &slug);
        fits.push_str(":{\"form\":");
        write_string(&mut fits, active.name);
        fits.push_str(",\"params\":");
        params_list(&mut fits, &active_params);
        fits.push_str(",\"forms\":{");
        for (i, (form, params)) in archive.iter().enumerate() {
            if i > 0 {
                fits.push(',');
            }
            write_string(&mut fits, form.name);
            fits.push(':');
            params_list(&mut fits, params);
        }
        fits.push_str("}}");

        let weights = build_weights(&data, &config, &tables)?;
        for hard in [false, true] {
            // Expected mode consults f, so each form is its own ranking.
            // Entropy never does, so one ranking stands for all of them and
            // gets written out under every name the menu can send.
            for (form, params) in &archive {
                let mut solver = Solver::new(
                    &table,
                    &weights,
                    params.clone(),
                    form.clone(),
                    Mode::Expected,
                )
                .with_hard(hard);
                let top = solver.top_guesses(&everything, TOP, Some(6));
                let key = settings_key(&config, Mode::Expected, hard, form.name);
                write_opening(
                    &mut openings,
                    &mut first_opening,
                    &key,
                    &top,
                    &data,
                    form.name == active.name,
                );
            }

            let mut solver = Solver::new(
                &table,
                &weights,
                active_params.clone(),
                active.clone(),
                Mode::Entropy,
            )
            .with_hard(hard);
            let top = solver.top_guesses(&everything, TOP, Some(6));
            for (form, _) in &archive {
                let key = settings_key(&config, Mode::Entropy, hard, form.name);
                write_opening(
                    &mut openings,
                    &mut first_opening,
                    &key,
                    &top,
                    &data,
                    form.name == active.name,
                );
            }
        }
    }
    fits.push('}');
    openings.push('}');

    fs::write(format!("{out_dir}/fits.json"), fits)?;
    fs::write(format!("{out_dir}/openings.json"), openings)?;
    println!("  wrote fits.json and openings.json");
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let out_dir = match args.next() {
        Some(dir) => dir,
        None => {
            println!("  FAIL: usage: build_site <repo root> <output directory>");
            return ExitCode::FAILURE;
        }
    };
    match run(&root, &out_dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("  FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
