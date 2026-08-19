//! Assembles everything the static page needs that is not already a file.
//!
//! Two things: the fitted `f` for every prior, gathered out of
//! `estimator_cache/` into one object, and the opening ranking for every set of
//! switches the page can select. The opening is the one position a browser
//! cannot afford live -- 12,947 guesses against 12,947 candidates is 167
//! million scorings -- and it is also a constant, so it is computed here once.
//!
//!     build_site <repo root> <output directory>

use std::env;
use std::fs;
use std::process::ExitCode;

use wordle_core::estimator::load_fit;
use wordle_core::json::{write_f64, write_string};
use wordle_core::priors::{all_configs, build_weights, PriorConfig, ShapeTables};
use wordle_core::solver::{Mode, Solver};
use wordle_core::source::Table;
use wordle_core::WordleData;

/// How many guesses the page lists: the pick plus its alternatives.
const TOP: usize = 5;

/// The key the page's settings collapse to, and the key `openings.json` is
/// indexed by. Must match `settings_key` in the WASM `api` module.
fn settings_key(config: &PriorConfig, mode: Mode, hard: bool) -> String {
    format!(
        "{}|{}|{}",
        config.slug(),
        if mode == Mode::Entropy {
            "entropy"
        } else {
            "expected"
        },
        if hard { "hard" } else { "free" }
    )
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

    for config in all_configs() {
        let slug = config.slug();
        let (form, params) = load_fit(root, &slug)?;

        if !first_fit {
            fits.push(',');
        }
        first_fit = false;
        write_string(&mut fits, &slug);
        fits.push_str(":{\"form\":");
        write_string(&mut fits, form.name);
        fits.push_str(",\"params\":[");
        for (i, value) in params.iter().enumerate() {
            if i > 0 {
                fits.push(',');
            }
            write_f64(&mut fits, *value);
        }
        fits.push_str("]}");

        let weights = build_weights(&data, &config, &tables)?;
        for mode in [Mode::Expected, Mode::Entropy] {
            for hard in [false, true] {
                let mut solver = Solver::new(&table, &weights, params.clone(), form.clone(), mode)
                    .with_hard(hard);
                let top = solver.top_guesses(&everything, TOP, Some(6));

                if !first_opening {
                    openings.push(',');
                }
                first_opening = false;
                write_string(&mut openings, &settings_key(&config, mode, hard));
                openings.push_str(":[");
                for (i, &(guess, score)) in top.iter().enumerate() {
                    if i > 0 {
                        openings.push(',');
                    }
                    openings.push_str("{\"word\":");
                    write_string(&mut openings, data.words.word(guess));
                    openings.push_str(",\"score\":");
                    write_f64(&mut openings, score);
                    openings.push('}');
                }
                openings.push(']');
                println!(
                    "  {:<46} {}",
                    settings_key(&config, mode, hard),
                    data.words.word(top[0].0)
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
