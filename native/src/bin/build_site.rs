//! Assembles everything the static page needs that is not already a file.
//!
//! Two things: every fit in `estimator_cache/`, gathered into one object, and
//! the rankings for the positions a browser should not be working out itself.
//! The opening is one -- 12,947 guesses against 12,947 candidates is 167
//! million scorings, which no tab can afford live -- and it is a constant per
//! setting, so it is computed here once for every set of switches.
//!
//! Turn two is the other, and for the four the prior switches reach. A tab
//! can afford it, but visibly: an opening splits the list into ~190 buckets and
//! most are small enough to engage the exact endgame, which is about a second
//! of sitting still. The other two menus stay at their defaults, expected mode
//! and whichever form each prior was fitted in, because doing all 180 settings
//! would take an hour here and several megabytes there.
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
use wordle_core::feedback::{pattern_text, N_PATTERNS, WIN_PATTERN};
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
fn write_ranking(
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
    let data = WordleData::load(root, 0)?;
    let tables = ShapeTables::load(root)?;
    let table = Table::build(&data.words);
    let everything: Vec<usize> = (0..data.n_words()).collect();

    let mut fits = String::from("{");
    let mut openings = String::from("{");
    let mut first_fit = true;
    // the word each prior opens with under the page's other defaults, which
    // is the board turn two is worked out behind
    let mut page_openings: Vec<(String, usize)> = Vec::new();
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
                if !hard && form.name == active.name {
                    page_openings.push((slug.clone(), top[0].0));
                }
                write_ranking(
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
                write_ranking(
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
    // The four the two prior switches can reach, built as `api::parse_config`
    // builds them. Each has its own opening and so its own set of patterns.
    for repeat in [false, true] {
        for curation in [false, true] {
            let config = PriorConfig {
                repeat,
                curation,
                ..Default::default()
            };
            let slug = config.slug();
            let Some(&(_, opening)) = page_openings.iter().find(|(at, _)| *at == slug) else {
                continue;
            };
            let (form, params) = load_fit(root, &slug)?;
            let key = settings_key(&config, Mode::Expected, false, form.name);
            let weights = build_weights(&data, &config, &tables)?;
            // one solver for all the patterns, so what each bucket works out is
            // still in the memo when the next one asks
            let mut solver = Solver::new(&table, &weights, params, form, Mode::Expected);

            let mut written = 0;
            for pattern in 0..N_PATTERNS as u8 {
                // all green ends the board, and the page asks for no ranking there
                if pattern == WIN_PATTERN {
                    continue;
                }
                let candidates = solver.filter(opening, pattern, &everything);
                if candidates.is_empty() {
                    continue;
                }
                let top = solver.top_guesses(&candidates, TOP, Some(5));
                write_ranking(
                    &mut openings,
                    &mut first_opening,
                    &format!("{key}|{}", pattern_text(pattern)),
                    &top,
                    &data,
                    false,
                );
                written += 1;
            }
            println!(
                "  {:<56} {written} patterns behind {}",
                format!("{key}|*"),
                data.words.word(opening)
            );
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
