//! Asks the solver for a guess at the positions Python handed over.
//!
//! What matters is which *branch* answered: the dispatch has five ways out,
//! and the arms turn each of them on over positions from one candidate to
//! the whole word list.
//!
//!     parity_solver <repo root> <sets file> <output directory> [arm masks]
//!
//! The masks are one comma-separated bitmask per set, saying which arms visit
//! it. Absent, every arm visits every set.

use std::env;
use std::fs;
use std::process::ExitCode;

use wordle_core::estimator::load_fit;
use wordle_core::priors::{build_weights, PriorConfig, ShapeTables};
use wordle_core::solver::{Mode, Solver};
use wordle_core::source::Table;
use wordle_core::WordleData;
use wordle_native::{read_sets, Sink};

/// `(mode, hard, use_exact, lookahead)`. Between them these reach every branch
/// of the dispatch; entropy mode never consults the exact search at all, which
/// is why it needs no separate row for it.
const ARMS: [(Mode, bool, bool, usize); 6] = [
    (Mode::Expected, false, true, 0),
    (Mode::Expected, true, true, 0),
    (Mode::Expected, false, false, 0), // heuristic even in the endgame
    (Mode::Expected, false, true, 1),  // one ply of lookahead
    (Mode::Entropy, false, true, 0),
    (Mode::Entropy, true, true, 0),
];

/// Including the last turn, where the guess must be a possible answer.
const TURNS: [usize; 4] = [1, 2, 4, 6];

fn arm_name(mode: Mode, hard: bool, use_exact: bool, lookahead: usize) -> String {
    format!(
        "{}{}{}{}",
        if mode == Mode::Expected {
            "expected"
        } else {
            "entropy"
        },
        if hard { "+hard" } else { "" },
        if use_exact { "" } else { "+noexact" },
        if lookahead > 0 { "+lookahead" } else { "" }
    )
}

fn run(
    root: &str,
    sets_path: &str,
    out_dir: &str,
    masks: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let data = WordleData::load(root, 0)?;
    let config = PriorConfig::default();
    let tables = ShapeTables::load(root)?;
    let weights = build_weights(&data, &config, &tables)?;
    let (form, params) = load_fit(root, &config.slug())?;
    let table = Table::build(&data.words);
    let sets = read_sets(sets_path)?;
    let masks: Vec<u32> = match masks {
        Some(text) => text.split(',').filter_map(|m| m.parse().ok()).collect(),
        None => vec![u32::MAX; sets.len()],
    };

    let mut out = Sink::create(out_dir, "solver")?;
    let mut asked = 0usize;

    for (arm, &(mode, hard, use_exact, lookahead)) in ARMS.iter().enumerate() {
        for (set_index, set) in sets.iter().enumerate() {
            if masks.get(set_index).copied().unwrap_or(u32::MAX) >> arm & 1 == 0 {
                continue;
            }
            let mut solver = Solver::new(&table, &weights, params.clone(), form.clone(), mode)
                .with_hard(hard)
                .with_use_exact(use_exact)
                .with_lookahead(lookahead)
                .with_loss_penalty(wordle_core::solver::DEFAULT_LOSS_PENALTY);
            for &turns in TURNS.iter() {
                let (guess, score) = solver.best_guess(set, Some(turns));
                out.int(guess as i64)?;
                out.float(score)?;
                // whether the answer is a true value or an estimate is part of
                // what the page shows, so it is part of what has to agree
                out.int(solver.solved_exactly(set, Some(turns)) as i64)?;
                asked += 1;
            }
            // the siblings of the pick: one guess spelled several ways
            let (guess, _) = solver.best_guess(set, Some(6));
            out.indices(&solver.split_siblings(guess, set))?;
        }
    }

    out.flush()?;
    println!("  {} guesses over {} arms", asked, ARMS.len());
    for &(mode, hard, use_exact, lookahead) in ARMS.iter() {
        println!("    {}", arm_name(mode, hard, use_exact, lookahead));
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let (sets, out_dir) = match (args.next(), args.next()) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            println!("  FAIL: usage: parity_solver <repo root> <sets file> <output directory>");
            return ExitCode::FAILURE;
        }
    };
    match run(&root, &sets, &out_dir, args.next()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("  FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
