//! Solves the endgame positions Python handed over, for Python to diff.
//!
//! The search has to return the identical guess *and* the identical value. A
//! guess of equal value is not good enough: `SPLIT_TIE_TOL` is 1e-9 and the
//! closest real pair of scores measured over a whole scan is 1.1e-9, so "same
//! value" and "same word" are nearly the same question here, and only the
//! stricter one catches a branch order that has drifted.
//!
//!     parity_exact <repo root> <sets file> <output directory>
//!
//! The sets file is `[u32 count]` then, per set, `[u32 len][len x u64]`.

use std::env;
use std::fs;
use std::process::ExitCode;

use wordle_core::exact::ExactEndgame;
use wordle_core::priors::{build_weights, PriorConfig};
use wordle_core::source::Table;
use wordle_core::WordleData;
use wordle_native::{read_sets, Sink};

/// `(loss penalty, hard mode)`, matching what the parity script asks for.
const ARMS: [(f64, bool); 3] = [(5.0, false), (1.0, false), (5.0, true)];
const TURNS: [usize; 4] = [3, 4, 5, 6];
const TOP_K: usize = 5;

fn run(root: &str, sets_path: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let data = WordleData::load(root, 1)?;
    let config = PriorConfig::default();
    let weights = build_weights(&data, &config, root)?;
    let table = Table::build(&data.words);
    let sets = read_sets(sets_path)?;

    let mut out = Sink::create(out_dir, "exact")?;
    let mut solved = 0usize;

    for &(penalty, hard) in ARMS.iter() {
        for set in &sets {
            // a fresh search per arm: values depend on the penalty, so the memo
            // must not carry across one
            let mut search = ExactEndgame::new(&table, &weights, penalty).with_hard(hard);
            for &turns in TURNS.iter() {
                let (guess, value) = search.solve(set, turns)?;
                out.int(guess)?;
                out.float(value)?;

                let top = search.top(set, turns, TOP_K)?;
                out.indices(&top.iter().map(|&(g, _)| g as usize).collect::<Vec<_>>())?;
                out.all_floats(&top.iter().map(|&(_, v)| v).collect::<Vec<_>>())?;
                solved += 1;
            }
        }
    }

    out.flush()?;
    println!(
        "  {} positions over {} arms and {} turn counts",
        solved,
        ARMS.len(),
        TURNS.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let (sets, out_dir) = match (args.next(), args.next()) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            println!("  FAIL: usage: parity_exact <repo root> <sets file> <output directory>");
            return ExitCode::FAILURE;
        }
    };
    match run(&root, &sets, &out_dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("  FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
