//! Writes numpy's draws as this port reproduces them, for Python to diff.
//!
//! Doubles straight off PCG64 come first, then the answers `choice` picks from
//! each prior's real weight vector. The second is what training actually uses
//! and the first is what explains it when the second disagrees.
//!
//!     parity_rng <repo root> <output directory>

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use wordle_core::priors::{all_configs, build_weights, ShapeTables};
use wordle_core::rng::{sample_answers, Pcg64};
use wordle_core::WordleData;

const MASK_LAST: usize = 0;
const DOUBLES: usize = 64;
const GAMES: usize = 2000;
const SEEDS: [u64; 3] = [0, 1, 12345];

fn write_f64(path: &str, values: &[f64]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn run(root: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    for seed in SEEDS {
        let mut rng = Pcg64::new(seed);
        let draws: Vec<f64> = (0..DOUBLES).map(|_| rng.next_double()).collect();
        write_f64(&format!("{out_dir}/doubles_{seed}.f64"), &draws)?;
    }

    let data = WordleData::load(root, MASK_LAST)?;
    let tables = ShapeTables::load(root)?;
    for config in all_configs() {
        let weights = build_weights(&data, &config, &tables)?;
        for seed in SEEDS {
            let answers = sample_answers(&weights, GAMES, seed);
            let as_f64: Vec<f64> = answers.iter().map(|&i| i as f64).collect();
            write_f64(
                &format!("{out_dir}/answers_{}_{seed}.f64", config.slug()),
                &as_f64,
            )?;
        }
        println!("  sampled {} for {}", GAMES, config.slug());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: parity_rng <repo root> <output directory>");
        return ExitCode::from(2);
    }
    match run(&args[1], &args[2]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
