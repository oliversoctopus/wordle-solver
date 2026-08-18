//! Writes the prior weight vector for every configuration, for Python to diff.
//!
//! The comparison lives in `tests/rust_parity.py` rather than here, because the
//! oracle is `priors.build_weights` and numpy is where comparing two vectors of
//! 12,947 floats is one line. This side just has to emit them losslessly, so
//! they go out as raw little-endian f64 rather than as text.
//!
//!     parity_priors <repo root> <output directory>

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use wordle_core::priors::{all_configs, build_weights};
use wordle_core::WordleData;

/// Matches `wordle_data.load_data`'s default: today's answer is kept out of the
/// "already used" prior while you are still solving today's puzzle.
const MASK_LAST: usize = 1;

fn write_weights(path: &str, weights: &[f64]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    for value in weights {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn run(root: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let data = WordleData::load(root, MASK_LAST)?;
    println!(
        "  {} words, {} past answers, {} curated",
        data.n_words(),
        data.past_solutions.len(),
        data.curated.len()
    );

    for config in all_configs() {
        let weights = build_weights(&data, &config, root)?;
        let slug = config.slug();
        write_weights(&format!("{out_dir}/{slug}.f64"), &weights)?;
        println!("  wrote {slug} ({} weights)", weights.len());
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let out_dir = match args.next() {
        Some(dir) => dir,
        None => {
            println!("  FAIL: usage: parity_priors <repo root> <output directory>");
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
