//! Writes `f` and `effective_size` over fixed inputs, for Python to diff.
//!
//! `f` is exercised twice per form: once on a synthetic parameter vector, which
//! covers bases no fit happens to use, and once on the fit the repository
//! actually cached, which is what really gets played.
//!
//!     parity_estimator <repo root> <output directory>

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use wordle_core::estimator::{effective_size, f_eval_x, forms, load_fit};
use wordle_core::priors::{all_configs, build_weights, ShapeTables};
use wordle_core::WordleData;

/// x = log2(n) runs from 0 at one candidate to ~13.66 at the whole word list;
/// the grid overshoots so nothing sits on the edge.
const GRID_STEPS: usize = 1401;
const GRID_STEP: f64 = 0.01;

/// Sizes to take the leading weights of, for effective_size.
const PREFIXES: [usize; 12] = [1, 2, 3, 5, 8, 13, 64, 128, 129, 1000, 5000, 12947];

fn grid() -> Vec<f64> {
    (0..GRID_STEPS).map(|i| i as f64 * GRID_STEP).collect()
}

/// Deterministic coefficients, mixed in sign so hinges and negative
/// contributions are both exercised. Python builds the same vector.
fn synthetic_params(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.15 * i as f64).collect()
}

fn write_f64(path: &str, values: &[f64]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn run(root: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let xs = grid();

    for form in forms() {
        let params = synthetic_params(form.codes.len());
        let values: Vec<f64> = xs.iter().map(|&x| f_eval_x(x, &params, &form)).collect();
        write_f64(&format!("{out_dir}/f_{}.f64", form.name), &values)?;
    }
    println!("  {} forms on the synthetic parameters", forms().len());

    let data = WordleData::load(root, 0)?;
    let mut fits = 0;
    let tables = ShapeTables::load(root)?;
    for config in all_configs() {
        let slug = config.slug();
        let (form, params) = load_fit(root, &slug)?;
        let values: Vec<f64> = xs.iter().map(|&x| f_eval_x(x, &params, &form)).collect();
        write_f64(&format!("{out_dir}/fit_{slug}.f64"), &values)?;
        fits += 1;

        // effective_size over the leading weights of this prior
        let weights = build_weights(&data, &config, &tables)?;
        let sizes: Vec<f64> = PREFIXES
            .iter()
            .map(|&k| effective_size(&weights[..k.min(weights.len())]))
            .collect();
        write_f64(&format!("{out_dir}/eff_{slug}.f64"), &sizes)?;
    }
    println!("  {fits} cached fits, and effective_size on each prior");
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let out_dir = match args.next() {
        Some(dir) => dir,
        None => {
            println!("  FAIL: usage: parity_estimator <repo root> <output directory>");
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
