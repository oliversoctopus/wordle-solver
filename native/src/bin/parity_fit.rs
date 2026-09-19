//! Writes fitted parameters over fixed datasets, for Python to diff.
//!
//! The three datasets exercise the paths that differ: one wide enough for the
//! full basis, one so narrow that a spline's upper hinges are columns of zeros
//! and the solve has to return the minimum-norm answer, and one whose targets
//! fall with `n` so the monotonicity guard drops every term and lands on the
//! flat fallback.
//!
//!     parity_fit <repo root> <output directory>

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use wordle_core::estimator::forms;
use wordle_core::fit::fit;

const SAMPLES: usize = 900;
const N_MAX: f64 = 12947.0;

/// Deterministic stand-ins for rollouts. Integer arithmetic picks the points so
/// that Python lands on the same doubles.
fn dataset(kind: usize) -> (Vec<f64>, Vec<f64>) {
    let mut n_eff = Vec::with_capacity(SAMPLES);
    let mut targets = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let step = ((i * 7919) % 1000) as f64;
        let n = if kind == 1 {
            1.0 + step * 0.03
        } else {
            1.0 + step * 12.9
        };
        let x = n.log2();
        let wobble = ((i * 37) % 13) as f64 / 13.0 - 0.5;
        let target = if kind == 2 {
            6.0 - 0.35 * x + wobble
        } else {
            1.0 + 0.42 * x + wobble
        };
        n_eff.push(n);
        targets.push(target);
    }
    (n_eff, targets)
}

fn write_f64(path: &str, values: &[f64]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: parity_fit <repo root> <output directory>");
        return ExitCode::from(2);
    }
    let out_dir = &args[2];
    if let Err(error) = fs::create_dir_all(out_dir) {
        eprintln!("cannot write to {out_dir}: {error}");
        return ExitCode::from(1);
    }

    for kind in 0..3 {
        let (n_eff, targets) = dataset(kind);
        for form in forms() {
            let (params, terms) = fit(&n_eff, &targets, &form, N_MAX);
            let mut row = params.clone();
            row.push(terms as f64);
            let path = format!("{out_dir}/fit_{kind}_{}.f64", form.name);
            if let Err(error) = write_f64(&path, &row) {
                eprintln!("cannot write {path}: {error}");
                return ExitCode::from(1);
            }
        }
    }
    println!("ok");
    ExitCode::SUCCESS
}
