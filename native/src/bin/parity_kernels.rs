//! Scores the candidate sets Python handed over, for Python to diff.
//!
//! The sets come from the Python side rather than being generated here, so that
//! a disagreement is always about the kernels and never about the inputs. They
//! straddle both thresholds that change how a kernel behaves: `RESCAN_MAX`,
//! where the reduce switches from walking candidates to sweeping patterns, and
//! `MAX_PACKED`, where a signature stops being the partition itself.
//!
//!     parity_kernels <repo root> <sets file> <output directory>
//!
//! The sets file is `[u32 count]` then, per set, `[u32 len][len x u64]`.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::process::ExitCode;

use wordle_core::estimator::load_fit;
use wordle_core::kernels;
use wordle_core::priors::{build_weights, PriorConfig};
use wordle_core::source::{Live, Source, Table};
use wordle_core::WordleData;

/// Bigger than this and the split helpers are not worth running: the Python
/// only ever asks them about endgame-sized sets.
const SPLIT_MAX: usize = 200;

fn read_sets(path: &str) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut at = 0;
    let take4 = |at: &mut usize| -> u32 {
        let v = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    };
    let count = take4(&mut at) as usize;
    let mut sets = Vec::with_capacity(count);
    for _ in 0..count {
        let len = take4(&mut at) as usize;
        let mut set = Vec::with_capacity(len);
        for _ in 0..len {
            set.push(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize);
            at += 8;
        }
        sets.push(set);
    }
    Ok(sets)
}

struct Out {
    f64s: BufWriter<File>,
    i64s: BufWriter<File>,
}

impl Out {
    fn floats(&mut self, values: &[f64]) -> std::io::Result<()> {
        for v in values {
            self.f64s.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// A variable-length index list, length-prefixed so Python can split it.
    fn indices(&mut self, values: &[usize]) -> std::io::Result<()> {
        self.i64s.write_all(&(values.len() as u32).to_le_bytes())?;
        for &v in values {
            self.i64s.write_all(&(v as u64).to_le_bytes())?;
        }
        Ok(())
    }
}

fn run(root: &str, sets_path: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let data = WordleData::load(root, 1)?;
    let config = PriorConfig::default();
    let weights_all = build_weights(&data, &config, root)?;
    let (form, params) = load_fit(root, &config.slug())?;
    let sets = read_sets(sets_path)?;

    let table = Table::build(&data.words);
    let live = Live::new(&data.words);

    // the two sources must agree, or the browser and the sweeps play different
    // games; spot-check before trusting the fast one for everything below
    for guess in (0..data.n_words()).step_by(997) {
        for candidate in (0..data.n_words()).step_by(991) {
            if table.score(guess, candidate) != live.score(guess, candidate) {
                return Err(format!("table and live disagree at {guess}, {candidate}").into());
            }
        }
    }
    println!("  table and live sources agree on the sampled pairs");

    let mut out = Out {
        f64s: BufWriter::new(File::create(format!("{out_dir}/scores.f64"))?),
        i64s: BufWriter::new(File::create(format!("{out_dir}/splits.i64"))?),
    };

    for set in &sets {
        let weights: Vec<f64> = set.iter().map(|&c| weights_all[c]).collect();

        out.floats(&kernels::ig_all(&table, set, &weights, None))?;
        out.floats(&kernels::expected_guesses_all(
            &table, set, &weights, &params, &form, None,
        ))?;
        out.floats(&kernels::two_turn_scores(&table, set, &weights, 1.0, None))?;
        out.floats(&kernels::two_turn_scores(&table, set, &weights, 5.0, None))?;

        // hard mode is the same kernels with the pool set to the candidates
        out.floats(&kernels::two_turn_scores(
            &table,
            set,
            &weights,
            5.0,
            Some(set),
        ))?;

        if set.len() <= SPLIT_MAX {
            out.indices(&kernels::distinct_partitions(&table, set))?;
            out.indices(&kernels::same_split(&table, set, set[0], None))?;
            if set.len() <= kernels::MAX_PACKED {
                let keys = kernels::signature_keys(&table, set);
                // the packed key is the split itself below eight candidates
                out.indices(&keys.iter().map(|&k| k as usize).collect::<Vec<_>>())?;
            }
        }
        // one filter per set, on the pattern the first candidate produces
        let pattern = table.score(set[0], set[0]);
        out.indices(&kernels::filter_candidates(&table, set[0], pattern, set))?;
    }

    out.f64s.flush()?;
    out.i64s.flush()?;
    println!("  {} candidate sets scored", sets.len());
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());
    let (sets, out_dir) = match (args.next(), args.next()) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            println!("  FAIL: usage: parity_kernels <repo root> <sets file> <output directory>");
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
