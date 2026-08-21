//! Plays every word in the list as the answer, and reports what it cost.
//!
//! This is the acceptance test for the whole port, and the sharpest one
//! available: the solver is deterministic, so playing all 12,947 words once and
//! weighting by the prior *computes* the mean rather than estimating it. There
//! is no sampling error to hide behind, and README's *Results* table is the
//! number to hit.
//!
//!     every_answer <repo root> [--mode entropy|expected] [--hard] [--fused]
//!                              [--no-exact] [--prior <slug>] [--json]
//!                              [--dump <path>]

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use wordle_core::estimator::load_fit;
use wordle_core::game::{every_answer, exact_stats, MAX_GUESSES};
use wordle_core::priors::{all_configs, build_weights, PriorConfig, ShapeTables};
use wordle_core::solver::{Mode, Solver};
use wordle_core::source::{Live, Table, Transposed};
use wordle_core::WordleData;

struct Options {
    root: String,
    mode: Mode,
    hard: bool,
    use_exact: bool,
    prior: Option<String>,
    json: bool,
    /// Score pairs on the spot instead of building the table, as WASM must.
    live: bool,
    fused: bool,
    /// Where to write the per-answer guess counts, for Python to diff.
    dump: Option<String>,
}

fn parse() -> Result<Options, String> {
    let mut out = Options {
        root: ".".to_string(),
        mode: Mode::Expected,
        hard: false,
        use_exact: true,
        prior: None,
        json: false,
        live: false,
        fused: wordle_core::fused::enabled_by_default(),
        dump: None,
    };
    let mut args = env::args().skip(1);
    let mut positional = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                out.mode = match args.next().as_deref() {
                    Some("entropy") => Mode::Entropy,
                    Some("expected") => Mode::Expected,
                    other => return Err(format!("unknown mode {other:?}")),
                }
            }
            "--hard" => out.hard = true,
            "--no-exact" => out.use_exact = false,
            "--prior" => out.prior = args.next(),
            "--json" => out.json = true,
            "--live" => out.live = true,
            "--fused" => out.fused = true,
            "--dump" => out.dump = args.next(),
            _ if positional == 0 => {
                out.root = arg;
                positional += 1;
            }
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    Ok(out)
}

/// Plays every answer, whichever way feedback is coming from.
fn sweep<S: wordle_core::source::Source>(
    source: &S,
    fused: Option<&Transposed>,
    weights: &[f64],
    params: Vec<f64>,
    form: wordle_core::estimator::Form,
    options: &Options,
    n_words: usize,
) -> (Vec<f64>, usize) {
    let mut solver = Solver::new(source, weights, params, form, options.mode)
        .with_fused(fused)
        .with_hard(options.hard)
        .with_use_exact(options.use_exact);
    let lengths = every_answer(&mut solver, MAX_GUESSES);
    let opening = solver.best_guess(&(0..n_words).collect::<Vec<_>>(), Some(MAX_GUESSES));
    (lengths, opening.0)
}

fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let data = WordleData::load(&options.root, 1)?;
    let config = match &options.prior {
        None => PriorConfig::default(),
        Some(slug) => all_configs()
            .into_iter()
            .find(|c| &c.slug() == slug)
            .ok_or_else(|| format!("no prior with slug {slug:?}"))?,
    };

    let tables = ShapeTables::load(&options.root)?;
    let weights = build_weights(&data, &config, &tables)?;
    let (form, params) = load_fit(&options.root, &config.slug())?;

    let built = Instant::now();
    let table = if options.live {
        None
    } else {
        Some(Table::build(&data.words))
    };
    let build_secs = built.elapsed().as_secs_f64();

    // the answer-major twin, only where it was asked for: another 168 MB
    let turned = Instant::now();
    let transposed = table
        .as_ref()
        .filter(|_| options.fused)
        .map(Transposed::build);
    let fused_secs = transposed.as_ref().map_or(0.0, |_| turned.elapsed().as_secs_f64());

    let started = Instant::now();
    let (lengths, opening_index) = match &table {
        Some(table) => sweep(
            table,
            transposed.as_ref(),
            &weights,
            params,
            form,
            options,
            data.n_words(),
        ),
        None => sweep(
            &Live::new(&data.words),
            None,
            &weights,
            params,
            form,
            options,
            data.n_words(),
        ),
    };
    let secs = started.elapsed().as_secs_f64();

    if let Some(path) = &options.dump {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for value in &lengths {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }

    let (mean, loss) = exact_stats(&lengths, &weights, MAX_GUESSES);
    let opening_word = data.words.word(opening_index);

    if options.json {
        println!(
            "{{\"slug\": \"{}\", \"mean\": {:.17}, \"loss_probability\": {:.17}, \
             \"opening\": \"{}\", \"seconds\": {:.3}, \"table_seconds\": {:.3}, \"fused_seconds\": {:.3}}}",
            config.slug(),
            mean,
            loss,
            opening_word,
            secs,
            build_secs,
            fused_secs
        );
    } else {
        let totals = wordle_core::profile::totals();
        if !totals.is_empty() {
            let mut inside = 0.0;
            println!();
            println!("  {:<24} {:>8} {:>10}", "entry point", "seconds", "calls");
            for (name, (elapsed, calls)) in wordle_core::profile::NAMES.iter().zip(&totals) {
                if *calls > 0 {
                    println!("  {name:<24} {elapsed:>8.2} {calls:>10}");
                }
                if *name != "table build" {
                    inside += elapsed;
                }
            }
            println!("  {:<24} {:>8.2}", "--- inside kernels", inside);
            println!(
                "  {:<24} {:>8.2}  ({:.0}%)",
                "--- everything else",
                secs - inside,
                100.0 * (secs - inside) / secs
            );
            println!();
        }
        println!("  prior          {}", config.slug());
        println!("  opening        {opening_word}");
        println!("  mean           {mean:.4}");
        println!("  loss           {:.4}%", loss * 100.0);
        println!(
            "  feedback       {}",
            if options.live { "scored live" } else { "table" }
        );
        println!("  table built in {build_secs:.2} s");
        println!("  swept in       {secs:.2} s");
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse().map_err(|e| e.into()).and_then(|o| run(&o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("  FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
