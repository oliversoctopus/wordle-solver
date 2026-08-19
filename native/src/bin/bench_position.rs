//! One position, scored both ways: what a browser turn actually costs.
//!
//! The sweep is throughput over 12,947 games; a page solves one position and
//! waits. The worst case after the opening is the biggest turn-two bucket, 864
//! candidates against all 12,947 guesses.
use std::time::Instant;
use wordle_core::estimator::load_fit;
use wordle_core::feedback::score_guess;
use wordle_core::kernels::expected_guesses_all;
use wordle_core::priors::{build_weights, PriorConfig};
use wordle_core::source::{Live, Source, Table};
use wordle_core::WordleData;

fn time<S: Source>(
    source: &S,
    label: &str,
    set: &[usize],
    w: &[f64],
    p: &[f64],
    f: &wordle_core::estimator::Form,
) {
    expected_guesses_all(source, set, w, p, f, None);
    let n = 20;
    let t0 = Instant::now();
    for _ in 0..n {
        expected_guesses_all(source, set, w, p, f, None);
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    println!("    {label:<12} {:>7.1} ms", per * 1e3);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let data = WordleData::load(&root, 1)?;
    let config = PriorConfig::default();
    let weights = build_weights(&data, &config, &root)?;
    let (form, params) = load_fit(&root, &config.slug())?;
    let table = Table::build(&data.words);
    let live = Live::new(&data.words);

    let salet = data.words.index_of("salet").unwrap();
    let mut by_size: Vec<(usize, Vec<usize>)> = Vec::new();
    for pattern in 0..243u8 {
        let set: Vec<usize> = (0..data.n_words())
            .filter(|&c| score_guess(data.words.letters(salet), data.words.letters(c)) == pattern)
            .collect();
        if !set.is_empty() {
            by_size.push((set.len(), set));
        }
    }
    by_size.sort_by_key(|(n, _)| *n);
    let median = by_size[by_size.len() / 2].clone();
    let worst = by_size.last().unwrap().clone();

    for (label, (size, set)) in [("median", median), ("worst", worst)] {
        println!("  turn two after salet, {label} bucket ({size} candidates):");
        let w: Vec<f64> = set.iter().map(|&c| weights[c]).collect();
        time(&table, "table", &set, &w, &params, &form);
        time(&live, "live", &set, &w, &params, &form);
    }
    Ok(())
}
