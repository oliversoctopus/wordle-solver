//! The answer prior, assembled from two independently toggleable assumptions.
//!
//! The editor's process is modeled as: with probability `repeat_prob` reuse a
//! word that has already been an answer; otherwise pick an unused word, which
//! is off the curated 2022 list with probability `noncurated_prob`. Step one is
//! the *repeat* prior, step two the *curation* prior, and either can be off.
//!
//! Each group also has a *shape*, which is how its mass spreads over its
//! members. Ported from `priors.py`, which stays the oracle; `README.md` has
//! what the shapes are worth and why they ship.

use std::collections::HashMap;
use std::error::Error;

use crate::data::{load_table, parse_table, WordleData};
use crate::sum::pairwise;

pub const DEFAULT_REPEAT_PROB: f64 = 0.10;
pub const DEFAULT_NONCURATED_PROB: f64 = 0.10;

/// 'binned': a Laplace-smoothed histogram of repeats over the source index,
/// meaning where in the original answer order the reused word came from.
pub const BIN_WIDTH: i64 = 200;
pub const BIN_ALPHA: f64 = 0.5;

/// 'frequency': weight ~ exp(FREQ_TEMPERATURE * answer-like log-odds). The
/// optimum is flat between 0.2 and 0.5; this is the middle of it.
pub const FREQ_TEMPERATURE: f64 = 0.35;

/// 'wordlebot': mass kept outside the list, for the answers the list misses.
pub const OFFLIST_HEDGE: f64 = 0.05;

/// Score given to a word the answer-like table does not mention.
const UNSCORED: f64 = -50.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatShape {
    Uniform,
    Binned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurationShape {
    Uniform,
    Frequency,
    Wordlebot,
}

impl RepeatShape {
    pub fn name(self) -> &'static str {
        match self {
            RepeatShape::Uniform => "uniform",
            RepeatShape::Binned => "binned",
        }
    }
}

impl CurationShape {
    pub fn name(self) -> &'static str {
        match self {
            CurationShape::Uniform => "uniform",
            CurationShape::Frequency => "frequency",
            CurationShape::Wordlebot => "wordlebot",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorConfig {
    pub repeat: bool,
    pub curation: bool,
    pub repeat_prob: f64,
    pub noncurated_prob: f64,
    pub repeat_shape: RepeatShape,
    pub curation_shape: CurationShape,
}

impl Default for PriorConfig {
    fn default() -> Self {
        PriorConfig {
            repeat: true,
            curation: true,
            repeat_prob: DEFAULT_REPEAT_PROB,
            noncurated_prob: DEFAULT_NONCURATED_PROB,
            repeat_shape: RepeatShape::Binned,
            curation_shape: CurationShape::Wordlebot,
        }
    }
}

impl PriorConfig {
    /// Filename-safe key, one per prior, shapes included. The shape belongs in
    /// the key because the best form of f depends on it.
    pub fn slug(&self) -> String {
        let mut parts = vec![
            format!("repeat{}", self.repeat as u8),
            format!("curation{}", self.curation as u8),
        ];
        if self.repeat && self.repeat_shape == RepeatShape::Binned {
            parts.push("r-binned".to_string());
        }
        if self.curation {
            match self.curation_shape {
                CurationShape::Frequency => parts.push("c-frequency".to_string()),
                CurationShape::Wordlebot => parts.push("c-wordlebot".to_string()),
                CurationShape::Uniform => {}
            }
        }
        parts.join("_")
    }
}

/// The five configurations the repository carries a fitted `f` for.
pub fn all_configs() -> Vec<PriorConfig> {
    let mut out = Vec::new();
    for repeat in [false, true] {
        for curation in [false, true] {
            out.push(PriorConfig {
                repeat,
                curation,
                ..Default::default()
            });
        }
    }
    out.push(PriorConfig {
        curation_shape: CurationShape::Frequency,
        ..Default::default()
    });
    out
}

/// The rates the two default probabilities are set from, measured rather than
/// written down, so they follow `solutions.json` as it grows.
pub struct Regime {
    pub start_puzzle: i64,
    pub puzzles: usize,
    pub regime_repeat: f64,
    pub regime_noncurated: f64,
    pub all_time_repeat: f64,
    pub all_time_noncurated: f64,
    pub words: usize,
    pub curated: usize,
    pub used: usize,
}

/// The regime starts at the first repeat in the record: nothing repeated before
/// it, and nothing was off-list before it either. Off-list is conditional on
/// not being a repeat, the same order the model draws in.
pub fn regime_stats(data: &WordleData) -> Regime {
    let words = &data.past_solutions;
    let curated: std::collections::HashSet<&str> =
        data.curated.iter().map(|w| w.as_str()).collect();

    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut repeats = Vec::with_capacity(words.len());
    for word in words {
        repeats.push(seen.contains(word.as_str()));
        seen.insert(word.as_str());
    }
    let start = repeats.iter().position(|&r| r).unwrap_or(0);

    let rates = |lo: usize| -> (f64, f64) {
        let window = lo..words.len();
        let count = window.len().max(1);
        let repeat = window.clone().filter(|&i| repeats[i]).count() as f64 / count as f64;
        let fresh: Vec<usize> = window.filter(|&i| !repeats[i]).collect();
        let noncurated = fresh
            .iter()
            .filter(|&&i| !curated.contains(words[i].as_str()))
            .count() as f64
            / fresh.len().max(1) as f64;
        (repeat, noncurated)
    };

    let (regime_repeat, regime_noncurated) = rates(start);
    let (all_time_repeat, all_time_noncurated) = rates(0);
    Regime {
        start_puzzle: data.puzzle_numbers.get(start).copied().unwrap_or(1),
        puzzles: words.len() - start,
        regime_repeat,
        regime_noncurated,
        all_time_repeat,
        all_time_noncurated,
        words: data.n_words(),
        curated: curated.len(),
        // the repeat group is the used words the solver can actually name
        used: seen
            .iter()
            .filter(|w| data.words.index_of(w).is_some())
            .count(),
    }
}

/// The two word-weight tables the sharpened curation shapes read.
///
/// A parameter rather than a path, because WASM has no filesystem: the page
/// fetches these and hands them over as text. `uniform` needs neither.
#[derive(Default)]
pub struct ShapeTables {
    pub answer_like: HashMap<String, f64>,
    pub wordlebot: HashMap<String, f64>,
}

impl ShapeTables {
    pub fn load(root: &str) -> Result<ShapeTables, Box<dyn Error>> {
        Ok(ShapeTables {
            answer_like: load_table(&format!("{root}/data/answer_like.json"))?,
            wordlebot: load_table(&format!("{root}/data/wordlebot_weights.json"))?,
        })
    }

    pub fn from_json(answer_like: &str, wordlebot: &str) -> Result<ShapeTables, Box<dyn Error>> {
        Ok(ShapeTables {
            answer_like: parse_table(answer_like)?,
            wordlebot: parse_table(wordlebot)?,
        })
    }
}

/// Relative weights within the used group, by where the word first appeared.
///
/// Reuse is not uniform over past answers: it concentrates near the start of
/// the run, so bin the used words by first-use index and weight each bin by how
/// many repeats have come out of it. The `BIN_ALPHA` keeps a bin that has never
/// produced a repeat from being written off, which is also what stops any
/// word's weight reaching zero.
fn repeat_shape(data: &WordleData) -> Option<Vec<f64>> {
    let mut first: HashMap<&str, i64> = HashMap::new();
    let mut order: Vec<(&str, i64)> = Vec::new();
    let mut sources: Vec<i64> = Vec::new();

    for (i, word) in data.past_solutions.iter().enumerate() {
        match first.get(word.as_str()) {
            Some(&start) => sources.push(start),
            None => {
                first.insert(word.as_str(), i as i64);
                order.push((word.as_str(), i as i64));
            }
        }
    }
    if sources.is_empty() {
        return None;
    }

    // only words the solver can actually name take part in the bin sizes
    let mut inds: Vec<usize> = Vec::new();
    let mut starts: Vec<i64> = Vec::new();
    for (word, start) in &order {
        if let Some(ind) = data.words.index_of(word) {
            inds.push(ind);
            starts.push(start / BIN_WIDTH);
        }
    }
    if inds.is_empty() {
        return None;
    }

    let max_start = *starts.iter().max().unwrap();
    let max_source = sources.iter().map(|s| s / BIN_WIDTH).max().unwrap();
    let n_bins = (max_start.max(max_source) + 1) as usize;

    let mut size = vec![0.0f64; n_bins];
    for &bin in &starts {
        size[bin as usize] += 1.0;
    }
    let mut hits = vec![0.0f64; n_bins];
    for &source in &sources {
        hits[(source / BIN_WIDTH) as usize] += 1.0;
    }
    let per_bin: Vec<f64> = (0..n_bins)
        .map(|b| (hits[b] + BIN_ALPHA) / size[b].max(1.0))
        .collect();

    let mut rel = vec![0.0f64; data.n_words()];
    for (k, &ind) in inds.iter().enumerate() {
        rel[ind] = per_bin[starts[k] as usize];
    }
    Some(rel)
}

/// Relative weight by how answer-like each word is.
fn answer_like(words: &[&str], table: &HashMap<String, f64>) -> Vec<f64> {
    let scores: Vec<f64> = words
        .iter()
        .map(|w| table.get(*w).copied().unwrap_or(UNSCORED))
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    scores
        .iter()
        .map(|s| (FREQ_TEMPERATURE * (s - top)).exp())
        .collect()
}

/// Relative weights within the non-curated group.
fn curation_shape(
    data: &WordleData,
    mask: &[bool],
    shape: CurationShape,
    tables: &ShapeTables,
) -> Option<Vec<f64>> {
    let members: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
    let words: Vec<&str> = members.iter().map(|&i| data.words.word(i)).collect();
    let mut rel = vec![0.0f64; mask.len()];

    if shape == CurationShape::Frequency {
        for (k, value) in answer_like(&words, &tables.answer_like)
            .into_iter()
            .enumerate()
        {
            rel[members[k]] = value;
        }
        return Some(rel);
    }

    // 'wordlebot': the mass goes to WordleBot's answerable words in proportion to its
    // weight, less a flat hedge over everything else (analysis/curation_shapes.py)
    let listed: Vec<f64> = words
        .iter()
        .map(|w| tables.wordlebot.get(*w).copied().unwrap_or(0.0))
        .collect();
    // the Python sums `listed[inside]`, so the compacted vector is what is
    // reduced and the order is the words' own
    let inside: Vec<f64> = listed.iter().copied().filter(|&v| v > 0.0).collect();
    let inside_total = pairwise(&inside);
    if inside_total <= 0.0 {
        return None;
    }
    let outside = listed.iter().filter(|&&v| v <= 0.0).count();

    for (k, &value) in listed.iter().enumerate() {
        rel[members[k]] = if value > 0.0 {
            (1.0 - OFFLIST_HEDGE) * value / inside_total
        } else {
            OFFLIST_HEDGE / outside as f64
        };
    }
    Some(rel)
}

/// The prior over answers as a probability vector of length `n_words`.
///
/// Every word keeps a strictly positive weight so that log-weights stay finite.
pub fn build_weights(
    data: &WordleData,
    config: &PriorConfig,
    tables: &ShapeTables,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = data.n_words();

    let mut used = vec![false; n];
    for word in &data.past_solutions {
        if let Some(ind) = data.words.index_of(word) {
            used[ind] = true;
        }
    }
    let mut curated = vec![false; n];
    for word in &data.curated {
        if let Some(ind) = data.words.index_of(word) {
            curated[ind] = true;
        }
    }

    // (mask, mass, shape); a shape of None spreads the mass uniformly
    let mut groups: Vec<(Vec<bool>, f64, Option<Vec<f64>>)> = Vec::new();
    let rest: Vec<bool>;
    let rest_mass: f64;

    if config.repeat {
        let shape = match config.repeat_shape {
            RepeatShape::Binned => repeat_shape(data),
            RepeatShape::Uniform => None,
        };
        groups.push((used.clone(), config.repeat_prob, shape));
        rest = used.iter().map(|&u| !u).collect();
        rest_mass = 1.0 - config.repeat_prob;
    } else {
        rest = vec![true; n];
        rest_mass = 1.0;
    }

    if config.curation {
        let noncurated: Vec<bool> = (0..n).map(|i| rest[i] && !curated[i]).collect();
        let shape = match config.curation_shape {
            CurationShape::Uniform => None,
            other => curation_shape(data, &noncurated, other, tables),
        };
        let rest_curated: Vec<bool> = (0..n).map(|i| rest[i] && curated[i]).collect();
        groups.push((
            rest_curated,
            rest_mass * (1.0 - config.noncurated_prob),
            None,
        ));
        groups.push((noncurated, rest_mass * config.noncurated_prob, shape));
    } else {
        groups.push((rest, rest_mass, None));
    }

    let mut weights = vec![0.0f64; n];
    for (mask, mass, shape) in &groups {
        let count = mask.iter().filter(|&&m| m).count();
        if count == 0 {
            // e.g. every curated word has already been used
            continue;
        }
        match shape {
            None => {
                let each = mass / count as f64;
                for i in 0..n {
                    if mask[i] {
                        weights[i] = each;
                    }
                }
            }
            Some(rel) => {
                let held: Vec<f64> = (0..n).filter(|&i| mask[i]).map(|i| rel[i]).collect();
                let total = pairwise(&held);
                for i in 0..n {
                    if mask[i] {
                        weights[i] = if total > 0.0 {
                            mass * rel[i] / total
                        } else {
                            mass / count as f64
                        };
                    }
                }
            }
        }
    }

    let total = pairwise(&weights);
    if total <= 0.0 {
        return Err("prior has no mass; check the prior probabilities".into());
    }
    for w in weights.iter_mut() {
        *w /= total;
    }
    // phrased as the Python's `not np.all(weights > 0.0)`, which also rejects a
    // NaN
    if !weights.iter().all(|&w| w > 0.0) {
        return Err("prior assigns zero weight to some word".into());
    }
    Ok(weights)
}
