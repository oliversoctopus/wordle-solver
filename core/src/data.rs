//! The static data every other module reads: word lists, the curated answer
//! list, and the answer record.
//!
//! Mirrors `wordle_data.load_data`. The feedback table is deliberately absent
//! -- the native build fills one from `feedback::score_guess` and the WASM
//! build does without, so nothing here reads the 167 MB file.

use std::collections::HashMap;
use std::error::Error;

use crate::json;
use crate::words::WordList;

pub struct WordleData {
    pub words: WordList,
    /// The original 2022 answer list, as indices into `words` where they exist.
    pub curated: Vec<String>,
    /// Answers already used, in order, oldest first.
    pub past_solutions: Vec<String>,
    /// Wordle's own number for each of those, in step.
    pub puzzle_numbers: Vec<i64>,
}

impl WordleData {
    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    /// Loads from a repository root.
    ///
    /// `mask_last` drops the most recent N entries of `solutions.json`, which
    /// is how today's not-yet-played answer is kept out of the "already used"
    /// prior while you are solving today's puzzle.
    pub fn load(root: &str, mask_last: usize) -> Result<WordleData, Box<dyn Error>> {
        let words = WordList::load(&format!("{root}/data/derived/ind2word.json"))?;

        let curated_json = json::parse_file(&format!("{root}/data/wordles.json"))?;
        let curated: Vec<String> = curated_json
            .as_array()
            .ok_or("wordles.json is not an array")?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();

        let solutions_json = json::parse_file(&format!("{root}/data/solutions.json"))?;
        let rows = solutions_json
            .as_array()
            .ok_or("solutions.json is not an array")?;
        let keep = rows.len().saturating_sub(mask_last);

        let mut past_solutions = Vec::with_capacity(keep);
        let mut puzzle_numbers = Vec::with_capacity(keep);
        for (i, row) in rows[..keep].iter().enumerate() {
            let word = row
                .get("solution")
                .and_then(|v| v.as_str())
                .ok_or("a solutions.json entry has no solution")?;
            past_solutions.push(word.to_string());
            // the pre-NYT entries carry no day count, so fall back to position
            let day = row
                .get("days_since_launch")
                .and_then(|v| v.as_f64())
                .map(|d| d as i64)
                .unwrap_or(i as i64);
            puzzle_numbers.push(day + 1);
        }

        Ok(WordleData {
            words,
            curated,
            past_solutions,
            puzzle_numbers,
        })
    }
}

/// A `word -> number` table, as `wordlebot_weights.json` and `answer_like.json`
/// both are.
pub fn load_table(path: &str) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    let value = json::parse_file(path)?;
    let object = value.as_object().ok_or("expected a JSON object")?;
    Ok(object
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
        .collect())
}
