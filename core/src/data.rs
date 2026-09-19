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
    /// The `print_date` of the record's last entry, masked or not: the day the
    /// solver was last brought up to date, which the page shows.
    pub record_through: String,
}

impl WordleData {
    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    /// Loads from a repository root.
    ///
    /// `mask_last` drops the most recent N entries of `solutions.json`, which
    /// winds the used-word prior back to an earlier day.  The record itself
    /// stops at yesterday, so 0 is what everything passes.
    pub fn load(root: &str, mask_last: usize) -> Result<WordleData, Box<dyn Error>> {
        WordleData::from_json(
            &std::fs::read_to_string(format!("{root}/data/derived/ind2word.json"))?,
            &std::fs::read_to_string(format!("{root}/data/wordles.json"))?,
            &std::fs::read_to_string(format!("{root}/data/solutions.json"))?,
            mask_last,
        )
    }

    /// The same, from the file contents rather than their paths.
    ///
    /// This is the constructor that matters: WASM has no filesystem, so every
    /// input reaches the core as a string the page fetched.
    pub fn from_json(
        ind2word: &str,
        wordles: &str,
        solutions: &str,
        mask_last: usize,
    ) -> Result<WordleData, Box<dyn Error>> {
        let words = WordList::from_json(ind2word)?;

        let curated_json = json::parse(wordles)?;
        let curated: Vec<String> = curated_json
            .as_array()
            .ok_or("wordles.json is not an array")?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();

        let solutions_json = json::parse(solutions)?;
        let rows = solutions_json
            .as_array()
            .ok_or("solutions.json is not an array")?;
        let keep = rows.len().saturating_sub(mask_last);
        let record_through = rows
            .last()
            .and_then(|row| row.get("print_date"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

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
            record_through,
        })
    }
}

/// A `word -> number` table, as `wordlebot_weights.json` and `answer_like.json`
/// both are.
pub fn load_table(path: &str) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    parse_table(&std::fs::read_to_string(path)?)
}

/// The same, from the file's contents.
pub fn parse_table(text: &str) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    let value = json::parse(text)?;
    let object = value.as_object().ok_or("expected a JSON object")?;
    Ok(object
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
        .collect())
}
