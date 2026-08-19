//! The word list, and the index that stands in for a word everywhere else.
//!
//! After loading, a word *is* its index: the feedback table is indexed by two
//! of them, a candidate set is an array of them, and text appears again only
//! when something is printed. `data/derived/ind2word.json` is that numbering.

use std::collections::HashMap;
use std::fmt;

use crate::feedback::{score_guess, WORD_LEN};

#[derive(Debug)]
pub enum WordsError {
    Io(std::io::Error),
    Empty,
    NotAWord(String),
    Unterminated,
}

impl fmt::Display for WordsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WordsError::Io(e) => write!(f, "cannot read word list: {e}"),
            WordsError::Empty => write!(f, "no words in the list"),
            WordsError::NotAWord(w) => write!(f, "not five lowercase letters: {w}"),
            WordsError::Unterminated => write!(f, "unterminated string in word list"),
        }
    }
}

impl std::error::Error for WordsError {}

impl From<std::io::Error> for WordsError {
    fn from(e: std::io::Error) -> Self {
        WordsError::Io(e)
    }
}

/// Words packed five bytes each, so `letters` is contiguous and the scorer
/// reads it without touching a `String`.
pub struct WordList {
    packed: Vec<u8>,
    index: HashMap<[u8; WORD_LEN], u32>,
}

impl WordList {
    /// Reads a JSON array of five-letter strings.
    ///
    /// The file is machine-written and holds nothing but plain five-letter
    /// strings, so the quoted runs are the whole of it: no escapes to unpick,
    /// no nesting to track.
    pub fn load(path: &str) -> Result<WordList, WordsError> {
        WordList::from_json(&std::fs::read_to_string(path)?)
    }

    /// The same, from the file's contents rather than its path.
    pub fn from_json(text: &str) -> Result<WordList, WordsError> {
        let mut packed = Vec::new();
        let mut index = HashMap::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while let Some(open) = bytes[i..].iter().position(|&c| c == b'"') {
            let start = i + open + 1;
            let close = bytes[start..]
                .iter()
                .position(|&c| c == b'"')
                .ok_or(WordsError::Unterminated)?;
            let word = &bytes[start..start + close];
            if word.len() != WORD_LEN || !word.iter().all(|c| c.is_ascii_lowercase()) {
                return Err(WordsError::NotAWord(
                    String::from_utf8_lossy(word).into_owned(),
                ));
            }
            let mut key = [0u8; WORD_LEN];
            key.copy_from_slice(word);
            index.insert(key, (packed.len() / WORD_LEN) as u32);
            packed.extend_from_slice(word);
            i = start + close + 1;
        }

        if packed.is_empty() {
            return Err(WordsError::Empty);
        }
        Ok(WordList { packed, index })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.packed.len() / WORD_LEN
    }

    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    #[inline]
    pub fn letters(&self, i: usize) -> &[u8] {
        &self.packed[i * WORD_LEN..(i + 1) * WORD_LEN]
    }

    pub fn word(&self, i: usize) -> &str {
        std::str::from_utf8(self.letters(i)).expect("validated as ASCII on load")
    }

    /// Index of a word, or `None`.
    pub fn index_of(&self, word: &str) -> Option<usize> {
        let bytes = word.as_bytes();
        if bytes.len() != WORD_LEN {
            return None;
        }
        let mut key = [0u8; WORD_LEN];
        key.copy_from_slice(bytes);
        self.index.get(&key).map(|&i| i as usize)
    }

    #[inline]
    pub fn score(&self, guess: usize, answer: usize) -> u8 {
        score_guess(self.letters(guess), self.letters(answer))
    }
}
