//! Where a kernel gets `feedback[guess, candidate]` from.
//!
//! Two answers, and the difference is the whole reason this port can be hosted
//! for free. The native build fills the 167 MB table once and reads it; the
//! WASM build has no table and scores the pair on the spot.
//!
//! Live scoring is not the poor relation it sounds. A scattered table read
//! spends a whole 64-byte cache line to use one byte of it, while `score_guess`
//! is a couple of dozen integer operations on data already in registers, and
//! the browser only ever sees scattered sets: after the opening the largest
//! bucket is 864 candidates and the median is 13.
//!
//! This is a generic parameter rather than a trait object on purpose. The inner
//! loop runs up to 167 million times, and a vtable lookup there would cost more
//! than the work it dispatches.

use crate::feedback::score_guess;
use crate::words::WordList;

pub trait Source {
    /// Pattern number for this pair, in `[0, 243)`.
    fn score(&self, guess: usize, candidate: usize) -> u8;
    /// How many guessable words there are, so `pool = None` can mean all.
    fn n_guesses(&self) -> usize;
}

/// Scores on demand. No memory beyond the word list, which is what the browser
/// build wants.
pub struct Live<'a> {
    words: &'a WordList,
}

impl<'a> Live<'a> {
    pub fn new(words: &'a WordList) -> Live<'a> {
        Live { words }
    }
}

impl Source for Live<'_> {
    #[inline(always)]
    fn score(&self, guess: usize, candidate: usize) -> u8 {
        score_guess(self.words.letters(guess), self.words.letters(candidate))
    }

    #[inline(always)]
    fn n_guesses(&self) -> usize {
        self.words.len()
    }
}

/// The precomputed guess-major table, as `feedback_table.npy` holds it.
pub struct Table {
    data: Vec<u8>,
    n: usize,
}

impl Table {
    /// Fills the table from the word list. Around two seconds for the full
    /// 12,947 x 12,947, which is cheaper than reading the file it replaces.
    pub fn build(words: &WordList) -> Table {
        let n = words.len();
        let mut data = vec![0u8; n * n];
        for guess in 0..n {
            let letters = words.letters(guess);
            let row = &mut data[guess * n..(guess + 1) * n];
            for (answer, slot) in row.iter_mut().enumerate() {
                *slot = score_guess(letters, words.letters(answer));
            }
        }
        Table { data, n }
    }

    pub fn from_raw(data: Vec<u8>, n: usize) -> Table {
        assert_eq!(data.len(), n * n, "table is not n by n");
        Table { data, n }
    }
}

impl Source for Table {
    #[inline(always)]
    fn score(&self, guess: usize, candidate: usize) -> u8 {
        self.data[guess * self.n + candidate]
    }

    #[inline(always)]
    fn n_guesses(&self) -> usize {
        self.n
    }
}
