//! Where a kernel gets `feedback[guess, candidate]` from.
//!
//! Two answers, and the difference is the whole reason this port can be hosted
//! for free. The native build fills the 167 MB table once and reads it; the
//! WASM build has no table and scores the pair on the spot.
//!
//! Live scoring is the slower of the two and is still the right answer for the
//! browser. Measured on one position -- `native/src/bin/bench_position.rs` --
//! the table wins 0.2 ms against 0.7 ms on a median turn-two bucket and 5.3 ms
//! against 18.5 ms on the worst, so about 3.5x. Reasoning from cache lines
//! suggested otherwise and was wrong: a scattered read does cost a whole line,
//! but the rows a position touches stay resident and most reads then hit.
//!
//! What settles it is the absolute number rather than the ratio. 18.5 ms is the
//! worst a browser turn can cost, natively, against a table that would have to
//! be downloaded at 167 MB or rebuilt in the tab at 167 million scorings. The
//! ratio is real and irrelevant.
//!
//! This is a generic parameter rather than a trait object on purpose. The inner
//! loop runs up to 167 million times, and a vtable lookup there would cost more
//! than the work it dispatches.

use crate::feedback::score_guess;
use crate::words::WordList;

/// `Sync` so the native build can score a pool across threads; both
/// implementations are immutable once built, so it costs nothing.
pub trait Source: Sync {
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
