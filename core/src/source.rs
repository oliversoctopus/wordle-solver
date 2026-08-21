//! Where a kernel gets `feedback[guess, candidate]` from.
//!
//! Two answers, and the difference is the whole reason this port can be hosted
//! for free. The native build fills the 167 MB table once and reads it; the
//! WASM build has no table and scores the pair on the spot. `README.md`
//! § *Without a server* has what live scoring costs and why it is affordable.
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

/// The same table indexed the other way round, `tt[answer, guess]`.
///
/// A second 168 MB and no file, since transposing the one already held is
/// cheaper than reading another. Only `fused` reads it, and only where a
/// position is wide enough for the layout to pay; `README.md` § *Without a
/// server* and SOLVER_EXPLAINED §8 have the band and why it ends where it does.
pub struct Transposed {
    data: Vec<u8>,
    n: usize,
}

impl Transposed {
    /// A blocked transpose: naively, one side of the copy strides 12,947 bytes
    /// per element and every write lands on a line of its own.
    pub fn build(table: &Table) -> Transposed {
        const TILE: usize = 64;
        let n = table.n;
        let mut data = vec![0u8; n * n];
        for guess0 in (0..n).step_by(TILE) {
            let guess_end = (guess0 + TILE).min(n);
            for answer0 in (0..n).step_by(TILE) {
                let answer_end = (answer0 + TILE).min(n);
                for guess in guess0..guess_end {
                    for answer in answer0..answer_end {
                        data[answer * n + guess] = table.data[guess * n + answer];
                    }
                }
            }
        }
        Transposed { data, n }
    }

    #[inline(always)]
    pub fn at(&self, answer: usize, guess: usize) -> u8 {
        self.data[answer * self.n + guess]
    }

    pub fn n_guesses(&self) -> usize {
        self.n
    }
}

/// Every guess scored against one small candidate set, held for as long as a
/// search runs.
///
/// The endgame rescans the whole pool at every node it visits, and every
/// candidate it ever sees belongs to the set it started from, since each bucket
/// is a subset of the one above. One scan up front therefore turns all the rest
/// into a byte load. `README.md` § *Without a server* has what that saves.
pub struct Columns {
    /// Guess-major, so a kernel walking the pool reads its candidates from
    /// consecutive bytes rather than striding.
    rows: Vec<u8>,
    /// Word index to column, `NOT_HELD` for a word the set left out.
    slot: Vec<u16>,
    held: usize,
    n_guesses: usize,
}

const NOT_HELD: u16 = u16::MAX;

impl Columns {
    pub fn build<S: Source>(source: &S, candidates: &[usize]) -> Columns {
        let n_guesses = source.n_guesses();
        let held = candidates.len();
        let mut rows = vec![0u8; n_guesses * held];
        for guess in 0..n_guesses {
            let row = &mut rows[guess * held..(guess + 1) * held];
            for (cell, &candidate) in row.iter_mut().zip(candidates) {
                *cell = source.score(guess, candidate);
            }
        }
        let mut slot = vec![NOT_HELD; n_guesses];
        for (column, &candidate) in candidates.iter().enumerate() {
            slot[candidate] = column as u16;
        }
        Columns {
            rows,
            slot,
            held,
            n_guesses,
        }
    }
}

impl Source for Columns {
    #[inline(always)]
    fn score(&self, guess: usize, candidate: usize) -> u8 {
        let column = self.slot[candidate] as usize;
        debug_assert!(column != NOT_HELD as usize, "candidate outside the set");
        self.rows[guess * self.held + column]
    }

    #[inline(always)]
    fn n_guesses(&self) -> usize {
        self.n_guesses
    }
}
