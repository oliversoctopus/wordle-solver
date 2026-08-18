//! The Wordle solver core, ported from the Python in this repository.
//!
//! It compiles twice: to WASM, where it backs a static page with no server and
//! no feedback table, and to a native binary the analysis scripts drive. The
//! Python remains the oracle for all of it -- `tests/rust_parity.py` runs the
//! two against the same inputs and diffs them.
//!
//! `README.md` is the reference for what the solver does and
//! `SOLVER_EXPLAINED.md` derives why it is built this way.

pub mod feedback;
pub mod words;

pub use feedback::{score_guess, N_PATTERNS, WIN_PATTERN, WORD_LEN};
pub use words::WordList;
