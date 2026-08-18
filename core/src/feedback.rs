//! The Wordle coloring rule, and the base-3 encoding of a pattern.
//!
//! This is the rule `feedback_table.npy` is built from. In the native build it
//! fills that table once; in the WASM build there is no table and this runs in
//! the kernels' inner loop instead, so it stays inlinable and allocation free.
//! `wordle_data.score_guess` is the Python it has to agree with.

pub const WORD_LEN: usize = 5;
pub const N_PATTERNS: usize = 243; // 3^5
pub const WIN_PATTERN: u8 = 242; // "GGGGG"

/// Feedback for one guess as a base-3 integer, N=0, Y=1, G=2, leftmost tile
/// most significant.
///
/// Greens are taken first and consume the answer's copy of the letter, so a
/// second copy in the guess goes uncolored unless the answer has one spare.
/// Both slices must be `WORD_LEN` lowercase ASCII letters.
#[inline]
pub fn score_guess(guess: &[u8], answer: &[u8]) -> u8 {
    let mut remaining = [0u8; 26];
    let mut tile = [0u8; WORD_LEN];

    for i in 0..WORD_LEN {
        if guess[i] == answer[i] {
            tile[i] = 2;
        } else {
            remaining[(answer[i] - b'a') as usize] += 1;
        }
    }
    for i in 0..WORD_LEN {
        if tile[i] == 0 {
            let letter = (guess[i] - b'a') as usize;
            if remaining[letter] > 0 {
                tile[i] = 1;
                remaining[letter] -= 1;
            }
        }
    }

    let mut num = 0u8;
    for &colored in tile.iter() {
        num = num * 3 + colored;
    }
    num
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two worked examples in SOLVER_EXPLAINED's section on the feedback
    // rule, which exist because the duplicate-letter case is the subtle one.
    #[test]
    fn duplicate_letters_follow_the_rule() {
        assert_eq!(score_guess(b"speed", b"abide"), 10); // NNYNY
        assert_eq!(score_guess(b"geese", b"abide"), 2); // NNNNG
    }

    #[test]
    fn a_word_against_itself_is_all_green() {
        assert_eq!(score_guess(b"crane", b"crane"), WIN_PATTERN);
    }

    #[test]
    fn nothing_shared_is_all_gray() {
        assert_eq!(score_guess(b"squid", b"champ"), 0);
    }
}
