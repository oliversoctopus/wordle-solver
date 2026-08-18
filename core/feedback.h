// The Wordle coloring rule, and the base-3 encoding of a pattern.
//
// This is the rule feedback_table.npy is built from.  In the native build it
// fills that table once; in the WASM build there is no table and this runs in
// the kernels' inner loop instead, so it has to stay inlinable and allocation
// free.  See wordle_data.score_guess for the Python it must agree with.

#ifndef WORDLE_FEEDBACK_H
#define WORDLE_FEEDBACK_H

#include <cstdint>

namespace wordle {

constexpr int WORD_LEN = 5;
constexpr int N_PATTERNS = 243;   // 3^5
constexpr uint8_t WIN_PATTERN = 242;   // "GGGGG"

// Feedback for one guess as a base-3 integer, N=0, Y=1, G=2, leftmost tile
// most significant.  Greens are taken first and consume the answer's copy of
// the letter, so a second copy in the guess goes uncolored unless the answer
// has one spare.
inline uint8_t score_guess(const char* guess, const char* answer) {
    int remaining[26] = {0};
    uint8_t tile[WORD_LEN];

    for (int i = 0; i < WORD_LEN; ++i) {
        if (guess[i] == answer[i]) {
            tile[i] = 2;
        } else {
            tile[i] = 0;
            ++remaining[answer[i] - 'a'];
        }
    }
    for (int i = 0; i < WORD_LEN; ++i) {
        if (tile[i] == 0) {
            int letter = guess[i] - 'a';
            if (remaining[letter] > 0) {
                tile[i] = 1;
                --remaining[letter];
            }
        }
    }

    int num = 0;
    for (int i = 0; i < WORD_LEN; ++i) {
        num = num * 3 + tile[i];
    }
    return static_cast<uint8_t>(num);
}

}  // namespace wordle

#endif  // WORDLE_FEEDBACK_H
