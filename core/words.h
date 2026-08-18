// The word list, and the index that stands in for a word everywhere else.
//
// After loading, a word *is* its index: the feedback table is indexed by two of
// them, a candidate set is an array of them, and text appears again only when
// something is printed.  data/derived/ind2word.json is that numbering.

#ifndef WORDLE_WORDS_H
#define WORDLE_WORDS_H

#include <cstddef>
#include <string>
#include <vector>

#include "feedback.h"

namespace wordle {

// Words packed five bytes each, so letters(i) is contiguous and the scorer
// reads it without touching a std::string.
class WordList {
public:
    // Reads a JSON array of five-letter strings.  Throws std::runtime_error on
    // a missing file or a word that is not five lowercase letters.
    static WordList load(const std::string& path);

    std::size_t size() const { return count_; }
    const char* letters(std::size_t i) const { return &packed_[i * WORD_LEN]; }
    std::string word(std::size_t i) const {
        return std::string(letters(i), WORD_LEN);
    }

    // Index of a word, or -1.  Linear; callers wanting many lookups should
    // build their own map.
    long index_of(const std::string& word) const;

    uint8_t score(std::size_t guess, std::size_t answer) const {
        return score_guess(letters(guess), letters(answer));
    }

private:
    std::vector<char> packed_;
    std::size_t count_ = 0;
};

}  // namespace wordle

#endif  // WORDLE_WORDS_H
