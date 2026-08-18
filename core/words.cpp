#include "words.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wordle {

namespace {

bool is_word(const std::string& text) {
    if (text.size() != static_cast<std::size_t>(WORD_LEN)) return false;
    for (char c : text) {
        if (c < 'a' || c > 'z') return false;
    }
    return true;
}

// The file is a machine-written array of plain five-letter strings, so the
// quoted runs are the whole of it -- no escapes, no nesting to track.
std::vector<std::string> quoted_strings(const std::string& text) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (true) {
        std::size_t open = text.find('"', i);
        if (open == std::string::npos) break;
        std::size_t close = text.find('"', open + 1);
        if (close == std::string::npos) {
            throw std::runtime_error("unterminated string in word list");
        }
        out.push_back(text.substr(open + 1, close - open - 1));
        i = close + 1;
    }
    return out;
}

}  // namespace

WordList WordList::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open word list: " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();

    std::vector<std::string> words = quoted_strings(buffer.str());
    if (words.empty()) throw std::runtime_error("no words in " + path);

    WordList list;
    list.count_ = words.size();
    list.packed_.resize(words.size() * WORD_LEN);
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (!is_word(words[i])) {
            throw std::runtime_error("not five lowercase letters: " + words[i]);
        }
        for (int k = 0; k < WORD_LEN; ++k) {
            list.packed_[i * WORD_LEN + k] = words[i][k];
        }
    }
    return list;
}

long WordList::index_of(const std::string& text) const {
    if (!is_word(text)) return -1;
    for (std::size_t i = 0; i < count_; ++i) {
        const char* w = letters(i);
        bool same = true;
        for (int k = 0; k < WORD_LEN; ++k) {
            if (w[k] != text[k]) { same = false; break; }
        }
        if (same) return static_cast<long>(i);
    }
    return -1;
}

}  // namespace wordle
