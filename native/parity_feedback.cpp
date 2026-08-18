// Checks core/feedback.h against the shipped feedback_table.npy, entry by
// entry.  The table is the oracle here and it is a complete one: every pair the
// solver can ever ask about is in it, so agreeing on all 12,947^2 of them
// leaves nothing about the coloring rule untested.
//
//     parity_feedback [repo root]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../core/words.h"

namespace {

// Reads a version 1.0 .npy of uint8, returning the raw bytes.  Only the shapes
// this repo writes are supported: C order, one dtype, two dimensions.
std::vector<uint8_t> read_npy_u1(const std::string& path, std::size_t* rows,
                                 std::size_t* cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);

    char magic[8];
    file.read(magic, 8);
    if (std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("not a .npy file: " + path);
    }
    uint16_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    if (header.find("'|u1'") == std::string::npos) {
        throw std::runtime_error("expected uint8 data in " + path);
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("expected C order in " + path);
    }
    std::size_t open = header.find('(');
    if (open == std::string::npos) throw std::runtime_error("no shape");
    if (std::sscanf(header.c_str() + open, "(%zu, %zu)", rows, cols) != 2) {
        throw std::runtime_error("cannot read shape from " + path);
    }

    std::vector<uint8_t> data(*rows * *cols);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!file) throw std::runtime_error("short read on " + path);
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    try {
        wordle::WordList words =
            wordle::WordList::load(root + "/data/derived/ind2word.json");

        std::size_t rows = 0, cols = 0;
        std::vector<uint8_t> table =
            read_npy_u1(root + "/data/derived/feedback_table.npy", &rows, &cols);

        std::cout << "  words " << words.size() << ", table " << rows << " x "
                  << cols << "\n";
        if (rows != words.size() || cols != words.size()) {
            std::cout << "  FAIL: table does not match the word list\n";
            return 1;
        }

        std::size_t mismatches = 0;
        for (std::size_t g = 0; g < rows; ++g) {
            const uint8_t* row = table.data() + g * cols;
            for (std::size_t a = 0; a < cols; ++a) {
                if (words.score(g, a) != row[a]) {
                    if (mismatches < 5) {
                        std::cout << "  " << words.word(g) << " vs "
                                  << words.word(a) << ": got "
                                  << int(words.score(g, a)) << ", table says "
                                  << int(row[a]) << "\n";
                    }
                    ++mismatches;
                }
            }
        }

        std::size_t total = rows * cols;
        if (mismatches) {
            std::cout << "  FAIL: " << mismatches << " of " << total
                      << " entries disagree\n";
            return 1;
        }
        std::cout << "  ok: all " << total << " entries agree\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "  FAIL: " << error.what() << "\n";
        return 1;
    }
}
