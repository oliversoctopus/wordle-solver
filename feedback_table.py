"""Builds ``feedback_table.npy`` and the two index maps from the word lists.

Every legal guess is also a possible answer, so the table is square:
``feedback_table[guess][answer]`` is `wordle_data.score_guess`.
``ind2word.json`` is the row numbering and ``word2ind.json`` its inverse.

    python feedback_table.py            report what a rebuild would change
    python feedback_table.py --write    rebuild
"""

import argparse
import json
import os

import numpy as np

from wordle_data import (CURATED_PATH, FEEDBACK_TABLE_PATH, IND2WORD_PATH,
                         NONCURATED_PATH, WORD2IND_PATH, score_guess,
                         write_json)


def all_words():
    """The curated answers, then every other legal guess."""
    with open(CURATED_PATH, "r") as file:
        words = json.load(file)
    with open(NONCURATED_PATH, "r") as file:
        words.extend(json.load(file))
    if len(set(words)) != len(words):
        raise ValueError("a word appears in both lists")
    return words


def build_table(words):
    table = np.zeros((len(words), len(words)), dtype=np.uint8)
    for i, guess in enumerate(words):
        for j, answer in enumerate(words):
            table[i, j] = score_guess(guess, answer)
    return table


def status(path, same):
    """'unchanged', 'differs' or 'missing', for one file on disk."""
    if not os.path.exists(path):
        return "missing"
    return "unchanged" if same() else "differs"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="write the table and the index maps")
    args = parser.parse_args()

    words = all_words()
    word2ind = {word: i for i, word in enumerate(words)}
    print(f"{len(words)} words")
    table = build_table(words)

    def shipped(path):
        with open(path, "r") as file:
            return json.load(file)

    for path, same in (
            (IND2WORD_PATH, lambda: shipped(IND2WORD_PATH) == words),
            (WORD2IND_PATH, lambda: shipped(WORD2IND_PATH) == word2ind),
            (FEEDBACK_TABLE_PATH,
             lambda: np.array_equal(np.load(FEEDBACK_TABLE_PATH), table))):
        print(f"  {os.path.basename(path)}: {status(path, same)}")

    if args.write:
        write_json(IND2WORD_PATH, words)
        write_json(WORD2IND_PATH, word2ind)
        np.save(FEEDBACK_TABLE_PATH, table)
        print("written")
    else:
        print("not written; pass --write to replace them")


if __name__ == "__main__":
    main()
