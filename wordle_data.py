"""Loading of the static word / feedback data used by every other module, and
the one JSON writer everything else goes through."""

import functools
import json
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# curated inputs and the answer record in data/, anything feedback_table.py
# regenerates in data/derived/
DATA_DIR = os.path.join(HERE, "data")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")

FEEDBACK_TABLE_PATH = os.path.join(DERIVED_DIR, "feedback_table.npy")
WORD2IND_PATH = os.path.join(DERIVED_DIR, "word2ind.json")
IND2WORD_PATH = os.path.join(DERIVED_DIR, "ind2word.json")
CURATED_PATH = os.path.join(DATA_DIR, "wordles.json")
NONCURATED_PATH = os.path.join(DATA_DIR, "nonwordles.json")
SOLUTIONS_PATH = os.path.join(DATA_DIR, "solutions.json")

MAX_GUESSES = 6

# "GGGGG" as a base-3 number with N=0, Y=1, G=2
WIN_PATTERN = 242
N_PATTERNS = 243

TILE_VALS = {"N": 0, "Y": 1, "G": 2}


def write_json(path, payload, indent=4, sort_keys=False):
    """JSON to disk, LF-terminated whatever the platform.

    ``newline=""`` stops text mode writing CRLF on Windows, where every line
    of the file would register as changed.
    """
    with open(path, "w", encoding="utf-8", newline="") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False,
                  sort_keys=sort_keys)
        file.write("\n")
    return path


def feedback_to_num(feedback):
    """'NYGNN' -> base-3 integer."""
    feedback = feedback.strip().upper()
    if len(feedback) != 5 or any(c not in TILE_VALS for c in feedback):
        raise ValueError(f"invalid feedback string: {feedback!r}")
    num = 0
    for c in feedback:
        num = num * 3 + TILE_VALS[c]
    return num


def score_guess(guess, answer):
    """Feedback for one guess as a base-3 integer, duplicate letters and all.

    This is the rule ``feedback_table.npy`` is built from; it exists separately
    so that anything only needing a single score (the game, say) does not have
    to load the table.
    """
    result = ["N"] * 5
    remaining = Counter(answer)
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = "G"
            remaining[answer[i]] -= 1
    for i in range(5):
        if result[i] != "G" and remaining[guess[i]] > 0:
            result[i] = "Y"
            remaining[guess[i]] -= 1
    return feedback_to_num("".join(result))


def num_to_feedback(num):
    """Base-3 integer -> 'NYGNN'."""
    chars = "NYG"
    out = []
    for _ in range(5):
        out.append(chars[num % 3])
        num //= 3
    return "".join(reversed(out))


@dataclass
class WordleData:
    all_words: list          # index -> word
    word2ind: dict           # word -> index
    curated_list: list       # the original 2022 answer list
    past_solutions: list     # answers that have already been used, in order
    puzzle_numbers: list     # Wordle's own number for each of those, in step
    feedback_table: np.ndarray  # uint8 [guess, answer] -> pattern number
    _letter_presence: np.ndarray = None

    @property
    def n_words(self):
        return len(self.all_words)

    def all_candidates(self):
        return np.arange(self.n_words, dtype=np.int64)

    def letter_presence(self):
        """bool [word, letter] -- does the word contain that letter at all.
        It is built once and kept, which turns a weighted count over any
        candidate set into a single matrix product."""
        if self._letter_presence is None:
            grid = np.frombuffer("".join(self.all_words).encode(), dtype=np.uint8)
            grid = grid.reshape(self.n_words, -1).astype(np.intp) - ord("a")
            presence = np.zeros((self.n_words, 26), dtype=bool)
            np.put_along_axis(presence, grid, True, axis=1)
            self._letter_presence = presence
        return self._letter_presence


_CACHE = {}


@functools.lru_cache(maxsize=None)
def record_through():
    """The print date of the record's last entry: the day the solver was last
    brought up to date, which the page shows."""
    with open(SOLUTIONS_PATH, "r") as file:
        return json.load(file)[-1].get("print_date", "")


def load_data(mask_last=0, use_cache=True, with_feedback_table=True):
    """Load everything.

    ``mask_last`` drops the most recent N entries of solutions.json, which
    winds the used-word prior back to an earlier day.  It defaults to 0:
    ``solutions.py`` stops the record at yesterday, so every entry is a puzzle
    already played and there is nothing to hold back.

    ``with_feedback_table=False`` leaves ``feedback_table`` as None, which is
    enough for the word lists and the priors and skips the table read.
    """
    key = (mask_last, with_feedback_table)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    feedback_table = np.load(FEEDBACK_TABLE_PATH) if with_feedback_table else None
    with open(WORD2IND_PATH, "r") as file:
        word2ind = json.load(file)
    with open(IND2WORD_PATH, "r") as file:
        all_words = json.load(file)
    with open(CURATED_PATH, "r") as file:
        curated_list = json.load(file)
    with open(SOLUTIONS_PATH, "r") as file:
        solutions_so_far = json.load(file)

    if mask_last:
        solutions_so_far = solutions_so_far[:-mask_last]
    past_solutions = [entry["solution"] for entry in solutions_so_far]
    # the pre-NYT entries carry no day count, so fall back to their position
    puzzle_numbers = [entry.get("days_since_launch", i) + 1
                      for i, entry in enumerate(solutions_so_far)]

    data = WordleData(
        all_words=all_words,
        word2ind=word2ind,
        curated_list=curated_list,
        past_solutions=past_solutions,
        puzzle_numbers=puzzle_numbers,
        feedback_table=feedback_table,
    )
    if use_cache:
        _CACHE[key] = data
    return data
