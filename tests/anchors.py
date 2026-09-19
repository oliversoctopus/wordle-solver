"""The facts everything else here is measured against.

Each claim is worked out by hand from a definition outside the repository:
Wordle's coloring rule, the base-3 pattern encoding, Shannon's entropy, and
what optimal play is worth in a position small enough to settle on paper.
The scoring sections run on a six-word game whose table `score_guess` builds,
and each asserts the structure it rests on before the value.  `--full` adds
`score_guess` against `feedback_table.npy`, which nothing else compares.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernels
from exact import ExactEndgame
from wordle_data import (N_PATTERNS, WIN_PATTERN, feedback_to_num, load_data,
                         num_to_feedback, score_guess)

problems = []


def check(claim, ok, detail=""):
    print("  %s %s%s" % ("ok  " if ok else "FAIL", claim,
                         "" if not detail else "   " + detail))
    if not ok:
        problems.append(claim + ("" if not detail else "   " + detail))


# --------------------------------------------------------------- the coloring

# (guess, answer, the pattern the rule gives, what it pins down)
COLORING = [
    ("crane", "crane", "GGGGG", "the win"),
    ("fudgy", "crane", "NNNNN", "nothing shared"),
    ("eerie", "crane", "NNYNG",
     "a green takes the only e, so both earlier ones stay gray"),
    ("sassy", "grass", "YYNGN",
     "one s left after the green, and the leftmost unplaced one takes it"),
    ("abbey", "babes", "YYGGN",
     "greens come out first, and the spare b is still yellow"),
    ("leant", "melee", "YGNNN",
     "the answer's three e's give the guess's one e nothing extra"),
    ("elegy", "geese", "YNGYN",
     "two e's left over one green, and the first of them takes one"),
]


def coloring():
    """Wordle's rule: greens first, each eating a letter of the answer, then
    yellows from the left over whatever the greens left behind."""
    for guess, answer, pattern, why in COLORING:
        got = num_to_feedback(score_guess(guess, answer))
        check("%s against %s is %s" % (guess, answer, pattern), got == pattern,
              why if got == pattern else "the rule gives " + got)


def encoding():
    """A pattern is a base-3 numeral, N=0 Y=1 G=2, read from the left."""
    check("NNNNN is 0", feedback_to_num("NNNNN") == 0)
    check("NNNNY is 1", feedback_to_num("NNNNY") == 1,
          "the last tile is the units")
    check("YNNNN is 81", feedback_to_num("YNNNN") == 81,
          "and the first is 3^4")
    check("GGGGG is 242", feedback_to_num("GGGGG") == 242, "2 * (3^5 - 1) / 2")
    check("the win pattern is that one", WIN_PATTERN == 242)
    check("there are 3^5 patterns", N_PATTERNS == 243)
    check("every one of the %d survives the round trip" % N_PATTERNS,
          all(feedback_to_num(num_to_feedback(n)) == n
              for n in range(N_PATTERNS)))


# ------------------------------------------------------------ a miniature game

WORDS = ["crane", "slate", "ghost", "plumb", "brick", "fudgy"]


def mini(words=None):
    """A whole feedback table for a handful of words, scored the real way."""
    words = WORDS if words is None else words
    table = np.array([[score_guess(g, a) for a in words] for g in words],
                     dtype=np.uint8)
    return words, table


def indices(words, *names):
    return np.array([words.index(name) for name in names], dtype=np.int64)


def information():
    """Information gain is Shannon's, in bits: the prior entropy less what is
    expected to remain once the guess has been colored."""
    words, table = mini()

    def bits(guess, candidates, weights=None):
        pool = indices(words, guess)
        cand = indices(words, *candidates)
        if weights is None:
            weights = np.ones(len(candidates))
        return float(kernels.ig_all(table, cand, np.asarray(weights),
                                    pool=pool)[0])

    def buckets(guess, candidates):
        row = table[words.index(guess)]
        return len({int(row[words.index(c)]) for c in candidates})

    # a guess that tells every candidate apart leaves nothing to learn, so it
    # is worth the whole prior entropy: log2 of however many there were
    four = ["crane", "slate", "ghost", "brick"]
    check("crane splits those four apart", buckets("crane", four) == 4)
    check("and is therefore worth log2(4) = 2 bits",
          bits("crane", four) == 2.0)

    two = ["slate", "brick"]
    check("crane splits those two apart", buckets("crane", two) == 2)
    check("and is therefore worth exactly 1 bit", bits("crane", two) == 1.0)

    three = ["crane", "slate", "brick"]
    check("crane splits those three apart", buckets("crane", three) == 3)
    check("and is therefore worth log2(3) bits",
          abs(bits("crane", three) - math.log2(3)) < 1e-12)

    # a guess that colors alike whatever the answer says nothing at all
    same = ["crane", "plumb"]
    check("ghost colors both of those the same", buckets("ghost", same) == 1)
    check("and is therefore worth exactly 0 bits", bits("ghost", same) == 0.0)

    # weights are probabilities, so an uneven split is worth less than a bit
    uneven = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    check("splitting 3:1 is worth H(3/4, 1/4) = %.6f bits" % uneven,
          abs(bits("crane", two, [3.0, 1.0]) - uneven) < 1e-12)


def endgame():
    """What optimal play is worth, in positions small enough to prove."""
    words, table = mini()
    search = ExactEndgame(table, np.ones(len(words)))

    guess, value = search.solve(indices(words, "crane"), 3)
    check("one candidate costs one guess", value == 1.0)
    check("and the guess is that candidate", words[guess] == "crane")

    # guess either: right half the time, and the other half you now know
    pair = ["crane", "slate"]
    guess, value = search.solve(indices(words, *pair), 3)
    check("two equal candidates cost (1 + 2) / 2 = 1.5 guesses", value == 1.5)
    check("and the guess is one of them", words[guess] in pair)

    # the same position, but one of them is three times as likely: guess that
    # one, and be done immediately three times in four
    weights = np.zeros(len(words))
    weights[words.index("crane")] = 3.0
    weights[words.index("slate")] = 1.0
    weighted = ExactEndgame(table, weights)
    guess, value = weighted.solve(indices(words, *pair), 3)
    check("at 3:1 they cost 0.75 + 2 * 0.25 = 1.25 guesses", value == 1.25)
    check("and the guess is the likely one", words[guess] == "crane")

    # one candidate can be had in one guess and the rest need two, so (1+2+2)/3
    # is a floor for any three; a candidate separating the other two reaches it
    three = ["crane", "slate", "brick"]

    def separates(word):
        row = table[words.index(word)]
        return len({int(row[words.index(c)]) for c in three}) == 3

    check("some of those three tell all three apart",
          any(separates(word) for word in three))
    guess, value = search.solve(indices(words, *three), 3)
    check("three equal candidates cost (1 + 2 + 2) / 3 guesses",
          abs(value - 5.0 / 3.0) < 1e-12)
    # all three separate here, so which one comes back is the cap's tie-break
    # and not a fact; what has to hold is that the guess could win outright
    check("and the guess is a candidate that separates them",
          words[guess] in three and separates(words[guess]),
          "it picked " + words[guess])


def bounds(seed, rounds):
    """Two bounds that hold of any position at all, over positions nobody
    chose: a guess cannot be worth more than the prior entropy, and optimal
    play cannot beat winning outright on the likeliest word and needing a
    second guess for every other."""
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefg"))     # few letters, so patterns collide
    broken, checked = [], 0
    for _ in range(rounds):
        n = int(rng.integers(3, 7))
        words = ["".join(rng.choice(letters, 5)) for _ in range(n)]
        if len(set(words)) < n:
            continue
        words, table = mini(words)
        weights = rng.random(n) + 0.05
        cand = np.arange(n, dtype=np.int64)

        prior = weights / weights.sum()
        entropy = float(-(prior * np.log2(prior)).sum())
        gains = kernels.ig_all(table, cand, weights)
        if gains.max() > entropy + 1e-9 or gains.min() < -1e-9:
            broken.append("%s: a gain of %.6f outside [0, %.6f]"
                          % (words, gains.max(), entropy))

        value = ExactEndgame(table, weights).solve(cand, 4)[1]
        floor = 2.0 - prior.max()
        if value < floor - 1e-9:
            broken.append("%s: optimal play at %.6f beats its floor %.6f"
                          % (words, value, floor))
        checked += 1
    check("%d drawn positions stay inside both bounds" % checked,
          not broken, "seed %d%s" % (seed, "" if not broken
                                     else "; " + broken[0]))


def against_the_table(seed, words):
    """`score_guess` against `feedback_table.npy`, which `wordle_result` built.
    Weighted towards repeated letters, where a transcription goes wrong."""
    data = load_data()
    rng = np.random.default_rng(seed)
    doubled = [w for w in data.all_words if len(set(w)) < 5]
    picked = list(rng.choice(doubled, words // 2, replace=False))
    picked += list(rng.choice(data.all_words, words // 2, replace=False))
    print("  %d of %d words carry a repeated letter; %d drawn, half from "
          "those" % (len(doubled), data.n_words, len(picked)))

    wrong = []
    for guess in picked:
        row = data.feedback_table[data.word2ind[guess]]
        for answer in picked:
            if int(row[data.word2ind[answer]]) != score_guess(guess, answer):
                wrong.append("%s/%s" % (guess, answer))
    check("%d pairs score the same both ways" % (len(picked) ** 2),
          not wrong, "" if not wrong else "differs on " + ", ".join(wrong[:5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true",
                        help="also score against the table, which it loads")
    parser.add_argument("--seed", type=int, default=20260827,
                        help="which positions and words get drawn")
    parser.add_argument("--rounds", type=int, default=60,
                        help="how many random positions the bounds see")
    parser.add_argument("--words", type=int, default=400,
                        help="how many --full draws; every pair is scored")
    args = parser.parse_args()

    print("the coloring rule")
    coloring()
    print("\nthe pattern encoding")
    encoding()
    print("\ninformation gain, in bits")
    information()
    print("\nwhat optimal play is worth")
    endgame()
    print("\nbounds that hold of any position")
    bounds(args.seed, args.rounds)
    if args.full:
        print("\nthe rule against the shipped table")
        against_the_table(args.seed, args.words)

    print()
    if problems:
        print("%d claim%s failed" % (len(problems),
                                     "" if len(problems) == 1 else "s"))
        return 1
    print("every anchor holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
