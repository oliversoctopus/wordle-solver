"""What hard mode costs, in guesses.

Every word is played twice, once with the guess free and once restricted to words
that could still be the answer, and both arms are weighted by the same prior, so
the difference between them is exact rather than a paired estimate.  `f` is the
fit cached for free play in both arms -- the restriction is not part of the slug,
so there is no hard-mode fit to read.  Backs README *Hard mode*.
"""

import argparse

import numpy as np

import common
from priors import ALL_CONFIGS, PriorConfig
from solver import ENTROPY, EXPECTED, WordleSolver
from wordle_data import load_data

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=(EXPECTED, ENTROPY), default=EXPECTED)
parser.add_argument("--all-priors", action="store_true",
                    help="every combination, not just the shipped one")
args = parser.parse_args()


def opening(solver):
    return solver.data.all_words[
        solver.best_guess(solver.all_candidates(), MAX_GUESSES)[0]]


data = load_data()
configs = ALL_CONFIGS if args.all_priors else [PriorConfig()]

for config in configs:
    common.header(config.slug)
    arms = {}
    for hard in (False, True):
        solver = WordleSolver(data=data, config=config, mode=args.mode, hard=hard)
        arms[hard] = (common.every_answer(solver), opening(solver), solver.weights)

    (free, free_open, weights), (rest, rest_open, _) = arms[False], arms[True]
    per_word = rest - free
    print(f"  every word as the answer, weighted by this prior, "
          f"mode={args.mode}\n")
    print(f"  {'guesses':14s} {'mean':>7s} {'solved':>8s} {'opening':>8s}")
    for name, lengths, word in (("free", free, free_open),
                                ("hard mode", rest, rest_open)):
        mean, loss = common.exact_stats(lengths, weights)
        print(f"  {name:14s} {mean:7.4f} {(1 - loss) * 100:7.2f}% {word:>8s}")
    print(f"\n  cost of the restriction: {float(weights @ per_word):+.4f} guesses"
          f"  (exact, not an estimate)")
    print(f"  words changing length: {int((per_word != 0).sum())} of "
          f"{len(per_word)} ({int((per_word < 0).sum())} better, "
          f"{int((per_word > 0).sum())} worse)")
    print(f"  losing words: {int((free > MAX_GUESSES).sum())} free, "
          f"{int((rest > MAX_GUESSES).sum())} hard\n")
