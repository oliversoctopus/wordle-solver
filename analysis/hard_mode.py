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
common.add_prior_filter(parser)
args = parser.parse_args()


def opening(solver):
    return solver.data.all_words[
        solver.best_guess(solver.all_candidates(), MAX_GUESSES)[0]]


data = load_data()
configs = common.filter_priors(
    ALL_CONFIGS if args.all_priors else [PriorConfig()], args)

LABELS = {"repeat0_curation0": "none", "repeat0_curation1_c-wordlebot": "curation only",
          "repeat1_curation0_r-binned": "repeat only",
          "repeat1_curation1_r-binned_c-wordlebot": "both",
          "repeat1_curation1_r-binned_c-frequency": "both, `frequency`"}
TAG = "" if args.mode == EXPECTED else "entropy."
table, costs = [], {}
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
    label = LABELS.get(config.slug, config.slug)
    key = TAG + label.replace(", `", "_").replace("`", "").replace(" ", "_") + "."
    free_mean, free_loss = common.exact_stats(free, weights)
    hard_mean, hard_loss = common.exact_stats(rest, weights)
    cost = float(weights @ per_word)
    costs[label] = cost
    common.record(key + "free", free_mean)
    common.record(key + "hard", hard_mean)
    common.record(key + "cost", cost)
    common.record(key + "free_solved", (1 - free_loss) * 100)
    common.record(key + "hard_solved", (1 - hard_loss) * 100)
    cells = (label, f"{free_mean:.4f}", f"{hard_mean:.4f}",
             common.neg(f"{cost:+.4f}"),
             f"{(1 - free_loss) * 100:.2f}% \u2192 {(1 - hard_loss) * 100:.2f}%")
    if label == "both":
        cells = ("**both (the default)**", *(f"**{c}**" for c in cells[1:]))
    table.append(cells)

if args.all_priors:
    common.record(TAG + "table", common.md_table(
        ["priors", "free", "hard", "cost", "solve rate, free \u2192 hard"], table))
    if "none" in costs and "both" in costs:
        common.record(TAG + "none_over_both", costs["none"] / costs["both"])
