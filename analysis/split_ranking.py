"""How much of a ranking of words is one guess spelled several ways.

A guess *is* its split of the candidate set: two words giving identical feedback
against every candidate still standing play the same game from here on.  So a
list of the five best-scoring *words* can be fewer than five guesses, and the
ones it leaves out are the ones a reader would find interesting -- a different
guess, not a different spelling of the one above.  `WordleSolver.top_guesses`
therefore ranks splits, and `WordleSolver.split_siblings` names the rest of each.

This measures both halves of that.  How often a word ranking would have doubled
up, by candidate count, and how large the groups are; then what the exact
search's `top` costs over `solve`, since asking it for runners-up loosens the
cutoff that prunes the search and so is not free.

Backs README *Alternatives* under the web page, and the closing paragraph of
README *The exact endgame*.  The timings are a probe rather than a measurement:
they move with the machine, so read the ratio.

    python analysis/split_ranking.py                 both, shipped prior
    python analysis/split_ranking.py --all-priors    all four
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import DEFAULT_EXACT_MAX, ENTROPY, EXPECTED, MAX_TURNS, WordleSolver, sample_answers
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=300,
                    help="games drawn on to collect positions from")
parser.add_argument("--positions", type=int, default=200,
                    help="positions kept per prior, per band")
parser.add_argument("--top", type=int, default=5, help="length of the ranking")
parser.add_argument("--exact-max", type=int, default=DEFAULT_EXACT_MAX)
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--all-priors", action="store_true")
args = parser.parse_args()

data = load_data()
configs = ([PriorConfig(repeat=r, curation=c)
            for r in (False, True) for c in (False, True)]
           if args.all_priors else [PriorConfig()])

# a split stops being shared long before the candidates run out; these are where
# it happens, not an even division of the range
BANDS = ((2, 4), (5, 12), (13, 40), (41, 12947))


def band_of(size):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= size <= hi)


def words_ranking(solver, candidates, k):
    """The k best *words*, which is what a ranking without the dedupe is."""
    pool = solver.guess_pool(candidates)
    scores = solver.score_all(candidates, pool=pool)
    order = np.argsort(scores)
    if solver.mode == ENTROPY:
        order = order[::-1]
    return [int(i) if pool is None else int(pool[i]) for i in order[:k]]


def duplicate_splits(solver, candidates, k):
    """(distinct guesses among the k best words, words in their splits)."""
    listed = words_ranking(solver, candidates, k)
    table = solver.data.feedback_table
    splits = {tuple(table[guess, candidates]) for guess in listed}
    named = sum(len(solver.split_siblings(guess, candidates)) for guess in listed)
    return len(splits), named / max(1, len(listed))


for config in configs:
    for mode in (EXPECTED, ENTROPY):
        solver = WordleSolver(data=data, config=config, mode=mode,
                              exact_max=args.exact_max)
        answers = sample_answers(solver.weights, args.games, seed=args.seed)
        found, counts = common.collect_positions(
            solver, answers, lambda s, c, t: len(c) > 1)
        rng = np.random.default_rng(args.seed)

        common.header(f"{config.slug}, {mode}: how much of a top {args.top} "
                      f"is one guess twice")
        print(f"  {len(found)} distinct positions from {args.games} games\n")
        print("  candidates    positions   distinct guesses   words per split")
        for band, (lo, hi) in enumerate(BANDS):
            rows = [(t, c) for t, c in found if band_of(len(c)) == band]
            if not rows:
                continue
            if len(rows) > args.positions:
                keep = rng.choice(len(rows), size=args.positions, replace=False)
                rows = [rows[i] for i in sorted(keep)]
            counted = [duplicate_splits(solver, c, args.top) for _, c in rows]
            distinct = np.mean([d for d, _ in counted])
            named = np.mean([n for _, n in counted])
            label = f"{lo}-{hi}" if hi < 12947 else f"{lo}+"
            print(f"  {label:>11}  {len(rows):>10}   {distinct:>16.2f}   "
                  f"{named:>15.1f}")
        print()

# ------------------------------------------------- what the runners-up cost

solver = WordleSolver(data=data, config=PriorConfig(), mode=EXPECTED,
                      exact_max=args.exact_max)
answers = sample_answers(solver.weights, args.games, seed=args.seed)
deep, _ = common.collect_positions(
    solver, answers,
    lambda s, c, t: t >= 3 and 3 <= len(c) <= s.exact_max)
if len(deep) > args.positions:
    keep = np.random.default_rng(args.seed).choice(len(deep), size=args.positions,
                                                   replace=False)
    deep = [deep[i] for i in sorted(keep)]

common.header("what asking the exact search for runners-up costs")
sizes = [len(c) for _, c in deep]
print(f"  {len(deep)} positions the deep search takes on, "
      f"{min(sizes)}-{max(sizes)} candidates\n")
print("  asked for      total     mean     worst")


def timed(call):
    total, worst = 0.0, 0.0
    for turn, candidates in deep:
        solver.exact.memo.clear()
        started = time.perf_counter()
        call(candidates, MAX_TURNS - turn + 1)
        elapsed = time.perf_counter() - started
        total, worst = total + elapsed, max(worst, elapsed)
    return total, worst


solver.exact.top(deep[0][1], 3, args.top)   # compile the kernels first
for label, call in (("the pick", lambda c, t: solver.exact.solve(c, t)),
                    ("top 1", lambda c, t: solver.exact.top(c, t, 1)),
                    ("top 3", lambda c, t: solver.exact.top(c, t, 3)),
                    (f"top {args.top}", lambda c, t: solver.exact.top(c, t, args.top))):
    total, worst = timed(call)
    print(f"  {label:>9}  {total:>8.2f}s  {total / len(deep) * 1000:>6.0f}ms  "
          f"{worst * 1000:>6.0f}ms")
