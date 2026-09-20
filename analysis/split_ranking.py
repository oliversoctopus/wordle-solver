"""How much of a ranking of words is one guess spelled several ways.

Two words giving identical feedback against every candidate play the same
game from here on, so `WordleSolver.top_guesses` ranks splits and
`WordleSolver.split_siblings` names the rest of each.  This measures how often
a word ranking would have doubled up, by candidate count, and what the exact
search's `top` costs over `solve`.  `--all-priors` runs all four and
`--exact-root-branch 50` the narrow cap.  Backs README *In a browser* and
*The exact endgame*; the timings are a probe and move with the machine.
"""

import argparse
import time

import numpy as np

import common
from exact import DEFAULT_ROOT_BRANCH
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
parser.add_argument("--exact-root-branch", type=int, default=DEFAULT_ROOT_BRANCH,
                    help="splits the ranked node may try; what this section "
                         "is timing, so 50 gives the pre-widening figures")
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--all-priors", action="store_true")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data()
configs = common.filter_priors(
    [PriorConfig(repeat=r, curation=c)
     for r in (False, True) for c in (False, True)]
    if args.all_priors else [PriorConfig()], args)

# a split stops being shared long before the candidates run out; these are where
# it happens, not an even division of the range
BANDS = ((2, 4), (5, 12), (13, 40), (41, 12947))


def band_of(size):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= size <= hi)


def words_ranking(solver, candidates, k):
    """The k best *words*: a ranking without the dedupe."""
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
            if config == PriorConfig() and mode == EXPECTED:
                common.record(f"band{lo}_{hi}.distinct", float(distinct))
                common.record(f"band{lo}_{hi}.words", float(named))
        print()

# ------------------------------------------------- what the runners-up cost

solver = WordleSolver(data=data, config=PriorConfig(), mode=EXPECTED,
                      exact_max=args.exact_max,
                      exact_root_branch=args.exact_root_branch)
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
    if args.exact_root_branch == DEFAULT_ROOT_BRANCH and not args.all_priors:
        key = "pick" if label == "the pick" else label.replace(" ", "")
        common.record(f"cost.{key}.mean_ms", total / len(deep) * 1000)
        common.record(f"cost.{key}.worst_ms", worst * 1000)
        common.record("cost.positions", len(deep))
