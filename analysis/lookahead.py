"""What one ply of branch-capped lookahead is worth, per prior and blended.

The same answers are played twice, once with `--lookahead 0` and once with 1 at
the default branch cap, so the difference is paired.  Both arms read the fit
cached for that prior: lookahead sits above `f` rather than replacing it, and is
not part of the cache key.

Backs README *Lookahead*.  Its value moves with the quality of the base policy,
so a refit of `f` moves this measurement -- read it against a run of the same
vintage, not against a published number.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import (DEFAULT_LOOKAHEAD_BRANCH, EXPECTED, WordleSolver,
                    sample_answers)
from wordle_data import load_data

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=5000, help="answers per prior")
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--branch", type=int, default=DEFAULT_LOOKAHEAD_BRANCH)
parser.add_argument("--shipped-only", action="store_true")
args = parser.parse_args()

data = load_data()
configs = ([PriorConfig()] if args.shipped_only else
           [PriorConfig(repeat=r, curation=c) for r in (False, True) for c in (False, True)])


def lengths_of(solver, answers):
    """Guesses taken per game, a failure counted as one over the limit."""
    out = np.empty(len(answers))
    for i, answer in enumerate(answers):
        guesses, _ = solver.play(int(answer), max_guesses=MAX_GUESSES,
                                 deadline=MAX_GUESSES)
        solved = guesses[-1] == int(answer)
        out[i] = len(guesses) if solved else MAX_GUESSES + 1
    return out


diffs = []
for config in configs:
    common.header(config.slug)
    answers = sample_answers(
        WordleSolver(data=data, config=config, mode=EXPECTED).weights,
        args.games, seed=args.seed)

    arms = {}
    for ply in (0, 1):
        solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                              lookahead=ply, lookahead_max_branch=args.branch)
        start = time.perf_counter()
        arms[ply] = lengths_of(solver, answers)
        print(f"  ply {ply}: mean {arms[ply].mean():.4f}   "
              f"[{time.perf_counter() - start:.0f}s]")

    delta, se, per_game = common.paired(arms[1], arms[0])
    diffs.append(per_game)
    changed = int((per_game != 0).sum())
    print(f"\n  one ply at width {args.width}: {delta:+.4f} +- {se:.4f} guesses")
    print(f"  games changing length: {changed} of {args.games} "
          f"({int((per_game < 0).sum())} better, {int((per_game > 0).sum())} worse)\n")

pooled = np.concatenate(diffs)
se = pooled.std(ddof=1) / np.sqrt(len(pooled))
common.header("blended")
print(f"  {len(pooled)} paired games over {len(configs)} prior(s): "
      f"{pooled.mean():+.4f} +- {se:.4f}  ({abs(pooled.mean()) / se:.1f} sigma)")
print(f"  games changing length: {int((pooled != 0).sum())} "
      f"({int((pooled < 0).sum())} better, {int((pooled > 0).sum())} worse)")
