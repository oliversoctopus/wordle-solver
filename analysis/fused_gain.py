"""What the answer-major dispatch is worth in whole games.

`table_layout.py` measures the two layouts kernel by kernel and projects a total
from the mix.  This measures the total directly: the same games played by a
solver holding an answer-major copy and by one without, wall clock, everything
included.

Each arm gets a fresh solver so neither inherits the other's memo, and the arms
are interleaved and taken at their best, because a run of this length drifts with
whatever else the machine is doing.  The opening is reported apart from the rest
for the reason it always is -- the memo answers it after the first game, so how
much it weighs is a fact about the length of the run.
"""

import argparse
import time

import numpy as np

import common
import fused_kernels
from solver import ENTROPY, EXPECTED, WordleSolver
from wordle_data import WIN_PATTERN, load_data


def play(solver, answers):
    """Total time, and the share of it spent on the first guess of a game."""
    opening = 0.0
    total = 0.0
    turns = 0
    for answer in answers:
        live = solver.all_candidates()
        for turn in range(6):
            started = time.perf_counter()
            guess, _ = solver.best_guess(live, 6 - turn)
            taken = time.perf_counter() - started
            total += taken
            turns += 1
            if turn == 0:
                opening += taken
            pattern = solver.feedback(guess, int(answer))
            if pattern == WIN_PATTERN:
                break
            live = solver.filter(guess, pattern, live)
    return total, opening, turns


def arm(data, answers, mode, fused):
    solver = WordleSolver(data=data, mode=mode, fused=fused)
    return play(solver, answers)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=120)
parser.add_argument("--rounds", type=int, default=3,
                    help="interleaved repeats; the best of each arm is taken")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mode", choices=(EXPECTED, ENTROPY), default=EXPECTED)
args = parser.parse_args()

data = load_data()
rng = np.random.default_rng(args.seed)
answers = rng.choice(data.all_candidates(), size=args.games, replace=False)

common.header("building the answer-major copy")
started = time.perf_counter()
transposed = fused_kernels.build_transposed(data.feedback_table)
build = time.perf_counter() - started
print(f"  {transposed.nbytes / 1e6:.0f} MB in {build:.2f}s, once per process, "
      f"no file on disk")
print(f"  dispatched to for candidate sets of {fused_kernels.FUSED_MIN} to "
      f"{fused_kernels.FUSED_MAX}")
del transposed

# both arms once before either is timed, so neither pays a compile
warm = answers[:1]
arm(data, warm, args.mode, False)
arm(data, warm, args.mode, True)

best = {}
for round_ in range(args.rounds):
    for fused in (False, True):
        total, opening, turns = arm(data, answers, args.mode, fused)
        key = "fused" if fused else "shipped"
        if key not in best or total < best[key][0]:
            best[key] = (total, opening, turns)
    print(f"  round {round_ + 1}: shipped {best['shipped'][0]:.2f}s, "
          f"fused {best['fused'][0]:.2f}s")

common.header(f"{args.games} games, {args.mode} mode, best of {args.rounds}")
print(f"  {'':<22} {'shipped':>10} {'fused':>10} {'gain':>8}")
rows = [
    ("whole games", 0, 0),
    ("first guesses only", 1, 1),
    ("every guess after", 0, 1),
]
for label, whole, minus in rows:
    a = best["shipped"][minus] if whole == 1 else best["shipped"][0] - best["shipped"][1] * minus
    b = best["fused"][minus] if whole == 1 else best["fused"][0] - best["fused"][1] * minus
    print(f"  {label:<22} {a:>9.2f}s {b:>9.2f}s {a / b:>7.2f}x")

turns = best["shipped"][2]
print(f"\n  {turns} guesses in all, "
      f"{(best['shipped'][0] - best['shipped'][1]) / (turns - args.games) * 1e3:.2f}ms "
      f"vs {(best['fused'][0] - best['fused'][1]) / (turns - args.games) * 1e3:.2f}ms "
      f"per guess after the first")
print(f"  the {build:.2f}s build pays for itself after "
      f"{build / max(1e-9, (best['shipped'][0] - best['fused'][0]) / args.games):.0f} "
      f"games")
