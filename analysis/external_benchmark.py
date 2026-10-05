"""The solver against the published optimum, over the curated answer list.

Fits `f` uniform over the list with the candidates restricted to it, then
plays every answer in four arms.  Backs README *Against the known optimum*.
"""

import argparse
import time

import numpy as np

import common
import train
from priors import PriorConfig
from solver import DERIVED_WIN_BONUS, ENTROPY, EXPECTED, MAX_TURNS, WordleSolver
from wordle_data import load_data

# Selby's exact optimum, normal mode, 2,315 answers and 12,972 allowed guesses:
# https://sonorouschocolate.com/notes/index.php/The_best_strategies_for_Wordle
OPTIMUM_TOTAL = 7920
OPTIMUM_ANSWERS = 2315
OPTIMUM_ALLOWED = 12972
OPTIMUM_OPENING = "salet"
OPTIMUM_WORST = 5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refit", action="store_true",
                    help="fit f again instead of reusing the recorded fit")
parser.add_argument("--slice", type=int, default=0,
                    help="smoke run over the first N answers; records nothing")
args = parser.parse_args()

data = load_data()
answers = np.array([data.word2ind[w] for w in data.curated_list], dtype=np.int64)
if args.slice:
    answers = answers[:args.slice]


def note(key, value):
    """Record a figure, except in a smoke run."""
    return value if args.slice else common.record(key, value)


def lengths(solver, first=None):
    """Guesses each answer on the list takes, a loss counting as seven."""
    out = np.empty(len(answers))
    for i, answer in enumerate(answers):
        live = solver.all_candidates()
        for turn in range(1, MAX_TURNS + 1):
            if turn == 1 and first is not None:
                guess = first
            else:
                guess, _ = solver.best_guess(live, MAX_TURNS - turn + 1)
            if guess == answer:
                out[i] = turn
                break
            live = solver.filter(guess, solver.feedback(guess, answer), live)
        else:
            out[i] = MAX_TURNS + 1
    return out


# ------------------------------------------------------------------ the fit

fit = None if args.refit or args.slice else common.result(
    "external_benchmark", "fit_params")
if fit is None:
    common.header("fitting f under a uniform prior over the list")
    started = time.perf_counter()
    params, meta = train.policy_iteration(
        PriorConfig(), data=data,
        n_games=200 if args.slice else train.DEFAULT_GAMES,
        max_iters=1 if args.slice else train.DEFAULT_MAX_ITERS,
        solver_options={"answers": answers}, save=False, use_rust=False)
    print(f"  fitted in {time.perf_counter() - started:.0f}s")
    fit = [float(p) for p in params]
note("fit_params", fit)

# ------------------------------------------------------------------ the arms

fitted = WordleSolver(data=data, mode=EXPECTED, params=fit, answers=answers)
arms = [
    ("fitted f", fitted, None),
    ("fitted f, opening salet", fitted, data.word2ind[OPTIMUM_OPENING]),
    ("greedy entropy", WordleSolver(data=data, mode=ENTROPY, answers=answers),
     None),
    ("IG + 2·p_win", WordleSolver(data=data, mode=ENTROPY, answers=answers,
                                  win_bonus=DERIVED_WIN_BONUS), None),
]

optimum = OPTIMUM_TOTAL / OPTIMUM_ANSWERS
note("optimum.mean", optimum)
note("optimum.total", OPTIMUM_TOTAL)
note("optimum.answers", OPTIMUM_ANSWERS)
note("optimum.allowed", OPTIMUM_ALLOWED)
note("optimum.worst", OPTIMUM_WORST)
note("answers", len(answers))
note("allowed", data.n_words)

common.header(f"every one of the {len(answers)} answers, against "
              f"{optimum:.4f} over {OPTIMUM_ANSWERS}")
print(f"  {'arm':<26} {'opening':>8} {'mean':>8} {'gap':>8} {'lost':>5} "
      f"{'worst':>6}  distribution 1..6")
rows, means = [], []
for n, (label, solver, first) in enumerate(arms):
    started = time.perf_counter()
    got = lengths(solver, first)
    opening = data.all_words[first if first is not None
                             else solver.best_guess(solver.all_candidates(),
                                                    MAX_TURNS)[0]]
    mean, lost = float(got.mean()), int((got > MAX_TURNS).sum())
    counts = [int((got == k).sum()) for k in range(1, MAX_TURNS + 1)]
    assert sum(counts) + lost == len(answers)
    print(f"  {label:<26} {opening:>8} {mean:8.4f} {mean - optimum:+8.4f} "
          f"{lost:>5} {int(got.max()):>6}  {counts}"
          f"   ({time.perf_counter() - started:.0f}s)")
    means.append(mean)
    key = ("fitted", "salet", "entropy", "ig")[n]
    note(f"{key}.mean", mean)
    note(f"{key}.gap", mean - optimum)
    note(f"{key}.lost", lost)
    note(f"{key}.sixes", counts[MAX_TURNS - 1])
    note(f"{key}.opening", opening)
    rows.append((label, f"`{opening}`", f"{mean:.4f}",
                 f"{mean - optimum:+.4f}".replace("-", "−"), str(lost),
                 " / ".join(str(c) for c in counts)))

# what the fitted policy's own opening gives up against salet behind it
print(f"\n  salet over the fitted opening: "
      f"{note('opening_regret', means[0] - means[1]):+.4f} guesses")
note("table", common.md_table(
    ["policy", "opening", "mean", "vs. optimum", "lost",
     "games in 1 / 2 / 3 / 4 / 5 / 6"], rows))
