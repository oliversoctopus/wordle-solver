"""Which form of f to fit when none is asked for.

Every archived fit is replayed on a holdout it has never seen -- a fresh seed,
not the one `compare` selected the winners on -- with the same answers for
every form within a prior, so the differences are paired.  Backs README
*Which form is the default*.
"""

import argparse
import json
import os

import numpy as np

import common
import estimator
from priors import ALL_CONFIGS
from solver import WordleSolver, sample_answers

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=3000)
parser.add_argument("--seed", type=int, default=777, help="not compare's 10000")
args = parser.parse_args()


def archived(config, form):
    """The fit `compare` left for this (prior, form), or None."""
    path = estimator.form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def lengths_of(solver, answers):
    """Guesses taken per game, a failure counted as one over the limit."""
    out = []
    for answer in answers:
        guesses, _ = solver.play(int(answer), max_guesses=MAX_GUESSES,
                                 deadline=MAX_GUESSES)
        solved = guesses[-1] == int(answer)
        out.append(len(guesses) if solved else MAX_GUESSES + 1)
    return np.array(out, dtype=float)


forms = sorted(estimator.FORMS)
results = {}

for config in ALL_CONFIGS:
    common.header(config.slug)
    runs, openings = {}, {}
    answers = None
    for form in forms:
        record = archived(config, form)
        if record is None:
            continue
        solver = WordleSolver(config=config, params=record["params"], form=form)
        if answers is None:   # identical within a prior, so the rows are paired
            answers = sample_answers(solver.weights, args.games, seed=args.seed)
        runs[form] = lengths_of(solver, answers)
        openings[form] = record.get("opening_word", "?")

    best = min(runs, key=lambda f: runs[f].mean())
    print(f"  {args.games} unseen answers, seed {args.seed}, paired against "
          f"{best}\n")
    print(f"  {'form':11s} {'mean':>7s} {'solved':>8s} {'gap':>8s} "
          f"{'se(gap)':>8s} {'opening':>8s}")
    order = sorted(runs, key=lambda f: runs[f].mean())
    for rank, form in enumerate(order, start=1):
        gap, se, _ = common.paired(runs[form], runs[best])
        results.setdefault(form, []).append((rank, gap, se))
        print(f"  {form:11s} {runs[form].mean():7.4f} "
              f"{(runs[form] <= MAX_GUESSES).mean() * 100:7.2f}% {gap:+8.4f} "
              f"{se:8.4f} {openings[form]:>8s}")
    print()

common.header("over every prior")
print("  gap is to the best form for that prior; a form has to be good under a")
print("  prior nobody has fitted it for, so the worst case is the one to read\n")
print(f"  {'form':11s} {'mean gap':>9s} {'worst gap':>10s} {'worst rank':>11s} "
      f"{'mean rank':>10s}")
rows = []
for form, seen in results.items():
    ranks = [r for r, _, _ in seen]
    gaps = [g for _, g, _ in seen]
    rows.append((float(np.mean(gaps)), max(gaps), max(ranks),
                 float(np.mean(ranks)), form))
for mean_gap, worst_gap, worst_rank, mean_rank, form in sorted(rows):
    print(f"  {form:11s} {mean_gap:+9.4f} {worst_gap:+10.4f} {worst_rank:11d} "
          f"{mean_rank:10.1f}")
print()
print(f"  current default: {estimator.DEFAULT_FORM}")
