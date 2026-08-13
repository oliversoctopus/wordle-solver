"""Which form of f to fit when none is asked for.

Every archived fit is replayed over every possible answer, weighted by the prior
it was fitted for, so each row is that form's exact performance under that prior
rather than its score on a sample.  There is no holdout because there is nothing
held out: the evaluation covers the whole space a game can be drawn from.  Backs
README *Which form is the default*.
"""

import argparse
import json
import os

import numpy as np

import common
import estimator
from priors import ALL_CONFIGS
from solver import WordleSolver

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
args = parser.parse_args()


def archived(config, form):
    """The fit `compare` left for this (prior, form), or None."""
    path = estimator.form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


forms = sorted(estimator.FORMS)
results = {}

for config in ALL_CONFIGS:
    common.header(config.slug)
    runs, openings = {}, {}
    for form in forms:
        record = archived(config, form)
        if record is None:
            continue
        solver = WordleSolver(config=config, params=record["params"], form=form)
        runs[form] = common.exact_stats(common.every_answer(solver), solver.weights)
        openings[form] = record.get("opening_word", "?")

    best = min(runs, key=lambda f: runs[f][0])
    print(f"  every word as the answer, weighted by this prior; gaps to {best}\n")
    print(f"  {'form':11s} {'mean':>7s} {'solved':>8s} {'gap':>8s} {'opening':>8s}")
    order = sorted(runs, key=lambda f: runs[f][0])
    for rank, form in enumerate(order, start=1):
        mean, loss = runs[form]
        gap = mean - runs[best][0]
        results.setdefault(form, []).append((rank, gap))
        print(f"  {form:11s} {mean:7.4f} {(1 - loss) * 100:7.2f}% {gap:+8.4f} "
              f"{openings[form]:>8s}")
    print()

common.header("over every prior")
print("  gap is to the best form for that prior; a form has to be good under a")
print("  prior nobody has fitted it for, so the worst case is the one to read\n")
print(f"  {'form':11s} {'mean gap':>9s} {'worst gap':>10s} {'worst rank':>11s} "
      f"{'mean rank':>10s}")
rows = []
for form, seen in results.items():
    ranks = [r for r, _ in seen]
    gaps = [g for _, g in seen]
    rows.append((float(np.mean(gaps)), max(gaps), max(ranks),
                 float(np.mean(ranks)), form))
for mean_gap, worst_gap, worst_rank, mean_rank, form in sorted(rows):
    print(f"  {form:11s} {mean_gap:+9.4f} {worst_gap:+10.4f} {worst_rank:11d} "
          f"{mean_rank:10.1f}")
print()
print(f"  current default: {estimator.DEFAULT_FORM}")
