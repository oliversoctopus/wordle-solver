"""The two tables in README *Results*, replayed over every possible answer.

Part one scores every archived form under the shipped prior.  Part two plays each
prior combination in both modes.  Both weight by the prior in question rather
than sampling from it, so the means are computed rather than estimated and the
gaps between rows carry no sampling error at all -- see the note on enumerating
in `analysis/README.md`.

Both parts evaluate what is cached; neither refits.  Run `wordle_solver.py
train` first if the fits are stale.
"""

import argparse
import json
import os

import numpy as np

import common
import estimator
from priors import PriorConfig
from solver import ENTROPY, EXPECTED, WordleSolver

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--part", choices=("forms", "priors", "both"), default="both")
args = parser.parse_args()


def archived(config, form):
    path = estimator.form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def opening_of(solver):
    guess, _ = solver.best_guess(solver.all_candidates(), MAX_GUESSES)
    return solver.data.all_words[guess]


if args.part in ("forms", "both"):
    config = PriorConfig()
    common.header(f"forms under {config.describe()}")
    print("  every word as the answer, weighted by this prior; exact\n")

    runs, meta, weights = {}, {}, None
    for form in sorted(estimator.FORMS):
        record = archived(config, form)
        if record is None:
            continue
        solver = WordleSolver(config=config, params=record["params"], form=form)
        weights = solver.weights
        runs[form] = common.every_answer(solver)
        meta[form] = (f"{record.get('terms_used')}/{len(record.get('terms', []))}",
                      record.get("opening_word", "?"))

    greedy = WordleSolver(config=config, mode=ENTROPY)
    runs["greedy entropy"] = common.every_answer(greedy)
    meta["greedy entropy"] = ("--", opening_of(greedy))

    scored = {f: common.exact_stats(runs[f], weights) for f in runs}
    best = min(scored, key=lambda f: scored[f][0])
    print(f"  {'form':15s} {'terms':>6s} {'opening':>8s} {'mean':>8s} "
          f"{'vs best':>9s} {'loss prob':>10s}")
    for form in sorted(scored, key=lambda f: scored[f][0]):
        mean, loss = scored[form]
        print(f"  {form:15s} {meta[form][0]:>6s} {meta[form][1]:>8s} "
              f"{mean:8.4f} {mean - scored[best][0]:+9.4f} {loss * 100:9.4f}%")
    print(f"\n  best: {best}")

if args.part in ("priors", "both"):
    common.header("prior combinations, both modes")
    print("  every word as the answer, weighted by that row's own prior, so the "
          "rows are\n  not comparable to each other; both modes in a row are "
          "exact under it\n")
    print(f"  {'priors':14s} {'opening':>8s} {'entropy':>9s} {'expected':>9s} "
          f"{'solved':>8s} {'form':>11s}")
    for label, config in [
        ("none", PriorConfig(repeat=False, curation=False)),
        ("curation only", PriorConfig(repeat=False)),
        ("repeat only", PriorConfig(curation=False)),
        ("both", PriorConfig()),
    ]:
        means = {}
        for mode in (ENTROPY, EXPECTED):
            solver = WordleSolver(config=config, mode=mode)
            means[mode] = common.exact_stats(common.every_answer(solver),
                                             solver.weights)
            if mode == EXPECTED:
                opening, form = opening_of(solver), solver.form.name
        print(f"  {label:14s} {opening:>8s} {means[ENTROPY][0]:9.4f} "
              f"{means[EXPECTED][0]:9.4f} {(1 - means[EXPECTED][1]) * 100:7.2f}% "
              f"{form:>11s}")
