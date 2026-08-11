"""The two tables in README *Results*, replayed.

Part one scores every archived form under the shipped prior, on one set of
answers split into blocks, so the spread between forms can be read against the
block-to-block noise.  Part two plays each prior combination in both modes on
answers drawn from that combination's own prior.

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
from solver import ENTROPY, EXPECTED, WordleSolver, sample_answers

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--part", choices=("forms", "priors", "both"), default="both")
parser.add_argument("--forms-games", type=int, default=30000)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--prior-games", type=int, default=20000)
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()


def lengths_of(solver, answers):
    """Guesses taken per game, a failure counted as one over the limit."""
    out = []
    for answer in answers:
        guesses, _ = solver.play(int(answer), max_guesses=MAX_GUESSES,
                                 deadline=MAX_GUESSES)
        solved = guesses[-1] == int(answer)
        out.append(len(guesses) if solved else MAX_GUESSES + 1)
    return np.array(out, dtype=float)


def archived(config, form):
    path = estimator.form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def opening_of(solver):
    guess, _ = solver.best_guess(solver.all_candidates(), MAX_GUESSES)
    return solver.data.all_words[guess]


def by_block(lengths, blocks):
    return np.array([b.mean() for b in np.array_split(lengths, blocks)])


if args.part in ("forms", "both"):
    config = PriorConfig()
    common.header(f"forms under {config.describe()}")
    print(f"  {args.forms_games} answers in {args.blocks} blocks, seed "
          f"{args.seed}, identical for every row\n")

    runs, meta = {}, {}
    answers = None
    for form in sorted(estimator.FORMS):
        record = archived(config, form)
        if record is None:
            continue
        solver = WordleSolver(config=config, params=record["params"], form=form)
        if answers is None:
            answers = sample_answers(solver.weights, args.forms_games, seed=args.seed)
        runs[form] = lengths_of(solver, answers)
        meta[form] = (f"{record.get('terms_used')}/{len(record.get('terms', []))}",
                      record.get("opening_word", "?"))

    greedy = WordleSolver(config=config, mode=ENTROPY)
    runs["greedy entropy"] = lengths_of(greedy, answers)
    meta["greedy entropy"] = ("--", opening_of(greedy))

    best = min(runs, key=lambda f: runs[f].mean())
    print(f"  {'form':15s} {'terms':>6s} {'opening':>8s} {'mean':>8s} "
          f"{'vs best':>9s} {'block sd':>9s} {'solved':>8s}")
    for form in sorted(runs, key=lambda f: runs[f].mean()):
        gap = runs[form].mean() - runs[best].mean()
        paired_blocks = by_block(runs[form], args.blocks) - by_block(runs[best], args.blocks)
        sd = paired_blocks.std(ddof=1) if args.blocks > 1 else float("nan")
        print(f"  {form:15s} {meta[form][0]:>6s} {meta[form][1]:>8s} "
              f"{runs[form].mean():8.4f} {gap:+9.4f} {sd:9.4f} "
              f"{(runs[form] <= MAX_GUESSES).mean() * 100:7.2f}%")
    print(f"\n  best: {best}")

if args.part in ("priors", "both"):
    common.header("prior combinations, both modes")
    print(f"  {args.prior_games} answers per row, drawn from that row's own "
          f"prior, so the rows are\n  not comparable to each other; the two "
          f"modes within a row see the same answers\n")
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
            answers = sample_answers(solver.weights, args.prior_games, seed=args.seed)
            lengths = lengths_of(solver, answers)
            means[mode] = lengths
            if mode == EXPECTED:
                opening, form = opening_of(solver), solver.form.name
        solved = (means[EXPECTED] <= MAX_GUESSES).mean() * 100
        print(f"  {label:14s} {opening:>8s} {means[ENTROPY].mean():9.4f} "
              f"{means[EXPECTED].mean():9.4f} {solved:7.2f}% {form:>11s}")
