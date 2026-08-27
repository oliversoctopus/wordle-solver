"""Why the fresh arm and the real fresh answers disagree.

The three-arm table's fresh column WAS a census of the 497 curated words still
unused; it is the regime's own 154 fresh answers now, and this script is why.
No word is in both sets -- one leaves the unused pile the moment the editor uses
it -- and sharpening the repeat shape cost nothing on the census against about
an eighth of a guess on the 154, which is what made the old blend in
`replay.py --shapes` disagree with the plain replay above it.

Two candidates, and the script separates them:

1. WHEN.  The census plays as if today; the 154 played on their own days.  So
   play the census words on the real answers' days too, same solver, and see
   whether the gap survives.
2. WHICH f.  Each arm of the table carries the f fitted for its own prior, and
   `train.policy_iteration` draws its training answers FROM that prior -- whose
   fresh-curated group is, at fitting time, exactly the 497.  Every f in the
   repo is therefore tuned on the census and has never seen a word the editor
   actually picked.  So run the same contrast with f forced to one value in both
   arms, at each of the two fits, and see how much of the gap is left.

The second is the one that carries it, which made the census a quiet case of the
thing SOLVER_EXPLAINED 14.1 opens by warning against: answers sampled from the
prior, smuggled back in through the estimator rather than through the arm.  It
was training on the test set, and that is why the arm was replaced.
Note that this locates the effect rather than proving the mechanism -- the
confirming run is an f fitted on a distribution that includes the editor's
picks, which is a policy iteration this does not do.

Backs the fresh-arm paragraph of SOLVER_EXPLAINED 14.1 and the caveat under
README *What the two shapes are worth together*.
"""

import argparse
import copy
import json
import os
import time

import numpy as np

import common
from priors import PriorConfig
from solver import WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--per-day", type=int, default=10,
                    help="census words played on each of the 154 days")
args = parser.parse_args()
STARTED = time.time()

data = load_data(mask_last=0)
sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
START, END = common.regime_start(sol), len(words)

seen, days = set(words[:START]), []
for t in range(START, END):
    word = words[t]
    if word not in seen and word in curated:
        days.append((t, word))
    seen.add(word)
census = sorted(curated - set(words))
print(f"{len(days)} real fresh answers, {len(census)} unused curated words, "
      f"{len(set(w for _, w in days) & set(census))} in both")

FLAT = PriorConfig(repeat_shape="uniform", curation_shape="uniform")
SHARP = PriorConfig(repeat_shape="binned", curation_shape="uniform")


def fit(slug):
    path = os.path.join(common.ROOT, "estimator_cache", f"params_{slug}.json")
    with open(path, "r") as file:
        record = json.load(file)
    return record["params"], record["form"]


FLAT_FIT, SHARP_FIT = fit("repeat1_curation1"), fit("repeat1_curation1_r-binned")
PAIRINGS = (("each prior its own f", FLAT_FIT, SHARP_FIT),
            ("both on the flat prior's f", FLAT_FIT, FLAT_FIT),
            ("both on the sharp prior's f", SHARP_FIT, SHARP_FIT))


def as_of(t):
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:t]
    return view


def length(solver, word):
    out = solver.play(data.word2ind[word])
    return len(out[0] if isinstance(out, tuple) else out)


common.header("the census played as if today, which is what the table used to do")
built = {cfg.slug: WordleSolver(data=data, config=cfg, params=p, form=f)
         for cfg, (p, f) in ((FLAT, FLAT_FIT), (SHARP, SHARP_FIT))}
at_end = np.array([length(built[SHARP.slug], w) - length(built[FLAT.slug], w)
                   for w in census])
print(f"  {len(census)} census words, each prior its own f   "
      f"{at_end.mean():+.4f} +- {at_end.std(ddof=1) / np.sqrt(len(at_end)):.4f}")

rng = np.random.default_rng(0)
got = {label: {"real": [], "census": []} for label, _, _ in PAIRINGS}
for i, (t, answer) in enumerate(days):
    view = as_of(t)
    picks = [census[j] for j in rng.choice(len(census), args.per_day,
                                           replace=False)]
    made = {}
    for label, flat_fit, sharp_fit in PAIRINGS:
        pair = []
        for cfg, (params, form) in ((FLAT, flat_fit), (SHARP, sharp_fit)):
            key = (cfg.slug, id(params))
            if key not in made:
                made[key] = WordleSolver(data=view, config=cfg,
                                         params=params, form=form)
            pair.append(made[key])
        flat, sharp = pair
        got[label]["real"].append(length(sharp, answer) - length(flat, answer))
        got[label]["census"].append(
            np.mean([length(sharp, w) - length(flat, w) for w in picks]))
    if i % 50 == 0:
        print(f"  day {i}/{len(days)}   [{time.time() - STARTED:.0f}s]")

common.header("the same days for both, so only the words differ")
print(f"  {'':>28s} {f'{len(days)} real':>18s} "
      f"{f'{len(days) * args.per_day} census':>18s} {'gap':>10s}\n")
for label, _, _ in PAIRINGS:
    cells, means = "", []
    for key in ("real", "census"):
        arr = np.array(got[label][key])
        means.append(arr.mean())
        cells += f" {arr.mean():+10.4f}+-{arr.std(ddof=1)/np.sqrt(len(arr)):.4f}"
    print(f"  {label:>28s}{cells} {means[0] - means[1]:+10.4f}")

print("\n  Sharpening the repeat shape, against the same prior with the shape")
print("  left flat.  Holding f equal is not a configuration anyone ships -- a")
print("  prior the estimator has not been fitted to is not one you would run --")
print("  so the first row is the honest figure and the other two say what is")
print("  behind it.")
print(f"\n  {time.time() - STARTED:.0f} s")
