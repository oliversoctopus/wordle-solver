"""Why the fresh arm and the real fresh answers disagree.

The three-arm table's fresh column is the regime's own fresh answers, not a
census of the curated words still unused, and this script is why.  Two
candidates are separated: WHEN, by playing the census on the real answers'
days; and WHICH f, by forcing one `f` on both arms, since policy iteration
draws its training answers from the prior and its fresh group is the census.
The second carries it.  Backs the fresh-arm paragraph of SOLVER_EXPLAINED 14.1
and README *What the two shapes are worth together*.
"""

import argparse
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
common.record("census", len(census))
common.record("real", len(days))
common.record("census_today", common.neg(
    f"{at_end.mean():+.3f} \u00b1 {at_end.std(ddof=1) / np.sqrt(len(at_end)):.3f}"))

rng = np.random.default_rng(0)
got = {label: {"real": [], "census": []} for label, _, _ in PAIRINGS}
for i, (t, answer) in enumerate(days):
    view = common.as_of(data, t)
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
KEYS = {"each prior its own f": "own", "both on the flat prior's f": "flat",
        "both on the sharp prior's f": "sharp"}
for label, _, _ in PAIRINGS:
    cells, means = "", []
    for key in ("real", "census"):
        arr = np.array(got[label][key])
        means.append(arr.mean())
        cells += f" {arr.mean():+10.4f}+-{arr.std(ddof=1)/np.sqrt(len(arr)):.4f}"
        common.record(f"{KEYS[label]}.{key}", common.neg(
            f"{arr.mean():+.3f} \u00b1 {arr.std(ddof=1)/np.sqrt(len(arr)):.3f}"))
    print(f"  {label:>28s}{cells} {means[0] - means[1]:+10.4f}")
    common.record(f"{KEYS[label]}.gap", float(means[0] - means[1]))

print("\n  Sharpening the repeat shape, against the same prior with the shape")
print("  left flat.  Holding f equal is not a configuration anyone ships -- a")
print("  prior the estimator has not been fitted to is not one you would run --")
print("  so the first row is the honest figure and the other two say what is")
print("  behind it.")
print(f"\n  {time.time() - STARTED:.0f} s")
