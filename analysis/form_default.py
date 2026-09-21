"""Which form of f to fit when none is asked for.

Every archived fit is replayed over every possible answer, weighted by the
prior it was fitted for.  `--shapes` adds the three priors the shape
comparison needs; `--promote` writes each prior's winner as its active fit,
ranking the fit already in play with the archives and keeping it when nothing
beats it.  Backs README *Which form is the default*.
"""

import argparse
import json
import os
import time

import numpy as np

import common
import estimator
from priors import ALL_CONFIGS, PriorConfig
from solver import WordleSolver
from wordle_data import write_json


# the three priors `replay.py --shapes` compares against the shipped one
SHAPE_CONFIGS = [
    PriorConfig(repeat_shape="uniform", curation_shape="uniform"),
    PriorConfig(repeat_shape="uniform"),
    PriorConfig(curation_shape="uniform"),
]

parser = argparse.ArgumentParser(description=__doc__)
common.add_prior_filter(parser)
parser.add_argument("--shapes", action="store_true",
                    help="rank the three shape-comparison priors as well")
parser.add_argument("--promote", action="store_true",
                    help="write each prior's best form as its active fit")
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
terms_used = {}      # form -> the term counts its fits kept, across priors
winners = []         # the best archive per prior, for the record
main_gap = 0.0       # the most the main set gives up against every form

ACTIVE = "(active)"

configs = ALL_CONFIGS + (SHAPE_CONFIGS if args.shapes else [])
for config in common.filter_priors(configs, args):
    common.header(config.slug)
    runs, openings, records = {}, {}, {}
    for form in forms:
        records[form] = archived(config, form)
    # the fit already in play competes too, so a refit can only ever improve
    # on it, unless it was made under other prior probabilities and cannot play
    current = estimator.load_record(config)
    if current is not None and current.get("prior") == config.fingerprint():
        records[ACTIVE] = current
    for label, record in records.items():
        if record is None:
            continue
        form = record["form"]
        solver = WordleSolver(config=config, params=record["params"], form=form)
        runs[label] = common.exact_stats(common.every_answer(solver), solver.weights)
        openings[label] = record.get("opening_word", "?")

    best = min(runs, key=lambda f: runs[f][0])
    # a tie goes to the fit in play, which an identical archive is a copy of
    if ACTIVE in runs and runs[ACTIVE][0] <= runs[best][0] + 1e-12:
        best = ACTIVE
    for form in forms:
        if records[form] is not None:
            terms_used.setdefault(form, set()).add(records[form]["terms_used"])
    archives = [f for f in forms if f in runs]
    best_archive = min(archives, key=lambda f: runs[f][0])
    winners.append(best_archive)
    in_main = [f for f in archives if f in estimator.MAIN_FORMS]
    if in_main:
        main_gap = max(main_gap, min(runs[f][0] for f in in_main)
                       - runs[best_archive][0])
    print(f"  every word as the answer, weighted by this prior; gaps to {best}\n")
    print(f"  {'form':11s} {'mean':>7s} {'solved':>8s} {'gap':>8s} {'opening':>8s}")
    order = sorted(runs, key=lambda f: runs[f][0])
    for rank, label in enumerate(order, start=1):
        mean, loss = runs[label]
        gap = mean - runs[best][0]
        # the summary stays over the five priors the page can select, and
        # over the archives, since the active fit is one of them twice
        if config in ALL_CONFIGS and label != ACTIVE:
            results.setdefault(label, []).append((rank, gap))
        shown = f"{records[label]['form']} {ACTIVE}" if label == ACTIVE else label
        print(f"  {shown:20s} {mean:7.4f} {(1 - loss) * 100:7.2f}% {gap:+8.4f} "
              f"{openings[label]:>8s}")
    if args.promote:
        if best == ACTIVE:
            # the kept fit carries the date it was ranked best on this record
            record = dict(records[ACTIVE],
                          ranked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            write_json(estimator.cache_path(config), record)
            print(f"\n  kept the active {records[ACTIVE]['form']}; no archive beats it")
        else:
            record = dict(records[best],
                          fitted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            path = write_json(estimator.cache_path(config), record)
            print(f"\n  promoted {best} -> {path}")
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
table = []
for mean_gap, worst_gap, worst_rank, mean_rank, form in sorted(rows):
    print(f"  {form:11s} {mean_gap:+9.4f} {worst_gap:+10.4f} {worst_rank:11d} "
          f"{mean_rank:10.1f}")
    kept = sorted(terms_used.get(form, set()))
    terms = f"{kept[0]}\u2013{kept[-1]}" if len(kept) > 1 else str(kept[0])
    shown = f"**{form}**" if form == estimator.DEFAULT_FORM else form
    table.append((shown, terms, f"+{mean_gap:.4f}", f"+{worst_gap:.4f}",
                  str(worst_rank)))
    common.record(f"{form}.mean_gap", mean_gap)
    common.record(f"{form}.worst_gap", worst_gap)
    common.record(f"{form}.worst_rank", worst_rank)
common.record("aggregate_table", common.md_table(
    ["form", "terms", "mean gap to the prior's best", "worst gap",
     f"worst rank of {len(forms)}"], table))
print()
print(f"  current default: {estimator.DEFAULT_FORM}")

# the top group: every form no prior punishes by a hundredth, and how wide
# it is on both columns
top = [r for r in sorted(rows) if r[1] < 0.01]
common.record("top_group", len(top))
common.record("top_group_spread_mean", top[-1][0] - top[0][0])
common.record("top_group_spread_worst", max(r[1] for r in top) - min(r[1] for r in top))
common.record("straight_line_gap", min(r[1] for r in rows
                                         if r[4] in ("loglinear", "quadratic")))
counts = {f: winners.count(f) for f in dict.fromkeys(winners)}
common.record("winners", ", ".join(f"`{f}` \u00d7{n}" if n > 1 else f"`{f}`"
                                   for f, n in sorted(counts.items(),
                                                      key=lambda kv: -kv[1])))
common.record("main_forms", ", ".join(f"`{f}`" for f in estimator.MAIN_FORMS))
common.record("main_forms_gap", main_gap)
if args.shapes:
    shape_wins = winners[len(winners) - len(SHAPE_CONFIGS):]
    common.record("shape_winners", ", ".join(f"`{f}`" for f in shape_wins))
print(f"  winners: {counts}; the main set gives up at most {main_gap:.4f}")
