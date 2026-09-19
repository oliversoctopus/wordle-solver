"""What a stale answer record costs, in guesses.

The same regime replay with the record truncated to `t - lag` for a range of
lags (`--lags`), paired on the same answers.  Nothing refits `f`: the cache is
keyed on the prior's slug, so a fresh record buys fresher weights only, and
only under `PriorConfig.repeat`.  A total-variation column beside the guesses
says how far the weights moved, against the flattened repeat shape as a
yardstick.  Backs README *How fresh the answer record has to be* and
SOLVER_EXPLAINED 14.1.
"""

import argparse
import copy
import time

import numpy as np

import common
from priors import PriorConfig, build_weights
from solver import WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lags", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 14, 30],
                    help="days of staleness to replay; the first is the "
                         "baseline every other one is paired against")
args = parser.parse_args()
STARTED = time.time()

LAGS = sorted(set(args.lags))
BASE = LAGS[0]

data = load_data(mask_last=0)
sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
START, END = common.regime_start(sol), len(words)

SHIPPED = PriorConfig(repeat_shape="binned", curation_shape="wordlebot")
# the same prior with the repeat shape left flat, which is one of the axes
# `prior_basin.py` sweeps -- here only to give the lags a yardstick in mass
FLAT_REPEAT = PriorConfig(repeat_shape="uniform", curation_shape="wordlebot")


def as_of(t):
    """`data` with its answer history truncated to strictly before puzzle t."""
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:max(t, 0)]
    return view


def total_variation(a, b):
    """Half the L1 distance, i.e. the share of the mass that sits elsewhere."""
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def guesses(view, answer):
    solver = WordleSolver(data=view, config=SHIPPED)
    out = solver.play(data.word2ind[answer])
    return len(out[0] if isinstance(out, tuple) else out)


common.header(f"a stale record over the {END - START} regime puzzles")
print(f"prior   {SHIPPED.slug}")
print(f"lags    {', '.join(str(lag) for lag in LAGS)} days\n")

runs = {lag: [] for lag in LAGS}
shift = {lag: [] for lag in LAGS}
repeat, offlist = [], []
seen = set(words[:START])
for t in range(START, END):
    word = words[t]
    fresh = build_weights(as_of(t - BASE), SHIPPED)
    for lag in LAGS:
        view = as_of(t - lag)
        runs[lag].append(guesses(view, word))
        shift[lag].append(total_variation(build_weights(view, SHIPPED), fresh))
    repeat.append(word in seen)
    # off-list *given not a repeat*, the order the model draws in
    offlist.append(word not in seen and word not in curated)
    seen.add(word)
    if (t - START) % 50 == 0:
        print(f"  puzzle {t - START}/{END - START}   "
              f"[{time.time() - STARTED:.0f}s]")

runs = {lag: np.array(run) for lag, run in runs.items()}
repeat, offlist = np.array(repeat), np.array(offlist)
rest = ~(repeat | offlist)
base = runs[BASE]

yard = total_variation(build_weights(data, FLAT_REPEAT),
                       build_weights(data, SHIPPED))
print(f"\nlag {BASE}, the record rebuilt every day   "
      f"{base.mean():.4f} guesses")
print("\nagainst it, paired on the same answers (negative = the stale one is "
      "better)\n")
print(f"  {'lag':>4s} {'mass moved':>11s} {'mean':>8s} {'paired':>19s}"
      f" {'changed':>8s} {'worse':>6s} {'better':>7s}")
table = []
for lag in LAGS[1:]:
    mean, se, d = common.paired(runs[lag], base)
    print(f"  {lag:4d} {np.mean(shift[lag]) * 100:10.2f}%"
          f" {runs[lag].mean():8.4f}"
          f" {mean:+9.4f} +- {se:.4f} {int((d != 0).sum()):8d}"
          f" {int((d > 0).sum()):6d} {int((d < 0).sum()):7d}")
    worse, better = int((d > 0).sum()), int((d < 0).sum())
    if lag in (1, 3, 7, 14, 30):
        split = f" ({worse} worse, {better} better)" if worse + better else ""
        table.append((f"{lag} day" + ("s" if lag > 1 else ""),
                      f"{np.mean(shift[lag]) * 100:.2f}%",
                      common.neg(f"{mean:+.4f} \u00b1 {se:.4f}"),
                      f"{worse + better}{split}"))
    common.record(f"lag{lag}.mass", float(np.mean(shift[lag]) * 100))
    common.record(f"lag{lag}.changed", worse + better)
    for name, mask in (("repeats", repeat), ("off-list", offlist),
                       ("other", rest)):
        if mask.any():
            m, s, _ = common.paired(runs[lag][mask], base[mask])
            print(f"  {'':4s} {'':11s} {'on the':>8s} {int(mask.sum()):3d} "
                  f"{name:9s} {m:+.4f} +- {s:.4f}")
    changed = [words[START + i] for i in range(len(base)) if d[i] != 0]
    print(f"  {'':4s} {'':11s} {', '.join(changed) if changed else 'none'}")

common.record("table", common.md_table(
    ["lag", "mass moved", "vs. a record rebuilt daily",
     f"games changed of {END - START}"], table))
common.record("puzzles", END - START)
common.record("yardstick", float(yard * 100))
print(f"\n  for scale, the repeat shape flattened moves {yard * 100:.2f}% of "
      f"the mass,")
print(f"  which `prior_basin.py` prices at "
      f"{common.result('prior_basin', 'repeat_flattened', 'a figure it has not recorded yet')} guesses.")

print("\nrefitting every k days, the lag then uniform over 0..k-1\n")
for k in range(2, max(LAGS[-1], 2) + 1):
    have = [lag for lag in range(k) if lag in runs]
    if len(have) < k:
        continue
    mean, se, _ = common.paired(np.stack([runs[lag] for lag in have]).mean(0),
                                base)
    print(f"  every {k:2d} days   {mean:+.5f} +- {se:.5f}")
    common.record(f"every{k}", common.neg(f"{mean:+.4f} \u00b1 {se:.4f}"))

print(f"\n[{time.time() - STARTED:.0f}s]")
