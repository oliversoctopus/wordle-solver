"""How much of the prior's setting reaches the board.

Two sweeps over the real regime puzzles with `f` held fixed: `--part shapes`
tempers each shape inside its group from uniform to what ships, and `--part
groups` moves the mass between groups by sweeping `repeat_prob` and
`noncurated_prob`.  The far ends read pessimistically, carrying a borrowed
`f`.  Backs README *The prior sits in a flat basin*.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np

import common
import priors
from priors import PriorConfig
from solver import WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--part", choices=("shapes", "groups", "both"),
                    default="both", help="which sweep to run")
args = parser.parse_args()
STARTED = time.time()

data = load_data(mask_last=0)
sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
START, END = common.regime_start(sol), len(words)

SHIPPED = PriorConfig(repeat_shape="binned", curation_shape="wordlebot")
UNIFORM = PriorConfig(repeat_shape="uniform", curation_shape="uniform")
# the shipped fit, then one fitted for a prior with neither shape sharpened
FITS = ("repeat1_curation1_r-binned_c-wordlebot", "repeat1_curation1")
LEVELS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.71, 0.90)
BASE_PROB = 0.10
TEMPERS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_fit(slug):
    path = os.path.join(common.ROOT, "estimator_cache", f"params_{slug}.json")
    with open(path, "r") as file:
        record = json.load(file)
    return record["params"], record["form"]


@contextlib.contextmanager
def tempered(t_rep, t_cur):
    """Raise each shape's relative weights to a power, for this block only.

    There is no `PriorConfig` field for this and there should not be -- it is a
    knob for asking whether the shipped sharpening is the right one, not a
    setting anything ships.  `build_weights` renormalizes inside each group, so
    powering the shape it is handed is the whole of the change.
    """
    rep, cur = priors._repeat_shape, priors._curation_shape
    priors._repeat_shape = lambda d, used: (
        None if rep(d, used) is None else rep(d, used) ** t_rep)
    priors._curation_shape = lambda d, mask, shape: cur(d, mask, shape) ** t_cur
    try:
        yield
    finally:
        priors._repeat_shape, priors._curation_shape = rep, cur


def regime(config, params, form):
    out = []
    for t in range(START, END):
        solver = WordleSolver(data=common.as_of(data, t), config=config,
                              params=params, form=form)
        played = solver.play(data.word2ind[words[t]])
        out.append(len(played[0] if isinstance(played, tuple) else played))
    return np.array(out)


seen, klass = set(words[:START]), []
for t in range(START, END):
    word = words[t]
    klass.append("repeat" if word in seen
                 else ("offlist" if word not in curated else "fresh"))
    seen.add(word)
klass = np.array(klass)
SIZES = {name: int((klass == name).sum())
         for name in ("repeat", "offlist", "fresh")}
HEAD = (f"  {'setting':>16s} {'mean':>8s} {'vs shipped':>17s} "
        + " ".join(f"{name + ' (' + str(SIZES[name]) + ')':>16s}"
                   for name in ("repeat", "offlist", "fresh")))


def row(label, run, base):
    """Print one setting's row; returns (whole, repeat, offlist, fresh) pairs."""
    mean, se, _ = common.paired(run, base)
    cells, out = "", [(mean, se)]
    for name in ("repeat", "offlist", "fresh"):
        mask = klass == name
        a, b, _ = common.paired(run[mask], base[mask])
        cells += f" {a:+9.3f}+-{b:.3f}"
        out.append((a, b))
    print(f"  {label:>16s} {run.mean():8.4f} {mean:+9.4f}+-{se:.4f}{cells}")
    return out


def pm(cell):
    return common.neg(f"{cell[0]:+.3f} \u00b1 {cell[1]:.3f}")


curation_span = []
if args.part in ("shapes", "both"):
    # t=0 has to rebuild the uniform shapes exactly, or the knob is not the knob
    with tempered(0.0, 0.0):
        flattened = priors.build_weights(data, SHIPPED)
    assert np.allclose(flattened, priors.build_weights(data, UNIFORM)), \
        "tempering to 0 does not reproduce the uniform shapes"

    grid = ([(t, 1.0) for t in TEMPERS[:-1]]
            + [(1.0, t) for t in TEMPERS[:-1]] + [(1.0, 1.0)])
    for slug in FITS:
        params, form = load_fit(slug)
        common.header(f"shapes tempered, f held at {slug} ({form})")
        runs = {}
        for point in grid:
            with tempered(*point):
                runs[point] = regime(SHIPPED, params, form)
        print(f"{HEAD}\n")
        for point in grid:
            got = row(f"t={point[0]:.2f}/{point[1]:.2f}", runs[point], runs[(1.0, 1.0)])
            if slug == FITS[0]:
                if point == (0.0, 1.0):
                    common.record("repeat_flattened", pm(got[0]))
                if point == (1.0, 0.0):
                    common.record("curation_flattened", pm(got[0]))
                if point[0] == 1.0 and 0.0 < point[1] < 1.0:
                    curation_span.append(abs(got[0][0]))
        print()
    common.record("curation_span", max(curation_span))

if args.part in ("groups", "both"):
    for name in ("repeat_prob", "noncurated_prob"):
        params, form = load_fit(FITS[0])
        common.header(f"{name}, the other held at {BASE_PROB}")
        runs = {p: regime(PriorConfig(repeat_shape="binned",
                                      curation_shape="wordlebot", **{name: p}),
                          params, form)
                for p in LEVELS}
        print(f"{HEAD}\n")
        got = {}
        for p in LEVELS:
            got[p] = row(f"{name.split('_')[0]}={p:.2f}", runs[p], runs[BASE_PROB])
        print()
        # the basin: every level whose whole-regime gap is inside a few
        # hundredths, and the four beyond it
        flat = [p for p in LEVELS if abs(got[p][0][0]) < 0.03]
        edge = [p for p in LEVELS if p > max(flat)]
        key = name.split("_")[0]
        common.record(f"{key}.flat_to", max(flat))
        common.record(f"{key}.flat_within", max(abs(got[p][0][0]) for p in flat))
        common.record(f"{key}.fresh_within", max(abs(got[p][3][0]) for p in flat))
        common.record(f"{key}.beyond", ", ".join(
            common.neg(f"{got[p][0][0]:+.3f}") for p in edge))
        common.record(f"{key}.repeat_lo", common.neg(f"{got[LEVELS[0]][1][0]:+.3f}"))
        common.record(f"{key}.repeat_hi", common.neg(f"{got[max(flat)][1][0]:+.3f}"))
        if key == "repeat":
            head = [f"`repeat_prob`", f"{LEVELS[0]:.2f}\u2013{max(flat):.2f}"] + \
                   [f"{p:.2f}" for p in edge]
            common.record("table", common.md_table(head, [
                (f"all {END - START}", f"within \u00b1{max(abs(got[p][0][0]) for p in flat):.2f}",
                 *(common.neg(f"{got[p][0][0]:+.3f}") for p in edge)),
                (f"the {int((klass == 'fresh').sum())} fresh",
                 f"within \u00b1{max(abs(got[p][3][0]) for p in flat):.2f}",
                 *(common.neg(f"{got[p][3][0]:+.3f}") for p in edge))]))

    rates = priors.regime_stats(data)["regime"]
    print(f"  the rates the regime really shows: repeat {rates['repeat']:.3f}, "
          f"off-list given not a repeat {rates['noncurated']:.3f}")
    common.record("puzzles", END - START)
    common.record("regime_repeat", float(rates["repeat"]))

print(f"\n  Every point plays the same {END - START} real answers, paired on each.  A")
print("  setting far from the one f was fitted for carries a borrowed f as")
print("  well as its own miscalibration, so the far ends read worse than a")
print("  refit would leave them, and the basin is at least this wide.")
print(f"\n  {time.time() - STARTED:.0f} s")
