"""How much of the prior's setting reaches the board.

Two sweeps over the real regime puzzles with `f` held fixed: `--part shapes`
tempers each shape inside its group from uniform to what ships, and `--part
groups` moves the mass between groups by sweeping `repeat_prob` and
`noncurated_prob`, scoring each level on the whole regime and on the last
`RECENT_PUZZLES`.  The far ends read pessimistically, carrying a borrowed
`f`.  Backs README *The prior sits in a flat basin* and the off-list default
in *The regime the defaults describe*.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np

import common
import priors
from priors import (DEFAULT_NONCURATED_PROB, DEFAULT_REPEAT_PROB, PriorConfig,
                    RECENT_PUZZLES)
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
BASE = {"repeat_prob": DEFAULT_REPEAT_PROB,
        "noncurated_prob": DEFAULT_NONCURATED_PROB}
# a level whose whole-regime gap is inside this is in the basin
FLAT = 0.05
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


def row(label, run, base, window=None):
    """Print one setting's row; returns (whole, repeat, offlist, fresh) pairs.

    `window` restricts the scoring to a boolean mask over the regime."""
    if window is not None:
        run, base, kinds = run[window], base[window], klass[window]
    else:
        kinds = klass
    mean, se, _ = common.paired(run, base)
    cells, out = "", [(mean, se)]
    for name in ("repeat", "offlist", "fresh"):
        mask = kinds == name
        if mask.sum() < 2:
            cells += f" {'--':>16s}"
            out.append((float("nan"), float("nan")))
            continue
        a, b, _ = common.paired(run[mask], base[mask])
        cells += f" {a:+9.3f}+-{b:.3f}"
        out.append((a, b))
    print(f"  {label:>16s} {run.mean():8.4f} {mean:+9.4f}+-{se:.4f}{cells}")
    return out


def pm(cell):
    return common.neg(f"{cell[0]:+.3f} \u00b1 {cell[1]:.3f}")


def series(points, cells):
    """One sweep's rows as plain lists, for the dev page to chart."""
    names = ("all", "repeat", "offlist", "fresh")
    out = {"points": list(points)}
    for i, name in enumerate(names):
        out[name] = [[None if np.isnan(v) else float(v) for v in cells[p][i]]
                     for p in points]
    return out


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
        runs, cells = {}, {}
        for point in grid:
            with tempered(*point):
                runs[point] = regime(SHIPPED, params, form)
        print(f"{HEAD}\n")
        for point in grid:
            got = row(f"t={point[0]:.2f}/{point[1]:.2f}", runs[point], runs[(1.0, 1.0)])
            cells[point] = got
            if slug == FITS[0]:
                if point == (0.0, 1.0):
                    common.record("repeat_flattened", pm(got[0]))
                if point == (1.0, 0.0):
                    common.record("curation_flattened", pm(got[0]))
                if point[0] == 1.0 and 0.0 < point[1] < 1.0:
                    curation_span.append(abs(got[0][0]))
        if slug == FITS[0]:
            rep = series(TEMPERS, {t: cells[(t, 1.0)] for t in TEMPERS})
            cur = series(TEMPERS, {t: cells[(1.0, t)] for t in TEMPERS})
            common.record("series.tempered.repeat", rep)
            common.record("series.tempered.curation", cur)
        print()
    common.record("curation_span", max(curation_span))

if args.part in ("groups", "both"):
    recent = np.arange(END - START) >= END - START - RECENT_PUZZLES
    for name in ("repeat_prob", "noncurated_prob"):
        params, form = load_fit(FITS[0])
        other = "noncurated_prob" if name == "repeat_prob" else "repeat_prob"
        common.header(f"{name}, {other} held at {BASE[other]}")
        runs = {p: regime(PriorConfig(repeat_shape="binned",
                                      curation_shape="wordlebot", **{name: p}),
                          params, form)
                for p in LEVELS}
        base = BASE[name]
        key = name.split("_")[0]
        print(f"{HEAD}\n")
        got = {}
        for p in LEVELS:
            got[p] = row(f"{key}={p:.2f}", runs[p], runs[base])
        print(f"\n  the same runs over the last {RECENT_PUZZLES} puzzles\n")
        late = {}
        for p in LEVELS:
            late[p] = row(f"{key}={p:.2f}", runs[p], runs[base], recent)
        print()
        common.record(f"series.{key}", {"base": base,
                                        "regime": series(LEVELS, got),
                                        "recent": series(LEVELS, late)})
        # the basin: every level whose whole-regime gap is inside FLAT, and
        # the levels beyond it
        flat = [p for p in LEVELS if abs(got[p][0][0]) < FLAT]
        edge = [p for p in LEVELS if p > max(flat)]
        common.record(f"{key}.flat_from", min(flat))
        common.record(f"{key}.flat_to", max(flat))
        common.record(f"{key}.flat_within", max(abs(got[p][0][0]) for p in flat))
        common.record(f"{key}.fresh_within", max(abs(got[p][3][0]) for p in flat))
        common.record(f"{key}.beyond", ", ".join(
            common.neg(f"{got[p][0][0]:+.3f}") for p in edge))
        common.record(f"{key}.repeat_lo", common.neg(f"{got[LEVELS[0]][1][0]:+.3f}"))
        common.record(f"{key}.repeat_hi", common.neg(f"{got[max(flat)][1][0]:+.3f}"))
        # the old off-list default against the shipped one, whole regime and
        # recent window, and the off-list arm at each
        for p, tag in ((0.10, "p10"), (0.30, "p30")):
            if p == base:
                continue
            common.record(f"{key}.{tag}.regime", pm(got[p][0]))
            common.record(f"{key}.{tag}.regime_offlist", pm(got[p][2]))
            common.record(f"{key}.{tag}.recent", pm(late[p][0]))
            common.record(f"{key}.{tag}.recent_offlist", pm(late[p][2]))
            common.record(f"{key}.{tag}.recent_fresh", pm(late[p][3]))
        if key == "repeat":
            head = ([f"`repeat_prob`", f"{LEVELS[0]:.2f}–{max(flat):.2f}"]
                    + [f"{p:.2f}" for p in edge])
            common.record("table", common.md_table(head, [
                (f"all {END - START}", f"within ±{max(abs(got[p][0][0]) for p in flat):.2f}",
                 *(common.neg(f"{got[p][0][0]:+.3f}") for p in edge)),
                (f"the {int((klass == 'fresh').sum())} fresh",
                 f"within ±{max(abs(got[p][3][0]) for p in flat):.2f}",
                 *(common.neg(f"{got[p][3][0]:+.3f}") for p in edge))]))
        else:
            shown = [p for p in LEVELS if p >= 0.05]
            head = ["`noncurated_prob`"] + [f"{p:.2f}" for p in shown]
            common.record("noncurated.table", common.md_table(head, [
                (f"all {END - START}",
                 *(common.neg(f"{got[p][0][0]:+.3f}") for p in shown)),
                (f"the last {RECENT_PUZZLES}",
                 *(common.neg(f"{late[p][0][0]:+.3f}") for p in shown)),
                (f"the {int(((klass == 'offlist') & recent).sum())} recent off-list",
                 *(common.neg(f"{late[p][2][0]:+.3f}") for p in shown))]))

    rates = priors.regime_stats(data)["regime"]
    print(f"  the rates the regime really shows: repeat {rates['repeat']:.3f}, "
          f"off-list given not a repeat {rates['noncurated']:.3f}")
    common.record("puzzles", END - START)
    common.record("recent_puzzles", RECENT_PUZZLES)
    common.record("regime_repeat", float(rates["repeat"]))
    common.record("recent_noncurated",
                  float(priors.regime_stats(data)["recent"]["noncurated"]))

print(f"\n  Every point plays the same {END - START} real answers, paired on each.  A")
print("  setting far from the one f was fitted for carries a borrowed f as")
print("  well as its own miscalibration, so the far ends read worse than a")
print("  refit would leave them, and the basin is at least this wide.")
print(f"\n  {time.time() - STARTED:.0f} s")
