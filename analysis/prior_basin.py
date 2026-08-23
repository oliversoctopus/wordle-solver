"""How much of the prior's setting actually reaches the board.

Two sweeps over the shipped prior, both replaying every real regime puzzle
with the prior rebuilt from strictly earlier puzzles each day:

1. `--part shapes` tempers each shape inside its own group, raising its relative
   weights to a power t and renormalizing.  t=0 is the uniform shape exactly and
   t=1 is what ships, so the sweep asks whether a partial sharpening plays
   better than a full one.
2. `--part groups` moves the mass *between* groups instead, sweeping
   `repeat_prob` and `noncurated_prob` from far below the regime's real rates to
   far above them.

`f` is held FIXED across every point, so a difference is the prior's doing and
not the fit's.  That is what makes the points comparable to each other, and it
is also why the far ends read pessimistically: a prior nothing was fitted for
carries a borrowed `f` as well as its own miscalibration.  The shapes sweep is
run under two different `f` for the same reason -- if the ranking held only
under the one the shipped prior was fitted for, it would be the fit talking.

Neither sweep finds anything to win, which is the point: the prior sits in a
basin far wider than the distance between its shipped setting and the truth.
Backs README *The prior sits in a flat basin*.
"""

import argparse
import contextlib
import copy
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


def as_of(t):
    """`data` with its answer history truncated to strictly before puzzle t."""
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:t]
    return view


def regime(config, params, form):
    out = []
    for t in range(START, END):
        solver = WordleSolver(data=as_of(t), config=config,
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
    mean, se, _ = common.paired(run, base)
    cells = ""
    for name in ("repeat", "offlist", "fresh"):
        mask = klass == name
        a, b, _ = common.paired(run[mask], base[mask])
        cells += f" {a:+9.3f}+-{b:.3f}"
    print(f"  {label:>16s} {run.mean():8.4f} {mean:+9.4f}+-{se:.4f}{cells}")


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
            row(f"t={point[0]:.2f}/{point[1]:.2f}", runs[point], runs[(1.0, 1.0)])
        print()

if args.part in ("groups", "both"):
    for name in ("repeat_prob", "noncurated_prob"):
        params, form = load_fit(FITS[0])
        common.header(f"{name}, the other held at {BASE_PROB}")
        runs = {p: regime(PriorConfig(repeat_shape="binned",
                                      curation_shape="wordlebot", **{name: p}),
                          params, form)
                for p in LEVELS}
        print(f"{HEAD}\n")
        for p in LEVELS:
            row(f"{name.split('_')[0]}={p:.2f}", runs[p], runs[BASE_PROB])
        print()

    rates = priors.regime_stats(data)["regime"]
    print(f"  the rates the regime really shows: repeat {rates['repeat']:.3f}, "
          f"off-list given not a repeat {rates['noncurated']:.3f}")

print(f"\n  Every point plays the same {END - START} real answers, paired on each.  A")
print("  setting far from the one f was fitted for carries a borrowed f as")
print("  well as its own miscalibration, so the far ends read worse than a")
print("  refit would leave them, and the basin is at least this wide.")
print(f"\n  {time.time() - STARTED:.0f} s")
