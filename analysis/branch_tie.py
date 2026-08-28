"""What the branch cap's tie tolerance costs, and whether 1e-11 sits in a band.

The cap keeps the best `max_branch` splits by their bound, and the bounds tie in
bulk -- one position holds 1,176 of its 1,177 at exactly 2.0 -- so which of them
survives is settled by the tie-break rather than by the bound.  Reading the bound
raw makes that a last-bit comparison, which two summation orders do not
reproduce; `exact.BRANCH_TIE_TOL` rounds it first, so bounds within a float's
noise of each other tie and the word index decides between them.  The bucket is
centred on a multiple of the tolerance rather than started there, because the
ties sit on whole values like 2.0 and an edge through one of those splits the
very cluster the rounding exists to gather -- put the edge there instead and
which outcome you get starts depending on the tolerance's decade.

Two questions, and the second is the one that matters.  How close do genuinely
distinct bounds get, which is the ceiling on any tolerance?  And over what range
of tolerances does play not move, which is where the choice is safe?  A tolerance
whose neighbours for decades either side play identically is a tolerance nothing
turns on; one that sits alone is a number someone tuned.

Read the guess *total* rather than the answers that moved.  Splits tied on the
bound are tied on what they cost, so choosing between them hands the same guesses
to different words -- the length distribution shifts, and every tolerance still
totals the same.  Which near-ties a given tolerance collapses depends on where
its grid lines fall against them, so the count of reassigned answers can wander while
nothing that matters does.

Everything runs in the Python, `WORDLE_NO_RUST=1` and all: the arms below are
counterfactual orderings that only `exact.py` has, so handing a sweep to the
native binary would measure the shipped rule five times over.  Backs README
*The exact endgame* and SOLVER_EXPLAINED SS12.
"""

import argparse
import os

os.environ["WORDLE_NO_RUST"] = "1"

import numpy as np

import common
import estimator
import exact
from priors import ALL_CONFIGS
from solver import EXPECTED, WordleSolver, sample_answers
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--tolerances", type=float, nargs="+",
                    default=[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
                             1e-6, 1e-5],
                    help="tolerances to floor the bound at (default a decade "
                         "either side of the shipped one)")
parser.add_argument("--probe-games", type=int, default=300,
                    help="games to read the gap distribution off (default "
                         "%(default)s)")
parser.add_argument("--priors", type=int, default=len(ALL_CONFIGS),
                    help="how many priors to sweep, after --repeat-only "
                         "(default all %(default)s)")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data(mask_last=1)
SHIPPED = exact.ExactEndgame._branch_order


def build(config):
    record = estimator.load_record(config)
    return WordleSolver(data=data, config=config, mode=EXPECTED,
                        params=np.asarray(record["params"]),
                        form=estimator.get_form(record["form"]))


def rounding(tol):
    """`_branch_order` as it stands, but at a tolerance of this caller's."""
    def order(self, bounds, cap):
        keys = np.floor(np.asarray(bounds, dtype=np.float64) / tol + 0.5)
        out = np.argsort(keys, kind="stable")
        return out if cap is None else out[:cap]
    return order


def raw(self, bounds, cap):
    """What the sort read before the bound was floored at all."""
    out = np.argsort(bounds, kind="stable")
    return out if cap is None else out[:cap]


def played(config, rule):
    exact.ExactEndgame._branch_order = rule
    try:
        solver = build(config)
        lengths = np.asarray(common.every_answer(solver))
        return lengths, solver.weights
    finally:
        exact.ExactEndgame._branch_order = SHIPPED


# ------------------------------------------- how close distinct bounds get
common.header("the gap between distinct bounds")
gaps = []


def recording(self, bounds, cap):
    ordered = np.sort(np.asarray(bounds, dtype=np.float64))
    steps = np.diff(ordered)
    nonzero = steps[steps > 0]
    if nonzero.size:
        gaps.append(nonzero.min())
    return SHIPPED(self, bounds, cap)


exact.ExactEndgame._branch_order = recording
probe = build(ALL_CONFIGS[3])
for answer in sample_answers(probe.weights, args.probe_games, seed=0):
    probe.play(int(answer), max_guesses=6, deadline=6)
exact.ExactEndgame._branch_order = SHIPPED

gaps = np.array(gaps)
print(f"  {gaps.size} nodes holding at least two distinct bounds, over "
      f"{args.probe_games} games")
for percentile in (0, 1, 25, 50):
    print(f"    {percentile:>3}th percentile smallest gap at a node: "
          f"{np.percentile(gaps, percentile):.3e}")
print("  a gap of one ulp is float scatter around a tie rather than a "
      "distinction, and the sweep below is what tells the two apart")


# ------------------------------------------ what rounding the bound changes
def shape(lengths):
    """The length distribution, as `3:2636` and so on."""
    counts = np.bincount(lengths.astype(np.int64))
    return " ".join(f"{n}:{c}" for n, c in enumerate(counts) if c)


moves = []
totals_agree = []
for config in common.filter_priors(ALL_CONFIGS, args)[:args.priors]:
    common.header(config.slug)
    unfloored, weights = played(config, raw)
    base_mean, base_loss = common.exact_stats(unfloored, weights)
    lost = int((unfloored > common.MAX_GUESSES).sum())
    print(f"  {'reading the bound raw':<22} {base_mean:.5f} guesses, "
          f"{int(unfloored.sum())} in all, {lost} lost ({base_loss * 100:.4f}%)")
    print(f"  {'':<22} {shape(unfloored)}")

    totals = set()
    for tol in args.tolerances:
        rule = SHIPPED if tol == exact.BRANCH_TIE_TOL else rounding(tol)
        lengths, _ = played(config, rule)
        mean, loss = common.exact_stats(lengths, weights)
        here = int((lengths > common.MAX_GUESSES).sum())
        totals.add(int(lengths.sum()))
        mark = "  <- shipped" if tol == exact.BRANCH_TIE_TOL else ""
        print(f"  rounded to {tol:.0e}{'':<9} {mean:.5f} guesses "
              f"({mean - base_mean:+.5f}), {int(lengths.sum())} in all, "
              f"{here} lost, {int((lengths != unfloored).sum()):>4} reassigned{mark}")
        if tol == exact.BRANCH_TIE_TOL:
            moves.append((config.slug, mean - base_mean, here - lost))
    totals_agree.append(len(totals) == 1)
    print(f"  every tolerance totals the same: {len(totals) == 1}"
          + ("" if len(totals) == 1 else f" -- {sorted(totals)}"))


common.header("over the priors")
deltas = np.array([m[1] for m in moves])
words = np.array([m[2] for m in moves])
print(f"  at the shipped {exact.BRANCH_TIE_TOL:.0e}, against reading the bound raw:")
print(f"    mean guesses  {deltas.mean():+.5f} on average, "
      f"{deltas.max():+.5f} at worst over {len(deltas)} priors")
print(f"    words lost    {words.sum():+d} in total, "
      f"{words.min():+d} to {words.max():+d} across them")
print(f"    better on {int((deltas < 0).sum())}, worse on {int((deltas > 0).sum())}, "
      f"level on {int((deltas == 0).sum())}")
print(f"    one guess total across every tolerance, on "
      f"{sum(totals_agree)} of {len(totals_agree)} priors")
print("  a sign that differs between priors is tie-break churn; one that agrees "
      "across all of them would be a cost, and is the thing to watch for here")
