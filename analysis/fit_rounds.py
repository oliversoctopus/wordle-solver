"""What policy iteration converges to, scored exactly rather than on a sample.

Every round of `train.policy_iteration` leaves a fitted `f` behind, and the loop
keeps the one whose policy measured best on the 2,000 training answers.  That
is an in-sample statistic chosen by argmin over correlated draws, so it invites
the suspicion that the kept fit is a lucky one.  Here every distinct `f` any
round produced is played against all 12,947 words, weighted by the prior, which
is computed rather than estimated and settles both questions at once: whether
the loop's own choice is the best it saw, and whether running to convergence
would have been better.

The same run prices fitting under the exact endgame search, since it fits each
prior twice -- at the `--exact-max 0` both front ends default to, and at the 60
`play` uses.  Backs README *What the loop converges to* and the search figures
in README *What it is worth*.
"""

import argparse
import time

import numpy as np

import common
import estimator
import train as training
from priors import ALL_CONFIGS
from solver import EXPECTED, WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=training.DEFAULT_GAMES,
                    help="training games per round (default %(default)s)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--reaches", type=int, nargs="+", default=[60, 0],
                    help="--exact-max settings to fit under (default 60 and 0)")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data(mask_last=1)


def exact_mean(config, params, form):
    """Prior-weighted guesses over every answer, under the shipped settings."""
    solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                          params=np.asarray(params, dtype=np.float64), form=form)
    return common.exact_stats(common.every_answer(solver), solver.weights)


kept_is_best = 0
worst_gap = (0.0, "", 0, 0.0, 0.0)
runs = 0
penalties = []
loops = {reach: 0.0 for reach in args.reaches}

for config in common.filter_priors(ALL_CONFIGS, args):
    common.header(config.slug)
    form = estimator.get_form(estimator.load_record(config)["form"])
    seen = {}
    picked = {}

    for reach in args.reaches:
        started = time.perf_counter()
        params, meta = training.policy_iteration(
            config, data=data, form=form, n_games=args.games, seed=args.seed,
            solver_options={"exact_max": reach}, save=False, verbose=False,
            # pinned to the Python: the native trainer returns a different valid
            # fit, and README quotes this script against the cached ones
            use_rust=False)
        loops[reach] += sum(round_["seconds"] for round_ in meta["history"])
        print(f"  --exact-max {reach:<3} {meta['iterations']} rounds in "
              f"{time.perf_counter() - started:5.1f} s, converged="
              f"{meta['converged']}, opening {meta['opening_word']}")
        for round_ in meta["history"]:
            seen.setdefault(tuple(round_["fitted_params"]), []).append(
                (reach, round_["label"]))
        picked[reach] = tuple(float(p) for p in params)
        seen.setdefault(picked[reach], []).append((reach, "kept"))

    scored = {}
    for params in seen:
        scored[params] = exact_mean(config, params, form)

    order = sorted(scored, key=lambda p: scored[p][0])
    print(f"\n  {'guesses':>9}{'loss':>9}   rounds that produced it")
    for params in order:
        mean, loss = scored[params]
        where = ", ".join(f"{reach}/{label}" for reach, label in seen[params])
        print(f"  {mean:9.5f}{loss * 100:8.4f}%   {where}")

    for reach in args.reaches:
        mine = [p for p in seen if any(r == reach for r, _ in seen[p])]
        best = min(scored[p][0] for p in mine)
        gap = scored[picked[reach]][0] - best
        runs += 1
        kept_is_best += gap < 1e-9
        print(f"  --exact-max {reach:<3} kept {scored[picked[reach]][0]:.5f}, "
              f"best of that run {best:.5f}, given up {gap:.5f}")
        if gap > worst_gap[0]:
            worst_gap = (gap, config.slug, reach, scored[picked[reach]][0], best)

    if len(args.reaches) == 2:
        wide, narrow = (max(args.reaches), min(args.reaches))
        penalties.append(scored[picked[narrow]][0] - scored[picked[wide]][0])

common.header("over the priors")
print(f"the kept fit is the exact best of its own run in {kept_is_best} of {runs} runs")
common.record("kept_is_best", kept_is_best)
common.record("runs", runs)
common.record("worst.gap", float(worst_gap[0]))
common.record("worst.slug", worst_gap[1])
common.record("worst.reach", worst_gap[2])
common.record("worst.kept", float(worst_gap[3]))
common.record("worst.best", float(worst_gap[4]))
if penalties:
    wide, narrow = (max(args.reaches), min(args.reaches))
    print(f"fitting at --exact-max {narrow} rather than {wide} costs "
          f"{np.mean(penalties):+.5f} guesses on average, {max(penalties):+.5f} at worst")
    print(f"and the fitting loop takes {loops[wide] / loops[narrow]:.1f}x as long "
          f"({loops[wide]:.1f} s against {loops[narrow]:.1f} s)")
    # what the search-fitted f is worth: negative where the narrow fit does better
    common.record("search_fit.mean", float(np.mean(penalties)))
    common.record("search_fit.worst", float(min(penalties)))
    common.record("search_fit.identical", int(sum(abs(p) < 1e-9 for p in penalties)))
    common.record("search_fit.priors", len(penalties))
    common.record("search_fit.loop_ratio", float(loops[wide] / loops[narrow]))
