"""Is the fit the native trainer returns as good as the one Python returns?

The two least-squares solves differ in their last bits and policy iteration
walks to different fixed points, so parity stops at the seed round and the
question is one for a measurement.  Both sides are fitted at `train`'s own
settings over several seeds, and each `f` is scored over every answer,
weighted by the prior.  Backs README *Fitting in Rust*.
"""

import argparse

import numpy as np

import common
import estimator
import train as training
from priors import ALL_CONFIGS
from solver import EXPECTED, WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds to pair over (default %(default)s)")
parser.add_argument("--games", type=int, default=training.DEFAULT_GAMES)
parser.add_argument("--reach", type=int, default=0,
                    help="--exact-max to fit at (default %(default)s, train's)")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data()


def fitted(config, form, seed, use_rust):
    params, meta = training.policy_iteration(
        config, data=data, form=form, n_games=args.games, seed=seed,
        solver_options={"exact_max": args.reach}, save=False, verbose=False,
        use_rust=use_rust)
    return np.asarray(params, dtype=np.float64), meta


def scored(config, params, form):
    solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                          params=params, form=form)
    return common.exact_stats(common.every_answer(solver), solver.weights)


deltas, losses, declined = [], [], 0
spread = {}      # slug -> how far the same form's fits range across seeds
for config in common.filter_priors(ALL_CONFIGS, args):
    common.header(config.slug)
    form = estimator.get_form(estimator.load_record(config)["form"])
    by_seed = []
    for seed in args.seeds:
        arms = {}
        for label, use_rust in (("python", False), ("rust", True)):
            params, meta = fitted(config, form, seed, use_rust)
            if use_rust and meta.get("fitted_by") != "rust":
                declined += 1
                arms = None
                break
            mean, loss = scored(config, params, form)
            arms[label] = (params, mean, loss, meta)
            print(f"  seed {seed} {label:<7} {meta['iterations']} rounds, "
                  f"opening {meta['opening_word']}, {mean:.5f} guesses, "
                  f"loss {loss * 100:.4f}%")
        if arms is None:
            print(f"  seed {seed}: the native trainer declined; not a pair")
            continue
        by_seed.append(arms["rust"][1])
        deltas.append(arms["rust"][1] - arms["python"][1])
        losses.append(arms["rust"][2] - arms["python"][2])
        identical = np.array_equal(arms["python"][0], arms["rust"][0])
        print(f"  seed {seed} rust - python {deltas[-1]:+.5f} guesses, "
              f"same parameters: {identical}")

    if len(by_seed) > 1:
        spread[config.slug] = max(by_seed) - min(by_seed)
        print(f"  the native fits range over {spread[config.slug]:.5f} guesses "
              f"across {len(by_seed)} seeds")

common.header("over every (prior, seed)")
if not deltas:
    print("  no pairs -- build the native trainer with ./build.sh")
    raise SystemExit(0)

deltas = np.array(deltas)
print(f"  {deltas.size} paired fits" +
      (f", {declined} declined" if declined else ""))
print(f"  rust - python: mean {deltas.mean():+.5f}, worst {deltas.max():+.5f}, "
      f"best {deltas.min():+.5f}")
print(f"  worse on {int((deltas > 1e-9).sum())}, "
      f"better on {int((deltas < -1e-9).sum())}, "
      f"level on {int((np.abs(deltas) <= 1e-9).sum())}")
if deltas.size > 1:
    half = 1.96 * deltas.std(ddof=1) / np.sqrt(deltas.size)
    print(f"  paired difference {deltas.mean():+.5f} +/- {half:.5f} (95%)")
print(f"  loss rate {np.mean(losses) * 100:+.4f} pp on average")
common.record("pairs", int(deltas.size))
common.record("worse", int((deltas > 1e-9).sum()))
common.record("better", int((deltas < -1e-9).sum()))
common.record("level", int((np.abs(deltas) <= 1e-9).sum()))
common.record("mean_delta", float(deltas.mean()))
if spread:
    common.record("seed_spread_max", max(spread.values()))
    common.record("seed_spread_min", min(spread.values()))
    common.record("seed_spread_widest", max(spread, key=spread.get))
    for slug, value in spread.items():
        common.record(f"seed_spread.{slug}", value)
    common.record("seeds", len(args.seeds))
print("  a difference with no sign is a different valid fit; one that is worse "
      "on most of the pairs would be a defect, and is what this looks for")
