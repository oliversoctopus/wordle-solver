"""How accurately the two valuations price a position, and whether it matters.

`min_g E(g, S)` and one-ply lookahead are scored against the truth computed by
`common.position_value`, each against its own play.  `--decisions` measures
the ordering within a position instead: the true value of the top `--top`
guesses, and what each policy's pick gives up.  Positions come from real games
where the heuristic decides, weighted by how many games reach them, and value
is deadline-free throughout.  Backs README *Lookahead* and SOLVER_EXPLAINED
*A better map, and next to no better score*; moves with the fit.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import (DEFAULT_LOOKAHEAD_BRANCH, EXPECTED, WordleSolver,
                    sample_answers)
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--positions", type=int, default=60,
                    help="positions scored per prior")
parser.add_argument("--games", type=int, default=400,
                    help="games drawn on to collect them from")
parser.add_argument("--branch", type=int, default=DEFAULT_LOOKAHEAD_BRANCH)
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--shipped-only", action="store_true")
parser.add_argument("--decisions", action="store_true",
                    help="rank the guesses within a position instead of scoring "
                         "the position")
parser.add_argument("--top", type=int, default=DEFAULT_LOOKAHEAD_BRANCH,
                    help="guesses played out per position under --decisions; at "
                         "the default these are exactly the ones the lookahead "
                         "re-ranks")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data()
configs = common.filter_priors(
    [PriorConfig()] if args.shipped_only else
    [PriorConfig(repeat=r, curation=c)
     for r in (False, True) for c in (False, True)], args)


def heuristic_decides(solver, candidates, turns_left):
    return not solver.solved_exactly(candidates, turns_left)


def errors(estimates, truths, weights):
    """Bias, mean absolute error and RMSE of one valuation, per game."""
    d = np.asarray(estimates, dtype=float) - np.asarray(truths, dtype=float)
    return (common.weighted(d, weights), common.weighted(np.abs(d), weights),
            np.sqrt(common.weighted(d ** 2, weights)))


def report_accuracy(title, arms, weights, tag=None):
    """``arms`` maps a label to its ``(estimates, truths)``."""
    common.header(title)
    count = len(next(iter(arms.values()))[0])
    print(f"  {count} positions, weighted by how often play reaches them\n")
    print(f"  {'valuation':<22} {'true value':>11} {'bias':>9} "
          f"{'mean abs err':>14} {'RMSE':>9}")
    rmses, table = {}, []
    for label, (estimates, truths) in arms.items():
        bias, mae, rmse = errors(estimates, truths, weights)
        rmses[label] = rmse
        print(f"  {label:<22} {common.weighted(truths, weights):>11.4f} "
              f"{bias:>+9.4f} {mae:>14.4f} {rmse:>9.4f}")
        shown = ("`min_g E(g, S)` (what the solver scores positions at)"
                 if label.startswith("min_g") else label)
        cell = f"{rmse:.4f}"
        table.append((shown, f"{common.weighted(truths, weights):.4f}",
                      f"+{bias:.4f}", f"{mae:.4f}",
                      f"**{cell}**" if label.startswith("one") else cell))
    base, look = rmses["min_g E(g, S)"], rmses["one-ply lookahead"]
    print(f"\n  lookahead is {base / look:.2f}x more accurate by RMSE\n")
    if tag:
        common.record(f"{tag}.positions", count)
        common.record(f"{tag}.table", common.md_table(
            ["valuation", "true value", "bias", "mean abs error", "RMSE"], table))
        common.record(f"{tag}.ratio", base / look)
        common.record(f"{tag}.base_rmse", base)
        common.record(f"{tag}.look_rmse", look)


COLUMNS = {}     # decisions per prior, gathered for one table across them


def report_decisions(title, rows, weights, k, column=None):
    common.header(title)
    if column:
        COLUMNS[column] = (
            common.weighted(rows["spread"], weights),
            common.weighted(rows["base"], weights),
            common.weighted(rows["look"], weights),
            100 * common.weighted(np.array(rows["base"]) <= 1e-12, weights),
            100 * common.weighted(rows["moved"], weights),
            common.weighted(rows["rho"], weights))
    print(f"  {len(rows['spread'])} positions, top {k} guesses played out each, "
          f"weighted by how often play reaches them\n")
    print(f"  spread of true value, best to worst   "
          f"{common.weighted(rows['spread'], weights):.4f}")
    print(f"  base regret, its pick vs the best     "
          f"{common.weighted(rows['base'], weights):.4f}")
    print(f"  lookahead regret                      "
          f"{common.weighted(rows['look'], weights):.4f}")
    print(f"  base already picks the best           "
          f"{100 * common.weighted(np.array(rows['base']) <= 1e-12, weights):.1f}% of games")
    print(f"  lookahead picks a different guess     "
          f"{100 * common.weighted(rows['moved'], weights):.1f}% of games")
    print(f"  rank correlation, base score vs truth "
          f"{common.weighted(rows['rho'], weights):+.3f}\n")


DECISION_KEYS = ("spread", "base", "look", "moved", "rho")
SHORT = {"repeat0_curation0": "none", "repeat0_curation1_c-wordlebot": "curation",
         "repeat1_curation0_r-binned": "repeat",
         "repeat1_curation1_r-binned_c-wordlebot": "default"}
if args.decisions:
    pooled = {key: [] for key in DECISION_KEYS}
else:
    pooled = {"min_g E(g, S)": ([], []), "one-ply lookahead": ([], [])}
pooled_weights = []

for config in configs:
    common.header(config.slug)
    base = WordleSolver(data=data, config=config, mode=EXPECTED)
    look = WordleSolver(data=data, config=config, mode=EXPECTED,
                        lookahead=1, lookahead_max_branch=args.branch)

    answers = sample_answers(base.weights, args.games, seed=args.seed)
    found, counts = common.collect_positions(base, answers, heuristic_decides)
    rng = np.random.default_rng(args.seed)
    if len(found) > args.positions:
        keep = sorted(rng.choice(len(found), size=args.positions, replace=False))
        found = [found[i] for i in keep]
        counts = counts[keep]
    sizes = [len(c) for _, c in found]
    print(f"  {len(found)} positions from {args.games} games, "
          f"{min(sizes)}-{max(sizes)} candidates, turns "
          f"{min(t for t, _ in found)}-{max(t for t, _ in found)}, "
          f"reached {int(counts.min())}-{int(counts.max())} times")
    # each prior weighs the same in the blended row
    pooled_weights.extend(counts / counts.sum())

    start = time.perf_counter()
    if args.decisions:
        rows = {key: [] for key in DECISION_KEYS}
        for _, candidates in found:
            words, scores = common.shortlist(base, candidates, args.top)
            # every guess is continued by the base policy, so what separates
            # these numbers is the choice made here and nothing downstream
            values = np.array([common.guess_value(base, candidates, g) for g in words])
            best = values.min()
            pick, _ = look.best_guess(candidates, None)
            look_value = (values[words.index(pick)] if pick in words
                          else common.guess_value(base, candidates, pick))
            rows["spread"].append(values.max() - best)
            rows["base"].append(values[0] - best)
            rows["look"].append(look_value - best)
            rows["moved"].append(float(pick != words[0]))
            rows["rho"].append(common.rank_correlation(scores, values)[0])
        print(f"  played out: {time.perf_counter() - start:.0f}s\n")
        report_decisions(f"{config.slug}: what is at stake in the choice",
                         rows, counts, args.top, column=SHORT[config.slug])
        for key, values in rows.items():
            pooled[key].extend(values)
    else:
        arms = {}
        for label, solver in (("min_g E(g, S)", base), ("one-ply lookahead", look)):
            # None leaves the turn count at the full horizon, matching the
            # deadline-free truth; neither valuation reaches the exact search here
            estimates = [solver.best_guess(c, None)[1] for _, c in found]
            truths = [common.position_value(solver, c) for _, c in found]
            arms[label] = (estimates, truths)
            pooled[label][0].extend(estimates)
            pooled[label][1].extend(truths)
        print(f"  played out: {time.perf_counter() - start:.0f}s\n")
        report_accuracy(f"{config.slug}: each valuation against its own play",
                        arms, counts,
                        tag="shipped" if config == PriorConfig() else None)

if len(configs) > 1:
    title = f"blended over {len(configs)} priors"
    if args.decisions:
        report_decisions(title, pooled, pooled_weights, args.top, column="blended")
        order = ["blended"] + [SHORT[c.slug] for c in configs]
        rows_out = [("spread of true value, best to worst", "{:.4f}"),
                    ("`min_g E(g, S)` regret", "{:.4f}"),
                    ("one-ply lookahead regret", "{:.4f}"),
                    ("the estimator picks the best", "{:.1f}%"),
                    ("lookahead picks a different guess", "{:.1f}%"),
                    ("rank correlation with the truth", "{:+.3f}")]
        common.record("decisions.table", common.md_table(
            [""] + order,
            [(name, *(fmt.format(COLUMNS[col][i]) for col in order))
             for i, (name, fmt) in enumerate(rows_out)]))
        common.record("decisions.positions", len(pooled["spread"]))
        for i, key in enumerate(("spread", "base_regret", "look_regret",
                                 "base_best", "moved", "rho")):
            common.record(f"decisions.{key}", float(COLUMNS["blended"][i]))
            common.record(f"decisions.shipped.{key}", float(COLUMNS["default"][i]))
    else:
        report_accuracy(title, pooled, pooled_weights, tag="blended")
