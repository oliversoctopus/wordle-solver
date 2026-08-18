"""How accurately the two valuations price a position, and whether it matters.

`min_g E(g, S)` is what the solver scores a position at; one-ply lookahead is
what sits above it.  Neither is checkable from a game's outcome, so this computes
the truth directly, by the argument `common.position_value` rests on.

The default run scores each valuation against what its *own* play achieves.  A
valuation's job is to predict what it will get, and the two disagree about the
guess in a fair share of positions, so holding both to the base policy's line
would charge lookahead for the disagreement rather than for any error in its
estimate.

`--decisions` measures the quantity that actually drives play, which the accuracy
table does not: an ordering of the guesses *within* one position.  A valuation
wrong by a constant at a position ranks perfectly and costs nothing, so accuracy
in the level says little about accuracy in the choice.  It plays out the true
value of each of the top `--top` guesses and reports what separates them, and
what each policy's pick gives up against the best of them.
`exact_decisions.py` asks the same question of the endgame search.

Positions come from real games, and only those where the heuristic actually
decides are kept: where the exact search reaches, both valuations return the same
true value and there is nothing left to measure.  In practice that is turns two
and three -- by the fourth the candidate set is under `--exact-max`.

A position is scored once and weighted by how many of the sampled games reach it,
so the averages are per game rather than per distinct position.  Every
per-position value is exact; it is the weighting that is sampled, so these means
carry the sampling error of `--games` and the enumerated scripts' do not.  Priors
are normalized to weigh equally in the blended row.

Value is deadline-free throughout.  That is the convention `f` is fitted under
(`solver.WordleSolver.play` with no deadline) and the only one `min_g E(g, S)`
can be estimating, since it is handed no turn count at all.  It also means a
playout always ends in a win, so no loss convention or penalty enters.

Backs the accuracy and decision tables in README *Lookahead* and
SOLVER_EXPLAINED *The result worth framing*.  Both move with the fit --
a refit of `f` changes the base valuation
directly -- so read this against a run of its own vintage.
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
args = parser.parse_args()

data = load_data()
configs = ([PriorConfig()] if args.shipped_only else
           [PriorConfig(repeat=r, curation=c) for r in (False, True) for c in (False, True)])


def heuristic_decides(solver, candidates, turns_left):
    return not solver.solved_exactly(candidates, turns_left)


def errors(estimates, truths, weights):
    """Bias, mean absolute error and RMSE of one valuation, per game."""
    d = np.asarray(estimates, dtype=float) - np.asarray(truths, dtype=float)
    return (common.weighted(d, weights), common.weighted(np.abs(d), weights),
            np.sqrt(common.weighted(d ** 2, weights)))


def report_accuracy(title, arms, weights):
    """``arms`` maps a label to its ``(estimates, truths)``."""
    common.header(title)
    count = len(next(iter(arms.values()))[0])
    print(f"  {count} positions, weighted by how often play reaches them\n")
    print(f"  {'valuation':<22} {'true value':>11} {'bias':>9} "
          f"{'mean abs err':>14} {'RMSE':>9}")
    rmses = {}
    for label, (estimates, truths) in arms.items():
        bias, mae, rmse = errors(estimates, truths, weights)
        rmses[label] = rmse
        print(f"  {label:<22} {common.weighted(truths, weights):>11.4f} "
              f"{bias:>+9.4f} {mae:>14.4f} {rmse:>9.4f}")
    base, look = rmses["min_g E(g, S)"], rmses["one-ply lookahead"]
    print(f"\n  lookahead is {base / look:.2f}x more accurate by RMSE\n")


def report_decisions(title, rows, weights, k):
    common.header(title)
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
                         rows, counts, args.top)
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
                        arms, counts)

if len(configs) > 1:
    title = f"blended over {len(configs)} priors"
    if args.decisions:
        report_decisions(title, pooled, pooled_weights, args.top)
    else:
        report_accuracy(title, pooled, pooled_weights)
