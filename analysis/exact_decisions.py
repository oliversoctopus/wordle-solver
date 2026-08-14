"""What the endgame search is worth as a valuation, at the level of the choice.

`lookahead_accuracy.py --decisions` asks how much a better valuation changes the
guess, and what the change is worth, for the one-ply search.  This asks the same
of `exact.ExactEndgame`, which is the strongest case available: where it reaches,
its value is not a better estimate but the true one, so any disagreement with
`min_g E(g, S)` is the estimator being wrong rather than two approximations
differing.

Positions are the ones the search actually takes over -- three or more turns left
and at most `--exact-max` candidates.  Below three turns both arms are exact and
there is nothing to compare; above the cap the search never runs.

Every guess is continued by the shipped policy, so what separates two of them is
the decision at this position and nothing downstream.  A loss cannot arise: the
continuation is deadline-free, so a playout always ends in a win.

Results are broken out by candidate count because that is where the old claim
sat -- that the smallest positions disagree often and at no cost, being genuine
ties.

Backs README *The exact endgame: what it is worth* and SOLVER_EXPLAINED section
12.  Moves with the fit, since the estimator arm is `f`.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import (DEFAULT_EXACT_MAX, DEFAULT_LOSS_PENALTY, EXPECTED, MAX_TURNS,
                    WordleSolver, sample_answers)
from wordle_data import load_data

# the closed forms at one and two candidates are the same either way
MIN_CANDIDATES = 3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--positions", type=int, default=150,
                    help="positions scored per prior")
parser.add_argument("--games", type=int, default=400,
                    help="games drawn on to collect them from")
parser.add_argument("--top", type=int, default=10,
                    help="guesses played out per position, best first by f")
parser.add_argument("--exact-max", type=int, default=DEFAULT_EXACT_MAX)
parser.add_argument("--loss-penalty", type=float, default=DEFAULT_LOSS_PENALTY,
                    help="at the shipped 5 the search optimizes a penalized "
                         "score rather than mean guesses, which is the whole of "
                         "its residual regret here; 1 is risk neutral")
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--shipped-only", action="store_true")
parser.add_argument("--who-decides", action="store_true",
                    help="count which valuation takes each decision, by turns "
                         "left, instead of scoring the choices")
args = parser.parse_args()

data = load_data()
configs = ([PriorConfig()] if args.shipped_only else
           [PriorConfig(repeat=r, curation=c) for r in (False, True) for c in (False, True)])

# where the search decides and the estimator would otherwise have had to
BANDS = ((3, 4), (5, 10), (11, args.exact_max))


def search_decides(solver, candidates, turns_left):
    return turns_left >= 3 and MIN_CANDIDATES <= len(candidates) <= solver.exact_max


def report(title, rows, weights):
    common.header(title)
    print(f"  {len(rows['spread'])} positions, top {args.top} guesses played out "
          f"each, weighted by how often play reaches them\n")
    print(f"  spread of true value, best to worst   "
          f"{common.weighted(rows['spread'], weights):.4f}")
    print(f"  estimator regret, its pick vs the best "
          f"{common.weighted(rows['heur'], weights):.4f}")
    print(f"  exact search regret                   "
          f"{common.weighted(rows['exact'], weights):.4f}")
    print(f"  the two differ on the guess           "
          f"{100 * common.weighted(rows['moved'], weights):.1f}% of games")
    print(f"  rank correlation, f vs truth          "
          f"{common.weighted(rows['rho'], weights):+.3f}\n")

    sizes = np.array(rows["size"])
    print(f"  {'candidates':>12} {'positions':>10} {'differ':>9} "
          f"{'estimator regret':>18} {'of which ties':>14}")
    for low, high in BANDS:
        pick = (sizes >= low) & (sizes <= high)
        if not pick.any():
            continue
        band_w = np.asarray(weights, dtype=float)[pick]
        moved = np.array(rows["moved"])[pick]
        heur = np.array(rows["heur"])[pick]
        # a disagreement costing nothing is the estimator finding an equal guess
        ties = ((moved > 0) & (heur <= 1e-12)).astype(float)
        print(f"  {f'{low}-{high}':>12} {int(pick.sum()):>10} "
              f"{100 * common.weighted(moved, band_w):>8.1f}% "
              f"{common.weighted(heur, band_w):>18.4f} "
              f"{100 * common.weighted(ties, band_w):>13.1f}%")
    print()


DECIDERS = ("f", "exact search", "closed form", "one candidate")


def decider(solver, candidates, turns_left):
    """Which of the four actually produces the guess at this position."""
    m = len(candidates)
    if m == 1:
        return "one candidate"
    if m == 2:
        return "closed form"
    return "exact search" if solver.solved_exactly(candidates, turns_left) else "f"


def who_decides():
    """Where each valuation is consulted, by turns left.

    Conditioning `f` on turns remaining can only change play where `f` is the one
    deciding *and* the deadline is close enough to bind.  Rather than fit such a
    curve to find out, this counts the positions: every decision in every game,
    not deduplicated, since what matters is how often a row is read.
    """
    total = {t: dict.fromkeys(DECIDERS, 0) for t in range(1, MAX_TURNS + 1)}
    for config in configs:
        solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                              exact_max=args.exact_max,
                              loss_penalty=args.loss_penalty)
        answers = sample_answers(solver.weights, args.games, seed=args.seed)
        counts = {t: dict.fromkeys(DECIDERS, 0) for t in range(1, MAX_TURNS + 1)}
        for answer in answers:
            answer = int(answer)
            candidates = solver.all_candidates()
            for turn in range(1, MAX_TURNS + 1):
                left = MAX_TURNS - turn + 1
                who = decider(solver, candidates, left)
                counts[left][who] += 1
                total[left][who] += 1
                guess, _ = solver.best_guess(candidates, left)
                if guess == answer:
                    break
                candidates = solver.filter(guess, solver.feedback(guess, answer),
                                           candidates)
        report_deciders(f"{config.slug}: who decides, over {args.games} games",
                        counts)
    if len(configs) > 1:
        report_deciders(f"all {len(configs)} priors, {args.games} games each",
                        total)


def report_deciders(title, counts):
    common.header(title)
    print(f"  {'turns left':>10} " + " ".join(f"{d:>13}" for d in DECIDERS)
          + f" {'total':>8}")
    for left in range(MAX_TURNS, 0, -1):
        row = counts[left]
        n = sum(row.values())
        if not n:
            continue
        print(f"  {left:>10} " + " ".join(f"{row[d]:>13,}" for d in DECIDERS)
              + f" {n:>8,}")
    binding = sum(counts[t]["f"] for t in (1, 2, 3))
    reached = sum(sum(counts[t].values()) for t in (1, 2, 3))
    print(f"\n  with three or fewer turns left, f decides {binding:,} of "
          f"{reached:,} positions\n")


KEYS = ("spread", "heur", "exact", "moved", "rho", "size")

if args.who_decides:
    who_decides()
    raise SystemExit

pooled = {key: [] for key in KEYS}
pooled_weights = []

for config in configs:
    common.header(config.slug)
    solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                          exact_max=args.exact_max,
                          loss_penalty=args.loss_penalty)

    answers = sample_answers(solver.weights, args.games, seed=args.seed)
    found, counts = common.collect_positions(solver, answers, search_decides)
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
    pooled_weights.extend(counts / counts.sum())

    start = time.perf_counter()
    rows = {key: [] for key in KEYS}
    for turn, candidates in found:
        turns_left = MAX_TURNS - turn + 1
        words, scores = common.shortlist(solver, candidates, args.top)
        values = [common.guess_value(solver, candidates, g) for g in words]
        # the search's own answer at this position, which is the true optimum
        # over every guess rather than over the shortlist
        pick, _ = solver.best_guess(candidates, turns_left)
        exact_value = (values[words.index(pick)] if pick in words
                       else common.guess_value(solver, candidates, pick))
        best = min(min(values), exact_value)
        rows["spread"].append(max(values) - min(values))
        rows["heur"].append(values[0] - best)
        rows["exact"].append(exact_value - best)
        rows["moved"].append(float(pick != words[0]))
        rows["rho"].append(common.rank_correlation(scores, values)[0])
        rows["size"].append(len(candidates))
    print(f"  played out: {time.perf_counter() - start:.0f}s\n")

    report(f"{config.slug}: the search against the estimator", rows, counts)
    for key, values in rows.items():
        pooled[key].extend(values)

if len(configs) > 1:
    report(f"blended over {len(configs)} priors", pooled, pooled_weights)
