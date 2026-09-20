"""What the endgame search is worth as a valuation, at the level of the choice.

Where `exact.ExactEndgame` reaches its value is the true one, so any
disagreement with `min_g E(g, S)` is the estimator being wrong.  Positions are
the ones the search takes over, three or more turns left and at most
`--exact-max` candidates; every guess is continued by the shipped policy with
no deadline.  Results are broken out by candidate count.  Backs README *What
it is worth* and SOLVER_EXPLAINED *A better map, and next to no better
score*; moves with the fit.
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
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data()
configs = common.filter_priors(
    [PriorConfig()] if args.shipped_only else
    [PriorConfig(repeat=r, curation=c)
     for r in (False, True) for c in (False, True)], args)

# where the search decides and the estimator would otherwise have had to
BANDS = ((3, 4), (5, 10), (11, args.exact_max))


def search_decides(solver, candidates, turns_left):
    return turns_left >= 3 and MIN_CANDIDATES <= len(candidates) <= solver.exact_max


def report(title, rows, weights, tag=None):
    """Print one prior's or the blend's report; `tag` says what to record it as."""
    rec = (lambda key, value: common.record(f"{tag}.{key}", value)) if tag \
        else (lambda key, value: None)
    common.header(title)
    rec("positions", len(rows["spread"]))
    rec("regret", float(common.weighted(rows["heur"], weights)))
    rec("search_regret", float(common.weighted(rows["exact"], weights)))
    rec("differ", float(100 * common.weighted(rows["moved"], weights)))
    rec("rho", float(common.weighted(rows["rho"], weights)))
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
    table = []
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
        differ = 100 * common.weighted(moved, band_w)
        tied = 100 * common.weighted(ties, band_w)
        table.append((f"{low}\u2013{high}", str(int(pick.sum())), f"{differ:.1f}%",
                      f"{common.weighted(heur, band_w):.4f}",
                      "all of them" if abs(tied - differ) < 0.05 else f"{tied:.1f}%"))
    rec("table", common.md_table(
        ["candidates", "positions", "estimator differs", "estimator regret",
         "of which ties"], table))
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
                        total, tag="who")
        common.record("who.games", args.games * len(configs))


def report_deciders(title, counts, tag=None):
    common.header(title)
    print(f"  {'turns left':>10} " + " ".join(f"{d:>13}" for d in DECIDERS)
          + f" {'total':>8}")
    table = []
    for left in range(MAX_TURNS, 0, -1):
        row = counts[left]
        n = sum(row.values())
        if not n:
            continue
        print(f"  {left:>10} " + " ".join(f"{row[d]:>13,}" for d in DECIDERS)
              + f" {n:>8,}")
        table.append((str(left), *(f"{row[d]:,}" for d in DECIDERS)))
    binding = sum(counts[t]["f"] for t in (1, 2, 3))
    reached = sum(sum(counts[t].values()) for t in (1, 2, 3))
    print(f"\n  with three or fewer turns left, f decides {binding:,} of "
          f"{reached:,} positions\n")
    if tag:
        common.record(f"{tag}.table", common.md_table(
            ["turns left", "`f`", "exact search", "closed form", "one candidate"],
            table))
        common.record(f"{tag}.f_late", binding)
        common.record(f"{tag}.late", reached)
        common.record(f"{tag}.f_at_four", counts[4]["f"])


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

    report(f"{config.slug}: the search against the estimator", rows, counts,
           tag="shipped" if config == PriorConfig() else None)
    for key, values in rows.items():
        pooled[key].extend(values)

if len(configs) > 1:
    report(f"blended over {len(configs)} priors", pooled, pooled_weights,
           tag="blended")
