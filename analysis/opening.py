"""Is the opening `f` picks the best one available to the policy behind it?

The root is the widest position in the game and the only one every game shares,
so the guess taken there is worth checking rather than trusting `f` on.  The
check is exact: the policy is deterministic, so playing every word as the answer
behind a forced opening *computes* that opening's mean instead of estimating it.
Guesses are counted as they are everywhere else -- a deadline of six, a loss
worth seven -- so the winning row reproduces the mean in README *Results*.

What it cannot do is look outside the shortlist.  The ranking is `f`'s, so an
opening that `f` prices badly enough to rank below `--top` is invisible here.
The band table is the evidence on whether that matters: if the true mean climbs
steadily with `f`'s rank, then `f` discriminates coarsely even where it cannot
order finely, and a good opening far down the list is unlikely.  If the bands are
flat, the shortlist is not doing its job and only an exhaustive sweep would
settle it.

Unlike its siblings this defaults to the shipped prior alone, since every prior
is another `--top` full enumerations and the four together run for hours.

Backs README *The opening is not where the estimator is weak*, and the third case
in SOLVER_EXPLAINED section 16.  It moves with the fit, the ranking being `f`'s.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import EXPECTED, MAX_TURNS, WordleSolver
from wordle_data import load_data

# how the shortlist is cut up in the trend table, in ranks by f
BANDS = ((1, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 500))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--top", type=int, default=200,
                    help="openings played out, best first by f")
parser.add_argument("--all-priors", action="store_true",
                    help="all four combinations rather than the shipped one")
parser.add_argument("--every-word", action="store_true",
                    help="ignore --top and sweep the whole list; hours, not minutes")
args = parser.parse_args()

data = load_data()
configs = ([PriorConfig(repeat=r, curation=c) for r in (False, True) for c in (False, True)]
           if args.all_priors else [PriorConfig()])


def opening_value(solver, first, candidates):
    """Prior-weighted mean and loss probability if the game opens with ``first``.

    Exact rather than sampled: every word is played as the answer once, and the
    solver's own policy takes every guess after the first.
    """
    lengths = np.empty(len(solver.weights))
    for answer in range(len(lengths)):
        guess, live = first, candidates
        for turn in range(1, MAX_TURNS + 1):
            if turn > 1:
                guess, _ = solver.best_guess(live, MAX_TURNS - turn + 1)
            if guess == answer:
                lengths[answer] = turn
                break
            live = solver.filter(guess, solver.feedback(guess, answer), live)
        else:
            lengths[answer] = MAX_TURNS + 1
    lost = lengths > MAX_TURNS
    return float(solver.weights @ lengths), float(solver.weights[lost].sum())


def report(solver, words, scores, means, losses):
    best = int(np.argmin(means))
    common.header(f"the {len(words)} best openings by f, played out")
    print(f"  {'rank':>5} {'opening':>8} {'f score':>9} {'true mean':>11} "
          f"{'loss prob':>10}")
    shown = sorted(range(len(words)), key=lambda i: means[i])[:10]
    for i in sorted(set([0, best] + shown)):
        mark = "  <- f picks it" if i == 0 else ""
        print(f"  {i + 1:>5} {data.all_words[words[i]]:>8} {scores[i]:9.4f} "
              f"{means[i]:11.4f} {losses[i] * 100:9.4f}%{mark}")

    print(f"\n  f picks {data.all_words[words[0]]} at a true {means[0]:.4f}")
    print(f"  best of the {len(words)} is {data.all_words[words[best]]} "
          f"at {means[best]:.4f}")
    print(f"  regret of the opening: {means[0] - means[best]:+.4f} guesses")
    beat = int((np.asarray(means) < means[0] - 1e-12).sum())
    print(f"  openings beating f's pick: {beat} of {len(words)}")

    rho, p = common.rank_correlation(scores, means)
    print(f"\n  rank correlation, f vs truth: {rho:+.3f} (p={p:.4f})")
    print(f"  {'ranks by f':>12} {'openings':>9} {'mean true value':>17} "
          f"{'worst':>9} {'best in band':>20}")
    for low, high in BANDS:
        pick = [i for i in range(len(words)) if low <= i + 1 <= high]
        if not pick:
            continue
        band = np.array([means[i] for i in pick])
        top = pick[int(band.argmin())]
        best_in = f"{data.all_words[words[top]]} {band.min():.4f}"
        print(f"  {f'{low}-{high}':>12} {len(pick):>9} {band.mean():>17.4f} "
              f"{band.max():>9.4f} {best_in:>20}")


for config in configs:
    common.header(config.slug)
    solver = WordleSolver(data=data, config=config, mode=EXPECTED)
    candidates = solver.all_candidates()

    scores_all = solver.score_all(candidates, pool=None)
    take = len(scores_all) if args.every_word else args.top
    order = np.argsort(scores_all)[:take]
    words = [int(i) for i in order]
    scores = [float(scores_all[i]) for i in order]

    start = time.perf_counter()
    means, losses = [], []
    for n, first in enumerate(words, start=1):
        # two openings partition the list differently, so almost nothing carries
        # over between them; kept across all of them, the finished openings'
        # positions are dead weight competing for the cap with live ones
        solver.reset_memo()
        mean, loss = opening_value(solver, first, candidates)
        means.append(mean)
        losses.append(loss)
        if n % 25 == 0:
            leader = data.all_words[words[int(np.argmin(means))]]
            print(f"    {n}/{len(words)} openings, "
                  f"{time.perf_counter() - start:.0f}s, "
                  f"best so far {leader} {min(means):.4f}")
    print(f"  played out: {time.perf_counter() - start:.0f}s\n")

    report(solver, words, scores, means, losses)
