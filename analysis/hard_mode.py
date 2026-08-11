"""What hard mode costs, in guesses.

The same answers are played twice, once with the guess free and once restricted
to words that could still be the answer, so the difference is paired.  `f` is
the fit cached for free play in both arms -- the restriction is not part of the
slug, so there is no hard-mode fit to read.  Backs README *Hard mode*.
"""

import argparse

import numpy as np

import common
from priors import ALL_CONFIGS, PriorConfig
from solver import ENTROPY, EXPECTED, WordleSolver, sample_answers
from wordle_data import load_data

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=5000)
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--mode", choices=(EXPECTED, ENTROPY), default=EXPECTED)
parser.add_argument("--all-priors", action="store_true",
                    help="every combination, not just the shipped one")
args = parser.parse_args()


def lengths_of(solver, answers):
    """Guesses taken per game, a failure counted as one over the limit."""
    out = []
    for answer in answers:
        guesses, _ = solver.play(int(answer), max_guesses=MAX_GUESSES,
                                 deadline=MAX_GUESSES)
        solved = guesses[-1] == int(answer)
        out.append(len(guesses) if solved else MAX_GUESSES + 1)
    return np.array(out, dtype=float)


def opening(solver):
    return solver.data.all_words[
        solver.best_guess(solver.all_candidates(), MAX_GUESSES)[0]]


data = load_data()
configs = ALL_CONFIGS if args.all_priors else [PriorConfig()]

for config in configs:
    common.header(config.slug)
    arms = {}
    answers = sample_answers(
        WordleSolver(data=data, config=config, mode=args.mode).weights,
        args.games, seed=args.seed)
    for hard in (False, True):
        solver = WordleSolver(data=data, config=config, mode=args.mode, hard=hard)
        arms[hard] = (lengths_of(solver, answers), opening(solver))

    (free, free_open), (rest, rest_open) = arms[False], arms[True]
    delta, se, per_game = common.paired(rest, free)
    print(f"  {args.games} answers drawn from this prior, seed {args.seed}, "
          f"mode={args.mode}\n")
    print(f"  {'guesses':14s} {'mean':>7s} {'solved':>8s} {'opening':>8s}")
    for name, lengths, word in (("free", free, free_open),
                                ("hard mode", rest, rest_open)):
        print(f"  {name:14s} {lengths.mean():7.4f} "
              f"{(lengths <= MAX_GUESSES).mean() * 100:7.2f}% {word:>8s}")
    print(f"\n  paired cost of the restriction: {delta:+.4f} +- {se:.4f} guesses")
    print(f"  games changing length: {int((per_game != 0).sum())} "
          f"({int((per_game < 0).sum())} better, {int((per_game > 0).sum())} worse)")
    print(f"  failures: {int((free > MAX_GUESSES).sum())} free, "
          f"{int((rest > MAX_GUESSES).sum())} hard\n")
