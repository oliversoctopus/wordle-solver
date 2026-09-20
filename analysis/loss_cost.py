"""What a lost game would have cost, in guesses.

Every measured result scores a loss as 7, the claim a penalty of 1 makes.
This plays every word under the six-turn deadline, takes the games
that lose, and carries each on from its turn-6 position with the deadline
lifted until the solver finds the word.  It prices the risk-neutral baseline
and never the choice of 5.  Backs the continuation figures in
SOLVER_EXPLAINED *Why the penalty is 5 and not 1*.
"""

import argparse

import numpy as np

import common
from priors import PriorConfig
from solver import EXPECTED, WordleSolver
from wordle_data import MAX_GUESSES, load_data

# the same four combinations `loss_penalty.py` blends its lines over, so the
# continuation cost lands on the population the crossovers are computed on
TOGGLE_CONFIGS = [PriorConfig(repeat=r, curation=c)
                  for r in (False, True) for c in (False, True)]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--penalties", type=float, nargs="+", default=[1, 5],
                    help="decision-time penalties to measure the losses of")
parser.add_argument("--shipped-only", action="store_true",
                    help="the default prior alone, rather than the blend")
args = parser.parse_args()

data = load_data()
configs = [PriorConfig()] if args.shipped_only else TOGGLE_CONFIGS
words = data.all_words


def losing_answers(solver):
    """The words this solver fails to find in six, and the prior mass on them."""
    lengths = common.every_answer(solver, MAX_GUESSES)
    lost = np.flatnonzero(lengths > MAX_GUESSES)
    return lost, float(solver.weights[lost].sum())


def continuation(solver, answer):
    """Guesses beyond the sixth this game needs once the deadline is lifted.

    Turns 1-6 are replayed exactly as the deadline had them, so the position
    carried on from is the one the game really lost in.
    """
    candidates = solver.all_candidates()
    for turn in range(1, MAX_GUESSES + 1):
        guess, _ = solver.best_guess(candidates, MAX_GUESSES - turn + 1)
        if guess == answer:
            raise AssertionError(f"{words[answer]} was not a loss after all")
        candidates = solver.filter(guess, solver.feedback(guess, answer),
                                   candidates)
    return common.playout(solver, candidates, answer)


common.header("what a loss costs once the deadline is lifted")

print(f"  {'penalty':>7s} {'prior':>32s} {'losses':>7s} {'loss prob':>10s} "
      f"{'extra':>7s} {'worst':>6s}")
print("  " + "-" * 76)

blended, worst = {}, {}
for penalty in args.penalties:
    rows = []
    for config in configs:
        solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                              loss_penalty=penalty)
        lost, mass = losing_answers(solver)
        extra = np.array([continuation(solver, int(a)) for a in lost],
                         dtype=float)
        w = solver.weights[lost]
        mean_extra = float(w @ extra / w.sum()) if len(lost) else float("nan")
        rows.append((mass, mean_extra))
        print(f"  {penalty:7.0f} {config.slug:>32s} {len(lost):7d} "
              f"{mass:9.4%} {mean_extra:7.2f} {extra.max():6.0f}")
        worst[penalty] = max(worst.get(penalty, 0), int(extra.max()) if len(lost) else 0)
    # each arm contributes its own losses, so the blend weights by loss mass
    mass = float(np.mean([m for m, _ in rows]))
    blended[penalty] = sum(m * e for m, e in rows) / sum(m for m, _ in rows)
    print(f"  {penalty:7.0f} {'blended over ' + str(len(configs)):>32s} "
          f"{'':7s} {mass:9.4%} {blended[penalty]:7.2f}")
    print()

print("  What a loss is really worth, in guesses:")
for penalty, extra in blended.items():
    print(f"    losses under penalty {penalty:g}: L = 6 + {extra:.2f} "
          f"= {6 + extra:.2f}")
    common.record(f"penalty{penalty:g}.extra", float(extra))
    common.record(f"penalty{penalty:g}.worst", worst[penalty])
print()
print("  Weighted over the losing words only -- what the games that actually")
print("  lose would have needed, not an average over all answers.")
