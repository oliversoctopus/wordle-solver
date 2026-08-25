"""How much of f's error is the feedback pattern that produced the bucket.

`f` reads one scalar, so it cannot tell ten candidates in the `-atch` family
from ten well-separated ones.  The pattern is the one structural feature
already in hand where `f` is evaluated, so the question worth asking is whether
a per-pattern offset would have anything to read.  It would: the bias rises
with the green count and is large under the priors with no curation shape.

Two rival causes have to be separated, which is what the depth dummies are for.
A four-green bucket arrives late in a game, so a class effect and a depth
effect are confounded in the marginal table; entering both leaves each read
against the other, and `f`'s own basis is in the design throughout so neither
can stand in for the curve.

What this does *not* reproduce is the verdict.  Playing the correction needs a
kernel that carries the offsets, and that patch was not kept -- README
*Conditioning f on the pattern* has what it measured.  Backs the residual
figures in that section.
"""

import argparse

import numpy as np

import common
import estimator
import train as training
from priors import ALL_CONFIGS
from solver import EXPECTED, WordleSolver, sample_answers
from wordle_data import load_data, num_to_feedback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=2000,
                    help="rollout games per prior (default %(default)s)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

GREENS = np.array([num_to_feedback(p).count("G") for p in range(243)])
MIN_LEVEL = 30          # states a level needs before it gets a dummy
data = load_data(mask_last=1)


def rollout(solver, answers):
    """train.collect_rollouts, plus the pattern and depth of each state."""
    n_eff, counts, targets, greens, depths = [], [], [], [], []
    for answer in answers:
        answer = int(answer)
        candidates = solver.all_candidates()
        states, pattern = [], None
        for _ in range(12):
            states.append((solver.effective_size(candidates), len(candidates), pattern))
            guess, _ = solver.best_guess(candidates, None)
            if guess == answer:
                break
            pattern = solver.feedback(guess, answer)
            candidates = solver.filter(guess, pattern, candidates)
        for depth, (size, count, produced_by) in enumerate(states[1:], start=1):
            n_eff.append(size)
            counts.append(count)
            targets.append(len(states) - depth)
            greens.append(GREENS[produced_by])
            depths.append(depth)
    return [np.array(column) for column in
            (n_eff, counts, targets, greens, depths)]


def least_squares(design, y):
    """Coefficients, their standard errors, and the residual RMSE."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = max(1, len(y) - design.shape[1])
    cov = float(residual @ residual) / dof * np.linalg.pinv(design.T @ design)
    return coef, np.sqrt(np.diag(cov)), float(np.sqrt(np.mean(residual ** 2)))


def dummies(values, levels):
    return np.column_stack([(values == level).astype(float) for level in levels])


for config in ALL_CONFIGS:
    common.header(config.slug)
    params, form = estimator.load_active(config)
    solver = WordleSolver(data=data, config=config, mode=EXPECTED, params=params,
                          form=form, exact_max=0)
    answers = sample_answers(solver.weights, args.games, seed=args.seed)
    n_eff, counts, targets, greens, depths = rollout(solver, answers)

    keep = counts >= training.MIN_COUNT_FOR_FIT
    n_eff, targets = n_eff[keep], targets[keep]
    greens, depths = greens[keep], depths[keep]

    basis = estimator.design_matrix(n_eff, form)
    y = targets - form.base
    g_levels = [g for g in range(1, 5) if (greens == g).sum() >= MIN_LEVEL]
    d_levels = [d for d in range(2, 7) if (depths == d).sum() >= MIN_LEVEL]
    g_dummies, d_dummies = dummies(greens, g_levels), dummies(depths, d_levels)

    _, _, plain = least_squares(basis, y)
    alone, se_alone, with_class = least_squares(np.hstack([basis, g_dummies]), y)
    _, _, with_depth = least_squares(np.hstack([basis, d_dummies]), y)
    both, se_both, with_either = least_squares(
        np.hstack([basis, g_dummies, d_dummies]), y)

    print(f"  {form.name} over {keep.sum()} states where f is consulted")
    print(f"  fit RMSE {plain:.4f} plain, {with_class:.4f} with the classes, "
          f"{with_depth:.4f} with depth, {with_either:.4f} with both")
    print(f"\n  {'greens':>7}{'states':>8}{'alone':>20}{'depth controlled':>22}")
    k = basis.shape[1]
    for i, level in enumerate(g_levels):
        print(f"  {level:>7}{int((greens == level).sum()):>8}"
              f"{alone[k + i]:>11.4f} +-{se_alone[k + i]:.4f}"
              f"{both[k + i]:>14.4f} +-{se_both[k + i]:.4f}")
