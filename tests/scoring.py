"""What each scoring rule claims to be, checked against what it is.

Information gain, expected guesses under `f`, and information gain with a
price on being a possible answer share one code path.  The price is
recomputed from the weights with numpy; the rules are compared to a solver
with the bonus switched off; and the ranking's first entry is checked against
`best_guess`, the other route to the same decision.  Loads the table, so it
is one of the slow ones.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import DERIVED_WIN_BONUS, ENTROPY, EXPECTED, MODES, WordleSolver
from wordle_data import load_data

failures = 0


def check(what, ok, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"{'ok  ' if ok else 'FAIL'}  {what}")
    if not ok and detail:
        print(f"        {detail}")


def positions(solver, answer, deadline=6):
    """Every position a game against ``answer`` passes through."""
    candidates = solver.all_candidates()
    for turn in range(1, deadline + 1):
        if len(candidates) <= 1:
            return
        yield candidates, deadline - turn + 1
        guess, _ = solver.best_guess(candidates, deadline - turn + 1)
        if guess == answer:
            return
        candidates = solver.filter(guess, solver.feedback(guess, answer),
                                   candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=3,
                        help="answers to play out, per mode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_data()
    rng = np.random.default_rng(args.seed)
    answers = [data.word2ind[word] for word in
               rng.choice(data.curated_list, size=args.games, replace=False)]

    plain = WordleSolver(data=data, mode=ENTROPY)
    priced = WordleSolver(data=data, mode=ENTROPY, win_bonus=DERIVED_WIN_BONUS)

    print(f"the price on a possible answer, over {args.games} games' positions")
    for answer in answers:
        for candidates, _ in positions(plain, answer):
            where = f"{len(candidates)} left"
            weights = plain.weights[candidates]
            # what the bonus is defined to be, from the weights alone
            paid = DERIVED_WIN_BONUS * weights / weights.sum()
            moved = priced.score_all(candidates) - plain.score_all(candidates)

            check(f"{where}: every candidate is paid its own win chance",
                  np.allclose(moved[candidates], paid))
            others = np.setdiff1d(np.arange(len(moved)), candidates)
            check(f"{where}: nothing that cannot win is paid at all",
                  not np.any(moved[others]))
            # a pooled score is the same score, whichever words are asked for
            pool = np.concatenate((candidates[:3], others[:3]))
            check(f"{where}: a pooled score matches the whole pass",
                  np.array_equal(priced.score_all(candidates, pool=pool),
                                 priced.score_all(candidates)[pool]))
    print()

    print("switching the price off leaves the mode exactly as it was")
    off = WordleSolver(data=data, mode=ENTROPY, win_bonus=0.0)
    for answer in answers:
        for candidates, _ in positions(plain, answer):
            check(f"{len(candidates)} left: scores are bit-identical",
                  np.array_equal(off.score_all(candidates),
                                 plain.score_all(candidates)))
    print()

    print("the ranking opens on the guess the solver would play")
    for mode in MODES:
        for win_bonus in (0.0, DERIVED_WIN_BONUS):
            if win_bonus and mode != ENTROPY:
                continue
            solver = WordleSolver(data=data, mode=mode, win_bonus=win_bonus)
            arm = f"{mode}{' + price' if win_bonus else ''}"
            for answer in answers:
                for candidates, guesses_left in positions(solver, answer):
                    best, value = solver.best_guess(candidates, guesses_left)
                    top = solver.top_guesses(candidates, k=5,
                                             guesses_left=guesses_left)
                    check(f"{arm}, {len(candidates)} left, {guesses_left} to "
                          f"play: the ranking's first is best_guess's pick",
                          top[0][0] == data.all_words[best],
                          f"{top[0][0]} against {data.all_words[best]}")
                    check(f"{arm}, {len(candidates)} left, {guesses_left} to "
                          f"play: carrying the same value",
                          abs(top[0][1] - value) < 1e-9,
                          f"{top[0][1]} against {value}")
    print()

    print("the price is priced in bits, so the other objective refuses it")
    try:
        WordleSolver(data=data, mode=EXPECTED, win_bonus=DERIVED_WIN_BONUS)
        check("expected mode rejects a win_bonus", False, "it was accepted")
    except ValueError:
        check("expected mode rejects a win_bonus", True)

    print("\nall passed" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
