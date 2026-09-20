"""One word per split in the ranking, and the words it stands for.

``top_guesses`` names one word per distinct split of the candidates and
``split_siblings`` the rest of it.  Whether two words split alike is settled
by comparing their table columns with numpy; whether the ranking is worth what
it says by scoring a split again, or by asking the exact search for the pick.
Loads the table, so it is one of the slow ones.

    python tests/splits.py --games 8
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import ENTROPY, MODES, WordleSolver
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
    """Every position a real game passes through, played out at ``answer``."""
    candidates = solver.all_candidates()
    for turn in range(1, deadline + 1):
        if len(candidates) == 1:
            return
        yield candidates, deadline - turn + 1
        guess, _ = solver.best_guess(candidates, deadline - turn + 1)
        if guess == answer:
            return
        candidates = solver.filter(guess, solver.feedback(guess, answer),
                                   candidates)


def split_by_hand(solver, guess, candidates):
    """Every word that splits the candidates as ``guess`` does, found by
    comparing the table's columns outright.  What the kernel answers to."""
    table = solver.data.feedback_table
    pool = (candidates if solver.hard
            else np.arange(table.shape[0], dtype=np.int64))
    # a word splitting them alike agrees on any few, so narrow on a handful and
    # compare those in full, without a copy of the table
    head = candidates[:8]
    near = pool[(table[pool[:, None], head] == table[guess, head]).all(1)]
    return {int(word) for word in near
            if np.array_equal(table[word, candidates],
                              table[guess, candidates])} - {int(guess)}


def check_position(solver, candidates, guesses_left, k=5):
    data = solver.data
    top = solver.top_guesses(candidates, k=k, guesses_left=guesses_left)
    where = f"{len(candidates)} left, {guesses_left} to play"
    listed = [data.word2ind[word] for word, _ in top]

    splits = [tuple(data.feedback_table[guess, candidates]) for guess in listed]
    check(f"{where}: the {len(top)} listed all split the candidates differently",
          len(set(splits)) == len(splits), f"{[word for word, _ in top]}")

    named, agreed = 0, True
    for guess in listed:
        siblings = set(int(w) for w in solver.split_siblings(guess, candidates))
        if siblings != split_by_hand(solver, guess, candidates):
            check(f"{where}: {data.all_words[guess]}'s split is the table's", False)
            return
        named += len(siblings)
        # interchangeable has to mean interchangeable: the same buckets, so the
        # same score, whichever of them is played
        score = solver.score_one(guess, candidates)
        agreed = agreed and all(solver.score_one(other, candidates) == score
                                for other in sorted(siblings)[:8])
    check(f"{where}: their splits hold the {named} words the table says", True)
    check(f"{where}: every word in a split scores as the word standing for it",
          agreed)

    if solver.solved_exactly(candidates, guesses_left):
        best, value = solver.best_guess(candidates, guesses_left)
        check(f"{where}: settled exactly, and the pick is best_guess's",
              listed[0] == best,
              f"{top[0][0]} against {data.all_words[best]}")
        check(f"{where}: carrying the value best_guess gives it",
              abs(top[0][1] - value) < 1e-9, f"{top[0][1]} against {value}")

    scores = [score for _, score in top]
    ordered = (all(a >= b - 1e-12 for a, b in zip(scores, scores[1:]))
               if solver.mode == ENTROPY
               else all(a <= b + 1e-12 for a, b in zip(scores, scores[1:])))
    check(f"{where}: the list runs best first", ordered, f"{scores}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=4,
                        help="answers to play out, per mode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_data()
    rng = np.random.default_rng(args.seed)
    answers = [data.word2ind[word] for word in
               rng.choice(data.curated_list, size=args.games, replace=False)]
    print(f"{args.games} games per mode, every position they pass through, "
          f"against the table read the slow way")

    for mode in MODES:
        print(f"\n{mode}")
        solver = WordleSolver(data=data, mode=mode)
        for answer in answers:
            for candidates, guesses_left in positions(solver, answer):
                check_position(solver, candidates, guesses_left)
                # two turns left is a closed form scoring every guess, and a real game reaches it
                # with a handful of candidates; ask it here too
                if len(candidates) >= 3 and guesses_left != 2:
                    check_position(solver, candidates, 2)

    print("\nall passed" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
