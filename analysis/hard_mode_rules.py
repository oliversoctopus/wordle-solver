"""Which hard-mode rule, and what the difference between them is worth.

The NYT's hard mode is a rule about letters: keep every green where it is, and
use every revealed letter somewhere, with as many copies as have been revealed.
It says nothing about gray letters, so a legal guess may spend letters already
ruled out, carry more copies of a letter than the answer holds, or put a yellow
back in the square where it came up yellow.  The solver's `--hard` reads the
rule as "every guess must be a word that could still be the answer", which is
strictly narrower: a possible answer keeps the greens and holds every revealed
letter, so it is always legal, while a legal guess is frequently impossible.

So the shipped arm plays a subset of the moves the NYT allows, and can only do
worse or the same.  This plays every word as the answer under both readings and
prices the gap, and counts the thing that would cause one: how often the wider
pool is worth leaving the candidates for.

Two things keep this a floor rather than the whole of it.  The endgame search
narrows to the candidates in both arms, since its nodes would need a constraint
history it is never given, and the pool a lookahead sees is the root's rather
than the one that node would really have.  Both understate the wider rule.

    python analysis/hard_mode_rules.py
    python analysis/hard_mode_rules.py --answers 300     a slice, to try it out
"""

import argparse

import numpy as np

import common
from memo import Memo
from priors import ALL_CONFIGS, PriorConfig
from solver import ENTROPY, EXPECTED, MAX_TURNS, WordleSolver
from wordle_data import load_data, num_to_feedback

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=(EXPECTED, ENTROPY), default=EXPECTED)
parser.add_argument("--all-priors", action="store_true",
                    help="every combination, not just the shipped one")
parser.add_argument("--answers", type=int, default=0,
                    help="play only the first N words, to smoke-test the loop")
parser.add_argument("--free", action="store_true",
                    help="also play the unrestricted arm, for scale")
parser.add_argument("--check", action="store_true",
                    help="first show this loop reproducing every_answer")
parser.add_argument("--no-endgame", action="store_true",
                    help="turn the exact search off, so the pool decides "
                         "every turn rather than the first two")
common.add_prior_filter(parser)
args = parser.parse_args()


class NytHard(WordleSolver):
    """The solver, with the guess pool handed to it from outside.

    `guess_pool` is given only the candidates, and the NYT's rule is not a
    function of those -- it is a function of the rows played.  The memo is
    keyed on the candidates alone for the same reason, so the constraints join
    the key here: two games can reach one candidate set by routes that leave
    different letters revealed.
    """

    pool = None
    signature = ()

    def guess_pool(self, candidates):
        # with no pool handed in this is the stock solver, free or restricted
        if self.pool is not None:
            return self.pool
        return super().guess_pool(candidates)

    def best_guess(self, candidates, guesses_left=None):
        turns = (MAX_TURNS if guesses_left is None
                 else min(int(guesses_left), MAX_TURNS))
        key = (self.signature, candidates.tobytes(), turns)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        result = self._best_guess_uncached(candidates, guesses_left)
        self._memo.put(key, result)
        return result


class Legality:
    """Which words the NYT's rule allows, given what has been revealed."""

    def __init__(self, words):
        grid = np.frombuffer("".join(words).encode(), dtype=np.uint8)
        self.letters = grid.reshape(len(words), -1).astype(np.int64) - ord("a")
        self.counts = np.zeros((len(words), 26), dtype=np.int64)
        for column in range(self.letters.shape[1]):
            np.add.at(self.counts,
                      (np.arange(len(words)), self.letters[:, column]), 1)
        self.cache = {}

    @staticmethod
    def revealed(rows):
        """(green letter per position, least copies of each letter), as a key.

        A row reveals as many copies of a letter as it colors green or yellow;
        the requirement is the most any single row has shown, since a later row
        that repeats less has not unlearned the earlier one.
        """
        greens, needs = {}, {}
        for guess, pattern in rows:
            shown = {}
            for position, tile in enumerate(num_to_feedback(pattern)):
                if tile == "N":
                    continue
                letter = ord(guess[position]) - ord("a")
                if tile == "G":
                    greens[position] = letter
                shown[letter] = shown.get(letter, 0) + 1
            for letter, count in shown.items():
                needs[letter] = max(needs.get(letter, 0), count)
        return (tuple(sorted(greens.items())), tuple(sorted(needs.items())))

    def pool(self, signature):
        if signature in self.cache:
            return self.cache[signature]
        greens, needs = signature
        allowed = np.ones(len(self.letters), dtype=bool)
        for position, letter in greens:
            allowed &= self.letters[:, position] == letter
        for letter, count in needs:
            allowed &= self.counts[:, letter] >= count
        pool = np.flatnonzero(allowed).astype(np.int64)
        self.cache[signature] = pool
        return pool


def play(solver, answer, legality=None, max_guesses=MAX_GUESSES):
    """One game.  Returns (guesses taken, guesses that were not candidates)."""
    words = solver.data.all_words
    candidates = solver.all_candidates()
    rows, off_pool = [], 0
    for turn in range(1, max_guesses + 1):
        if legality is not None:
            solver.signature = legality.revealed(rows)
            solver.pool = legality.pool(solver.signature)
        left = max(1, max_guesses - turn + 1)
        guess, _ = solver.best_guess(candidates, left)
        if guess == answer:
            return turn, off_pool
        if legality is not None and not np.isin(guess, candidates):
            off_pool += 1
        pattern = solver.feedback(guess, answer)
        rows.append((words[guess], pattern))
        candidates = solver.filter(guess, pattern, candidates)
    return max_guesses + 1, off_pool


def sweep(solver, answers, legality=None):
    lengths = np.empty(len(answers))
    off_pool = 0
    for i, answer in enumerate(answers):
        lengths[i], left = play(solver, int(answer), legality)
        off_pool += left
    return lengths, off_pool


def check_loop(data, config):
    """The loop here plays the policy itself rather than handing the sweep to
    `common.every_answer`, since neither that nor the native binary can be told
    about a pool.  So show the two agreeing where they can: with no pool handed
    in, game for game, both arms."""
    for hard in (False, True):
        reference = WordleSolver(data=data, config=config, mode=args.mode,
                                 hard=hard)
        mine, _ = sweep(NytHard(data=data, config=config, mode=args.mode,
                                hard=hard), np.arange(len(data.all_words)))
        theirs = common.every_answer(reference)
        same = bool((mine == theirs).all())
        print(f"  hard={str(hard):5s} {'agrees' if same else 'DIFFERS'} with "
              f"common.every_answer over all {len(mine)} games")


data = load_data()
legality = Legality(data.all_words)
configs = common.filter_priors(
    ALL_CONFIGS if args.all_priors else [PriorConfig()], args)
answers = np.arange(args.answers or len(data.all_words))

if args.check:
    common.header("this loop against common.every_answer")
    check_loop(data, configs[0])

for config in configs:
    common.header(config.slug)
    weights = None
    rows = []

    def arm(name, hard, legal):
        global weights
        solver = NytHard(data=data, config=config, mode=args.mode, hard=hard,
                         use_exact=not args.no_endgame)
        weights = solver.weights
        lengths, off_pool = sweep(solver, answers, legal)
        rows.append((name, lengths, off_pool))
        return lengths

    if args.free:
        arm("free", False, None)
    shipped = arm("possible answers only", True, None)
    nyt = arm("the NYT's rule", True, legality)

    share = weights[answers] / weights[answers].sum()
    print(f"  {len(answers)} words as the answer, weighted by this prior, "
          f"mode={args.mode}\n")
    print(f"  {'pool':22s} {'mean':>7s} {'solved':>8s}"
          f" {'off-pool guesses':>18s}")
    for name, lengths, off_pool in rows:
        lost = lengths > MAX_GUESSES
        mean = float(share @ lengths)
        print(f"  {name:22s} {mean:7.4f} {(1 - share[lost].sum()) * 100:7.2f}%"
              f" {off_pool:18d}")

    per_word = nyt - shipped
    print(f"\n  what the wider rule is worth: "
          f"{float(share @ per_word):+.4f} guesses"
          f"  (a floor; see the docstring)")
    print(f"  words changing length: {int((per_word != 0).sum())} of "
          f"{len(per_word)} ({int((per_word < 0).sum())} better, "
          f"{int((per_word > 0).sum())} worse)")
    print(f"  losing words: {int((shipped > MAX_GUESSES).sum())} restricted, "
          f"{int((nyt > MAX_GUESSES).sum())} under the NYT's rule\n")
    if config == PriorConfig() and not args.answers:
        policy = ("entropy" if args.mode == ENTROPY
                  else "no_endgame" if args.no_endgame else "shipped")
        off_pool = dict((name, off) for name, _, off in rows)["the NYT's rule"]
        common.record(f"{policy}.off_pool", int(off_pool))
        common.record(f"{policy}.changed", int((per_word != 0).sum()))
        common.record(f"{policy}.better", int((per_word < 0).sum()))
        common.record(f"{policy}.worse", int((per_word > 0).sum()))
        common.record(f"{policy}.worth", float(share @ per_word))
