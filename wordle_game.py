"""Play Wordle in the terminal, against a chosen or a sampled answer.

    python wordle_game.py                  pick how to play from a menu
    python wordle_game.py --answer crane   play against a known answer
    python wordle_game.py --random         answer drawn from the prior

The prior toggles select what ``--random`` draws from.  While playing, ``?``
asks the solver and ``quit`` gives up; ``--assist`` shows the pick every turn,
``--auto`` lets the solver play, and ``--share`` prints the emoji grid.
"""

import argparse
import sys

import numpy as np

from priors import PriorConfig, build_weights
from wordle_data import (MAX_GUESSES, TILE_VALS, WIN_PATTERN, load_data,
                         num_to_feedback, score_guess)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

RESET = "\x1b[0m"

# 24-bit copies of the emoji's own colors.  The palette's bright yellow is far
# paler than U+1F7E8 (#F9F1A5 against #FFB02E on this console), so the gold is
# spelled out rather than left to the terminal to choose.
GOLD = (255, 176, 46)      # as U+1F7E8
LEAF = (0, 210, 106)       # as U+1F7E9
SLATE = (120, 124, 126)    # Wordle's own gray; U+2B1B would be near-invisible


def ink(rgb):
    return "\x1b[38;2;{};{};{}m".format(*rgb)


def wash(rgb):
    return "\x1b[48;2;{};{};{}m".format(*rgb)


GREEN, YELLOW, GRAY = "\x1b[30;102m", "\x1b[30m" + wash(GOLD), "\x1b[30;47m"
TILE_COLOR = {"G": GREEN, "Y": YELLOW, "N": GRAY}

# The share grid is drawn as blocks of color, since the console sets the emoji
# ragged.  A square tile is three cells wide: a full line plus U+2580 over the
# top half of the next, and the half-line left over is the gap below.
BLOCK_RGB = {"G": LEAF, "Y": GOLD, "N": SLATE}
TILE_BLOCK = {"G": "\U0001F7E9", "Y": "\U0001F7E8", "N": "⬛"}


def enable_color():
    """True if the terminal can show the colored tiles."""
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:  # ask the console for VT processing (Windows 10 and later)
        import ctypes

        kernel32 = ctypes.windll.kernel32
        return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))
    except Exception:
        return False


def render_row(word, pattern, color):
    tiles = num_to_feedback(pattern)
    if color:
        return "".join(f"{TILE_COLOR[t]} {c.upper()} {RESET}" for c, t in zip(word, tiles))
    return "  ".join(c.upper() for c in word) + "\n  " + "  ".join(tiles)


class Keyboard:
    """Best thing known about each letter so far."""

    def __init__(self):
        self.state = {}

    def update(self, word, pattern):
        for letter, tile in zip(word, num_to_feedback(pattern)):
            if TILE_VALS[tile] > TILE_VALS.get(self.state.get(letter), -1):
                self.state[letter] = tile

    def describe(self):
        parts = []
        for tile, label in (("G", "correct"), ("Y", "present"), ("N", "absent")):
            letters = sorted(c.upper() for c, t in self.state.items() if t == tile)
            if letters:
                parts.append(f"{label}: {' '.join(letters)}")
        untried = [c.upper() for c in ALPHABET if c not in self.state]
        if untried:
            parts.append(f"untried: {' '.join(untried)}")
        return "\n  ".join(parts)


# ------------------------------------------------------------------- answers

PRIOR_CHOICES = [
    ("both priors: repeats and off-list answers are both rarer",
     PriorConfig(repeat=True, curation=True)),
    ("repeat prior only: already-used answers are rarer",
     PriorConfig(repeat=True, curation=False)),
    ("curation prior only: off-list answers are rarer",
     PriorConfig(repeat=False, curation=True)),
    ("no priors: uniform over every word",
     PriorConfig(repeat=False, curation=False)),
]


def pick_random_answer(data, config, seed=None):
    weights = build_weights(data, config)
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(weights), p=weights))


def prompt_menu(title, options):
    """Returns the chosen option's value, or None on EOF.  Empty input takes
    the first option."""
    print(f"\n{title}")
    for i, (label, _) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return None
        if not raw:
            return options[0][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"  enter a number from 1 to {len(options)}.")


def choose_answer(data, args):
    """Returns ``(answer index, description)``, or ``(None, None)`` to stop."""
    if args.answer:
        word = args.answer.strip().lower()
        if word not in data.word2ind:
            print(f"'{word}' is not in the word list.")
            return None, None
        return data.word2ind[word], "the one you gave"

    if args.random:
        config = PriorConfig(repeat=not args.no_repeat_prior,
                             curation=not args.no_curation_prior)
        return pick_random_answer(data, config, args.seed), f"drawn from {config.describe()}"

    how = prompt_menu("how should the answer be chosen?", [
        ("type it in yourself (handy for testing the solver)", "typed"),
        ("draw one at random from the answer prior", "random"),
    ])
    if how is None:
        return None, None

    if how == "typed":
        while True:
            try:
                word = input("answer: ").strip().lower()
            except EOFError:
                return None, None
            if word in data.word2ind:
                return data.word2ind[word], "the one you gave"
            print(f"  '{word}' is not in the word list; try again.")

    config = prompt_menu("which prior should it be drawn from?", PRIOR_CHOICES)
    if config is None:
        return None, None
    return pick_random_answer(data, config, args.seed), f"drawn from {config.describe()}"


# ---------------------------------------------------------------------- play

class SolverHint:
    """Wraps the solver, loading it (and the feedback table) only if a
    recommendation is actually asked for.  Guesses made before that point are
    replayed when it loads, so a hint asked for on turn 4 is still correct."""

    def __init__(self, mask_last):
        self.mask_last = mask_last
        self.history = []
        self._solver = None
        self._candidates = None

    def _load(self):
        if self._solver is not None:
            return
        print("  (loading the solver...)", flush=True)
        from solver import WordleSolver

        full = load_data(mask_last=self.mask_last)
        self._solver = WordleSolver(data=full)
        self._candidates = self._solver.all_candidates()
        for word, pattern in self.history:
            self._candidates = self._solver.filter(
                self._solver.data.word2ind[word], pattern, self._candidates)

    def recommend(self, guesses_left):
        """``(word, expected score, candidates remaining)``."""
        self._load()
        guess, score = self._solver.best_guess(self._candidates, guesses_left)
        return self._solver.data.all_words[guess], score, len(self._candidates)

    def observe(self, word, pattern):
        self.history.append((word, pattern))
        if self._solver is not None:
            self._candidates = self._solver.filter(
                self._solver.data.word2ind[word], pattern, self._candidates)


def describe_hint(word, score, left, name=True):
    lead = f"solver: {word}  " if name else ""
    return (f"{lead}({score:.4f} expected guesses to finish, "
            f"{left} candidate{'' if left == 1 else 's'} left)")


def play(data, answer, description, args):
    color = not args.no_color and enable_color()
    answer_word = data.all_words[answer]
    hint = SolverHint(args.mask_last)
    keyboard = Keyboard()
    rows = []

    print(f"\nwordle: {MAX_GUESSES} guesses, {len(data.all_words)} valid words. "
          f"the answer is {description}.")
    print("type ? for the solver's pick, quit to give up."
          if not args.auto else "the solver is playing.")
    print()

    for turn in range(1, MAX_GUESSES + 1):
        guesses_left = MAX_GUESSES - turn + 1
        guess = None

        suggested = None
        if args.auto or args.assist:
            word, score, left = hint.recommend(guesses_left)
            if args.auto:
                print(f"guess {turn}/{MAX_GUESSES}: {word}   "
                      f"{describe_hint(word, score, left, name=False)}")
                guess = word
            else:
                print("  " + describe_hint(word, score, left))
                suggested = word

        prompt = (f"guess {turn}/{MAX_GUESSES}"
                  + (f" [{suggested}]: " if suggested else ": "))
        while guess is None:
            try:
                raw = input(prompt).strip().lower()
            except EOFError:
                print(f"\ngave up. the answer was {answer_word.upper()}.")
                return 1
            if not raw and suggested:
                guess = suggested
                continue
            if raw in ("quit", "q", "exit"):
                print(f"gave up. the answer was {answer_word.upper()}.")
                return 1
            if raw in ("?", "h", "hint"):
                word, score, left = hint.recommend(guesses_left)
                print("  " + describe_hint(word, score, left))
                suggested = word
                prompt = f"guess {turn}/{MAX_GUESSES} [{suggested}]: "
                continue
            if not raw:
                print("  type a five letter word, ? for a hint, or quit.")
                continue
            if raw not in data.word2ind:
                print(f"  '{raw}' is not in the word list.")
                continue
            guess = raw

        pattern = score_guess(guess, answer_word)
        rows.append((guess, pattern))
        keyboard.update(guess, pattern)
        hint.observe(guess, pattern)

        print("  " + render_row(guess, pattern, color))
        if pattern == WIN_PATTERN:
            print(f"\nsolved in {turn} guess{'' if turn == 1 else 'es'}.")
            print_share(rows, turn, color, args.share)
            return 0
        print("  " + keyboard.describe() + "\n")

    print(f"\nout of guesses. the answer was {answer_word.upper()}.")
    print_share(rows, None, color, args.share)
    return 1


def print_utf8(line):
    """``print``, but survives a stdout that cannot encode the emoji.  Piping
    the grid into a file is the point of ``--share``, and a redirected stdout
    on Windows encodes as cp1252, which has none of these characters."""
    try:
        print(line)
    except UnicodeEncodeError:
        sys.stdout.flush()
        sys.stdout.buffer.write(line.encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()


def print_share(rows, turns, color, share=False):
    """The board in miniature, and with ``--share`` the emoji grid to go with
    it.  The emoji are left to whatever you paste them into to render, so they
    are the plain ones everybody else's grid uses."""
    header = f"Wordle {turns if turns else 'X'}/{MAX_GUESSES}"

    if color:
        print(f"\n{header}")
        for _, pattern in rows:
            tiles = num_to_feedback(pattern)
            print("  " + " ".join(f"{wash(BLOCK_RGB[t])}   {RESET}" for t in tiles))
            print("  " + " ".join(f"{ink(BLOCK_RGB[t])}▀▀▀{RESET}" for t in tiles))

    if share:
        print(f"\n{header}")
        for _, pattern in rows:
            print_utf8("".join(TILE_BLOCK[t] for t in num_to_feedback(pattern)))


# ---------------------------------------------------------------------- main

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--answer", metavar="WORD",
                        help="play against this answer instead of asking")
    source.add_argument("--random", action="store_true",
                        help="draw the answer from the prior instead of asking")
    parser.add_argument("--no-repeat-prior", action="store_true",
                        help="with --random: do not make already-used answers rarer")
    parser.add_argument("--no-curation-prior", action="store_true",
                        help="with --random: do not make off-list answers rarer")
    parser.add_argument("--seed", type=int, default=None,
                        help="with --random: seed for a reproducible answer")
    parser.add_argument("--assist", action="store_true",
                        help="show the solver's recommendation every turn")
    parser.add_argument("--auto", action="store_true",
                        help="let the solver play the whole game")
    parser.add_argument("--no-color", action="store_true", help="plain text tiles")
    parser.add_argument("--share", action="store_true",
                        help="also print the emoji grid, for pasting elsewhere")
    parser.add_argument("--mask-last", type=int, default=0,
                        help="ignore the last N entries of solutions.json "
                             "(default %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # the game only needs the word lists; the feedback table is loaded lazily,
    # and only if the solver is asked for
    data = load_data(mask_last=args.mask_last, with_feedback_table=False)
    answer, description = choose_answer(data, args)
    if answer is None:
        return 1
    return play(data, answer, description, args)


if __name__ == "__main__":
    sys.exit(main())
