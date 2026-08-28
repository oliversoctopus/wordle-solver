"""Shared loading for the measurement scripts.

Importing this puts the project root and this directory on ``sys.path``, so the
scripts can be run from anywhere:

    python analysis/repeat_shapes.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# both, so the scripts import the same way whether they are run as
# `python analysis/foo.py` or reached as `analysis.common` from tests/
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import rust  # noqa: E402  -- needs the path above
from priors import (WORDLEBOT_SNAPSHOT_DATE,  # noqa: E402  -- likewise
                    depends_on_answer_record)

DATA = os.path.join(ROOT, "data")

# the first repeat in Wordle's history, and the start of the current regime
REGIME_DATE = "2026-02-02"
# the Wayback snapshot the WordleBot list is validated against; `wordlebot.py`
# fetches it and names the file, so the date is not written down twice
SNAPSHOT_DATE = WORDLEBOT_SNAPSHOT_DATE


def add_prior_filter(parser):
    """The flag that drops the arms an extended answer record cannot have moved."""
    parser.add_argument("--repeat-only", action="store_true",
                        help="sweep only the priors whose weights read the "
                             "answer record, i.e. skip the repeat-off arms, "
                             "which solutions.py --write cannot move")
    return parser


def keep_prior(config, args):
    """Whether ``--repeat-only`` keeps this arm."""
    return (not getattr(args, "repeat_only", False)
            or depends_on_answer_record(config))


def filter_priors(items, args):
    """The arms to sweep, given configs or (label, config) pairs.

    Says which it dropped, so a table assembled from two runs says so in both.
    """
    items = list(items)
    if not getattr(args, "repeat_only", False):
        return items
    kept = [it for it in items
            if keep_prior(it[1] if isinstance(it, tuple) else it, args)]
    if not kept:
        raise SystemExit("--repeat-only left nothing to sweep")
    print(f"--repeat-only: {len(kept)} of {len(items)} arms; the rest do not "
          f"read the answer record, so their last figures still stand\n")
    return kept


def load(name):
    with open(os.path.join(DATA, name), "r") as file:
        return json.load(file)


def solutions():
    return load("solutions.json")


def answers():
    return [row["solution"] for row in solutions()]


def puzzle_no(row, index):
    """Wordle's puzzle number.  The pre-NYT rows carry no day count."""
    return row.get("days_since_launch", index) + 1


def regime_start(sol=None):
    sol = sol or solutions()
    return next(i for i, r in enumerate(sol) if r["print_date"] == REGIME_DATE)


def repeat_events(words=None):
    """(when, word, first_used) for every answer that had already been used."""
    words = words if words is not None else answers()
    first, events = {}, []
    for i, word in enumerate(words):
        if word in first:
            events.append((i, word, first[word]))
        else:
            first[word] = i
    return events


def fresh_events(words=None, curated=None, start=None):
    """(when, word) for every first use of a curated word, from `start` on.

    Cut to the regime where the other two arms are not, because a fresh curated
    answer is most of every era and the pre-regime ones would be scored under a
    prior with no repeat events in it at all.
    """
    words = words if words is not None else answers()
    curated = curated if curated is not None else set(load("wordles.json"))
    start = regime_start() if start is None else start
    seen, events = set(words[:start]), []
    for i in range(start, len(words)):
        if words[i] not in seen and words[i] in curated:
            events.append((i, words[i]))
        seen.add(words[i])
    return events


def offlist_events(words=None, curated=None):
    """(when, word) for every first use of a word off the curated list."""
    words = words if words is not None else answers()
    curated = curated if curated is not None else set(load("wordles.json"))
    seen, events = set(), []
    for i, word in enumerate(words):
        if word not in seen and word not in curated:
            events.append((i, word))
        seen.add(word)
    return events


def state_at(words, t):
    """First-use index of every word used before t, and the repeats before t."""
    first, prior = {}, []
    for i, word in enumerate(words[:t]):
        if word in first:
            prior.append((i, first[word]))
        else:
            first[word] = i
    return first, prior


def paired(a, b):
    """Mean difference and its standard error, for two paired sequences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d


MAX_GUESSES = 6


def every_answer(solver, max_guesses=MAX_GUESSES):
    """Guesses taken with each of the 12,947 words in turn as the answer.

    The solver is deterministic, so a game's outcome is a pure function of the
    answer: this is the entire sample space, not a draw from it.  A loss counts
    as one guess over the limit, the convention every measured result here uses.

    Answered by the Rust where it was built and can reproduce these settings,
    which is most of a sweep-driven script's wall clock; ``WORDLE_NO_RUST=1``
    forces the loop below.  The two return the same lengths and
    ``tests/rust_parity.py --every-answer`` is what says so.
    """
    lengths = rust.every_answer(solver, max_guesses)
    if lengths is not None:
        return lengths
    out = np.empty(len(solver.weights))
    for answer in range(len(out)):
        guesses, _ = solver.play(answer, max_guesses=max_guesses,
                                 deadline=max_guesses)
        solved = guesses[-1] == answer
        out[answer] = len(guesses) if solved else max_guesses + 1
    return out


def exact_stats(lengths, weights, max_guesses=MAX_GUESSES):
    """Prior-weighted mean and loss probability -- computed, not estimated."""
    lost = lengths > max_guesses
    return float(weights @ lengths), float(weights[lost].sum())


# ------------------------------------------------- positions and their values
#
# The shared parts of `lookahead_accuracy.py` and `exact_decisions.py`: what a
# position is worth, computed by playing the policy out rather than estimated.

# a deadline-free playout ends when one candidate is left, so this is a backstop
# against a bug rather than a real limit
PLAYOUT_CAP = 12


def collect_positions(solver, answers, keep):
    """Distinct positions from real games where ``keep(solver, candidates,
    turns_left)`` holds, and how many of the games reach each.

    Turn 1 is skipped: it is the same position in every game, and its value
    costs a playout per word in the list.
    """
    index, out, counts = {}, [], []
    for answer in answers:
        answer = int(answer)
        candidates = solver.all_candidates()
        for turn in range(1, MAX_GUESSES + 1):
            left = MAX_GUESSES - turn + 1
            if turn > 1 and keep(solver, candidates, left):
                key = candidates.tobytes()
                if key in index:
                    counts[index[key]] += 1
                else:
                    index[key] = len(out)
                    out.append((turn, candidates))
                    counts.append(1)
            guess, _ = solver.best_guess(candidates, left)
            if guess == answer:
                break
            candidates = solver.filter(guess, solver.feedback(guess, answer), candidates)
    return out, np.array(counts, dtype=float)


def playout(solver, candidates, answer):
    """Guesses this solver's policy needs from here, with no deadline."""
    for used in range(1, PLAYOUT_CAP + 1):
        guess, _ = solver.best_guess(candidates, None)
        if guess == answer:
            return used
        candidates = solver.filter(guess, solver.feedback(guess, answer), candidates)
    return PLAYOUT_CAP


def position_value(solver, candidates):
    """Expected guesses from here under this policy: computed, not estimated."""
    weights = solver.weights[candidates]
    used = np.array([playout(solver, candidates, int(a)) for a in candidates],
                    dtype=float)
    return float(weights @ used / weights.sum())


def guess_value(solver, candidates, guess):
    """True expected guesses from playing ``guess`` here, then following the
    solver's policy.  Every guess is continued the same way, so the difference
    between two of them is the decision at this position and nothing else."""
    weights = solver.weights[candidates]
    used = np.empty(len(candidates), dtype=float)
    for i, answer in enumerate(candidates):
        answer = int(answer)
        if guess == answer:
            used[i] = 1.0
            continue
        rest = solver.filter(guess, solver.feedback(guess, answer), candidates)
        used[i] = 1.0 + playout(solver, rest, answer)
    return float(weights @ used / weights.sum())


def shortlist(solver, candidates, k):
    """The k best guesses by this solver's own heuristic score, best first."""
    pool = solver.guess_pool(candidates)
    scores = solver.score_all(candidates, pool=pool)
    order = np.argsort(scores)[:k]
    words = [int(i) if pool is None else int(pool[i]) for i in order]
    return words, scores[order]


def weighted(values, weights):
    """Weighted mean, i.e. the per-game figure rather than the per-position one."""
    w = np.asarray(weights, dtype=float)
    return float(np.asarray(values, dtype=float) @ w / w.sum())


def effective_support(weight):
    """2^H of a weight vector: how many equally likely words it spreads over."""
    p = np.asarray(weight, dtype=float)
    p = p / p.sum()
    return 2.0 ** -(p * np.log2(np.maximum(p, 1e-300))).sum()


def rank_correlation(x, y):
    """Spearman's rho, plus a normal-approximation two-sided p."""
    from math import erf, sqrt

    def ranks(a):
        order = np.asarray(a).argsort()
        out = np.empty(len(a), dtype=float)
        out[order] = np.arange(1, len(a) + 1)
        return out

    rho = np.corrcoef(ranks(x), ranks(y))[0, 1]
    n = len(x)
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    return rho, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def slope_se(x, y):
    """Least-squares slope of y on x, and its standard error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    return slope, se


def header(title):
    print("=" * 74)
    print(title)
    print("=" * 74)
