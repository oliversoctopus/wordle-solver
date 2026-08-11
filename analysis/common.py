"""Shared loading for the measurement scripts.

Importing this puts the project root on ``sys.path``, so the scripts can be run
from anywhere:

    python analysis/repeat_shapes.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")

# the first repeat in Wordle's history, and the start of the current regime
REGIME_DATE = "2026-02-02"
# the Wayback snapshot the WordleBot list is validated against
SNAPSHOT_DATE = "2023-10-12"


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
