"""Rebuild `answer_like.json`: how answer-like each non-curated word is.

The curated list is 2,309 labeled examples of "a word an editor considered
answerable" and it is independent of the off-list events, so fit a logistic
regression with the curated words as positives and the non-curated words as
negatives and keep the log-odds.  `SOLVER_EXPLAINED.md` 4.4 has the derivation
and reads the fitted weights.

This reproduces the shipped `answer_like.json` exactly.  It writes nothing
unless you pass `--write`, since the solver reads the file.

The fit is sensitive to details that are easy to guess wrong -- the feature set,
the language list, the ridge on the intercept.  Change any of them and it still
fits, still correlates at r ~ 0.98, and no longer reproduces the file.

Needs `wordfreq`, which nothing else in the project uses:

    pip install wordfreq
"""

import argparse
import json
import os

import numpy as np
from wordfreq import zipf_frequency

from wordle_data import CURATED_PATH, DATA_DIR, IND2WORD_PATH, write_json

ANSWER_LIKE_PATH = os.path.join(DATA_DIR, "answer_like.json")

# the "is this actually a name" lift: a word as common abroad as at home is a
# name, not an English word.  Ten languages, not twenty -- adding more drags
# the lift toward zero for genuinely international words like `laser`.
FOREIGN_LANGUAGES = ("fr", "de", "es", "it", "pl", "tr", "nl", "sv", "pt", "id")

SUFFIXES = ("s", "es", "ed", "er", "y", "ly")
FEATURE_NAMES = (("intercept", "zipf", "zipf^2", "zero-freq", "foreign-lift",
                  "foreign-max") + tuple(f"ends {s}" for s in SUFFIXES)
                 + ("dup letters",))

# ridge on every coefficient, the intercept included; exempting it shifts the
# whole score and changes nothing that exp(t * score) does not renormalize away
DEFAULT_L2 = 1.0
ITERATIONS = 300

DECIMALS = 3


def features(words, english, foreign):
    z = np.array([english[w] for w in words])
    fz = np.array([foreign[w] for w in words])
    columns = [np.ones(len(words)), z, z * z, (z == 0).astype(float),
               fz.mean(1) - z, fz.max(1)]
    for suffix in SUFFIXES:
        columns.append(np.array([w.endswith(suffix) for w in words], dtype=float))
    # a count, not a flag: `mommy` has one repeat, `sissy` two
    columns.append(np.array([5 - len(set(w)) for w in words], dtype=float))
    return np.column_stack(columns)


def fit_logistic(X, y, l2=DEFAULT_L2, iterations=ITERATIONS):
    """Newton / IRLS.  Thirteen columns, so the Hessian solve is free."""
    if l2 <= 0.0:
        # the curated words are nearly separable, so the weights on the Hessian
        # collapse and it goes singular; the ridge is what keeps it invertible
        raise ValueError("l2 must be positive")
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        hessian = (X * (p * (1.0 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        beta += np.linalg.solve(hessian, X.T @ (y - p) - l2 * beta)
    return beta


def effective_support(score, temperature):
    """2^H of the weight this score turns into: how many words it spreads over."""
    weight = np.exp(temperature * (score - score.max()))
    p = weight / weight.sum()
    return 2.0 ** -(p * np.log2(np.maximum(p, 1e-300))).sum()


def build(l2=DEFAULT_L2):
    with open(IND2WORD_PATH, "r") as file:
        words = json.load(file)
    with open(CURATED_PATH, "r") as file:
        curated_list = json.load(file)
    curated = set(curated_list)
    noncurated = [w for w in words if w not in curated]

    english = {w: zipf_frequency(w, "en") for w in words}
    foreign = {w: [zipf_frequency(w, lang) for lang in FOREIGN_LANGUAGES]
               for w in words}

    X = features(curated_list + noncurated, english, foreign)
    y = np.concatenate([np.ones(len(curated_list)), np.zeros(len(noncurated))])
    beta = fit_logistic(X, y, l2=l2)

    scores = X[len(curated_list):] @ beta
    table = {w: round(float(s), DECIMALS) for w, s in zip(noncurated, scores)}
    return table, dict(zip(FEATURE_NAMES, beta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2,
                        help=f"ridge penalty (default {DEFAULT_L2})")
    parser.add_argument("--temperature", type=float, default=0.35,
                        help="temperature to report effective support at")
    parser.add_argument("--write", action="store_true",
                        help="overwrite answer_like.json, which the solver reads")
    args = parser.parse_args()
    if args.l2 <= 0.0:
        parser.error("--l2 must be positive; at zero the Hessian goes singular")

    print(f"scoring in {1 + len(FOREIGN_LANGUAGES)} languages, this takes a minute...")
    table, weights = build(l2=args.l2)
    print("fitted weights")
    for name, value in weights.items():
        print(f"  {name:<20s} {value:+7.3f}")

    mine = np.array(list(table.values()))
    print(f"\nscores run {mine.min():.3f} to {mine.max():.3f} over {len(table)} words")
    print(f"effective support at t={args.temperature}: "
          f"{effective_support(mine, args.temperature):.1f}")

    identical = False
    if os.path.exists(ANSWER_LIKE_PATH):
        with open(ANSWER_LIKE_PATH, "r") as file:
            shipped = json.load(file)
        common = [w for w in table if w in shipped]
        a = np.array([table[w] for w in common])
        b = np.array([shipped[w] for w in common])
        differing = int((a != b).sum())
        identical = set(table) == set(shipped) and differing == 0
        print(f"\nagainst the shipped answer_like.json ({len(common)} words in common)")
        print(f"  identical         {identical}")
        print(f"  differing values  {differing}")
        print(f"  correlation       {np.corrcoef(a, b)[0, 1]:.6f}")
        print(f"  max |difference|  {np.abs(a - b).max():.3f}")

    if args.write:
        write_json(ANSWER_LIKE_PATH, table, sort_keys=True)
        print(f"\nwrote {ANSWER_LIKE_PATH}")
    elif identical:
        print("\nnot written; it already matches the shipped file exactly")
    else:
        print("\nnot written; pass --write to replace answer_like.json")


if __name__ == "__main__":
    main()
