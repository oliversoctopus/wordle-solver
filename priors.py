"""The answer prior, assembled from two independently toggleable assumptions.

The editor's process is modeled as:

    1. with probability ``repeat_prob`` pick a word that has already been an
       answer;
    2. otherwise pick an unused word, which is off the curated 2022 answer list
       with probability ``noncurated_prob``.

Step 1 is the *repeat* prior, step 2 is the *curation* prior.  Either can be
switched off, in which case that split is simply not made:

    repeat=1 curation=1   the full model above
    repeat=1 curation=0   used vs. unused only, uniform inside each
    repeat=0 curation=1   curated vs. non-curated only, over every word
    repeat=0 curation=0   uniform over all words

Each group also has a *shape*: how its mass is spread over its members.  Both
default to the sharpened shape, worth -0.064 +- 0.032 guesses together; passing
``uniform`` gives the flat spread the two priors used on their own.  They were
adopted as a pair because together they measured far better than the sum of
their parts; on current data they measure merely additive -- see README.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from wordle_data import DATA_DIR

WORDLEBOT_PATH = os.path.join(DATA_DIR, "wordlebot_weights.json")
ANSWER_LIKE_PATH = os.path.join(DATA_DIR, "answer_like.json")

DEFAULT_REPEAT_PROB = 0.10
DEFAULT_NONCURATED_PROB = 0.10

REPEAT_SHAPES = ("uniform", "binned")
CURATION_SHAPES = ("uniform", "frequency", "wordlebot")

# 'binned': a Laplace-smoothed histogram of repeats over the source index, i.e.
# where in the original answer order the reused word came from
BIN_WIDTH = 200
BIN_ALPHA = 0.5

# 'frequency': weight ~ exp(FREQ_TEMPERATURE * answer-like log-odds).  The
# optimum is flat between 0.2 and 0.5; this is the middle of it.
FREQ_TEMPERATURE = 0.35

# 'wordlebot': mass kept outside the list, since it missed 2 of the 41 off-list
# answers that followed the snapshot it was validated against
OFFLIST_HEDGE = 0.05

_TABLES = {}


def _table(path):
    if path not in _TABLES:
        with open(path, "r") as file:
            _TABLES[path] = json.load(file)
    return _TABLES[path]


@dataclass(frozen=True)
class PriorConfig:
    """Which priors are enabled, plus the probabilities they use."""

    repeat: bool = True
    curation: bool = True
    repeat_prob: float = DEFAULT_REPEAT_PROB
    noncurated_prob: float = DEFAULT_NONCURATED_PROB
    repeat_shape: str = "binned"
    curation_shape: str = "wordlebot"

    def __post_init__(self):
        if self.repeat_shape not in REPEAT_SHAPES:
            raise ValueError(f"repeat_shape must be one of {REPEAT_SHAPES}")
        if self.curation_shape not in CURATION_SHAPES:
            raise ValueError(f"curation_shape must be one of {CURATION_SHAPES}")

    @property
    def slug(self):
        """Filename-safe key; one per prior, shapes included.

        The shape belongs in the key because the best form of f depends on it:
        'sqrt' is right for the uniform shapes and picks the wrong opening word
        for the sharpened ones.
        """
        parts = [f"repeat{int(self.repeat)}", f"curation{int(self.curation)}"]
        if self.repeat and self.repeat_shape != "uniform":
            parts.append(f"r-{self.repeat_shape}")
        if self.curation and self.curation_shape != "uniform":
            parts.append(f"c-{self.curation_shape}")
        return "_".join(parts)

    def describe(self):
        parts = []
        parts.append(
            f"repeat prior {'ON' if self.repeat else 'OFF'}"
            + (f" (p={self.repeat_prob:.3g}, {self.repeat_shape})"
               if self.repeat else "")
        )
        parts.append(
            f"curation prior {'ON' if self.curation else 'OFF'}"
            + (f" (p={self.noncurated_prob:.3g}, {self.curation_shape})"
               if self.curation else "")
        )
        return ", ".join(parts)

    def fingerprint(self):
        """Everything a cached fit depends on besides the rollouts themselves."""
        return {
            "repeat": self.repeat,
            "curation": self.curation,
            "repeat_prob": self.repeat_prob if self.repeat else None,
            "noncurated_prob": self.noncurated_prob if self.curation else None,
            "repeat_shape": self.repeat_shape if self.repeat else None,
            "curation_shape": self.curation_shape if self.curation else None,
        }


ALL_CONFIGS = [
    PriorConfig(repeat=r, curation=c)
    for r in (False, True)
    for c in (False, True)
] + [
    # the fallback curation shape, which needs no vendored word list
    PriorConfig(curation_shape="frequency"),
]


def _repeat_shape(data, used_mask):
    """Relative weights within the used group, by where the word first appeared.

    Reuse is not uniform over past answers: it concentrates near the start of
    the run, so bin the used words by first-use index and weight each bin by
    how many repeats have come out of it.
    """
    first, sources = {}, []
    for i, word in enumerate(data.past_solutions):
        if word in first:
            sources.append(first[word])
        else:
            first[word] = i
    if not sources:
        return None

    inds, starts = [], []
    for word, start in first.items():
        ind = data.word2ind.get(word)
        if ind is not None:
            inds.append(ind)
            starts.append(start)
    inds = np.array(inds, dtype=np.int64)
    starts = np.array(starts, dtype=np.int64) // BIN_WIDTH
    n_bins = int(max(starts.max(), max(sources) // BIN_WIDTH)) + 1

    size = np.bincount(starts, minlength=n_bins).astype(np.float64)
    hits = np.bincount(np.array(sources) // BIN_WIDTH,
                       minlength=n_bins).astype(np.float64)
    per_bin = (hits + BIN_ALPHA) / np.maximum(size, 1.0)

    rel = np.zeros(len(used_mask), dtype=np.float64)
    rel[inds] = per_bin[starts]
    return rel


def _curation_shape(data, mask, shape):
    """Relative weights within the non-curated group."""
    words = [data.all_words[i] for i in np.flatnonzero(mask)]
    rel = np.zeros(len(mask), dtype=np.float64)

    if shape == "frequency":
        table = _table(ANSWER_LIKE_PATH)
        score = np.array([table.get(w, -50.0) for w in words])
        rel[mask] = np.exp(FREQ_TEMPERATURE * (score - score.max()))
        return rel

    # 'wordlebot': the mass goes to the words WordleBot considers answerable,
    # in proportion to its weight, less a hedge spread over everything else
    table = _table(WORDLEBOT_PATH)
    listed = np.array([table.get(w, 0.0) for w in words])
    inside = listed > 0.0
    if not inside.any():
        return None
    share = np.zeros(len(words), dtype=np.float64)
    share[inside] = (1.0 - OFFLIST_HEDGE) * listed[inside] / listed[inside].sum()
    if (~inside).any():
        share[~inside] = OFFLIST_HEDGE / int((~inside).sum())
    rel[mask] = share
    return rel


def build_weights(data, config):
    """Return the prior over answers as a length-``n_words`` probability vector.

    Every word keeps a strictly positive weight so that log-weights stay finite.
    """
    n = data.n_words

    used = np.zeros(n, dtype=bool)
    for word in data.past_solutions:
        ind = data.word2ind.get(word)
        if ind is not None:
            used[ind] = True

    curated = np.zeros(n, dtype=bool)
    for word in data.curated_list:
        ind = data.word2ind.get(word)
        if ind is not None:
            curated[ind] = True

    # (mask, mass, shape) triples; shape None spreads the mass uniformly
    groups = []
    if config.repeat:
        shape = _repeat_shape(data, used) if config.repeat_shape == "binned" else None
        groups.append((used, config.repeat_prob, shape))
        rest, rest_mass = ~used, 1.0 - config.repeat_prob
    else:
        rest, rest_mass = np.ones(n, dtype=bool), 1.0

    if config.curation:
        noncurated = rest & ~curated
        shape = (None if config.curation_shape == "uniform"
                 else _curation_shape(data, noncurated, config.curation_shape))
        groups.append((rest & curated, rest_mass * (1.0 - config.noncurated_prob), None))
        groups.append((noncurated, rest_mass * config.noncurated_prob, shape))
    else:
        groups.append((rest, rest_mass, None))

    weights = np.zeros(n, dtype=np.float64)
    for mask, mass, shape in groups:
        count = int(mask.sum())
        if count == 0:  # e.g. every curated word has already been used
            continue
        if shape is None:
            weights[mask] = mass / count
            continue
        rel = shape[mask]
        total = rel.sum()
        weights[mask] = mass * rel / total if total > 0.0 else mass / count

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("prior has no mass; check the prior probabilities")
    weights /= total

    if not np.all(weights > 0.0):
        raise ValueError("prior assigns zero weight to some word")
    return weights
