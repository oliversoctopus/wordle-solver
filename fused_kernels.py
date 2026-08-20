"""Scoring kernels that read an answer-major copy of the feedback table.

The shipped kernels in `kernels.py` fix a guess and walk the candidates, which
`feedback_table[guess, answer]` makes a row: contiguous for a whole candidate
set, and a cache line fetched per candidate for a scattered handful.  That is
the bill at every mid-sized position, and an answer-major copy is what pays it
off -- one candidate against a contiguous run of guesses.

Reading that way on its own is not enough, because the reduce is per guess and
walking guesses in the inner loop leaves all of them accumulating at once, which
puts the accumulator out of L1 and costs more than the reads save.  These
kernels avoid the trade instead of taking it.  For a block of G guesses they
read the G x m tile out of the answer-major table -- contiguous, G bytes per
candidate -- and transpose it into a small buffer on the way in.  Each guess's
feedback vector is then a contiguous row of that tile, so the reduce is exactly
the shipped one, over a single 243-bucket accumulator reused across the block.

Contiguous reads and an L1-resident accumulator, at the price of writing and
re-reading the tile.  That price is nothing while the tile is in cache and grows
with the candidate count, which is why `FUSED_MAX` exists: above it the shipped
kernels are faster, and at the widest positions they cannot be beaten at all,
since there they already read the table a full cache line at a time.

Everything here is a fallback to `kernels` when the transposed table is absent
or the position is outside the band, and the summation order is the shipped
order throughout, so scores are bit-identical rather than merely close.
`analysis/table_layout.py` is where the band and the block rule come from.
"""

import os

import numpy as np
from numba import get_num_threads, njit, prange

import kernels
from estimator import f_eval_x
from wordle_data import N_PATTERNS, WIN_PATTERN

# candidate counts the fused path is used for.  Below the floor a position is
# too cheap for the dispatch to be worth taking; above the ceiling the tile
# stops fitting in cache and the shipped kernels win.  SOLVER_EXPLAINED
# *Storing the table the other way round* has where the ceiling comes from.
FUSED_MIN = 13
FUSED_MAX = 600

# guesses per tile.  Wider tiles amortize the transpose but have to stay in
# cache, so the block is cut back as the candidate set grows.
TILE_BYTES = 32 * 1024
MIN_BLOCK = 16
MAX_BLOCK = 256

# set WORDLE_FUSED=1 to make it the default for every solver in the process
DEFAULT_ENABLED = os.environ.get("WORDLE_FUSED") == "1"


def build_transposed(feedback_table):
    """The answer-major copy, `tt[answer, guess]`.  About 0.3 s and another
    168 MB; there is no file for it, and it is not worth one."""
    if feedback_table is None:
        return None
    return np.ascontiguousarray(feedback_table.T)


def _plan(n_pool, m):
    """Guesses per tile, or None to leave the position to `kernels`."""
    if not FUSED_MIN <= m <= FUSED_MAX:
        return None
    block = min(MAX_BLOCK, max(MIN_BLOCK, TILE_BYTES // m))
    # a short pool must still reach every thread, or the fused path runs on one
    units = 4 * get_num_threads()
    block = max(MIN_BLOCK, min(block, -(-n_pool // units)))
    if n_pool < MIN_BLOCK:
        return None
    return block


def _resolve(feedback_table, pool):
    if pool is None:
        return np.arange(feedback_table.shape[0], dtype=np.int64)
    return np.ascontiguousarray(pool, dtype=np.int64)


# ------------------------------------------------------------------- entropy

@njit(parallel=True, cache=True)
def _ig(tt, candidates, weights, pool, block, threads):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        tile = np.empty((block, m), dtype=np.uint8)
        mass = np.zeros(N_PATTERNS)

        for b in range(t, n_blocks, threads):
            start = b * block
            here = min(start + block, n_out) - start

            for i in range(m):
                c = candidates[i]
                for j in range(here):
                    tile[j, i] = tt[c, pool[start + j]]

            for j in range(here):
                for i in range(m):
                    mass[tile[j, i]] += weights[i]

                h = 0.0
                if m < kernels._RESCAN_MAX:
                    for i in range(m):
                        r = tile[j, i]
                        if mass[r] > 0.0:
                            p = mass[r] / total
                            h -= p * np.log2(p)
                            mass[r] = 0.0
                else:
                    for r in range(N_PATTERNS):
                        if mass[r] > 0.0:
                            p = mass[r] / total
                            h -= p * np.log2(p)
                            mass[r] = 0.0
                out[start + j] = h
    return out


def ig_all(feedback_table, transposed, candidates, weights, pool=None):
    """`kernels.ig_all`, over the transposed table where that is faster."""
    if transposed is None:
        return kernels.ig_all(feedback_table, candidates, weights, pool=pool)
    rows = _resolve(feedback_table, pool)
    block = _plan(rows.shape[0], candidates.shape[0])
    if block is None:
        return kernels.ig_all(feedback_table, candidates, weights, pool=pool)
    return _ig(transposed, candidates, weights, rows, block, get_num_threads())


# ------------------------------------------------------------ expected guesses

@njit(cache=True, inline="always")
def _bucket(mass, count, wlw, heaviest, params, codes, knots, base):
    if count == 1.0:
        return 1.0
    if count == 2.0:
        return 2.0 - heaviest / mass
    return f_eval_x(np.log2(mass) - wlw / mass, params, codes, knots, base)


@njit(parallel=True, cache=True)
def _expected_guesses(tt, candidates, weights, wlw, params, codes, knots, base,
                      pool, block, threads):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        tile = np.empty((block, m), dtype=np.uint8)
        mass = np.zeros(N_PATTERNS)
        count = np.zeros(N_PATTERNS)
        w_log_w = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)

        for b in range(t, n_blocks, threads):
            start = b * block
            here = min(start + block, n_out) - start

            for i in range(m):
                c = candidates[i]
                for j in range(here):
                    tile[j, i] = tt[c, pool[start + j]]

            for j in range(here):
                for i in range(m):
                    r = tile[j, i]
                    w = weights[i]
                    mass[r] += w
                    count[r] += 1.0
                    w_log_w[r] += wlw[i]
                    if w > heaviest[r]:
                        heaviest[r] = w

                expected = 1.0
                if m < kernels._RESCAN_MAX:
                    for i in range(m):
                        r = tile[j, i]
                        if count[r] > 0.0:
                            if r != WIN_PATTERN:
                                expected += mass[r] / total * _bucket(
                                    mass[r], count[r], w_log_w[r], heaviest[r],
                                    params, codes, knots, base)
                            mass[r] = 0.0
                            count[r] = 0.0
                            w_log_w[r] = 0.0
                            heaviest[r] = 0.0
                else:
                    for r in range(N_PATTERNS):
                        if count[r] > 0.0:
                            if r != WIN_PATTERN:
                                expected += mass[r] / total * _bucket(
                                    mass[r], count[r], w_log_w[r], heaviest[r],
                                    params, codes, knots, base)
                            mass[r] = 0.0
                            count[r] = 0.0
                            w_log_w[r] = 0.0
                            heaviest[r] = 0.0
                out[start + j] = expected
    return out


def expected_guesses_all(feedback_table, transposed, candidates, weights,
                         params, codes, knots, base, pool=None):
    """`kernels.expected_guesses_all`, over the transposed table where faster."""
    if transposed is None:
        return kernels.expected_guesses_all(
            feedback_table, candidates, weights, params, codes, knots, base,
            pool=pool)
    rows = _resolve(feedback_table, pool)
    block = _plan(rows.shape[0], candidates.shape[0])
    if block is None:
        return kernels.expected_guesses_all(
            feedback_table, candidates, weights, params, codes, knots, base,
            pool=pool)
    return _expected_guesses(transposed, candidates, weights,
                             kernels._w_log_w(weights), params, codes, knots,
                             base, rows, block, get_num_threads())


# -------------------------------------------------------------- two turns left

@njit(parallel=True, cache=True)
def _two_turn(tt, candidates, weights, penalty, pool, block, threads):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        tile = np.empty((block, m), dtype=np.uint8)
        mass = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)

        for b in range(t, n_blocks, threads):
            start = b * block
            here = min(start + block, n_out) - start

            for i in range(m):
                c = candidates[i]
                for j in range(here):
                    tile[j, i] = tt[c, pool[start + j]]

            for j in range(here):
                for i in range(m):
                    r = tile[j, i]
                    w = weights[i]
                    mass[r] += w
                    if w > heaviest[r]:
                        heaviest[r] = w

                mass_sum = 0.0
                max_sum = 0.0
                if m < kernels._RESCAN_MAX:
                    for i in range(m):
                        r = tile[j, i]
                        if mass[r] > 0.0:
                            if r != WIN_PATTERN:
                                mass_sum += mass[r]
                                max_sum += heaviest[r]
                            mass[r] = 0.0
                            heaviest[r] = 0.0
                else:
                    for r in range(N_PATTERNS):
                        if mass[r] > 0.0:
                            if r != WIN_PATTERN:
                                mass_sum += mass[r]
                                max_sum += heaviest[r]
                            mass[r] = 0.0
                            heaviest[r] = 0.0
                out[start + j] = 1.0 + (mass_sum
                                        + (mass_sum - max_sum) * penalty) / total
    return out


def two_turn_scores(feedback_table, transposed, candidates, weights, penalty,
                    pool=None):
    """`kernels.two_turn_scores`, over the transposed table where faster."""
    if transposed is None:
        return kernels.two_turn_scores(feedback_table, candidates, weights,
                                       penalty, pool=pool)
    rows = _resolve(feedback_table, pool)
    block = _plan(rows.shape[0], candidates.shape[0])
    if block is None:
        return kernels.two_turn_scores(feedback_table, candidates, weights,
                                       penalty, pool=pool)
    return _two_turn(transposed, candidates, weights, penalty, rows, block,
                     get_num_threads())
