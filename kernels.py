"""Numba kernels that score guesses against a candidate set.

All three scorers share one shape: for each guessable word, bucket the current
candidates by the feedback pattern they would produce, then reduce those
buckets to a single number.  Entropy reduces them to bits gained, the expected
guess count reduces them through the fitted estimator f, and the two-turn value
reduces them to the exact score when two guesses remain.

Small candidate sets are scored far more often than large ones.  Every node of
the exact search and every late turn of every rollout scans a handful of
candidates against all ~13k guesses, and a scan carrying a cost fixed at the 243
possible patterns would spend all its time there.  Three things avoid that.  None
of them adds to the accumulate loop, which runs once per (guess, candidate) pair
and stays branch-free:

* the per-pattern accumulators are allocated once per block of guesses, not
  once per guess;
* below ``_RESCAN_MAX`` candidates the reduce walks the candidates a second
  time instead of sweeping all 243 patterns, taking each bucket the first time
  it is reached and zeroing it there.  The reduce *and* the clear for the next
  guess then both cost the number of candidates;
* ``w * log2(w)`` depends only on the candidate and never on the guess.
  Computing it once per candidate instead of once per (guess, candidate) pair
  was, on its own, most of the cost of scoring a wide position.

What is left is bound by memory, not arithmetic.  ``feedback_table[guess, c]``
for scattered ``c`` spends a whole cache line per candidate to use one byte of
it, and a small scattered candidate set costs several times what the same number
of adjacent candidates would.  That is why the second pass above is nearly free:
it re-reads what the first pass has already pulled in.

That penalty is bounded at both ends, which is most of why the layout is what it
is.  A row is 203 cache lines, so past a couple of hundred candidates a
scattered set can touch no more of one than a dense set does, and below about
twenty-five there is too little traffic either way to matter.  Indexing
``[answer, guess]`` instead does make the reads contiguous, and over the
positions in between it is worth between 1.2x and 2.4x on the sets real play
produces.  What it cannot do is win the widest ones, and that part is
structural: reducing a guess needs that guess's whole feedback vector, which
that layout holds as a strided column, so it has to either keep every guess in
the block accumulating at once -- putting the accumulator out of L1 -- or
rebuild the vector by transposing a tile, and both are additive on top of a scan
already reading the table a full cache line at a time.  Neither ordering
dominates the other.  ``analysis/table_layout.py`` measures it.

None of these carry ``fastmath``, and they must not.  The reduce reads a bucket
and then zeroes it for the next guess; under fastmath LLVM reorders the store
ahead of the load, and every bucket maximum reads back as zero.  It shows up as
``two_turn_scores`` returning 3.0 everywhere and the search preferring nonsense
words.  The two other scorers happen to survive only because a log or a call in
the reduce blocks the same transform, which is not something to rely on.
Leaving it off costs nothing on the widest positions and up to about 10% in the
middle.  That is a small price for scores that are what they claim.

Every kernel takes a ``pool`` of row indices into the feedback table and returns
one score per pool entry, in pool order; ``pool=None`` means every row.  Scoring
a restricted pool is what keeps a deeper search affordable, since a lookahead
never needs all 13k replies to a bucket of ninety candidates.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from estimator import f_eval_x
from wordle_data import N_PATTERNS, WIN_PATTERN  # 243, and "GGGGG" = 242

# guesses handed to one parallel work unit: small enough that a short pool still
# spreads across the threads, large enough that the per-block allocation is lost
# in the noise
_MAX_BLOCK = 64

# at or above this many candidates, the reduce sweeps all 243 patterns; below
# it, it walks the candidates a second time instead (see the kernels).  The test
# is on the candidate count, which is fixed for the whole scan, so the branch
# costs nothing per guess.  SOLVER_EXPLAINED §8 has why the threshold sits here
# and why the two branches disagreeing in their last bits is harmless.
_RESCAN_MAX = 128

_ALL_ROWS = {}


def _resolve_pool(n_rows, pool):
    """``(rows to score, guesses per work unit)``."""
    if pool is None:
        rows = _ALL_ROWS.get(n_rows)
        if rows is None:
            rows = np.arange(n_rows, dtype=np.int64)
            _ALL_ROWS[n_rows] = rows
        pool = rows
    else:
        pool = np.ascontiguousarray(pool, dtype=np.int64)
    n_units = 4 * get_num_threads()  # a few blocks each, so stragglers even out
    block = max(1, min(_MAX_BLOCK, -(-pool.shape[0] // n_units)))
    return pool, block


# ------------------------------------------------------------------- entropy

@njit(parallel=True, cache=True)
def _ig(feedback_table, candidates, weights, pool, block):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for b in prange(n_blocks):
        mass = np.zeros(N_PATTERNS)
        stop = min((b + 1) * block, n_out)

        for j in range(b * block, stop):
            guess = pool[j]
            for i in range(m):
                mass[feedback_table[guess, candidates[i]]] += weights[i]

            h = 0.0
            if m < _RESCAN_MAX:
                # walk the candidates again rather than all 243 patterns: each
                # bucket is reduced the first time it is reached and zeroed on
                # the spot, which both clears it for the next guess and stops it
                # being counted twice
                for i in range(m):
                    r = feedback_table[guess, candidates[i]]
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
            out[j] = h
    return out


def ig_all(feedback_table, candidates, weights, pool=None):
    """Expected information gain, in bits, of every pooled guess."""
    pool, block = _resolve_pool(feedback_table.shape[0], pool)
    return _ig(feedback_table, candidates, weights, pool, block)


# ------------------------------------------------------------ expected guesses

@njit(cache=True)
def _w_log_w(weights):
    """``w * log2(w)`` per candidate, the guess-independent half of a bucket's
    entropy.  Zero weights contribute nothing, as 0 log 0 = 0."""
    out = np.empty(weights.shape[0])
    for i in range(weights.shape[0]):
        w = weights[i]
        out[i] = w * np.log2(w) if w > 0.0 else 0.0
    return out


@njit(cache=True, inline="always")
def _bucket_guesses(mass, count, wlw, heaviest, params, codes, knots, base):
    """Guesses still needed from one bucket.  One and two candidates are worth
    exactly 1 and 2 - p_max, so the estimator is only consulted above that."""
    if count == 1.0:
        return 1.0
    if count == 2.0:
        # guess the likelier one: right with probability p_max, else one more
        return 2.0 - heaviest / mass
    return f_eval_x(np.log2(mass) - wlw / mass, params, codes, knots, base)


@njit(parallel=True, cache=True)
def _expected_guesses(feedback_table, candidates, weights, wlw,
                      params, codes, knots, base, pool, block):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for b in prange(n_blocks):
        mass = np.zeros(N_PATTERNS)
        count = np.zeros(N_PATTERNS)
        w_log_w = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)
        stop = min((b + 1) * block, n_out)

        for j in range(b * block, stop):
            guess = pool[j]
            for i in range(m):
                r = feedback_table[guess, candidates[i]]
                w = weights[i]
                mass[r] += w
                count[r] += 1.0
                w_log_w[r] += wlw[i]
                if w > heaviest[r]:
                    heaviest[r] = w

            expected = 1.0
            if m < _RESCAN_MAX:
                # as in _ig: reduce each bucket the first time the second pass
                # reaches it, clearing it on the spot
                for i in range(m):
                    r = feedback_table[guess, candidates[i]]
                    if count[r] > 0.0:
                        if r != WIN_PATTERN:
                            expected += mass[r] / total * _bucket_guesses(
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
                            expected += mass[r] / total * _bucket_guesses(
                                mass[r], count[r], w_log_w[r], heaviest[r],
                                params, codes, knots, base)
                        mass[r] = 0.0
                        count[r] = 0.0
                        w_log_w[r] = 0.0
                        heaviest[r] = 0.0
            out[j] = expected
    return out


def expected_guesses_all(feedback_table, candidates, weights, params,
                         codes, knots, base, pool=None):
    """Estimated total guesses to finish, for every pooled guess:

        E(g, S) = 1 + sum_{r != GGGGG} p_r * f(n_r)

    ``p_r`` is the prior mass of bucket r relative to the whole candidate set,
    and ``n_r`` is that bucket's effective size.  The GGGGG bucket contributes
    nothing, which is what makes a guess that could itself be the answer come
    out ahead of an equally informative guess that could not.
    """
    pool, block = _resolve_pool(feedback_table.shape[0], pool)
    wlw = _w_log_w(weights)
    return _expected_guesses(feedback_table, candidates, weights, wlw,
                             params, codes, knots, base, pool, block)


# -------------------------------------------------------------- two turns left

@njit(parallel=True, cache=True)
def _two_turn(feedback_table, candidates, weights, penalty, pool, block):
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for b in prange(n_blocks):
        mass = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)
        stop = min((b + 1) * block, n_out)

        for j in range(b * block, stop):
            guess = pool[j]
            for i in range(m):
                r = feedback_table[guess, candidates[i]]
                w = weights[i]
                mass[r] += w
                if w > heaviest[r]:
                    heaviest[r] = w

            mass_sum = 0.0
            max_sum = 0.0
            if m < _RESCAN_MAX:
                # as in _ig; a bucket of zero mass has a zero maximum too, so
                # mass alone says whether this bucket has already been taken
                for i in range(m):
                    r = feedback_table[guess, candidates[i]]
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
            out[j] = 1.0 + (mass_sum + (mass_sum - max_sum) * penalty) / total
    return out


def two_turn_scores(feedback_table, candidates, weights, penalty, pool=None):
    """Exact expected score of every pooled guess when two turns remain.

    With one turn left the best you can do in a bucket is play its heaviest
    member, worth ``1 + (1 - p_max) * penalty``, so two-turn play needs no
    estimator at all:

        V(g, 2) = 1 + sum_{r != GGGGG} p_r * (1 + (1 - p_max_r) * penalty)
                = 1 + (A + (A - B) * penalty) / total

    with A the mass and B the sum of bucket maxima over the non-winning
    buckets.  Called with ``penalty = min(loss_penalty, 1)`` this is also a
    valid lower bound on the value at any larger number of turns, which is what
    the exact search prunes with.
    """
    pool, block = _resolve_pool(feedback_table.shape[0], pool)
    return _two_turn(feedback_table, candidates, weights, penalty, pool, block)


# --------------------------------------------------------------------- splits

@njit(parallel=True, cache=True)
def signature_keys(feedback_table, candidates):
    """Pack each guess's feedback pattern over the candidates into one integer.

    Guesses sharing a key split the candidate set identically and are
    interchangeable, which is what collapses the exact search's branching
    factor from ~13k to the number of distinct partitions.  Needs 8 bits per
    candidate, so callers must keep to at most 7 candidates.
    """
    n_guesses = feedback_table.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_guesses, dtype=np.int64)
    for guess in prange(n_guesses):
        key = 0
        for i in range(m):
            key = (key << 8) | feedback_table[guess, candidates[i]]
        out[guess] = key
    return out


_FNV_OFFSET = np.int64(-3750763034362895579)  # 14695981039346656037, wrapped
_FNV_PRIME = np.int64(1099511628211)


@njit(parallel=True, cache=True)
def _partition_hash(feedback_table, candidates):
    """FNV-1a over each guess's feedback vector, for grouping equal splits."""
    n_guesses = feedback_table.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_guesses, dtype=np.int64)
    for guess in prange(n_guesses):
        h = _FNV_OFFSET
        for i in range(m):
            h = (h ^ np.int64(feedback_table[guess, candidates[i]])) * _FNV_PRIME
        out[guess] = h
    return out


@njit(parallel=True, cache=True)
def _splits_differently(feedback_table, candidates, heads):
    """Which guesses split the candidates differently from ``heads[guess]``."""
    n_guesses = heads.shape[0]
    m = candidates.shape[0]
    out = np.zeros(n_guesses, dtype=np.bool_)
    for guess in prange(n_guesses):
        head = heads[guess]
        if head != guess:
            for i in range(m):
                c = candidates[i]
                if feedback_table[guess, c] != feedback_table[head, c]:
                    out[guess] = True
                    break
    return out


@njit(parallel=True, cache=True)
def _same_split(feedback_table, candidates, guess, pool):
    """Which pooled guesses split the candidates exactly as ``guess`` does."""
    n = pool.shape[0]
    m = candidates.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for slot in prange(n):
        other = pool[slot]
        same = True
        for i in range(m):
            c = candidates[i]
            if feedback_table[other, c] != feedback_table[guess, c]:
                same = False
                break
        out[slot] = same
    return out


def same_split(feedback_table, candidates, guess, pool=None):
    """The words interchangeable with ``guess`` here: same feedback against
    every candidate, so the same buckets for the rest of the game.

    ``guess`` itself is in the result.  Only a candidate can come back all
    green, and only against itself, so such a group holds either one possible
    answer or any number of words that cannot be the answer.
    """
    pool, _ = _resolve_pool(feedback_table.shape[0], pool)
    return pool[_same_split(feedback_table, candidates, guess, pool)]


def distinct_partitions(feedback_table, candidates):
    """One representative guess per distinct split of the candidate set.

    Guesses are grouped by a 64-bit hash of their feedback vector.  Sorting the
    vectors themselves (``np.unique(..., axis=0)``) is a lexicographic sort with
    a pass per candidate, and costs an order of magnitude more.

    A hash collision would merge two different splits, so each guess is then
    checked against its group's representative and promoted to one of its own if
    it really does split differently.  The result is therefore exact, not
    probabilistic: a collision can only ever cost a redundant representative,
    never a missing one.
    """
    keys = _partition_hash(feedback_table, candidates)
    first, inverse = np.unique(keys, return_index=True, return_inverse=True)[1:]
    heads = first[inverse.reshape(-1)]
    extra = np.nonzero(_splits_differently(feedback_table, candidates, heads))[0]
    return first if extra.size == 0 else np.concatenate((first, extra))


@njit(cache=True)
def filter_candidates(feedback_table, guess, pattern, candidates):
    """Candidates still consistent with seeing ``pattern`` after ``guess``."""
    m = candidates.shape[0]
    keep = np.empty(m, dtype=np.int64)
    n_kept = 0
    for i in range(m):
        if feedback_table[guess, candidates[i]] == pattern:
            keep[n_kept] = candidates[i]
            n_kept += 1
    return keep[:n_kept]
