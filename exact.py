"""Exact endgame search.

Once the candidate set is small the value function is computed outright:

    V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
    V(S, 0) = loss_penalty

with ``t`` the guesses left including this one.  README *The exact endgame*
has what keeps it affordable and what each part costs.
"""

import os

import numpy as np

import fused_kernels
import kernels
from memo import Memo
from wordle_data import MAX_GUESSES

MAX_TURNS = MAX_GUESSES
# above this many candidates the packed signature no longer fits in an int64
MAX_PACKED = 7

# how many splits a node may try, best-bound first; None searches every one
DEFAULT_MAX_BRANCH = 50

# how many the node being *ranked* may try; WORDLE_ROOT_BRANCH narrows it, and
# the Rust reads the same variable
DEFAULT_ROOT_BRANCH = int(os.environ.get("WORDLE_ROOT_BRANCH") or 150)

# a backstop; with the branch cap on, real positions use a handful
DEFAULT_MAX_NODES = 2000

# two guesses that split the candidates identically score identically, so the
# feedback-vector comparison is gated at equality
SPLIT_TIE_TOL = 0.0

# the branch cap orders on the bound rounded to this, and a tie goes to the
# lower word index; `exact.rs` rounds the same way
BRANCH_TIE_TOL = 1e-11


class SearchBudgetExceeded(Exception):
    """Raised when a search visits more nodes than allowed; the caller falls
    back to the heuristic policy."""


class ExactEndgame:
    """Exact solver for a fixed prior.  Values depend only on the weights, so
    the memo survives changes to the estimator.

    ``hard`` narrows every node's choice to the candidates themselves, which is
    the exact search's half of the solver's hard mode.
    """

    def __init__(self, feedback_table, weights, loss_penalty=1.0,
                 max_nodes=DEFAULT_MAX_NODES, memo_limit=500_000,
                 max_branch=DEFAULT_MAX_BRANCH,
                 root_branch=DEFAULT_ROOT_BRANCH, hard=False, transposed=None):
        self.feedback_table = feedback_table
        # answer-major copy, or None to score the shipped way
        self.transposed = transposed
        self.weights = weights
        self.hard = bool(hard)
        self.loss_penalty = float(loss_penalty)
        # capping at 1 is what makes V non-decreasing in t: running out then
        # costs exactly one guess, so no deeper value undercuts the two-turn one
        self.bound_penalty = min(self.loss_penalty, 1.0)
        self.max_nodes = max_nodes
        self.max_branch = max_branch
        # asking for the proof means asking for it at the ranked node too, or
        # the list is still the capped one
        self.root_branch = None if max_branch is None else root_branch
        self.memo = Memo(memo_limit)
        self.nodes = 0
        self.searches = 0

    # ------------------------------------------------------------------ entry

    def solve(self, candidates, turns_left):
        """``(guess, expected score)`` under optimal play.  Raises
        SearchBudgetExceeded if the node budget runs out."""
        self.searches += 1
        self.nodes = 0
        return self._solve(np.asarray(candidates, dtype=np.int64),
                           min(int(turns_left), MAX_TURNS))

    def top(self, candidates, turns_left, k=5):
        """``[(guess, expected score), ...]`` for the k best splits of this
        position, best first, each score as true as ``solve``'s.

        The k-th best value stands in for the incumbent, so a guess is still
        abandoned the moment it cannot make the list.  Asking for several
        therefore costs more than asking for one -- the cutoff is looser, so
        fewer guesses die on their bound -- and much less than ranking the pool.

        Guesses are the search's own representatives: one word per distinct
        split, since the others are the same guess by another name.
        """
        self.searches += 1
        self.nodes = 0
        return self._top(np.asarray(candidates, dtype=np.int64),
                         min(int(turns_left), MAX_TURNS), max(1, int(k)))

    # ------------------------------------------------------------- recursion

    def _solve(self, candidates, turns):
        m = candidates.shape[0]
        if m == 0:
            raise ValueError("no candidates left; the feedback given is inconsistent")
        weights = self.weights[candidates]
        total = weights.sum()
        heaviest = int(weights.argmax())
        p_max = float(weights[heaviest] / total)

        if m == 1:
            return int(candidates[0]), 1.0
        if turns <= 1:
            # nothing to learn from the feedback; play the likeliest candidate
            return int(candidates[heaviest]), 1.0 + (1.0 - p_max) * self.loss_penalty
        if m == 2:
            # guess the likelier one; if wrong the other is known
            return int(candidates[heaviest]), kernels.two_candidates(p_max)
        key = (candidates.tobytes(), turns)
        cached = self.memo.get(key)
        if cached is not None:
            return cached

        # the same two-turn position is reached along many different guess
        # orders -- four calls in five are repeats -- so it is memoized like any
        # other, and the closed form below runs once per distinct position
        if turns == 2:
            pool = candidates if self.hard else None
            scores = fused_kernels.two_turn_scores(
                self.feedback_table, self.transposed, candidates, weights,
                self.loss_penalty, pool=pool)
            best = int(scores.argmin())
            guess = int(candidates[best]) if self.hard else best
            result = (guess, float(scores[best]))
            self.memo.put(key, result)
            return result

        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchBudgetExceeded(f"more than {self.max_nodes} nodes")

        representatives, rep_bounds = self._bounded_splits(candidates, weights)
        order = self._branch_order(rep_bounds, self.max_branch)

        best_value = np.inf
        best_guess = -1
        for j in order:
            if rep_bounds[j] >= best_value - 1e-12:
                break  # sorted, so nothing left can beat the incumbent
            guess = int(representatives[j])
            value = self._evaluate(guess, candidates, weights, total, turns, best_value)
            if value < best_value:
                best_value, best_guess = value, guess

        result = (best_guess, float(best_value))
        self.memo.put(key, result)
        return result

    def _top(self, candidates, turns, k):
        m = candidates.shape[0]
        if m == 0:
            raise ValueError("no candidates left; the feedback given is inconsistent")
        weights = self.weights[candidates]
        total = weights.sum()

        if m == 1:
            return [(int(candidates[0]), 1.0)]
        if turns <= 1:
            # every word that cannot be the answer loses alike, so the ranking
            # here is the candidates by weight and nothing else.  Stable, so
            # that equal weights leave the winner the one ``_solve`` names
            order = np.argsort(-weights, kind="stable")[:k]
            return [(int(candidates[i]),
                     1.0 + (1.0 - float(weights[i] / total)) * self.loss_penalty)
                    for i in order]
        if m == 2:
            # either order wins by the second turn; leading with the likelier
            # one is what makes the difference
            order = np.argsort(-weights, kind="stable")
            return [(int(candidates[i]),
                     kernels.two_candidates(float(weights[i] / total)))
                    for i in order]
        if turns == 2:
            # the closed form scores every guess rather than the distinct
            # splits, so the ranking is deduped on the way out; the winner is
            # left where it is, which is where ``_solve`` has it
            pool = candidates if self.hard else None
            scores = fused_kernels.two_turn_scores(
                self.feedback_table, self.transposed, candidates, weights,
                self.loss_penalty, pool=pool)
            out = []
            for j in np.argsort(scores, kind="stable"):
                guess = int(candidates[j]) if self.hard else int(j)
                if self._duplicates(guess, float(scores[j]), out, candidates):
                    continue
                out.append((guess, float(scores[j])))
                if len(out) == k:
                    break
            return out

        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchBudgetExceeded(f"more than {self.max_nodes} nodes")

        # the node the caller asked about is the only one that ranks, and
        # ``_evaluate`` recurses into ``_solve``, so this is the only ``top``
        representatives, rep_bounds = self._bounded_splits(candidates, weights)
        order = self._branch_order(rep_bounds, self.root_branch)

        best = []   # (value, guess), best first, never longer than k
        for j in order:
            cutoff = best[-1][0] if len(best) == k else np.inf
            if rep_bounds[j] >= cutoff - 1e-12:
                break  # sorted, so nothing left can reach the list
            guess = int(representatives[j])
            value = self._evaluate(guess, candidates, weights, total, turns, cutoff)
            if value < cutoff:
                best.append((value, guess))
                best.sort(key=lambda entry: entry[0])
                del best[k:]

        # the winner was found under a cutoff no tighter than solve's, so it is
        # the same word solve would return and worth leaving in the memo
        self.memo.put((candidates.tobytes(), turns), (best[0][1], float(best[0][0])))
        return [(guess, float(value)) for value, guess in best]

    def _branch_order(self, bounds, cap):
        """Splits in the order to try them, best bound first, keeping the best
        ``cap``.  Where the dedupe already did its job the cap never binds, so
        the endgame stays exact and only the wide positions are truncated.

        Ordered on the bound rounded to ``BRANCH_TIE_TOL`` and stable, so a tie
        goes to the lower word index rather than to whatever quicksort leaves it
        at.  The bounds tie in bulk -- a position can hold a thousand splits at
        one value -- so which of them the cap keeps rests entirely on this, and
        ``exact.rs`` has to be able to spell the same rule.

        Rounded rather than floored because the ties sit on whole values like
        2.0, which a floor would put a bucket edge through; ``floor(x + 0.5)``
        rather than ``round`` because numpy rounds a half to even and Rust
        rounds it away from zero, where floor is IEEE in both.
        """
        keys = np.floor(np.asarray(bounds, dtype=np.float64) / BRANCH_TIE_TOL
                        + 0.5)
        order = np.argsort(keys, kind="stable")
        return order if cap is None else order[:cap]

    def _duplicates(self, guess, score, listed, candidates):
        """Whether ``guess`` splits the candidates as something already listed
        does, and so would be that guess again under another spelling."""
        return any(abs(score - value) <= SPLIT_TIE_TOL
                   and np.array_equal(self.feedback_table[guess, candidates],
                                      self.feedback_table[other, candidates])
                   for other, value in listed)

    def _evaluate(self, guess, candidates, weights, total, turns, cutoff):
        """Exact value of one guess, abandoned early if it cannot beat
        ``cutoff``.  Starts from the per-bucket lower bounds and replaces them
        with exact values one bucket at a time."""
        patterns = self.feedback_table[guess, candidates]

        buckets = []
        running = 1.0
        for pattern in np.unique(patterns):
            if pattern == kernels.WIN_PATTERN:
                continue
            members = candidates[patterns == pattern]
            sub_weights = self.weights[members]
            mass = float(sub_weights.sum())
            share = mass / total
            size = members.shape[0]
            if size == 1:
                running += share  # certain win next turn
                continue
            sub_p_max = float(sub_weights.max() / mass)
            if size == 2:
                running += share * kernels.two_candidates(sub_p_max)
                continue
            bound = 1.0 + (1.0 - sub_p_max) * self.bound_penalty
            running += share * bound
            buckets.append((share, bound, members))

        if running >= cutoff:
            return np.inf
        # biggest contributors first: they tighten the total fastest
        buckets.sort(key=lambda b: -b[0])
        for share, bound, members in buckets:
            _, value = self._solve(members, turns - 1)
            running += share * (value - bound)
            if running >= cutoff:
                return np.inf
        return running

    # ------------------------------------------------------------------ split

    def _bounded_splits(self, candidates, weights):
        """``(representatives, two-turn bounds)`` -- one guess per distinct
        split, each with the lower bound the search orders and prunes by."""
        if self.hard:
            # nothing for the dedupe to merge: only a candidate itself can come
            # back all-green, so no two of them split S the same way
            representatives = candidates
        else:
            # bound only the representatives: they are about a quarter of the
            # pool, and scoring the rest would be scoring duplicate splits
            representatives = self._distinct_partitions(candidates)
        bounds = fused_kernels.two_turn_scores(
            self.feedback_table, self.transposed, candidates, weights,
            self.bound_penalty, pool=representatives)
        return representatives, bounds

    def _distinct_partitions(self, candidates):
        """One representative guess per distinct split of the candidate set."""
        if candidates.shape[0] <= MAX_PACKED:
            # the packed key *is* the split, so equal keys need no checking
            keys = kernels.signature_keys(self.feedback_table, candidates)
            _, first = np.unique(keys, return_index=True)
            return first
        return kernels.distinct_partitions(self.feedback_table, candidates)
