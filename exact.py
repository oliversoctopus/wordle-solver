"""Exact endgame search.

Once the candidate set is small the value function can be computed outright
instead of estimated:

    V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
    V(S, 0) = loss_penalty

with ``t`` the number of guesses left including this one.  Scoring a lost game
as 7 is the same as ``loss_penalty = 1`` -- running out of turns costs one more
unit than the six already spent -- so the deadline needs no separate objective.
Raising the penalty buys solve rate at the cost of mean score.

Note that V is non-*decreasing* in t under this convention: with more turns you
are obliged to actually finish rather than paying the (comparatively cheap)
penalty.  That is what makes the two-turn value a valid lower bound for pruning.

Four things keep it tractable, the last of which trades the proof of optimality
for a large amount of time:

* **partition dedupe** -- two guesses that split S identically are
  interchangeable, so the branching factor is the number of distinct splits of
  S rather than the ~13k words.  A guess that is itself a candidate always has
  a distinct split, since only it can produce GGGGG, so the dedupe never merges
  a possible answer with a probe.  Note this only bites while S is *very*
  small: at 8 candidates about 10.6k of the 13k guesses already split S
  differently, and by 12 candidates it is 12.7k, so beyond that the dedupe buys
  almost nothing and the branching factor is effectively the whole word list.
  That, not the node count, is what makes the occasional deep position slow --
  a 12-candidate set with four turns left can take tens of seconds.
* **exact leaves** -- one and two turns left, and one and two candidates left,
  are closed forms, so the recursion only has nodes at t >= 3 with three or
  more candidates.
* **branch and bound** -- guesses are ordered by their two-turn value (a lower
  bound), and within a guess each bucket's bound is replaced by its exact value
  one at a time, abandoning the guess as soon as the running total passes the
  incumbent.
* **branch cap** (``max_branch``) -- only the best ``max_branch`` splits by that
  bound are tried at all.  This is what tames the tail described above, and it
  is worth being clear about why it costs so little: branch and bound finds the
  best guess early and then spends nearly all of its time *proving* that nothing
  else beats it.  The cap buys the answer and skips the proof.

  Measured over 1275 real positions across the four priors, against the
  uncapped optimum: a cap of 25 or 50 picks the optimal guess every time, while
  a cap of 10 misses 5 of 1275 (mean regret 0.0002 guesses, worst 0.053).  The
  default of 50 leaves a margin over the smallest value that measured clean.
  Wall clock over those positions falls from 322s to 8s, and the worst single
  position from 30s to 1.9s.

  The value returned is therefore an upper bound on the true one rather than
  the true one, and ``solve`` is optimal-within-the-cap, not provably optimal.
  Set ``max_branch=None`` to get the proof back.
"""

import numpy as np

import kernels

MAX_TURNS = 6
# above this many candidates the packed signature no longer fits in an int64
MAX_PACKED = 7

# how many splits a node may try, best-bound first.  Set to None to search every
# one and get a provable optimum; see the note on the branch cap above.
DEFAULT_MAX_BRANCH = 50


class SearchBudgetExceeded(Exception):
    """Raised when a search visits more nodes than allowed; the caller falls
    back to the heuristic policy."""


class ExactEndgame:
    """Exact solver for a fixed prior.  Values depend only on the weights, so
    the memo survives changes to the estimator.

    ``hard`` narrows every node's choice to the candidates themselves, which is
    the exact search's half of the solver's hard mode.
    """

    # backstop only; with the branch cap on, real positions use a handful
    DEFAULT_MAX_NODES = 2000

    def __init__(self, feedback_table, weights, loss_penalty=1.0,
                 max_nodes=DEFAULT_MAX_NODES, memo_limit=500_000,
                 max_branch=DEFAULT_MAX_BRANCH, hard=False):
        self.feedback_table = feedback_table
        self.weights = weights
        self.hard = bool(hard)
        self.loss_penalty = float(loss_penalty)
        # the bound stays valid for any penalty: a guess costs 1, and being
        # wrong costs at least one more guess or the penalty, whichever is less
        self.bound_penalty = min(self.loss_penalty, 1.0)
        self.max_nodes = max_nodes
        self.max_branch = max_branch
        self.memo_limit = memo_limit
        self.memo = {}
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
            return int(candidates[heaviest]), 2.0 - p_max
        key = (candidates.tobytes(), turns)
        cached = self.memo.get(key)
        if cached is not None:
            return cached

        # the same two-turn position is reached along many different guess
        # orders -- four calls in five are repeats -- so it is memoized like any
        # other, and the closed form below runs once per distinct position
        if turns == 2:
            pool = candidates if self.hard else None
            scores = kernels.two_turn_scores(self.feedback_table, candidates,
                                             weights, self.loss_penalty, pool=pool)
            best = int(scores.argmin())
            guess = int(candidates[best]) if self.hard else best
            result = (guess, float(scores[best]))
            if len(self.memo) < self.memo_limit:
                self.memo[key] = result
            return result

        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchBudgetExceeded(f"more than {self.max_nodes} nodes")

        if self.hard:
            # nothing for the dedupe to merge: only a candidate itself can come
            # back all-green, so no two of them split S the same way
            representatives = candidates
            rep_bounds = kernels.two_turn_scores(self.feedback_table, candidates,
                                                 weights, self.bound_penalty,
                                                 pool=candidates)
        else:
            # bound only the representatives: they are about a quarter of the
            # pool, and scoring the rest would be scoring duplicate splits
            representatives = self._distinct_partitions(candidates)
            rep_bounds = kernels.two_turn_scores(self.feedback_table, candidates,
                                                 weights, self.bound_penalty,
                                                 pool=representatives)
        order = np.argsort(rep_bounds)
        if self.max_branch is not None:
            # keep only the most promising splits.  Where the dedupe already
            # did its job this never binds, so the endgame stays exact and only
            # the wide positions -- the ones that cost seconds -- are truncated.
            order = order[:self.max_branch]

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
        if len(self.memo) < self.memo_limit:
            self.memo[key] = result
        return result

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
                running += share * (2.0 - sub_p_max)
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

    def _distinct_partitions(self, candidates):
        """One representative guess per distinct split of the candidate set."""
        if candidates.shape[0] <= MAX_PACKED:
            # the packed key *is* the split, so equal keys need no checking
            keys = kernels.signature_keys(self.feedback_table, candidates)
            _, first = np.unique(keys, return_index=True)
            return first
        return kernels.distinct_partitions(self.feedback_table, candidates)
