"""The solver itself: pick a guess, apply feedback, play a game out."""

import numpy as np

import estimator
import fused_kernels
import kernels
from exact import (DEFAULT_MAX_BRANCH, DEFAULT_ROOT_BRANCH, MAX_TURNS,
                   SPLIT_TIE_TOL, ExactEndgame,
                   SearchBudgetExceeded)
from memo import Memo
from priors import PriorConfig, build_weights
from wordle_data import MAX_GUESSES, WIN_PATTERN, load_data

ENTROPY = "entropy"
EXPECTED = "expected"
MODES = (ENTROPY, EXPECTED)

# price of being a possible answer, in bits, for the reference policy
# IG(g) + a*p_win(g); README *Results* says where the 2 comes from
DERIVED_WIN_BONUS = 2.0

DEFAULT_DEADLINE = MAX_GUESSES

# largest candidate set the exact endgame search takes on with three or more turns
# left; with two or fewer it is always used
DEFAULT_EXACT_MAX = 60

# how many guesses the lookahead expands at each node it reaches
DEFAULT_LOOKAHEAD_BRANCH = 10

# cost of running out of turns, on top of the six spent; 1.0 is risk neutral
DEFAULT_LOSS_PENALTY = 5.0

# positions a memo holds before it starts dropping the least recently used
DEFAULT_MEMO_LIMIT = 200_000


def solver_options(args):
    """The keywords a front end's flags name.  ``--exact-max`` is a sentinel
    on the terminal, resolved per subcommand through ``default_exact_max``."""
    exact_max = args.exact_max
    if exact_max is None:
        exact_max = getattr(args, "default_exact_max", DEFAULT_EXACT_MAX)
    return {
        "use_exact": not args.no_exact,
        "exact_max": exact_max,
        "loss_penalty": args.loss_penalty,
        "exact_max_branch": None if args.exact_branch <= 0 else args.exact_branch,
        "exact_root_branch": args.exact_root_branch,
        "lookahead": args.lookahead,
        "lookahead_max_branch": args.lookahead_branch,
    }


class WordleSolver:
    """Chooses guesses under a given prior, in one of two modes.

    ``entropy`` maximizes expected information gain; ``expected`` minimizes
    E(g, S) = 1 + sum_{r != GGGGG} p_r f(n_r) with this prior's fitted f.  Both
    use the exact search where affordable and restrict the last guess to a
    possible answer.  README *Lookahead*, *Hard mode* and *Results* describe
    ``lookahead``, ``hard`` and ``win_bonus``.
    """

    def __init__(self, data=None, config=None, mode=EXPECTED, params=None,
                 form=None, load_cached=True,
                 use_exact=True, exact_max=DEFAULT_EXACT_MAX,
                 loss_penalty=DEFAULT_LOSS_PENALTY,
                 exact_max_branch=DEFAULT_MAX_BRANCH,
                 exact_root_branch=DEFAULT_ROOT_BRANCH, hard=False,
                 lookahead=0, lookahead_max_branch=DEFAULT_LOOKAHEAD_BRANCH,
                 lookahead_exact_max=None, fused=None, win_bonus=0.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if win_bonus and mode != ENTROPY:
            raise ValueError("win_bonus is priced in bits, so it only means "
                             "something against information gain")
        self.data = data if data is not None else load_data()
        self.config = config if config is not None else PriorConfig()
        self.mode = mode
        self.weights = build_weights(self.data, self.config)

        if params is not None:
            self.form = estimator.get_form(form)
            self.params = np.asarray(params, dtype=np.float64)
        elif mode == EXPECTED and load_cached:
            if form is None:
                self.params, self.form = estimator.load_active(self.config)
            else:
                self.form = estimator.get_form(form)
                self.params = estimator.load_form_params(self.config, self.form)
        else:
            # entropy mode never reads f, and load_cached=False asks for no
            # fit; expected play raises instead
            self.form = estimator.get_form(form)
            self.params = np.zeros(self.form.n_params, dtype=np.float64)

        self._codes, self._knots, self._base = self.form.arrays()
        self._memo = Memo(DEFAULT_MEMO_LIMIT)
        self._lookahead_memo = Memo(DEFAULT_MEMO_LIMIT)

        self.lookahead = int(lookahead)
        self.lookahead_max_branch = int(lookahead_max_branch)
        self.win_bonus = float(win_bonus)
        self.use_exact = use_exact
        self.exact_max = exact_max
        self.hard = bool(hard)
        self.lookahead_exact_max = (exact_max if lookahead_exact_max is None
                                    else int(lookahead_exact_max))
        # an answer-major copy of the table, or None to score the shipped way
        if fused is None:
            fused = fused_kernels.DEFAULT_ENABLED
        self.transposed = (fused_kernels.build_transposed(self.data.feedback_table)
                           if fused else None)
        self.exact = ExactEndgame(self.data.feedback_table, self.weights,
                                  loss_penalty=loss_penalty,
                                  max_branch=exact_max_branch,
                                  root_branch=exact_root_branch, hard=self.hard,
                                  transposed=self.transposed)
        self.exact_hits = 0
        self.exact_bailouts = 0

    # ------------------------------------------------------------------ state

    def set_params(self, params, form=None):
        """Swap in new estimator parameters (used between policy iterations)."""
        if form is not None:
            self.form = estimator.get_form(form)
            self._codes, self._knots, self._base = self.form.arrays()
        self.params = np.asarray(params, dtype=np.float64)
        self._memo.clear()
        self._lookahead_memo.clear()

    def reset_memo(self):
        """Drop the cached guesses, for a sweep playing many independent
        configurations in one process.  The exact search's memo is kept; its
        values depend only on the weights.
        """
        self._memo.clear()
        self._lookahead_memo.clear()

    def all_candidates(self):
        return self.data.all_candidates()

    def guess_pool(self, candidates):
        """Rows of the feedback table a guess may come from: every word, or in
        hard mode only the words that could still be the answer."""
        return candidates if self.hard else None

    @staticmethod
    def _row(pool, index):
        """The word a pooled score belongs to.  Without a pool the score's own
        position is already the word index."""
        return index if pool is None else int(pool[index])

    def effective_size(self, candidates):
        return estimator.effective_size(self.weights[candidates])

    def win_probability(self, candidates, guess):
        """Prior probability that ``guess`` is the answer, given the candidates."""
        weights = self.weights[candidates]
        total = weights.sum()
        if total <= 0.0:
            return 0.0
        return float(weights[candidates == guess].sum() / total)

    def filter(self, guess, pattern, candidates):
        return kernels.filter_candidates(self.data.feedback_table, guess, pattern, candidates)

    def feedback(self, guess, answer):
        return int(self.data.feedback_table[guess, answer])

    # ----------------------------------------------------------------- search

    def score_all(self, candidates, pool=None):
        """Score for every guessable word, or, with ``pool``, for the words it
        names and in its order.  Higher is better in entropy mode, lower in
        expected."""
        table = self.data.feedback_table
        weights = self.weights[candidates]
        if self.mode == ENTROPY:
            scores = fused_kernels.ig_all(table, self.transposed, candidates,
                                          weights, pool=pool)
            return (self._priced_win(scores, candidates, weights, pool)
                    if self.win_bonus else scores)
        return fused_kernels.expected_guesses_all(
            table, self.transposed, candidates, weights, self.params,
            self._codes, self._knots, self._base, pool=pool,
        )

    def _priced_win(self, scores, candidates, weights, pool):
        """Pay each guess that could itself be the answer for the chance it
        is one.  Candidates run ascending, so a search finds the pooled ones."""
        total = float(weights.sum())
        if total <= 0.0:
            return scores
        paid = self.win_bonus * weights / total
        scores = scores.copy()
        if pool is None:
            scores[candidates] += paid
            return scores
        slot = np.clip(np.searchsorted(candidates, pool), 0, len(candidates) - 1)
        hit = candidates[slot] == pool
        scores[hit] += paid[slot[hit]]
        return scores

    def score_one(self, guess, candidates):
        """The same score, for a single guess."""
        pool = np.array([guess], dtype=np.int64)
        return float(self.score_all(candidates, pool=pool)[0])

    def best_guess(self, candidates, guesses_left=None):
        """Return ``(guess_index, score)``.

        The score is bits of information in entropy mode, or estimated total
        guesses remaining (this one included) in expected mode.  ``guesses_left``
        is how many turns are left including this one; at 1 the guess must be a
        possible answer.
        """
        # keyed on the turn count rather than just "is this the last turn":
        # the exact search's answer depends on how many turns are really left
        turns = MAX_TURNS if guesses_left is None else min(int(guesses_left), MAX_TURNS)
        key = (candidates.tobytes(), turns)
        hit = self._memo.get(key)
        if hit is not None:
            return hit

        result = self._best_guess_uncached(candidates, guesses_left)
        self._memo.put(key, result)
        return result

    def _exact_applies(self, m, turns, cap=None):
        """Two turns or fewer costs one pass over the guesses, the same as
        scoring them heuristically, so it is always worth doing.  Deeper than
        that, only small candidate sets are affordable."""
        if not self.use_exact or self.mode != EXPECTED:
            # the exact value is a score in guesses, so it only composes with
            # the expected-guess objective; entropy mode stays a clean baseline
            return False
        return turns <= 2 or m <= (self.exact_max if cap is None else cap)

    def _best_guess_uncached(self, candidates, guesses_left):
        m = len(candidates)
        if m == 0:
            raise ValueError("no candidates left; the feedback given is inconsistent")

        if m == 1:
            return int(candidates[0]), (0.0 if self.mode == ENTROPY else 1.0)

        if m == 2:
            # exact either way: guess the likelier one, and if wrong the other is known
            weights = self.weights[candidates]
            p = weights / weights.sum()
            first = 0 if p[0] >= p[1] else 1
            if self.mode == ENTROPY:
                h = float(-(p * np.log2(p)).sum())
                return int(candidates[first]), h
            return int(candidates[first]), kernels.two_candidates(float(p[first]))

        turns = MAX_TURNS if guesses_left is None else int(guesses_left)
        if self._exact_applies(m, turns):
            try:
                guess, value = self.exact.solve(candidates, turns)
                self.exact_hits += 1
                return guess, value
            except SearchBudgetExceeded:
                self.exact_bailouts += 1

        if guesses_left == 1:
            # last turn: a guess that cannot be the answer loses outright, so
            # take the likeliest candidate however uninformative it is
            guess = int(candidates[self.weights[candidates].argmax()])
            return guess, self.score_one(guess, candidates)

        if self.lookahead > 0 and self.mode == EXPECTED:
            # the value is in guesses either way, so it composes with the exact
            # search's; entropy mode is left alone as a clean baseline
            return self._lookahead_guess(candidates, turns, self.lookahead)

        pool = self.guess_pool(candidates)
        scores = self.score_all(candidates, pool=pool)
        best_ind = int(scores.argmin() if self.mode == EXPECTED else scores.argmax())
        return self._row(pool, best_ind), float(scores[best_ind])

    # ------------------------------------------------------------- lookahead

    def _partition(self, guess, candidates):
        """``[(mass, members), ...]`` over the non-winning feedback buckets."""
        patterns = self.data.feedback_table[guess, candidates]
        order = np.argsort(patterns, kind="stable")
        sorted_patterns = patterns[order]
        cuts = np.flatnonzero(np.diff(sorted_patterns)) + 1
        out = []
        for group in np.split(order, cuts):
            # group holds indices into `candidates`, not into `sorted_patterns`
            if patterns[group[0]] == WIN_PATTERN:
                continue  # winning now costs nothing further
            members = candidates[group]
            out.append((float(self.weights[members].sum()), members))
        return out

    def _lookahead_value(self, candidates, turns, depth):
        """Value of a position under the branch-capped search."""
        m = candidates.shape[0]
        if m == 1:
            return 1.0
        # how far the exact search may reach inside a lookahead sub-position; its own knob
        if self._exact_applies(m, turns, cap=self.lookahead_exact_max):
            try:
                value = self.exact.solve(candidates, turns)[1]
                self.exact_hits += 1
                return value
            except SearchBudgetExceeded:
                self.exact_bailouts += 1
        if depth <= 0 or turns <= 1:
            return float(self.score_all(
                candidates, pool=self.guess_pool(candidates)).min())
        return self._lookahead_guess(candidates, turns, depth)[1]

    def _lookahead_guess(self, candidates, turns, depth):
        """``(guess, value)`` from expanding the best ``lookahead_max_branch``
        guesses and scoring each by what its buckets are really worth.

        Only the guess is pruned, never the feedback.  The cap limits the
        fan-out at this node, and every level below pays it again.
        """
        key = (candidates.tobytes(), turns, depth)
        hit = self._lookahead_memo.get(key)
        if hit is not None:
            return hit

        pool = self.guess_pool(candidates)
        scores = self.score_all(candidates, pool=pool)
        # stable, so a tie goes to the lower index, the rule ExactEndgame._branch_order
        # and the Rust share
        branch = np.argsort(scores, kind="stable")[:self.lookahead_max_branch]
        total = float(self.weights[candidates].sum())

        best_guess, best_value = self._row(pool, int(branch[0])), np.inf
        for slot in branch:
            guess = self._row(pool, int(slot))
            value = 1.0
            for mass, members in self._partition(guess, candidates):
                value += mass / total * self._lookahead_value(
                    members, turns - 1, depth - 1)
                if value >= best_value:
                    break  # the remaining buckets can only add to it
            if value < best_value:
                best_value, best_guess = value, guess

        result = (best_guess, float(best_value))
        self._lookahead_memo.put(key, result)
        return result

    def solved_exactly(self, candidates, guesses_left=None):
        """True if the guess for this position comes from somewhere other than
        the heuristic scorer, so its score is a true value rather than an
        estimate."""
        turns = MAX_TURNS if guesses_left is None else int(guesses_left)
        return (len(candidates) <= 2 or guesses_left == 1
                or self._exact_applies(len(candidates), turns))

    def split_siblings(self, guess, candidates):
        """Every other word that gives the same feedback as ``guess`` against
        every candidate, and so plays identically from here on.

        Hard mode narrows these the way it narrows the choice itself.  A group
        holds either one possible answer or any number of words that cannot be
        the answer.
        """
        pool = kernels.same_split(self.data.feedback_table, candidates,
                                  int(guess), pool=self.guess_pool(candidates))
        return pool[pool != guess]

    def top_guesses(self, candidates, k=5, guesses_left=None):
        """``[(word, score), ...]`` for the k best guesses, best first.

        One word per distinct split; ``split_siblings`` names the rest.  Where
        the solver settles the position exactly the scores are true values, and
        the list never mixes those with estimates.
        """
        if self.solved_exactly(candidates, guesses_left):
            exact_top = self._exact_top(candidates, k, guesses_left)
            if exact_top is not None:
                return [(self.data.all_words[g], float(s)) for g, s in exact_top]

        pool = self.guess_pool(candidates)
        scores = self.score_all(candidates, pool=pool)
        # stable, so a tie leaves the winner where argmin and argmax have it and
        # the ranking opens on the guess best_guess would play
        order = np.argsort(-scores if self.mode == ENTROPY else scores,
                           kind="stable")

        table = self.data.feedback_table
        kept = []
        for i in order:
            guess = self._row(pool, int(i))
            score = float(scores[i])
            split = table[guess, candidates]
            if any(abs(score - seen) <= SPLIT_TIE_TOL and np.array_equal(split, other)
                   for _, seen, other in kept):
                continue
            kept.append((guess, score, split))
            if len(kept) == k:
                break
        return [(self.data.all_words[g], s) for g, s, _ in kept]

    def _exact_top(self, candidates, k, guesses_left):
        """``[(guess, value), ...]`` for a position settled without the
        heuristic scorer, or None if the exact search ran out of budget and the
        caller should fall back to it.

        Mirrors ``_best_guess_uncached``: the same cases decided the same way,
        reporting the runners-up rather than the winner alone.
        """
        m = len(candidates)
        if m == 1:
            return [(int(candidates[0]), 0.0 if self.mode == ENTROPY else 1.0)]

        if m == 2:
            weights = self.weights[candidates]
            p = weights / weights.sum()
            # stable, so equal weights leave the winner where best_guess has it
            order = np.argsort(-p, kind="stable")
            if self.mode == ENTROPY:
                h = float(-(p * np.log2(p)).sum())
                return [(int(candidates[i]), h) for i in order]
            return [(int(candidates[i]), kernels.two_candidates(float(p[i])))
                    for i in order]

        turns = MAX_TURNS if guesses_left is None else int(guesses_left)
        if self._exact_applies(m, turns):
            try:
                top = self.exact.top(candidates, turns, k)
                self.exact_hits += 1
                return top
            except SearchBudgetExceeded:
                self.exact_bailouts += 1
                return None

        if guesses_left == 1:
            # a guess that cannot be the answer loses outright, so the ranking
            # is the candidates by weight however uninformative they are
            order = np.argsort(-self.weights[candidates], kind="stable")
            best = candidates[order[:k]]
            scores = self.score_all(candidates, pool=best)
            return [(int(g), float(s)) for g, s in zip(best, scores)]
        return None

    # --------------------------------------------------------------- rollouts

    def play(self, answer, max_guesses=12, deadline=None):
        """Play a full game against ``answer``.

        Returns ``(guesses, states)`` where ``states[i]`` is
        ``(effective_size, candidate_count)`` of the position *before* guess i.

        ``deadline`` is the turn on which the game must be won; the guess made
        then is restricted to possible answers.  None plays with no deadline,
        as training rollouts do.
        """
        candidates = self.all_candidates()
        guesses = []
        states = []
        for turn in range(1, max_guesses + 1):
            states.append((self.effective_size(candidates), len(candidates)))
            guesses_left = None if deadline is None else max(1, deadline - turn + 1)
            guess, _ = self.best_guess(candidates, guesses_left)
            guesses.append(guess)
            if guess == answer:
                return guesses, states
            candidates = self.filter(guess, self.feedback(guess, answer), candidates)
        return guesses, states


def sample_answers(weights, n_games, seed=0):
    """Draw answers from the prior, which is the distribution the estimator is
    supposed to be accurate under."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(weights), size=n_games, replace=True, p=weights)
