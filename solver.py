"""The solver itself: pick a guess, apply feedback, play a game out."""

import numpy as np

import estimator
import fused_kernels
import kernels
from exact import DEFAULT_MAX_BRANCH, MAX_TURNS, ExactEndgame, SearchBudgetExceeded
from memo import Memo
from priors import PriorConfig, build_weights
from wordle_data import WIN_PATTERN, load_data

ENTROPY = "entropy"
EXPECTED = "expected"
MODES = (ENTROPY, EXPECTED)

# guesses within this many bits of the best are treated as tied (entropy mode)
IG_TIE_TOL = 1e-4

DEFAULT_DEADLINE = 6

# largest candidate set the exact endgame search is allowed to take on when
# three or more turns remain; with two or fewer it is always used, since it
# costs no more than scoring the guesses heuristically
DEFAULT_EXACT_MAX = 60

# how many guesses the lookahead expands at each node it reaches
DEFAULT_BEAM_WIDTH = 10

# cost of running out of turns, on top of the six spent; 1.0 is risk neutral
DEFAULT_LOSS_PENALTY = 5.0

# positions a memo holds before it starts dropping the least recently used
DEFAULT_MEMO_LIMIT = 200_000


class WordleSolver:
    """Chooses guesses under a given prior, in one of two modes.

    ``entropy``   greedily maximize expected information gain, breaking near
                  ties in favor of a word that could be the answer.
    ``expected``  minimize E(g, S) = 1 + sum_{r != GGGGG} p_r f(n_r), using the
                  estimator fitted for this prior combination.

    Both are only fallbacks: when the candidate set is small enough, or few
    enough turns remain, the guess comes from the exact search in exact.py
    instead, which needs no estimator and subsumes the last-turn rule.  Where
    the search does not reach, one guess left still means the guess is
    restricted to words that could actually be the answer, since information is
    worthless on the last turn.

    ``lookahead`` puts a width-limited expectimax above that, expanding the best
    ``beam_width`` guesses and scoring each by what its buckets are really worth
    rather than by f alone.  It is off by default: one ply measures
    -0.0034 +- 0.0041 guesses over 20,000 paired games, which is nothing at 0.9
    sigma, for roughly 10x the wall clock, and two plies costs minutes per
    position.  See the README for the numbers.

    ``hard`` restricts every guess to a word that could still be the answer,
    which is a strictly smaller choice than the ~13k words the solver usually
    picks from; f is still the one fitted for unrestricted play, so its estimates
    are optimistic under the restriction.
    """

    def __init__(self, data=None, config=None, mode=EXPECTED, params=None,
                 form=None, load_cached=True,
                 use_exact=True, exact_max=DEFAULT_EXACT_MAX,
                 loss_penalty=DEFAULT_LOSS_PENALTY,
                 exact_max_branch=DEFAULT_MAX_BRANCH, hard=False,
                 lookahead=0, beam_width=DEFAULT_BEAM_WIDTH, beam_exact_max=None,
                 fused=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
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
            # entropy mode never reads f, and load_cached=False is a deliberate
            # ask for an unfitted one; expected play raises instead
            self.form = estimator.get_form(form)
            self.params = estimator.default_params(self.form)

        self._codes, self._knots, self._base = self.form.arrays()
        self._memo = Memo(DEFAULT_MEMO_LIMIT)
        self._beam_memo = Memo(DEFAULT_MEMO_LIMIT)

        self.lookahead = int(lookahead)
        self.beam_width = int(beam_width)
        self.use_exact = use_exact
        self.exact_max = exact_max
        self.hard = bool(hard)
        self.beam_exact_max = exact_max if beam_exact_max is None else int(beam_exact_max)
        # an answer-major copy of the table, or None to score the shipped way
        if fused is None:
            fused = fused_kernels.DEFAULT_ENABLED
        self.transposed = (fused_kernels.build_transposed(self.data.feedback_table)
                           if fused else None)
        self.exact = ExactEndgame(self.data.feedback_table, self.weights,
                                  loss_penalty=loss_penalty,
                                  max_branch=exact_max_branch, hard=self.hard,
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
        self._beam_memo.clear()

    def reset_memo(self):
        """Drop the cached guesses, for a sweep playing many independent
        configurations in one process.

        A finished run's positions are dead weight: the memo evicts by least
        recent use, so they linger and compete for the cap with the positions the
        next run is actually reusing.  Clearing between runs keeps it small and
        useful.  The exact search's memo is left alone: its values depend only on
        the weights, so they stay valid and stay worth sharing.
        """
        self._memo.clear()
        self._beam_memo.clear()

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
            return fused_kernels.ig_all(table, self.transposed, candidates,
                                        weights, pool=pool)
        return fused_kernels.expected_guesses_all(
            table, self.transposed, candidates, weights, self.params,
            self._codes, self._knots, self._base, pool=pool,
        )

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
            return int(candidates[first]), 2.0 - float(p[first])

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

        if self.mode == EXPECTED:
            best_ind = int(scores.argmin())
            return self._row(pool, best_ind), float(scores[best_ind])

        # entropy mode: among guesses within the tolerance of the best, prefer
        # one that could itself be the answer
        best_ind = int(scores.argmax())
        best = float(scores[best_ind])
        close = np.nonzero(scores >= best - IG_TIE_TOL)[0]
        words = close if pool is None else pool[close]
        close_possible = close[np.isin(words, candidates)]
        if close_possible.size:
            pick = int(close_possible[scores[close_possible].argmax()])
            return self._row(pool, pick), float(scores[pick])
        return self._row(pool, best_ind), best

    # ------------------------------------------------------------- lookahead

    def _partition(self, guess, candidates):
        """``[(mass, members), ...]`` over the non-winning feedback buckets.

        One sort rather than a pass per pattern, which matters at the opening
        where there are ~150 buckets over ~13k candidates.
        """
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
        """Value of a position under the width-limited search."""
        m = candidates.shape[0]
        if m == 1:
            return 1.0
        # the exact search gives a true value here, but the lookahead reaches
        # these sub-positions tens of thousands of times, so how far it is
        # allowed to reach *inside* the search is its own knob
        if self._exact_applies(m, turns, cap=self.beam_exact_max):
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
        """``(guess, value)`` from expanding the best ``beam_width`` guesses and
        scoring each by what its buckets are really worth.

        Only the guess is pruned, never the feedback: dropping a bucket would
        leave the sum something other than an expectation.
        """
        key = (candidates.tobytes(), turns, depth)
        hit = self._beam_memo.get(key)
        if hit is not None:
            return hit

        pool = self.guess_pool(candidates)
        scores = self.score_all(candidates, pool=pool)
        beam = np.argsort(scores)[:self.beam_width]
        total = float(self.weights[candidates].sum())

        best_guess, best_value = self._row(pool, int(beam[0])), np.inf
        for slot in beam:
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
        self._beam_memo.put(key, result)
        return result

    def solved_exactly(self, candidates, guesses_left=None):
        """True if the guess for this position comes from somewhere other than
        the heuristic scorer, so its score is a true value rather than an
        estimate."""
        turns = MAX_TURNS if guesses_left is None else int(guesses_left)
        return (len(candidates) <= 2 or guesses_left == 1
                or self._exact_applies(len(candidates), turns))

    def top_guesses(self, candidates, k=5, guesses_left=None):
        """``[(word, score), ...]`` for the k best guesses, best first.

        Collapses to the single best guess whenever that guess came from
        somewhere other than the heuristic scorer, so the list never mixes
        exact values with estimated ones.
        """
        if self.solved_exactly(candidates, guesses_left):
            guess, score = self.best_guess(candidates, guesses_left)
            return [(self.data.all_words[guess], score)]
        pool = self.guess_pool(candidates)
        scores = self.score_all(candidates, pool=pool)
        order = np.argsort(scores)
        if self.mode == ENTROPY:
            order = order[::-1]
        return [(self.data.all_words[self._row(pool, int(i))], float(scores[i]))
                for i in order[:k]]

    # --------------------------------------------------------------- rollouts

    def play(self, answer, max_guesses=12, deadline=None):
        """Play a full game against ``answer``.

        Returns ``(guesses, states)`` where ``states[i]`` is
        ``(effective_size, candidate_count)`` of the position *before* guess i.

        ``deadline`` is the turn on which the game must be won; the guess made
        then is restricted to possible answers.  Leaving it None (the default,
        used for training rollouts) plays with no deadline, so the recorded
        guess counts are unbiased estimates of the value function.
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
