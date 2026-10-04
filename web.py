"""Web front end for the Wordle solver.

    python web.py                 serve on http://127.0.0.1:8000 and open a browser
    python web.py --port 9000     somewhere else
    python web.py --no-browser    do not open one

Standard library only: an ``http.server`` and the static files in ``web/``.
The feedback table and the numba kernels are warmed in the background and the
page waits for them.  The whole solver sits behind one lock.
"""

import argparse
import json
import mimetypes
import os
import secrets
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import estimator
import kernels
import wording
from exact import DEFAULT_MAX_BRANCH, DEFAULT_ROOT_BRANCH
from priors import PriorConfig, regime_stats
from solver import (DEFAULT_EXACT_MAX, DEFAULT_LOOKAHEAD_BRANCH,
                    DEFAULT_LOSS_PENALTY, ENTROPY, EXPECTED, MODES,
                    WordleSolver, solver_options)
# the terminal game's answer-drawing and emoji grid, so both front ends deal
# the same answers and print the same thing at the end
from wordle_game import TILE_BLOCK, pick_random_answer
from wordle_data import (FEEDBACK_TABLE_PATH, MAX_GUESSES, TILE_VALS,
                         WIN_PATTERN, feedback_to_num, load_data,
                         num_to_feedback, record_through, score_guess)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "web")

MAX_SESSIONS = 200          # oldest are dropped; a session is a few dozen bytes
MAX_LISTED_CANDIDATES = 80  # possible answers sent to the page, likeliest first
MAX_LISTED_MATCHES = 120    # words the finder lists, best first by its sort
MAX_LISTED_SIBLINGS = 40    # words sharing a guess's split, likeliest first
MAX_BODY = 64 * 1024

# how the word finder can order what it found, and which way round each runs;
# the names are the page's
FINDER_SORTS = {"left": 1, "bits": -1, "weight": -1, "alpha": 1}
DEFAULT_SORT = "left"

# every string the page and this file say, and whatever editor.py has
# rewritten them to
WORDING = wording.Wording()


class SolverError(Exception):
    """Something the user can fix, reported as a 400 with this message."""


def table_size():
    """How big the feedback table on disk actually is, rather than how big it
    was when the sentence naming it was written."""
    try:
        return f"{int(os.path.getsize(FEEDBACK_TABLE_PATH) / 1e6)} MB"
    except OSError:
        return "the table"


# --------------------------------------------------------------------- solvers

class SolverPool:
    """Solvers, one per (prior, mode, form, hard), built on demand and kept.

    Building one costs a pass over the word lists; the feedback table behind
    them all is loaded once by ``load_data``'s own cache.
    """

    def __init__(self, mask_last=0, options=None):
        self.mask_last = mask_last
        self.options = options or {}
        self.lock = threading.Lock()
        self._solvers = {}
        # read without the lock by /api/status, so they are only ever replaced,
        # never mutated in place
        self.ready = False
        self.failed = None
        self.progress = "Starting up"

    def data(self):
        return load_data(mask_last=self.mask_last)

    def get(self, config, mode, form, hard=False):
        """Caller must hold ``self.lock``."""
        key = (config.slug, mode, form, bool(hard))
        solver = self._solvers.get(key)
        if solver is None:
            solver = WordleSolver(data=self.data(), config=config, mode=mode,
                                  form=form, hard=bool(hard), **self.options)
            self._solvers[key] = solver
        return solver

    def solver_for(self, settings):
        """The solver a ``Settings`` names.  Caller must hold ``self.lock``."""
        return self.get(settings.config, settings.mode, settings.form, settings.hard)

    def warm(self, config, mode, form):
        """Load the table, compile the kernels and compute the opening guess, so
        that the first thing the page asks for is not also the slowest."""
        try:
            self.progress = f"Loading the feedback table ({table_size()})"
            with self.lock:
                self.data()
                self.progress = "Compiling the scoring kernels"
                solver = self.get(config, mode, form)
                self.progress = "Computing the opening guess"
                solver.best_guess(solver.all_candidates(), MAX_GUESSES)
            self.progress = "Ready"
            self.ready = True
        except estimator.NotFitted as exc:
            # nothing to debug: the message says which command makes the fit
            self.failed = str(exc)
            self.progress = "Failed to start"
        except Exception:
            traceback.print_exc()
            self.failed = traceback.format_exc(limit=0).strip()
            self.progress = "Failed to start"


# -------------------------------------------------------------------- sessions

class Settings:
    """Everything about how the solver is asked, as one value.

    It travels with the request rather than living on the session, so that
    changing a setting mid-board re-runs the solver over the rows already played
    instead of starting again.
    """

    __slots__ = ("config", "mode", "form", "hard", "top")

    def __init__(self, config, mode, form, hard, top):
        self.config = config
        self.mode = mode
        self.form = form
        self.hard = hard
        self.top = top


class Session:
    """One board.  Only the rows played are kept; the candidate set is rebuilt
    from them on every request, which costs a few filters over 13k words and
    makes undo free."""

    def __init__(self, settings):
        self.settings = settings
        self.rows = []  # (guess index, pattern number)


class Game:
    """One game of Wordle played against the page.

    The answer is kept here and never sent until the game is over, so that
    reading the network traffic does not spoil it.

    ``config`` is the prior the answer was *drawn* from and never changes; what
    the hint is asked under comes with each hint request, so turning a prior off
    mid-game rethinks the advice rather than dealing a new word.
    """

    def __init__(self, answer, config, source):
        self.answer = answer
        self.config = config
        self.source = source
        self.rows = []       # (guess index, pattern number)
        self.gave_up = False

    @property
    def solved(self):
        return bool(self.rows) and self.rows[-1][1] == WIN_PATTERN

    @property
    def over(self):
        return self.solved or self.gave_up or len(self.rows) >= MAX_GUESSES


def share_grid(rows, solved):
    """The emoji grid, exactly as ``wordle_game.py --share`` prints it."""
    header = f"Wordle {len(rows) if solved else 'X'}/{MAX_GUESSES}"
    lines = ["".join(TILE_BLOCK[tile] for tile in num_to_feedback(pattern))
             for _, pattern in rows]
    return "\n".join([header, *lines])


def parse_config(body):
    return PriorConfig(repeat=bool(body.get("repeat", True)),
                       curation=bool(body.get("curation", True)))


def parse_mode(body):
    mode = str(body.get("mode", EXPECTED))
    if mode not in MODES:
        raise SolverError(f"Mode must be one of {', '.join(MODES)}")
    return mode


def parse_form(body):
    form = body.get("form") or None
    if form is not None and form not in estimator.FORMS:
        raise SolverError(f"Unknown form {form!r}")
    return form


def parse_settings(body):
    """The whole of how to ask the solver, read off one request.

    The page sends all of it every time, as it sends the whole board every time,
    so nothing on the server has to be kept in step with the switches.
    """
    return Settings(config=parse_config(body), mode=parse_mode(body),
                    form=parse_form(body), hard=bool(body.get("hard", False)),
                    top=min(10, max(1, int(body.get("top", 5)))))


def parse_answer(body, data):
    """The answer the board is being played against, or None.  Set, it colors
    every row itself; the page's tiles are then a display, not an input."""
    word = str(body.get("answer", "") or "").strip().lower()
    if not word:
        return None
    if word not in data.word2ind:
        raise SolverError(f"'{word}' is not in the word list")
    return data.word2ind[word]


def parse_sort(body):
    sort = str(body.get("sort", DEFAULT_SORT))
    if sort not in FINDER_SORTS:
        raise SolverError(f"Sort must be one of {', '.join(FINDER_SORTS)}")
    return sort


def parse_pattern(raw):
    """A five-tile pattern, as either 'NYGNN' or [0, 1, 2, 0, 0]."""
    if isinstance(raw, str):
        tiles = list(raw.strip().upper())
    elif isinstance(raw, list):
        tiles = ["NYG"[int(t)] if isinstance(t, int) and 0 <= t <= 2 else "?" for t in raw]
    else:
        raise SolverError("No feedback given for that guess")
    if len(tiles) != 5 or any(t not in TILE_VALS for t in tiles):
        raise SolverError("Feedback needs five tiles, each gray, yellow or green")
    return feedback_to_num("".join(tiles))


# ----------------------------------------------------------------------- state

def score_label(mode):
    return ("bits of expected information" if mode == ENTROPY
            else "expected guesses to finish")


def describe_prior(config, data):
    """The prior as the page explains it: which halves are on, how much mass
    each takes, how that mass is shaped, and the rates it is set from."""
    return {
        "summary": config.describe(),
        "repeat": config.repeat,
        "repeat_prob": config.repeat_prob,
        "repeat_shape": config.repeat_shape if config.repeat else None,
        "curation": config.curation,
        "noncurated_prob": config.noncurated_prob,
        "curation_shape": config.curation_shape if config.curation else None,
        "measured": regime_stats(data),
    }


def describe_estimator(solver):
    """The fitted f, in pieces, so the page can set it as an equation rather
    than as the one-line summary ``estimator.describe`` prints."""
    samples = [2, 5, 20, 100, 500, 2000]
    values = estimator.f_value(samples, solver.params, solver.form)
    return {
        "form": solver.form.name,
        # what "whatever is cached" resolves to for this prior, so the page can
        # say which form is the fitted one instead of naming one in advance
        "fitted": estimator.active_form_name(solver.config),
        "description": solver.form.description,
        "base": float(solver.form.base),
        "terms": [{"label": label, "coefficient": float(value)}
                  for label, value in zip(solver.form.term_labels, solver.params)],
        "samples": [{"n": n, "f": float(v)} for n, v in zip(samples, values)],
    }


def describe_guess(solver, word, score, candidates):
    """One entry of the ranking: a word, its score, and the words that play
    identically from here.

    The ranking names one word per distinct split, so the ones it stands for
    travel with it.  Likeliest first, which is the order that makes the
    familiar spelling of a split the one on show.
    """
    guess = solver.data.word2ind[word]
    siblings = solver.split_siblings(guess, candidates)
    order = np.argsort(-solver.weights[siblings], kind="stable")
    listed = siblings[order[:MAX_LISTED_SIBLINGS]]
    return {
        "word": word,
        "score": float(score),
        "same_split": [solver.data.all_words[i] for i in listed],
        "same_split_count": int(siblings.size),
    }


def build_state(pool, session_id, settings, rows, answer=None):
    """The whole board as the page wants it: what has been played, what the
    solver would play next, and what is still possible.

    ``rows`` and ``settings`` are passed in rather than read off the session so
    that a proposed board can be scored before it is adopted; a set of rows that
    cannot all be true raises rather than returning a state.  With ``answer``
    the rows' patterns are replaced by the true ones, so the board is the game
    the solver would have played against that word.
    """
    started = time.perf_counter()
    with pool.lock:
        solver = pool.solver_for(settings)
        data = solver.data
        if answer is not None:
            rows = [(guess, score_guess(data.all_words[guess], data.all_words[answer]))
                    for guess, _ in rows]

        candidates = solver.all_candidates()
        for index, (guess, pattern) in enumerate(rows):
            if pattern == WIN_PATTERN and index != len(rows) - 1:
                raise SolverError(f"Row {index + 1} already solves it; "
                                  f"clear the rows under it")
            candidates = solver.filter(guess, pattern, candidates)
            if len(candidates) == 0:
                raise SolverError(f"Row {index + 1} contradicts the rows above it "
                                  f"-- no word matches all of them")

        turn = len(rows) + 1
        solved = bool(rows) and rows[-1][1] == WIN_PATTERN
        out_of_guesses = not solved and turn > MAX_GUESSES

        state = {
            "session": session_id,
            "turn": turn,
            "guesses_left": max(0, MAX_GUESSES - len(rows)),
            "max_guesses": MAX_GUESSES,
            "rows": [{"word": data.all_words[g], "pattern": num_to_feedback(p)}
                     for g, p in rows],
            "solved": solved,
            "over": solved or out_of_guesses,
            "mode": settings.mode,
            "hard": settings.hard,
            "score_label": score_label(settings.mode),
            "prior": describe_prior(settings.config, data),
            "answer": None if answer is None else data.all_words[answer],
            "candidates_left": int(len(candidates)),
            "recommendation": None,
            "alternatives": [],
            "possible_answers": [],
            "estimator": describe_estimator(solver) if settings.mode == EXPECTED else None,
        }

        weights = solver.weights[candidates]
        order = candidates[weights.argsort()[::-1]]
        state["possible_answers"] = [data.all_words[i]
                                     for i in order[:MAX_LISTED_CANDIDATES]]
        state["effective_size"] = float(solver.effective_size(candidates))

        if state["over"]:
            return state

        guesses_left = MAX_GUESSES - len(rows)
        exact = bool(solver.solved_exactly(candidates, guesses_left))
        top = solver.top_guesses(candidates, k=max(1, settings.top),
                                 guesses_left=guesses_left)
        word, score = top[0]
        pick = data.word2ind[word]
        state["recommendation"] = {
            **describe_guess(solver, word, score, candidates),
            "exact": exact,
            # whether the pick could be the answer; a non-candidate is often the
            # right play
            "possible": bool((candidates == pick).any()),
            "win_probability": float(solver.win_probability(candidates, pick)),
        }
        state["alternatives"] = [describe_guess(solver, w, s, candidates)
                                 for w, s in top[1:]]
        state["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return state


def game_state(pool, session_id, game):
    data = pool.data()
    state = {
        "session": session_id,
        "rows": [{"word": data.all_words[g], "pattern": num_to_feedback(p)}
                 for g, p in game.rows],
        "turn": len(game.rows) + 1,
        "guesses_left": max(0, MAX_GUESSES - len(game.rows)),
        "max_guesses": MAX_GUESSES,
        "solved": game.solved,
        "gave_up": game.gave_up,
        "over": game.over,
        "source": game.source,
        "prior": describe_prior(game.config, data),
        "answer": None,
        "share": None,
    }
    if game.over:
        state["answer"] = data.all_words[game.answer]
        state["share"] = share_grid(game.rows, game.solved)
    else:
        state["letters"] = letter_odds(pool, game)
    return state


def letter_odds(pool, game):
    """The chance the answer contains each letter, likeliest first.

    The weighting is the prior the answer was drawn from and not the settings on
    the request, so turning a prior off mid-game rethinks the hint but leaves
    these numbers alone.  A letter no remaining candidate holds sits at 0.0 and
    is left out; the gray keys have said that already.  The ones at 1.0 stay,
    and they sort to the front.
    """
    with pool.lock:
        # entropy's solver will do: this wants weights and filtering, never a
        # fitted f, so it works under a prior nothing has been fitted for
        solver = pool.get(game.config, ENTROPY, None)
        candidates = solver.all_candidates()
        for guess, pattern in game.rows:
            candidates = solver.filter(guess, pattern, candidates)
        weights = solver.weights[candidates]
        total = float(weights.sum())
        if candidates.size == 0 or total <= 0.0:
            return []
        odds = (weights @ solver.data.letter_presence()[candidates]) / total

    absent = 1e-9   # this close to zero and no candidate holds the letter at all
    # stable, so equal odds stay alphabetical, which is how api.rs lists them
    return [{"letter": chr(ord("a") + i), "chance": float(odds[i])}
            for i in np.argsort(-odds, kind="stable") if odds[i] > absent]


def game_hint(pool, game, settings):
    """What the solver would play here -- the web version of the terminal
    game's ``?``.

    Every setting comes with the request rather than being fixed when the game
    was dealt, so changing one rethinks the advice for the position you are in
    instead of dealing a new answer.  That includes the priors: the answer was
    drawn from the ones in force at the time, and it stays what it is.
    """
    with pool.lock:
        solver = pool.solver_for(settings)
        candidates = solver.all_candidates()
        for guess, pattern in game.rows:
            candidates = solver.filter(guess, pattern, candidates)
        guesses_left = max(1, MAX_GUESSES - len(game.rows))
        top = solver.top_guesses(candidates, k=max(1, settings.top),
                                 guesses_left=guesses_left)
        word, score = top[0]
        return {
            **describe_guess(solver, word, score, candidates),
            "mode": settings.mode,
            "hard": settings.hard,
            "score_label": score_label(settings.mode),
            "candidates_left": int(len(candidates)),
            "exact": bool(solver.solved_exactly(candidates, guesses_left)),
            "alternatives": [describe_guess(solver, w, s, candidates)
                             for w, s in top[1:]],
        }


# ------------------------------------------------------------- the word finder

def find_guesses(pool, answer, pattern, config, sort):
    """Every word that would have come back as ``pattern`` against ``answer``.

    ``feedback_table[:, answer]`` is what each word scores against this one
    word, so the search is one column of the table.  Each match is measured
    three ways: ``left`` is how many words would still be possible after it,
    ``bits`` what the guess reveals on average over every answer, and
    ``weight`` the prior's own chance that the guess is the answer.
    """
    started = time.perf_counter()
    with pool.lock:
        # entropy needs no fitted f, so the finder works whatever is cached
        solver = pool.get(config, ENTROPY, None)
        data = solver.data
        candidates = solver.all_candidates()

        matches = np.nonzero(data.feedback_table[:, answer] == pattern)[0]
        matches = np.ascontiguousarray(matches, dtype=np.int64)
        if matches.size == 0:
            # a pattern no word can produce against this answer -- four greens
            # and a yellow, say
            return {"answer": data.all_words[answer],
                    "pattern": num_to_feedback(pattern), "sort": sort,
                    "total": 0, "shown": 0, "words": [],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000)}
        bits = kernels.ig_all(data.feedback_table, candidates,
                              solver.weights, pool=matches)
        left = np.array([len(solver.filter(int(g), pattern, candidates))
                         for g in matches], dtype=np.int64)
        weight = solver.weights[matches]
        words = [data.all_words[i] for i in matches]

    if sort == "alpha":
        order = np.argsort(words, kind="stable")
    else:
        key = {"left": left, "bits": bits, "weight": weight}[sort]
        order = np.argsort(FINDER_SORTS[sort] * key, kind="stable")

    return {
        "answer": data.all_words[answer],
        "pattern": num_to_feedback(pattern),
        "sort": sort,
        "total": int(matches.size),
        "shown": int(min(matches.size, MAX_LISTED_MATCHES)),
        "words": [{"word": words[i], "left": int(left[i]), "bits": float(bits[i]),
                   "weight": float(weight[i])}
                  for i in order[:MAX_LISTED_MATCHES]],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


# ------------------------------------------------------------------ the server

class Store:
    """Sessions, oldest evicted once there are too many."""

    def __init__(self, noun="board", limit=MAX_SESSIONS):
        self.noun = noun
        self.limit = limit
        self.lock = threading.Lock()
        self.sessions = OrderedDict()

    def create(self, session):
        session_id = secrets.token_hex(8)
        with self.lock:
            self.sessions[session_id] = session
            while len(self.sessions) > self.limit:
                self.sessions.popitem(last=False)
        return session_id

    def get(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SolverError(f"That {self.noun} has expired; start a new one")
            self.sessions.move_to_end(session_id)
            return session


class Handler(BaseHTTPRequestHandler):
    server_version = "wordle-solver"
    protocol_version = "HTTP/1.1"

    pool = None       # all set on the class in main()
    store = None      # solver boards
    games = None      # games being played

    # ---------------------------------------------------------------- plumbing

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            super().log_message(fmt, *args)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # an unread body leaves the connection unusable; say so
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        # this file's own literals were compiled before any edit, so they are
        # matched on the way out instead of at the point they were written
        self.send_json({"error": WORDING.message(message)}, status=status)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            # it is not going to be read, so the connection cannot be reused
            self.close_connection = True
            raise SolverError("Request too large")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise SolverError("Malformed request")

    # ------------------------------------------------------------------ routes

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            return self.send_json({
                "ready": self.pool.ready,
                "progress": WORDING.message(self.pool.progress),
                "error": self.pool.failed,
                "forms": sorted(estimator.FORMS),
                "modes": list(MODES),
                "max_guesses": MAX_GUESSES,
                "sorts": list(FINDER_SORTS),
                "record_through": record_through(),
            })
        return self.serve_static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        # read first, whoever ends up wanting it: an unread body desynchronizes
        # the kept-alive connection
        try:
            body = self.read_json()
        except SolverError as exc:
            return self.send_error_json(str(exc))

        routes = {"/api/new": self.api_new,
                  "/api/board": self.api_board,
                  "/api/draw": self.api_draw,
                  "/api/find": self.api_find,
                  "/api/game/new": self.api_game_new,
                  "/api/game/guess": self.api_game_guess,
                  "/api/game/hint": self.api_game_hint,
                  "/api/game/reveal": self.api_game_reveal}
        route = routes.get(path)
        if route is None:
            return self.send_error_json("No such endpoint", status=404)
        if not self.pool.ready:
            if self.pool.failed:
                return self.send_error_json(self.pool.failed, status=503)
            return self.send_error_json(f"Not ready yet \u2014 {self.pool.progress}", status=503)
        try:
            return route(body)
        except SolverError as exc:
            return self.send_error_json(str(exc))
        except estimator.NotFitted as exc:
            # a setting the page offers that nobody has fitted: say which, and
            # what to run, rather than logging a traceback nobody sees
            return self.send_error_json(str(exc), status=503)
        except Exception:
            traceback.print_exc()
            return self.send_error_json("The solver raised; see the server log",
                                        status=500)

    def api_new(self, body):
        session = Session(parse_settings(body))
        session_id = self.store.create(session)
        answer = parse_answer(body, self.pool.data())
        return self.send_json(
            build_state(self.pool, session_id, session.settings, [], answer))

    def api_draw(self, body):
        """One answer drawn from the prior, for the board to be played against."""
        config = parse_config(body)
        data = self.pool.data()
        with self.pool.lock:
            answer = pick_random_answer(data, config)
        return self.send_json({"word": data.all_words[answer]})

    def api_board(self, body):
        """Set the whole board, and the whole of how to solve it, at once.

        The page sends every row rather than only the newest one, so that
        correcting a tile four rows up is the same request as playing a new
        word.  The settings ride along for the same reason: changing a prior is
        then this request too, and the board survives it.  A board that cannot
        be true is refused whole -- the session keeps the rows and the settings
        it had, and the page keeps the edit for the user to fix.
        """
        session_id = str(body.get("session", ""))
        session = self.store.get(session_id)
        settings = parse_settings(body)

        raw_rows = body.get("rows")
        if not isinstance(raw_rows, list):
            raise SolverError("Malformed board")
        if len(raw_rows) > MAX_GUESSES:
            raise SolverError(f"A board is at most {MAX_GUESSES} rows")

        data = self.pool.data()
        answer = parse_answer(body, data)
        rows = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                raise SolverError("Malformed board")
            word = str(row.get("word", "")).strip().lower()
            if word not in data.word2ind:
                raise SolverError(f"Row {index + 1}: "
                                  + (f"'{word}' is not in the word list" if word
                                     else "needs a five letter word"))
            try:
                pattern = parse_pattern(row.get("pattern"))
            except SolverError as exc:
                raise SolverError(f"Row {index + 1}: {exc}")
            rows.append((data.word2ind[word], pattern))

        state = build_state(self.pool, session_id, settings, rows, answer)
        session.settings = settings
        session.rows = [(data.word2ind[row["word"]], parse_pattern(row["pattern"]))
                        for row in state["rows"]]
        return self.send_json(state)

    # -------------------------------------------------------- the word finder

    def api_find(self, body):
        """Which words could have produced this pattern against this answer.

        Backwards from the rest of the page: the answer is known and the guess
        is what is being looked for.
        """
        data = self.pool.data()
        answer = str(body.get("answer", "")).strip().lower()
        if answer not in data.word2ind:
            raise SolverError(f"'{answer}' is not in the word list" if answer
                              else "Type the answer, a five letter word")
        return self.send_json(find_guesses(
            self.pool, data.word2ind[answer], parse_pattern(body.get("pattern")),
            parse_config(body), parse_sort(body)))

    # -------------------------------------------------------------- play mode

    def api_game_new(self, body):
        """Start a game.  The answer is either the one asked for -- handy for
        setting the solver a position -- or drawn from the prior, exactly as
        ``wordle_game.py --random`` draws it."""
        config = parse_config(body)
        data = self.pool.data()
        wanted = str(body.get("answer", "") or "").strip().lower()
        if wanted:
            if wanted not in data.word2ind:
                raise SolverError(f"'{wanted}' is not in the word list")
            game = Game(data.word2ind[wanted], config, "the one you chose")
        else:
            with self.pool.lock:
                answer = pick_random_answer(data, config)
            game = Game(answer, config, f"drawn from {config.describe()}")
        return self.send_json(game_state(self.pool, self.games.create(game), game))

    def api_game_guess(self, body):
        session_id = str(body.get("session", ""))
        game = self.games.get(session_id)
        if game.over:
            raise SolverError("This game is over")

        word = str(body.get("word", "")).strip().lower()
        data = self.pool.data()
        if word not in data.word2ind:
            raise SolverError(f"'{word}' is not in the word list"
                              if word else "Type a five letter word")

        answer = data.all_words[game.answer]
        game.rows.append((data.word2ind[word], score_guess(word, answer)))
        return self.send_json(game_state(self.pool, session_id, game))

    def api_game_hint(self, body):
        session_id = str(body.get("session", ""))
        game = self.games.get(session_id)
        if game.over:
            raise SolverError("This game is over")
        return self.send_json(game_hint(self.pool, game, parse_settings(body)))

    def api_game_reveal(self, body):
        session_id = str(body.get("session", ""))
        game = self.games.get(session_id)
        game.gave_up = not game.solved
        return self.send_json(game_state(self.pool, session_id, game))

    # ------------------------------------------------------------------ static

    def serve_static(self, path):
        name = "index.html" if path == "/" else path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, name))
        # a prefix test is not enough: web/ and web.py share one.  relpath also
        # copes with a name naming another drive, which normpath leaves absolute
        try:
            inside = not os.path.relpath(full, STATIC_DIR).startswith(os.pardir)
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(full):
            return self.send_error_json("Not found", status=404)
        relative = os.path.relpath(full, HERE).replace(os.sep, "/")
        if relative in ("web/index.html", "web/app.js"):
            body = WORDING.served(relative).encode("utf-8")
        else:
            with open(full, "rb") as handle:
                body = handle.read()
        content_type = mimetypes.guess_type(full)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------------ main

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default %(default)s, this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--quiet", action="store_true", help="no request log")
    parser.add_argument("--wording-file", default=None, metavar="PATH",
                        help="where the rewrites editor.py saves are kept "
                             "(default wording.json)")
    parser.add_argument("--mask-last", type=int, default=0,
                        help="ignore the last N entries of solutions.json "
                             "(default %(default)s)")
    parser.add_argument("--no-exact", action="store_true",
                        help="never use the exact endgame search, only the estimator")
    parser.add_argument("--exact-max", type=int, default=DEFAULT_EXACT_MAX,
                        help="largest candidate set the exact search takes on when "
                             "three or more turns remain (default %(default)s)")
    parser.add_argument("--exact-branch", type=int, default=DEFAULT_MAX_BRANCH,
                        help="splits the exact search tries per position (default "
                             "%(default)s; 0 tries every one)")
    parser.add_argument("--exact-root-branch", type=int, default=DEFAULT_ROOT_BRANCH,
                        help="splits it tries at the position being ranked (default "
                             "%(default)s)")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="plies of branch-capped lookahead (default %(default)s, "
                             "off); 1 costs roughly 10x the wall clock per guess")
    parser.add_argument("--lookahead-branch", type=int,
                        default=DEFAULT_LOOKAHEAD_BRANCH,
                        help="guesses expanded per position when --lookahead is on; "
                             "a fan-out at each node, so a second ply multiplies "
                             "the work by it again")
    parser.add_argument("--loss-penalty", type=float, default=DEFAULT_LOSS_PENALTY,
                        help="cost of running out of turns, on top of the six spent "
                             "(default %(default)s; 1.0 is risk neutral)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    pool = SolverPool(mask_last=args.mask_last, options=solver_options(args))
    if args.wording_file:
        global WORDING
        WORDING = wording.Wording(path=os.path.abspath(args.wording_file))

    Handler.pool = pool
    Handler.store = Store("board")
    Handler.games = Store("game")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.quiet = args.quiet

    url = f"http://{args.host}:{server.server_port}/"
    print(f"Wordle solver on {url}")
    print("Rewrite what it says with editor.py, which edits the static build.")
    print("Warming up: the feedback table and the kernels take a moment.")

    threading.Thread(target=pool.warm, args=(PriorConfig(), EXPECTED, None),
                     daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
