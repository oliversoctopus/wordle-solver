"""Web front end for the Wordle solver.

    python web.py                 serve on http://127.0.0.1:8000 and open a browser
    python web.py --port 9000     somewhere else
    python web.py --no-browser    do not open one

The page does both of the terminal front ends, and switches between them:

* **solve** is ``wordle_solver.py play`` -- here is my recommendation, tell me
  what you guessed and what came back -- except that the feedback is entered by
  coloring tiles rather than typing ``NYGNN``, and the prior toggles and the
  mode are switches rather than flags.
* **play** is ``wordle_game.py``: an answer is drawn from the same prior (or
  typed in), you guess, and the solver is there for a hint if you want one.
  The answer stays on this side until the game ends.

Only the standard library is used, so there is nothing to install: the server is
``http.server``, and the page is three static files in ``web/``.  The 167 MB
feedback table and the numba kernels are warmed up in the background as soon as
the server starts, and the page waits for them.

The whole solver is behind one lock.  Scoring ~13k guesses already uses every
core, so letting two requests do it at once would only make both slower, and the
solver's memo tables are not written for concurrent use.
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

import estimator
import wording
from exact import DEFAULT_MAX_BRANCH
from priors import PriorConfig
from solver import (DEFAULT_BEAM_WIDTH, DEFAULT_EXACT_MAX, DEFAULT_LOSS_PENALTY,
                    ENTROPY, EXPECTED, MODES, WordleSolver)
# the terminal game's answer-drawing and emoji grid, so both front ends deal
# the same answers and print the same thing at the end
from wordle_game import TILE_BLOCK, pick_random_answer
from wordle_data import WIN_PATTERN, load_data, num_to_feedback, score_guess

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "web")

MAX_GUESSES = 6
MAX_SESSIONS = 200          # oldest are dropped; a session is a few dozen bytes
MAX_LISTED_CANDIDATES = 80  # possible answers sent to the page, likeliest first
MAX_BODY = 64 * 1024

TILE_VALS = {"N": 0, "Y": 1, "G": 2}

# every string the page and this file say, and whatever /editor has
# rewritten them to
WORDING = wording.Wording()


class SolverError(Exception):
    """Something the user can fix, reported as a 400 with this message."""


# --------------------------------------------------------------------- solvers

class SolverPool:
    """Solvers, one per (prior, mode, form), built on demand and kept.

    Building one costs a pass over the word lists; the feedback table behind
    them all is loaded once by ``load_data``'s own cache.
    """

    def __init__(self, mask_last=1, options=None):
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

    def get(self, config, mode, form):
        """Caller must hold ``self.lock``."""
        key = (config.slug, mode, form)
        solver = self._solvers.get(key)
        if solver is None:
            solver = WordleSolver(data=self.data(), config=config, mode=mode,
                                  form=form, quiet=True, **self.options)
            self._solvers[key] = solver
        return solver

    def warm(self, config, mode, form):
        """Load the table, compile the kernels and compute the opening guess, so
        that the first thing the page asks for is not also the slowest."""
        try:
            self.progress = "Loading the feedback table (167 MB)"
            with self.lock:
                self.data()
                self.progress = "Compiling the scoring kernels"
                solver = self.get(config, mode, form)
                self.progress = "Computing the opening guess"
                solver.best_guess(solver.all_candidates(), MAX_GUESSES)
            self.progress = "Ready"
            self.ready = True
        except Exception:
            traceback.print_exc()
            self.failed = traceback.format_exc(limit=0).strip()
            self.progress = "Failed to start"


# -------------------------------------------------------------------- sessions

class Session:
    """One board.  Only the rows played are kept; the candidate set is rebuilt
    from them on every request, which costs a few filters over 13k words and
    makes undo free."""

    def __init__(self, config, mode, form, top):
        self.config = config
        self.mode = mode
        self.form = form
        self.top = top
        self.rows = []  # (guess index, pattern number)


class Game:
    """One game of Wordle played against the page.

    The answer is kept here and never sent until the game is over, so that
    reading the network traffic does not spoil it.
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
    num = 0
    for tile in tiles:
        num = num * 3 + TILE_VALS[tile]
    return num


# ----------------------------------------------------------------------- state

def score_label(mode):
    return ("bits of expected information" if mode == ENTROPY
            else "expected guesses to finish")


def describe_prior(config):
    return {
        "summary": config.describe(),
        "repeat": config.repeat,
        "repeat_prob": config.repeat_prob,
        "curation": config.curation,
        "noncurated_prob": config.noncurated_prob,
    }


def describe_estimator(solver):
    """The fitted f, in pieces, so the page can set it as an equation rather
    than as the one-line summary ``estimator.describe`` prints."""
    samples = [2, 5, 20, 100, 500, 2000]
    values = estimator.f_value(samples, solver.params, solver.form)
    return {
        "form": solver.form.name,
        "description": solver.form.description,
        "base": float(solver.form.base),
        "terms": [{"label": label, "coefficient": float(value)}
                  for label, value in zip(solver.form.term_labels, solver.params)],
        "samples": [{"n": n, "f": float(v)} for n, v in zip(samples, values)],
    }


def build_state(pool, session_id, session, rows):
    """The whole board as the page wants it: what has been played, what the
    solver would play next, and what is still possible.

    ``rows`` is passed in rather than read off the session so that a proposed
    board can be scored before it is adopted; a set of rows that cannot all be
    true raises rather than returning a state.
    """
    started = time.perf_counter()
    with pool.lock:
        solver = pool.get(session.config, session.mode, session.form)
        data = solver.data

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
            "mode": session.mode,
            "score_label": score_label(session.mode),
            "prior": describe_prior(session.config),
            "candidates_left": int(len(candidates)),
            "recommendation": None,
            "alternatives": [],
            "possible_answers": [],
            "estimator": describe_estimator(solver) if session.mode == EXPECTED else None,
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
        top = solver.top_guesses(candidates, k=max(1, session.top),
                                 guesses_left=guesses_left)
        word, score = top[0]
        pick = data.word2ind[word]
        state["recommendation"] = {
            "word": word,
            "score": float(score),
            "exact": exact,
            # a guess that cannot be the answer is still often the right one:
            # worth saying which it is
            "possible": bool((candidates == pick).any()),
            "win_probability": float(solver.win_probability(candidates, pick)),
        }
        state["alternatives"] = [{"word": w, "score": float(s)} for w, s in top[1:]]
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
        "prior": describe_prior(game.config),
        "answer": None,
        "share": None,
    }
    if game.over:
        state["answer"] = data.all_words[game.answer]
        state["share"] = share_grid(game.rows, game.solved)
    return state


def game_hint(pool, game, mode=EXPECTED, form=None):
    """What the solver would play here -- the web version of the terminal
    game's ``?``.

    The prior is the one the answer was drawn from, so the hint is playing the
    same game you are.  The mode and the shape of f come with the request
    instead, so that changing either re-asks rather than dealing a new answer.
    """
    with pool.lock:
        solver = pool.get(game.config, mode, form)
        candidates = solver.all_candidates()
        for guess, pattern in game.rows:
            candidates = solver.filter(guess, pattern, candidates)
        guesses_left = max(1, MAX_GUESSES - len(game.rows))
        guess, score = solver.best_guess(candidates, guesses_left)
        return {
            "word": solver.data.all_words[guess],
            "score": float(score),
            "mode": mode,
            "score_label": score_label(mode),
            "candidates_left": int(len(candidates)),
            "exact": bool(solver.solved_exactly(candidates, guesses_left)),
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
    editable = True   # whether /editor and its routes exist

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
        if path == "/editor":
            if not self.editable:
                return self.send_error_json("Not found", status=404)
            return self.serve_static("/editor.html")
        if path == "/api/wording":
            if not self.editable:
                return self.send_error_json("Not found", status=404)
            return self.send_json(WORDING.manifest())
        if path == "/api/status":
            return self.send_json({
                "ready": self.pool.ready,
                "progress": WORDING.message(self.pool.progress),
                "error": self.pool.failed,
                "forms": sorted(estimator.FORMS),
                "default_form": estimator.DEFAULT_FORM,
                "modes": list(MODES),
                "max_guesses": MAX_GUESSES,
            })
        return self.serve_static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        # read first, whoever ends up wanting it: an unread body desynchronises
        # the kept-alive connection
        try:
            body = self.read_json()
        except SolverError as exc:
            return self.send_error_json(str(exc))

        # the wording can be edited before the solver has warmed up: it needs
        # no solver, and the loading screen is one of the things being worded
        if path in ("/api/wording/save", "/api/wording/reset", "/api/wording/apply"):
            if not self.editable:
                return self.send_error_json("Not found", status=404)
            try:
                return self.api_wording(path, body)
            except (SolverError, ValueError) as exc:
                return self.send_error_json(str(exc))

        routes = {"/api/new": self.api_new,
                  "/api/board": self.api_board,
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
        except Exception:
            traceback.print_exc()
            return self.send_error_json("The solver raised; see the server log",
                                        status=500)

    def api_new(self, body):
        session = Session(config=parse_config(body), mode=parse_mode(body),
                          form=parse_form(body),
                          top=min(10, max(1, int(body.get("top", 5)))))
        session_id = self.store.create(session)
        return self.send_json(build_state(self.pool, session_id, session, []))

    def api_board(self, body):
        """Set the whole board at once.

        The page sends every row rather than only the newest one, so that
        correcting a tile four rows up is the same request as playing a new
        word.  A board that cannot be true is refused whole: the session keeps
        the rows it had, and the page keeps the edit for the user to fix.
        """
        session_id = str(body.get("session", ""))
        session = self.store.get(session_id)

        raw_rows = body.get("rows")
        if not isinstance(raw_rows, list):
            raise SolverError("Malformed board")
        if len(raw_rows) > MAX_GUESSES:
            raise SolverError(f"A board is at most {MAX_GUESSES} rows")

        data = self.pool.data()
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

        state = build_state(self.pool, session_id, session, rows)
        session.rows = rows
        return self.send_json(state)

    # ------------------------------------------------------------ the wording

    def api_wording(self, path, body):
        """Save or drop rewrites.  Both take effect on the next page load: the
        sources are re-read and re-scanned on every request."""
        if path == "/api/wording/reset":
            keys = body.get("keys")
            if keys is not None and not isinstance(keys, list):
                raise ValueError("expected a list of keys")
            WORDING.forget(keys)
        elif path == "/api/wording/apply":
            # the source itself is rewritten here, so the rest of this process
            # is now running slightly older code than the file it came from
            return self.send_json({**WORDING.bake(), **WORDING.manifest()})
        else:
            WORDING.save(body.get("edits") or {})
        return self.send_json(WORDING.manifest())

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
        return self.send_json(
            game_hint(self.pool, game, parse_mode(body), parse_form(body)))

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
                        help="where /editor's rewrites are kept (default "
                             "wording.json).  Point a second server at a "
                             "scratch file to try the editor out without "
                             "touching the real one")
    parser.add_argument("--no-editor", action="store_true",
                        help="do not serve /editor, where every word on the "
                             "page can be rewritten and saved")
    parser.add_argument("--mask-last", type=int, default=1,
                        help="ignore the last N entries of solutions.json (default 1)")
    parser.add_argument("--no-exact", action="store_true",
                        help="never use the exact endgame search, only the estimator")
    parser.add_argument("--exact-max", type=int, default=DEFAULT_EXACT_MAX,
                        help="largest candidate set the exact search takes on when "
                             "three or more turns remain (default %(default)s)")
    parser.add_argument("--exact-branch", type=int, default=DEFAULT_MAX_BRANCH,
                        help="splits the exact search tries per position (default "
                             "%(default)s; 0 tries every one)")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="plies of width-limited lookahead (default %(default)s, "
                             "off); 1 costs roughly 10x the wall clock per guess")
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH,
                        help="guesses expanded per position when --lookahead is on")
    parser.add_argument("--loss-penalty", type=float, default=DEFAULT_LOSS_PENALTY,
                        help="cost of running out of turns, on top of the six spent "
                             "(default %(default)s; 1.0 is risk neutral)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    pool = SolverPool(mask_last=args.mask_last, options={
        "use_exact": not args.no_exact,
        "exact_max": args.exact_max,
        "loss_penalty": args.loss_penalty,
        "exact_max_branch": None if args.exact_branch <= 0 else args.exact_branch,
        "lookahead": args.lookahead,
        "beam_width": args.beam_width,
    })
    if args.wording_file:
        global WORDING
        WORDING = wording.Wording(path=os.path.abspath(args.wording_file))

    Handler.pool = pool
    # anyone who can reach /editor can rewrite the page, so it is offered only
    # when the server is listening to this machine alone
    Handler.editable = not args.no_editor and args.host in ("127.0.0.1", "::1", "localhost")
    Handler.store = Store("board")
    Handler.games = Store("game")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.quiet = args.quiet

    url = f"http://{args.host}:{server.server_port}/"
    print(f"Wordle solver on {url}")
    if Handler.editable:
        print(f"rewrite what it says on {url}editor")
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
