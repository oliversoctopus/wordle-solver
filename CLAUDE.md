# Working in this repo

A Wordle solver: a scoring engine over the full word list, an estimator `f(n)` fitted by policy iteration, an exact endgame search, two terminal front ends and a web page that does both. README *Layout* lists the files.

`README.md` is the reference: priors, objective, endgame, results. `SOLVER_EXPLAINED.md` is the derivation from scratch; `ABOUT_SOLVER_EXPLAINED.md` says who it is for, how to view it, and holds the *If you edit it* rules to read first. When behavior changes, update whichever document describes it.

## Never poll background work

* A turn whose message says "waiting" contains zero tool calls, since the harness notifies when a background task exits. Wait by editing, reading unread source or drafting, and ending the turn with nothing pending is correct.
* Before any tool call, ask whether it changes a file or yields something otherwise unavailable, and if neither, do not make it. Two consecutive calls returning the same output is the loop signature: stop, and an instruction to stop holds for the rest of the session.

## Conventions

* **Comments are one or two lines** saying what the line does. A file header says what the module is, in a few lines. No measurements, no rationale, no history anywhere in source. A decision that needs defending gets one line naming the README section that defends it.
* **Point at code by name**: `priors.build_weights`, `source::Columns`, never `priors.py:221`. `tests/doc_refs.py` resolves every pointer in the documents and the analysis docstrings.
* **Line endings are LF everywhere.** Write JSON through `wordle_data.write_json`. Anything else that rewrites a file passes `newline=""` to `open`.
* **Markdown is one line per paragraph. Python is hard-wrapped at about 80 columns.** `python unwrap_md.py` checks; `--write` converts, in its own commit.
* **Every measured number in a document is bound** to a results file. See *Numbers in the documents*.
* **A rule gets a check.** `tests/voice.py` checks *Voice* and the comment rules above. Propose a check with any new rule.

## Voice

Every sentence in this repo states a fact and stops, and the documents follow `VOICE.md` above that floor. `tests/voice.py` bans the phrases: a trailing justification ("…, which is what lets X"), evaluative filler ("genuinely"), "X rather than Y" where Y was never an option, the "Not X. Y." reversal, and narrated history ("used to", dates of edits).

Write the sentence a careful person would put in an email. A rule is one line. A comment is one or two.

Before finishing, reread the diff against `VOICE.md` and *Conventions*.

## Running it

The dependencies live in `.venv`; the system Python has no `numba`. Run `.venv/Scripts/python.exe` or activate the environment first.

```
.venv/Scripts/activate            PowerShell: .venv\Scripts\Activate.ps1
python web.py                     the page, on http://127.0.0.1:8000
python wordle_solver.py play      the solver in the terminal; the page's oracle
python wordle_game.py             a game in the terminal; web.py imports from it
./build_site.sh                   the same page as static files, in site/
./deploy.sh                       build that, and upload it to Vercel
python editor.py                  rewrite what the site says, on :8001/editor
```

* Startup loads the 167 MB table and compiles the numba kernels. `web.py` warms both in the background; poll `/api/status` until `ready`.
* `play` stays because `tests/front_ends_agree.py` diffs its parser against `web.py`'s. `web.py` imports `wordle_game.TILE_BLOCK` and `wordle_game.pick_random_answer`.
* `WORDLE_FUSED=1` is a speed knob and changes no score. `WORDLE_ROOT_BRANCH=50` narrows the ranked node's branch cap and moves the runners-up; WASM has no environment. README *The exact endgame*.
* The solver sits behind one lock. The kernels use every core.
* Expected-guesses play on a prior with no cached fit raises `estimator.NotFitted`, whose message names the `train` command that makes one. Entropy mode needs no fit.
* **A refit invalidates `site/`.** `build_site` bakes the openings into `site/openings.json`. Rerun `./build_site.sh` after any refit before reading a parity failure as a port bug. It calls bare `python`, so have `.venv` on PATH.

## The Rust port

`core/` is the solver, `wasm/` the linear-memory entry points, `native/` the binaries the scripts drive and the parity checks. `cargo build --release`, or `./build.sh` when cargo is not on PATH. `tests/rust_parity.py` diffs every ported piece against the Python, which stays the oracle. Reasons for the rules below: README *Keeping the port in step*.

* `analysis/rust.py` hands a whole-answer sweep to `every_answer` where the flags reproduce the solver, and declines otherwise. `WORDLE_NO_RUST=1` turns it off; anything using `common.every_answer` as an oracle sets it.
* `train` fits in Rust by default through `rust_train.py`; `--no-rust` fits in Python. The two produce different valid fits, and parity stops at the seed round. README *Fitting in Rust*.
* In `site/`, `web/app.js` is served unmodified; `web/wasm-api.js` stands in for `fetch` on `/api/*`, and `web/solver-worker.js` holds the module. The API lives in `wordle_core::api` so `native/src/bin/api_probe.rs` can drive an endpoint from a shell. README *Without a server*.
* Port the optimizations with the algorithms: every kernel parallel, the all-rows pool cached, `source::Columns` for the endgame. Measure with `--features wordle_core/profile` before guessing where time goes.
* Reduce a float vector through `sum::pairwise` wherever the Python spells `.sum()`. `kernels.rs` and `api::find` accumulate left to right to match their bucket totals. A numpy `@` cannot be matched; those fields are compared on a tolerance.
* Break a tie by word index over a key rounded to `BRANCH_TIE_TOL`. A cap or a top-k never stands on last bits.
* No fast math anywhere. `rustc` exposes none; do not add a crate that does.

## Where the data comes from

`data/` holds the curated inputs and the answer record; `data/derived/` holds what `feedback_table.py` regenerates. Every path comes from `wordle_data.py`.

* `answer_like.py`, `feedback_table.py`, `solutions.py`, `word_lists.py` and `wordlebot.py` write nothing without `--write`. Each reproduces the shipped copy, except that `solutions.json` is its own cache and `wordlebot_weights.json` follows the live site.
* `word_lists.py --write` invalidates `feedback_table.npy`. `wordlebot.py --snapshot-only` refreshes the frozen 2023 copy alone.
* `answer_like.py` needs `wordfreq` and reproduces its file only with its exact feature set, language list and ridge.

## The web page

Three views, solve, play and find, over one board and one set of switches. README *In a browser* describes them and *Editing the board* has the board rules.

* The page sends the whole board on every submit, never a delta.
* The settings ride along with every request as one `Settings` value; nothing about them lives on the session. Do not add a setting remembered at `/api/new` time.
* Taking a word off submits the shorter board itself. A board reached by removing words is indistinguishable from the same board played straight.
* One request is in flight at a time; a second is held, never dropped.
* A pattern crosses the wire as `"NYGNN"` or `[0, 1, 2, 0, 0]`; every reader takes both.
* The editor scans string literals in `index.html`, `app.js`, `web.py` and `core/src/api.rs`. Every message exists twice, in `web.py` and in the Rust; edit both.
* `wording_history.json` records past wording. Never rewrite it.

## Numbers in the documents

A script calls `common.record(key, value)` at the print site; the figure lands in `analysis/results/<script>.json`; the document marks the figure with an HTML comment naming the key, as `doc_numbers.py`'s docstring shows. `python doc_numbers.py` checks, `--write` rewrites and lists what moved, `--unbound` fails on an unmarked figure. Never type a measured number into a document.

The refresh: `solutions.py --write`, refit, rerun the live scripts, `doc_numbers.py --write`, reread what moved. `analysis/README.md` *Rerunning them* says which scripts are live and which are findings.

The refit: `wordle_solver.py compare` per shipped prior, then `analysis/form_default.py --promote`. The three shape-comparison priors are refitted only with `--shapes`. The estimator cache is keyed on the `PriorConfig` alone, so refit before `analysis/replay.py`. `priors.depends_on_answer_record` names the priors a record update cannot move; `--repeat-only` skips them. Which form wins a refit is fit noise, not a finding.

A cause needs a measurement. Before writing "because", name two candidate causes and run something that separates them, or label the claim as a guess. Say which kind of claim a sentence is: measured, follows from the code, or guessed.

## Testing

`python tests/run.py` runs the quick checks and `--full` the rest; `tests/README.md` describes them.

* Nothing asserts a number written by hand. Every check runs over inputs nobody chose, against a property, the answer record, or a second implementation.
* `tests/anchors.py` is the exception: facts from outside the repository. Pin the fact, never the behavior around it.
* Page logic follows `tests/board.test.js`: a fake DOM, a fake `fetch`, `web/app.js` unmodified, properties over a random walk of whole intentions.
* Check server endpoints with `http.client` over one keep-alive connection.
* Build a test board with `score_guess(guess, answer)`, never hand-written patterns.
