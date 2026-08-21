# Working in this repo

A Wordle solver: a scoring engine over the full word list, an estimator `f(n)` fitted by policy iteration, an exact endgame search, two terminal front ends and a web page that does both.

`README.md` is the reference — priors, objective, endgame, results. Read `SOLVER_EXPLAINED.md` for the derivation from scratch, by way of `ABOUT_SOLVER_EXPLAINED.md`, which says who it is written for and in what order to read it. Neither is a summary of the other and both are kept current; if you change behavior, update the one that documents it.

`ABOUT_SOLVER_EXPLAINED.md` is the front door to the second of those: what it is for, how to view its LaTeX, and the route through its sections. The viewing instructions live there and nowhere else — do not restore a copy of them to the top of `SOLVER_EXPLAINED.md`, and if you add or remove a section, its section map is the thing that goes stale.

**Read its *If you edit it* rules before changing `SOLVER_EXPLAINED.md`.** That document decays by accretion. A result gets appended, then a paragraph reconciling it with the one above, and eventually the whole thing reads as a changelog instead of an explanation. It has been rewritten once to undo exactly that. The short version: edit the sentence that carries a number instead of adding one that says the number moved, and never let the document mention its own revision history.

## Conventions

**Comments stay to a line or two.** No multi-paragraph blocks justifying a change, no measurement tables in source. The reasoning and the numbers go in `README.md`; the comment says what the line does. File headers explaining what a module is are the exception and are welcome.

**Vary how sentences are built.** One shape should not carry a whole paragraph: a clause, a comma, and a tail. "X, so Y", "X, rather than Y", ", which …", an em-dash aside. Each of those is good English by itself, and the trouble is only the rate at which it recurs. Mix in sentences that open on a subordinate clause, sentences joining two full clauses, and sentences that state a thing and stop. Chopping the long ones into short ones is not the fix; it reads worse than what it replaced.

**Hold that one loosely.** It is a smell worth checking when a paragraph reads badly, and not a rule to enforce by counting. Three sentences in a row can take the same shape if they read well, and prose that already works is better left alone than rewritten to a quota.

**Point at code by name, never by line number.** `priors.build_weights`, not `priors.py:221`. Line references go stale on the next edit and nothing checks them; a single pass over `SOLVER_EXPLAINED.md` found seven that had drifted. A `module.function` name is stable, greppable, and fails loudly instead of quietly when the symbol is renamed.

**Line endings are LF, everywhere, including `web/`.** The whole tree is normalized, so a CRLF file is a regression rather than a local convention.

Write JSON through `wordle_data.write_json`, which handles this. Anything else that rewrites a file must pass `newline=""` to `open` — Python's text mode writes CRLF on Windows, so without it every line of the file changes and the real edit disappears into a whole-file diff.

**Markdown is one line per paragraph. Python is hard-wrapped.** Markdown here is written unwrapped, because it is always read through something that wraps it for you: a browser, a gist, VS Code's preview, and VS Code's editor too, which ships `editor.wordWrap: on` for Markdown. The source width therefore reaches no reader, while hard wrapping costs a reflow on every edit, hides phrases that straddle a line break from `grep`, and turns a one-word change into four lines of diff. Python keeps its ~80 columns, where source *is* what gets read and a 200-character line is genuinely worse. Every Markdown file in the repo is converted, so a wrapped paragraph in one is a regression rather than a local style — `python unwrap_md.py` says so and exits non-zero, and `--write` fixes it. Converting a file is a whole-file diff, so it goes in its own commit touching nothing else.

## Running it

**The dependencies live in `.venv`, and nowhere else.** The system Python has no `numba`, so `python web.py` dies on the import; run `.venv/Scripts/python.exe` instead, or activate the environment first.

```
.venv/Scripts/activate            PowerShell: .venv\Scripts\Activate.ps1
python web.py                     the page, on http://127.0.0.1:8000
python wordle_solver.py play      the solver in the terminal
python wordle_game.py             a game in the terminal
./build_site.sh                   the same page as static files, in site/
./deploy.sh                       build that, and upload it to Vercel
python editor.py                  rewrite what the site says, on :8001/editor
```

`requirements.txt` splits in two: the solver needs `numpy` and `numba`, and the last two lines are for the one-off builders only. It is what `.venv` was built from; installing into the system Python is not the setup here.

Startup loads `feedback_table.npy` (167 MB) and compiles the numba kernels, which takes a while. `web.py` warms both in a background thread and the page waits behind its veil; `/api/status` reports progress and flips `ready`. Poll that before assuming the server is usable.

`WordleSolver(fused=True)`, or `WORDLE_FUSED=1` in the environment, builds a second answer-major copy of the table and dispatches mid-sized positions to `fused_kernels`. It is off by default and scores bit-identically either way, so it is a speed knob and never a behavior one. Worth 1.09× over whole games in expected mode and 1.63× in entropy, for 168 MB and 0.3 s at startup; `SOLVER_EXPLAINED.md` *Storing the table the other way round* says why it is a dispatch rather than a replacement.

`WORDLE_ROOT_BRANCH=50` narrows the exact search back to one cap, and unlike the one above it *is* a behavior knob — a deliberately cheap one, since the winner is exact at 50 either way and only a ranking's tail moves. It exists because the wider cap takes a top five from 12 ms to 18 ms on a mean position and from 501 ms to 1,125 ms on the worst, which buys the site something an experiment timing the endgame does not want to pay for. Both implementations read it, so parity holds under it; `--exact-root-branch` sets it per run on `web.py`, `wordle_solver.py` and `analysis/split_ranking.py`, and it reaches the ~30 places in `analysis/` that build a solver without a flag. There is no environment under WASM, which is the build that should never be narrowed.

The whole solver sits behind one lock. Scoring ~13k guesses already uses every core, so concurrent requests would only make both slower, and the memo tables are not written for concurrent use.

Expected-guesses play on a prior with no cached fit raises `estimator.NotFitted` rather than inventing an `f`, so a prior combination assembled on the fly needs `train --this-config-only` first — the exception's message spells the command out. Entropy mode needs no fit and runs on anything.

**The solver is being ported to Rust**, so it can compile to WASM behind a free static site and to a native binary. What the native side is worth is measured rather than assumed: a full `every_answer` sweep runs in about 6.8 s against the Python's 10.0, because two thirds of the Python's time is outside the kernels and that part is roughly twice as fast in Rust. The kernels themselves are memory-bound in both and will not move much. `cargo build --release`, or `./build.sh` since rustup was installed without touching PATH. The Python stays the oracle throughout and keeps `estimator.fit`, `answer_like.py`, the two fetchers and all of `analysis/`'s reporting; nothing is deleted until the Rust matches it in `tests/rust_parity.py`.

**The page runs off the Rust with no server at all.** `./build_site.sh` writes `site/`; `README.md` § *Without a server* has what is in it and what it gives up. Two rules hold there. `web/app.js` is served unmodified — `web/wasm-api.js` stands in for `fetch` on `/api/*` instead, because `tests/board.test.js` loads `app.js` through `vm` and a fork would mean the board tests cover a file that does not ship. And the API lives in `wordle_core::api`, not in the `wasm` crate, so `native/src/bin/api_probe.rs` can drive an endpoint from a shell; every endpoint bug so far was found that way and none of them were quick to see through a browser.

The module itself sits in `web/solver-worker.js` rather than on the page's thread. A wide position costs upwards of 150 ms, and on the main thread that is 150 ms in which nothing repaints or accepts a keystroke. The work takes just as long in the worker; what changes is that the page stays alive through it. The swap is invisible to `app.js` because `fetch` was always asynchronous — a request is now a `postMessage` and a reply, and the worker's event loop serializes them in arrival order for free.

**Port the optimizations, not just the algorithms.** `kernels.py` declares all seven of its kernels `parallel=True` and caches the all-rows pool in `_ALL_ROWS`; leaving either out of the Rust made a whole sweep slower than the Python it replaced, and the cause was invisible until it was measured. `cargo build --release --features wordle_core/profile` times every kernel entry point, which has settled this three times where guessing got it wrong three times; `api_probe` prints those totals for a single request, which is how the third one was found. The same trap caught `api::find` outside the kernels: fetching the guess's letters per candidate rather than once per guess cost 22×, and the loop looked perfectly ordinary. The browser also needs one the Python has no use for, having no table to read: `source::Columns` tabulates the searched set's columns once per search, because the endgame rescans the whole pool at every node it visits. `README.md` § *Without a server* has that and the memo that goes with it.

What the Rust still does not have is `fused_kernels`' answer-major dispatch. That is opt-in on the Python side too, so the two are comparable as they stand.

**Reduce a float vector through `sum::pairwise`, and break a tie by index.** Those two rules are what make the two implementations bit-identical rather than merely close, and both were bought rather than assumed. `np.sum` adds pairwise, so its error grows as log n where a left-to-right loop grows as n; `sum.rs` transcribes it, which is the more accurate of the two orders as well as the reproducible one. Use it wherever the Python spells `.sum()` on floats. The exception is `kernels.rs`, where the numba loops accumulate left to right and a naive sum is what matches — `api::find` stands in for `kernels.ig_all` and belongs to that exception too. The unreachable case is a numpy `@`: BLAS reduces in an order that is neither documented nor stable across builds, which is why the letter odds are the one field `tests/rust_parity.py` still compares on a tolerance.

The tie-break matters because the endgame's branch cap keeps the best fifty splits and the bounds tie in bulk — one position holds 1,176 of its 1,177 at exactly 2.0. Sorting them unstably means each side searches a different fifty, which is a real difference and not a cosmetic one: the two used to return runners-up 0.02 guesses apart. Both now sort stably, so a tie goes to the lower word index. Weights that differ in their last bits break a tie the same way, which is why the two rules only work together.

Fast math is not a hazard you can reach by accident here: `rustc` exposes no such flag, stable or unstable, and release-mode LLVM does not reassociate a float sum. If you ever add a crate or a nightly intrinsic that grants it, know what it cost in Python — it hoisted a store above a load, every bucket maximum read back as the zero its own loop had just written, and `two_turn_scores` returned a flat 3.0 for every guess. That is a wrong answer rather than a wobbly last bit, which is why `kernels.py` carries none of it.

## The pieces

| file | what it is |
| --- | --- |
| `solver.py` | scoring and guess choice; `WordleSolver` is the entry point |
| `kernels.py` | the numba inner loops |
| `fused_kernels.py` | the same scores off an answer-major copy of the table; opt-in |
| `exact.py` | exact endgame search |
| `memo.py` | the bounded LRU both of those cache positions in |
| `estimator.py` | `f(n)`, its forms, and fitting |
| `train.py` | policy iteration that fits `f` |
| `priors.py` | the repeat and curation priors |
| `wordle_data.py` | word lists, feedback scoring, the table |
| `wordle_solver.py` | the terminal front end: `play`, `train`, `benchmark`, `compare`, `info` |
| `wordle_game.py` | play Wordle yourself, against a chosen or a sampled answer |
| `web.py` | the HTTP server and JSON API |
| `web/app.js` | the whole page; served unmodified by both the Python and the static build |
| `web/wasm-api.js` | stands in for `fetch` on `/api/*` in the static build, and passes it to the worker |
| `web/solver-worker.js` | holds the module and answers those calls, off the page's thread |
| `build_site.sh` | assembles `site/`, the whole thing as static files |
| `deploy.sh` | runs that and uploads the result; `site/` is generated, not committed |
| `wording.py` | the editor's string scanner |
| `editor.py` | the editor: rewrites what the site says, previewing the static build |
| `unwrap_md.py` | checks that every `.md` is one line per paragraph; `--write` converts |
| `feedback_table.py` | builds `feedback_table.npy` and the index maps |
| `answer_like.py` | builds `answer_like.json`, the answer-like scores |
| `solutions.py` | brings `solutions.json` up to date from the NYT |
| `word_lists.py` | refetches `wordles.json` and `nonwordles.json` |
| `analysis/` | the scripts behind every measured number in the two documents |
| `core/` | the Rust port of the solver; compiles to WASM and to a native binary, and answers the page's API |
| `wasm/` | marshalling and nothing else: the module's linear-memory entry points |
| `native/` | Rust binaries the analysis scripts drive, and the parity checks against the Python |

## Where the data comes from

`data/` holds the curated inputs and the answer record; `data/derived/` holds only what `feedback_table.py` regenerates from the two word lists, so it is the part that is safe to throw away. Every path is resolved from `wordle_data.py`'s `DATA_DIR` and `DERIVED_DIR` rather than spelled out again per module.

Every data file has a builder except one. Three of them — `answer_like.py`, `solutions.py`, `word_lists.py` — **verify against the shipped file and write nothing** unless you pass `--write`. They rewrite inputs the solver reads, and `word_lists.py` invalidates `feedback_table.npy` outright, so the default is to report the difference and stop.

`feedback_table.py` is the exception and predates that rule: it has no `--write` and no `__main__` guard, so **merely importing it rebuilds the index maps and starts on the 167 MB table**. Run it deliberately or not at all.

| file | built by | reproduces the shipped copy? |
| --- | --- | --- |
| `data/wordles.json`, `data/nonwordles.json` | `word_lists.py` | yes, exactly |
| `data/solutions.json` | `solutions.py` | it *is* the cache — a rerun fetches only the days since the last one |
| `data/derived/*` — the table and the index maps | `feedback_table.py` | yes, deterministic |
| `data/answer_like.json` | `answer_like.py` | yes, exactly |
| `data/wordlebot_weights.json` | nothing | a third-party scrape of WordleBot; not rebuildable here |

`answer_like.py` needs `wordfreq`, which nothing else here does. Its fit is sensitive to details that are easy to guess wrong: thirteen features and not the four `SOLVER_EXPLAINED.md` singles out, exactly ten foreign languages, a duplicate-letter *count* where a flag would seem enough, and a ridge that hits the intercept too. Change any of them and it still fits, still correlates at r ≈ 0.98, and no longer reproduces the file.

## The web page

Three views — solve, play and find — over one board pane and one set of switches. The finder is the odd one: it is given the answer and looks for the guess, so it has its own two rows, its own panel, and `/api/find` instead of a session.

The server holds sessions; the page holds an editable copy of the board and sends **the whole board** on every submit, never a delta. The reply is the board as the solver accepted it, and the local copy is rebuilt from it, so the two cannot drift. Editing an earlier row is therefore the same request as playing a new one.

**The settings ride along with the board**, as one `Settings` value parsed per request. Nothing about them lives on the session. Changing one is then the same request as playing a row, which is what lets a switch re-run the solver over the rows already played instead of clearing them. Only the answer box deals a new game. Nothing on the server has to be kept in step with the switches; do not add a setting that is remembered at `/api/new` time.

Typing and coloring are local until Enter. Taking a word off is not, whether by *Delete row* or by backspacing the row empty: `refreshAfterRemoval` submits the shorter board itself, since a shadow and a panel still describing the word just removed read as no removal at all. A hole still open in the middle stays local, there being no valid board to send.

The rule that follows is worth stating whole: **a board reached by taking words off is indistinguishable from the same board played straight** — shadow, cursor, panel, and which rows can be typed in. `trimRows` is the last of those: exactly one free row after the last word, so nothing selectable is left below the shadow. `README.md` § *Editing the board* has the rest, with why they are what they are; they are not obvious and they were each a bug once.

Requests are serialized: one in flight, and a second made meanwhile is held, not dropped. Dropping it used to leave a view with nothing in it.

Patterns cross the wire as either `"NYGNN"` or `[0, 1, 2, 0, 0]`; the server takes both, and the page sends whichever it has to hand. A test double for the API has to accept both too.

`python editor.py` rewrites what the site says by scanning the source for user-visible strings. It means **string literals in `index.html`, `app.js`, `web.py` and `core/src/api.rs` are data**: changing one changes what the editor lists, and any saved override keyed to the old wording is orphaned. `wording.json` holds live overrides; *Write into the code* bakes them into the source and logs the phrases to `wording_history.json`. There is no file-level undo.

It previews `site/`, so what you are editing is what deploys, and it loads no word lists and no table — `web.py` used to serve it and had to warm the whole solver first, to change a sentence. A phrase baked into the Rust needs `./build_site.sh` before the preview says it, being inside the compiled module by then rather than in a file being served.

**Every message exists twice**, once in `web.py` and once in the Rust the static build answers from. Both are listed and both have to be edited; carrying an edit across was tried and does not work, since the two write their placeholders differently and `{index + 1}` is not a name Rust can have. `tests/rust_parity.py`'s message check drives both with the same bad requests and fails when one of them was rewritten and the other was not.

**`wording_history.json` is a record of the past.** Its `was:` fields quote what the code used to say. Do not sweep it with a project-wide rewrite — that makes it false. It is left exactly as written, even where the code it quotes has since been reworded.

## Numbers in the documents

Every measured claim about the priors is reproduced by a script in `analysis/`, and the claims move with `solutions.json` — rerun after `solutions.py --write`, and refit before rerunning `analysis/replay.py` or it measures a stale policy against fresh answers. Nothing warns you: the estimator cache is keyed on the `PriorConfig` alone, so a fit made against a shorter answer log loads silently.

`analysis/repeat_shapes.py --events 18` is the method check: it scores a frozen 18 repeats, so what it prints should not move at all. `analysis/README.md` § *Rerunning them* carries those figures and the other two fixed points, and is the only place they are written down — quote them from there rather than copying them here, and never read one of them back as though it were the live number the same script prints without `--events`.

## Testing

`tests/` holds eight files and `tests/README.md` describes them; `python tests/run.py` runs the lot in a few seconds, and `--full` adds what needs the table. They are deliberately narrow — the board's editing rules, the two front ends agreeing on how the solver is asked, whether the cached `f` is a fit of the answer record we have, whether the guesses a ranking names really are distinct splits, whether each scoring rule is the policy the documents describe, whether every pointer into the documents still resolves, whether the static build's shim refuses in a way the page survives, and whether the Rust core agrees with the Python it was ported from.

**Nothing there asserts a number written down by hand, and nothing new should.** Each check has an oracle instead: the other front end, the answer record, or a property that holds of every board. A figure copied into a test is a fourth place for it to go stale, and there are three already.

The board tests are the pattern to follow for page logic: a small fake DOM plus a fake `fetch`, `web/app.js` loaded unmodified via `vm`, and a getter object appended to reach its internals. Cursor and board-state bugs are where the page's bugs live. Add cases as **properties over a random walk of whole intentions** (*a word*, *clearing the row by hand*), not as more sequences you thought of: bare random keystrokes never clear a row, and the anchors only ever cover what someone already had in mind. The phantom free row was found by the walk, not by anyone writing a case for it.

Server endpoints are quickest to check with `http.client` over a single keep-alive connection — several bugs have been connection-reuse bugs, and one request per connection hides them.

Do not trust a solver change without checking it end to end: build a truthful board with `score_guess(guess, answer)` instead of hand-writing patterns, or the server rejects it as contradictory and the test proves nothing.
