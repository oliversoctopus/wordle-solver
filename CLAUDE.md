# Working in this repo

A Wordle solver: a scoring engine over the full word list, an estimator `f(n)` fitted by policy iteration, an exact endgame search, two terminal front ends and a web page that does both.

`README.md` is the reference — priors, objective, endgame, results. Read `SOLVER_EXPLAINED.md` for the derivation from scratch, by way of `ABOUT_SOLVER_EXPLAINED.md`, which says who it is written for and in what order to read it. Neither is a summary of the other and both are kept current; if you change behavior, update the one that documents it.

`ABOUT_SOLVER_EXPLAINED.md` is the front door to the second of those: what it is for, how to view its LaTeX, and the route through its sections. The viewing instructions live there and nowhere else — do not restore a copy of them to the top of `SOLVER_EXPLAINED.md`, and if you add or remove a section, its section map is the thing that goes stale.

**Read its *If you edit it* rules before changing `SOLVER_EXPLAINED.md`.** That document decays by accretion. A result gets appended, then a paragraph reconciling it with the one above, and eventually the whole thing reads as a changelog instead of an explanation. It has been rewritten once to undo exactly that. The short version: edit the sentence that carries a number instead of adding one that says the number moved, and never let the document mention its own revision history.

## Conventions

**American English**, everywhere: prose, comments, identifiers, CSS custom properties, UI strings. `color`, `gray`, `favor`, `labeled`, `modeled`, `minimize`/`normalize`/`optimize`. The project was written in British English and converted wholesale on 2026-08-09, so a British spelling in existing code is a leftover to fix, not a convention to match.

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
```

`requirements.txt` splits in two: the solver needs `numpy` and `numba`, and the last two lines are for the one-off builders only. It is what `.venv` was built from; installing into the system Python is not the setup here.

Startup loads `feedback_table.npy` (167 MB) and compiles the numba kernels, which takes a while. `web.py` warms both in a background thread and the page waits behind its veil; `/api/status` reports progress and flips `ready`. Poll that before assuming the server is usable.

`WordleSolver(fused=True)`, or `WORDLE_FUSED=1` in the environment, builds a second answer-major copy of the table and dispatches mid-sized positions to `fused_kernels`. It is off by default and scores bit-identically either way, so it is a speed knob and never a behavior one. Worth 1.09× over whole games in expected mode and 1.63× in entropy, for 168 MB and 0.3 s at startup; `SOLVER_EXPLAINED.md` *Storing the table the other way round* says why it is a dispatch rather than a replacement.

The whole solver sits behind one lock. Scoring ~13k guesses already uses every core, so concurrent requests would only make both slower, and the memo tables are not written for concurrent use.

Expected-guesses play on a prior with no cached fit raises `estimator.NotFitted` rather than inventing an `f`, so a prior combination assembled on the fly needs `train --this-config-only` first — the exception's message spells the command out. Entropy mode needs no fit and runs on anything.

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
| `web/app.js` | the whole page |
| `wording.py` | the `/editor` string scanner |
| `unwrap_md.py` | checks that every `.md` is one line per paragraph; `--write` converts |
| `feedback_table.py` | builds `feedback_table.npy` and the index maps |
| `answer_like.py` | builds `answer_like.json`, the answer-like scores |
| `solutions.py` | brings `solutions.json` up to date from the NYT |
| `word_lists.py` | refetches `wordles.json` and `nonwordles.json` |
| `analysis/` | the scripts behind every measured number in the two documents |

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

`/editor` rewrites what the site says by scanning the source for user-visible strings. It means **string literals in `index.html`, `app.js` and `web.py` are data**: changing one changes what `/editor` lists, and any saved override keyed to the old wording is orphaned. `wording.json` holds live overrides; *Write into the code* bakes them into the source and logs the phrases to `wording_history.json`. There is no file-level undo.

**`wording_history.json` is a record of the past.** Its `was:` fields quote what the code used to say. Do not sweep it with a project-wide rewrite — that makes it false. It was deliberately left British when everything else was converted.

## Numbers in the documents

Every measured claim about the priors is reproduced by a script in `analysis/`, and the claims move with `solutions.json` — rerun after `solutions.py --write`, and refit before rerunning `analysis/replay.py` or it measures a stale policy against fresh answers. Nothing warns you: the estimator cache is keyed on the `PriorConfig` alone, so a fit made against a shorter answer log loads silently.

`analysis/repeat_shapes.py --events 18` is the method check: it scores a frozen 18 repeats, so what it prints should not move at all. `analysis/README.md` § *Rerunning them* carries those figures and the other two fixed points, and is the only place they are written down — quote them from there rather than copying them here, and never read one of them back as though it were the live number the same script prints without `--events`.

## Testing

`tests/` holds seven files and `tests/README.md` describes them; `python tests/run.py` runs the lot in a few seconds, and `--full` adds what needs the table. They are deliberately narrow — the board's editing rules, the two front ends agreeing on how the solver is asked, whether the cached `f` is a fit of the answer record we have, whether the guesses a ranking names really are distinct splits, whether each scoring rule is the policy the documents describe, whether every pointer into the documents still resolves, and whether the C++ core agrees with the Python it was ported from.

**Nothing there asserts a number written down by hand, and nothing new should.** Each check has an oracle instead: the other front end, the answer record, or a property that holds of every board. A figure copied into a test is a fourth place for it to go stale, and there are three already.

The board tests are the pattern to follow for page logic: a small fake DOM plus a fake `fetch`, `web/app.js` loaded unmodified via `vm`, and a getter object appended to reach its internals. Cursor and board-state bugs are where the page's bugs live. Add cases as **properties over a random walk of whole intentions** (*a word*, *clearing the row by hand*), not as more sequences you thought of: bare random keystrokes never clear a row, and the anchors only ever cover what someone already had in mind. The phantom free row was found by the walk, not by anyone writing a case for it.

Server endpoints are quickest to check with `http.client` over a single keep-alive connection — several bugs have been connection-reuse bugs, and one request per connection hides them.

Do not trust a solver change without checking it end to end: build a truthful board with `score_guess(guess, answer)` instead of hand-writing patterns, or the server rejects it as contradictory and the test proves nothing.
