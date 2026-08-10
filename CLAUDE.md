# Working in this repo

A Wordle solver: a scoring engine over the full word list, an estimator `f(n)`
fitted by policy iteration, an exact endgame search, two terminal front ends and
a web page that does both.

`README.md` is the reference — priors, objective, endgame, results. Read
`SOLVER_EXPLAINED.md` for the derivation from scratch. Neither is a summary of
the other and both are kept current; if you change behavior, update the one that
documents it.

## Conventions

**American English**, everywhere: prose, comments, identifiers, CSS custom
properties, UI strings. `color`, `gray`, `favor`, `labeled`, `modeled`,
`minimize`/`normalize`/`optimize`. The project was written in British English
and converted wholesale on 2026-08-09, so a British spelling in existing code is
a leftover to fix, not a convention to match.

**Comments stay to a line or two.** No multi-paragraph blocks justifying a
change, no measurement tables in source. The reasoning and the numbers go in
`README.md`; the comment says what the line does. File headers explaining what a
module is are the exception and are welcome.

**Line endings are LF, everywhere, including `web/`.** They were mixed until
2026-08-10 and the rule that described them ("CRLF in `web/`") was never true of
either side of that line; the whole tree is normalized now.

Write JSON through `wordle_data.write_json`, which handles this. Anything else
that rewrites a file must pass `newline=""` to `open` — Python's text mode
writes CRLF on Windows, so without it every line of the file changes and the
real edit disappears into a whole-file diff.

## Running it

```
pip install -r requirements.txt
python web.py                     the page, on http://127.0.0.1:8000
python wordle_solver.py play      the solver in the terminal
python wordle_game.py             a game in the terminal
```

`requirements.txt` splits in two: the solver needs `numpy` and `numba`, and the
last two lines are for the one-off builders only.

Startup loads `feedback_table.npy` (167 MB) and compiles the numba kernels,
which takes a while. `web.py` warms both in a background thread and the page
waits behind its veil; `/api/status` reports progress and flips `ready`. Poll
that before assuming the server is usable.

The whole solver sits behind one lock. Scoring ~13k guesses already uses every
core, so concurrent requests would only make both slower, and the memo tables
are not written for concurrent use.

## The pieces

| file | what it is |
| --- | --- |
| `solver.py` | scoring and guess choice; `WordleSolver` is the entry point |
| `kernels.py` | the numba inner loops |
| `exact.py` | exact endgame search |
| `estimator.py` | `f(n)`, its forms, and fitting |
| `train.py` | policy iteration that fits `f` |
| `priors.py` | the repeat and curation priors |
| `wordle_data.py` | word lists, feedback scoring, the table |
| `web.py` | the HTTP server and JSON API |
| `web/app.js` | the whole page |
| `wording.py` | the `/editor` string scanner |
| `feedback_table.py` | builds `feedback_table.npy` and the index maps |
| `answer_like.py` | builds `answer_like.json`, the answer-like scores |
| `solutions.py` | brings `solutions.json` up to date from the NYT |
| `word_lists.py` | refetches `wordles.json` and `nonwordles.json` |

## Where the data comes from

`data/` holds the curated inputs and the answer record; `data/derived/` holds
only what `feedback_table.py` regenerates from the two word lists, so it is the
part that is safe to throw away. Every path is resolved from `wordle_data.py`'s
`DATA_DIR` and `DERIVED_DIR` rather than spelled out again per module.

Every data file has a builder except one. Three of them — `answer_like.py`,
`solutions.py`, `word_lists.py` — **verify against the shipped file and write
nothing** unless you pass `--write`. They rewrite inputs the solver reads, and
`word_lists.py` invalidates `feedback_table.npy` outright, so the default is to
report the difference and stop.

`feedback_table.py` is the exception and predates that rule: it has no `--write`
and no `__main__` guard, so **merely importing it rebuilds the index maps and
starts on the 167 MB table**. Run it deliberately or not at all.

| file | built by | reproduces the shipped copy? |
| --- | --- | --- |
| `data/wordles.json`, `data/nonwordles.json` | `word_lists.py` | yes, exactly |
| `data/solutions.json` | `solutions.py` | it *is* the cache — a rerun fetches only the days since the last one |
| `data/derived/*` — the table and the index maps | `feedback_table.py` | yes, deterministic |
| `data/answer_like.json` | `answer_like.py` | yes, exactly |
| `data/wordlebot_weights.json` | nothing | a third-party scrape of WordleBot; not rebuildable here |

`answer_like.py` needs `wordfreq`, which nothing else here does. Its fit is
sensitive to details that are easy to guess wrong: thirteen features rather than
the four `SOLVER_EXPLAINED.md` §4.4 names, ten foreign languages rather than
more, a duplicate-letter *count* rather than a flag, and the ridge applied to
the intercept too. Change any of them and it still fits, still correlates at
r ≈ 0.98, and no longer reproduces the file.

## The web page

The server holds sessions; the page holds an editable copy of the board and
sends **the whole board** on every submit, never a delta. The reply is the board
as the solver accepted it, and the local copy is rebuilt from it, so the two
cannot drift. Editing an earlier row is therefore the same request as playing a
new one.

Every board edit — typing, coloring, deleting a row — is local until Enter.
`README.md` § *Editing the board* has the rules that follow from this and why
they are what they are; they are not obvious and they were each a bug once.

Requests are serialized: one in flight, and a second made meanwhile is held
rather than dropped. Dropping it used to leave a view with nothing in it.

Patterns cross the wire as either `"NYGNN"` or `[0, 1, 2, 0, 0]`; the server
takes both, and the page sends whichever it has to hand. A test double for the
API has to accept both too.

`/editor` rewrites what the site says by scanning the source for user-visible
strings. It means **string literals in `index.html`, `app.js` and `web.py` are
data**: changing one changes what `/editor` lists, and any saved override keyed
to the old wording is orphaned. `wording.json` holds live overrides; *Write into
the code* bakes them into the source and logs the phrases to
`wording_history.json`. There is no file-level undo.

**`wording_history.json` is a record of the past.** Its `was:` fields quote what
the code used to say. Do not sweep it with a project-wide rewrite — that makes it
false. It was deliberately left British when everything else was converted.

## Testing

There is no test suite in the repo. For page logic, the approach that works is a
small fake DOM plus a fake `fetch`, loading `web/app.js` unmodified via `vm` and
appending a getter object to reach its internals. That catches the cursor and
board-state bugs, which is where they live. Server endpoints are quickest to
check with `http.client` over a single keep-alive connection — several bugs have
been connection-reuse bugs, and one request per connection hides them.

Do not trust a solver change without checking it end to end: build a truthful
board with `score_guess(guess, answer)` rather than hand-writing patterns, or
the server rejects it as contradictory and the test proves nothing.
