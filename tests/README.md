# Tests

```
python tests/run.py           a few seconds, needs nothing loaded
python tests/run.py --full    also the parts that load the 167 MB table
```

Nine files, and each says what it is for at the top. They cover the places where being wrong is *quiet*: the board can be left in a state that looks plausible, the solver can answer under a prior nobody chose, a list of words can be five spellings of one guess, a scoring rule can stop being the policy the documents describe, a pointer into the documents can stop pointing anywhere, and the deployed page can quietly be an older build than the source beside it — all without anything looking wrong. Nothing here tests whether the page looks right, and nothing here can.

None of them assert numbers written down by hand. Each has something else to check against — the second implementation, the answer record, or a property that holds of every board there is — because a number copied into a test is a fourth place for it to go stale, and `README.md` § *Numbers in the documents* already has three.

## `board.test.js` — the solve board's editing rules

`web/app.js` loaded unmodified through `vm`, over a fake DOM and a fake server. No dependencies; `node tests/board.test.js` is the whole of it.

Two halves. The **anchors** name a sequence and say what the board should look like afterwards — these are the rules in `README.md` § *Editing the board*, one check each. The **properties** are what has to be true of every board, checked after every keystroke of a random walk:

* at most one free row after the last word, so nothing is selectable below the shadow
* the cursor is on the board
* the board holds every row the solver was told about, and never more than six
* the shadow is in the free row, holds no letters, and appears nowhere else
* a board with nothing pending is exactly what the solver last returned
* every tile holds a letter it is allowed and a color that belongs to a letter

The walk plays whole intentions rather than keystrokes — *a word*, *clearing the row by hand*, *a word played* — each spent one press at a time. Bare random keystrokes never clear a row, since that takes six particular presses in a row, and the states behind it then go unvisited. Failures print the seed and the whole sequence, so a walk that finds something can be run again.

```
node tests/board.test.js [walks] [edits]     default 200 walks of 60
```

## `front_ends_agree.py` — the page and the terminal ask the same question

`wordle_solver.py play` builds a `PriorConfig` from flags; `web.py` builds one from a JSON body. Each has its own parser, nothing makes them agree, and a divergence is invisible from either side — the page would recommend under a prior nobody selected, with a panel describing that prior perfectly correctly. So each front end is the other's oracle: the flags they share start in the same place, all four switch combinations name the same prior, and the solver is built with the same options.

`--full` goes further and compares the recommendation itself over several boards, which is the only part that catches a divergence further down — a different candidate filter, a different `guesses_left`, a solver cached under a key that does not describe it. Boards are given as an answer and the words played at it, with the patterns from `score_guess`; a hand-written pattern contradicts itself, is refused, and then nothing has been compared.

## `splits.py` — one word per split, and the words it stands for

The ranking names one word per distinct split of the candidates, and `split_siblings` names the rest of that split: the words giving identical feedback against every candidate still standing, which play the same game from here on. Both come from numba kernels over the feedback table, where being wrong looks like a plausible list rather than a crash.

The table is its own oracle. Which words split the candidates alike is settled by comparing their columns with numpy — narrowed on a handful of candidates first, then compared in full, which names the same set without building a copy of the table — and that owes nothing to the kernels. What a split is worth is settled by scoring it again, or, where the solver settles the position outright, by asking `best_guess` for the pick alone. Every position of a played-out game is checked, in both modes.

Needs the table, so `run.py` keeps it behind `--full`.

```
python tests/splits.py --games 8     four games per mode by default
```

## `scoring.py` — each scoring rule is what it claims to be

Three rules share one path: information gain, expected guesses under `f`, and information gain with a price on being a possible answer. The last is new machinery on the first, and paying the wrong words — or paying them twice — still yields a ranking that reads perfectly sensibly.

Nothing is compared against itself. The price is recomputed from the prior weights with numpy, which owes the kernels nothing; switching it off must leave the mode bit-identical; a pooled score must equal the full pass restricted to that pool; and the ranking's first entry must be the guess `best_guess` plays, at the same value. That last one is the check that would have caught a real divergence — the page shows `top_guesses[0]` and the terminal plays `best_guess`, and until the entropy tie-break came out they could name different words on a tie.

Needs the table, so `run.py` keeps it behind `--full`.

```
python tests/scoring.py --games 6     three games per mode by default
```

## `doc_refs.py` — every pointer into the documents still points somewhere

The documents cross-reference each other, `analysis/README.md` pairs each script with the claim it backs, and the scripts repeat that in their docstrings. Nothing held any of it together, and it drifted: a heading gets reworded and the pointers at it quietly stop resolving. When this was written seven references pointed nowhere, four of them at a SOLVER_EXPLAINED numbering that had not existed for some time.

The headings are the oracle, so nothing here is written down by hand. A reference names its heading — `README *Results*` — and matches on a prefix, since a pointer often quotes a heading's opening words rather than all of it. SOLVER_EXPLAINED is also numbered, so a section sign or the word *section* followed by a number resolves against that numbering instead. Whitespace is collapsed first, which is the part that matters: Python is hard-wrapped, so a title in a docstring is routinely split across a line break, and a line-oriented `grep` misses exactly the references hardest to find by hand.

What it cannot do is tell you a reference points at the *right* section, only that it points at a real one. A pointer that survives a renumbering by landing on the wrong existing section still passes.

## `rust_parity.py` — the Rust core agrees with the Python it came from

The solver is being ported to Rust so it can compile to WASM for a static site and to a native binary for the sweeps. This runs each ported piece against the Python it replaces, on the same inputs, and diffs. It skips rather than fails when nothing is built, so a checkout without a toolchain still runs the rest; `cargo build --release` builds it, and `./build.sh` does the same when cargo is not on your PATH.

The coloring rule has a complete oracle and gets the strongest check there is: every one of the 12,947² pairs must equal `feedback_table.npy`, which is the file the Python rule generated. There is nothing about the rule left over once all 167,624,809 agree.

The prior weights are compared per configuration, the Rust writing raw little-endian f64 and the Python diffing with numpy — a vector of 12,947 floats is one line to compare there and a parser to get wrong here.

`f` is checked over a grid of `x` twice per form: once on synthetic coefficients, which reaches bases no fit happens to use, and once on the fit the repository actually cached, which is what really gets played. `effective_size` goes over the leading weights of each prior at sizes either side of the kernels' 128-candidate branch.

The kernels are driven over candidate sets **Python** builds, so a disagreement is always about the scoring and never about the inputs. They straddle both thresholds that change how a kernel behaves — 127/128/129 candidates either side of the reduce's `_RESCAN_MAX`, and 7/8 either side of the packed signature's `MAX_PACKED` — and three of them are real positions left by playing a word rather than the first n indices, which is not the same shape at all.

The split helpers are compared **in order**, not as sets. `distinct_partitions` decides which word represents a split and therefore the order the exact search tries them in, so under a branch cap a different order is a different search.

The Rust side also checks its two feedback sources against each other before trusting either: the table the native build fills, and the live scoring the browser build uses instead. If those ever diverged the site and the sweeps would be playing different games.

The endgame search is checked over positions played out two plies deep, at every turn count from three to six, under three arms — the shipped penalty, risk-neutral, and hard mode. Its *values* match to within an ulp.

Its *guess* is checked differently, and the reason is worth knowing: **a position routinely has several exactly optimal guesses, and which one comes back is arbitrary.** `_bounded_splits` leaves scores of representatives tied on the bound — 1,176 of 1,177 at exactly 2.0 in one of these positions — and `np.argsort` breaks those ties with quicksort, which is unstable and documented as giving no particular order. So Python's own pick is not reproducible across numpy versions, and demanding the Rust reproduce it would be testing an accident. What the check asks instead is whether the Rust pick is *also* optimal: where the two differ it values the Rust guess with Python's own `_evaluate` and requires the same number. One of 288 positions differs that way, at `smoko` against `hakam`, both worth 2.02282365417121035.

The solver dispatch is checked for **which branch answers**, not just what it answers with. It has five ways out — a closed form at one or two candidates, the exact search, the heaviest candidate on the last turn, one-ply lookahead, and the heuristic scorer — and six arms reach each of them: both modes, hard mode, the exact search switched off so the heuristic decides even in the endgame, and lookahead on. Positions run from one candidate to the whole word list, and `solved_exactly` is compared too, since whether a score is a true value or an estimate is part of what the page shows.

The whole sweep is the acceptance test: every one of the 12,947 words played as the answer, both sides, compared per answer rather than on the mean so that two games differing in opposite directions cannot cancel. Both reproduce the mean and loss probability README quotes.

They do not agree game by game, and the reason is worth knowing. About a thousand answers take a different number of guesses — and the differing games split **exactly evenly**, 549 longer against 549 shorter. That is the signature of the tied guesses `rust_parity.py` finds elsewhere: where a position has several exactly optimal guesses the two implementations take different ones, play different games from there, and arrive at the same expected value by another route. The aggregate moves by 9e-07.

What that does mean is that README's figures are exact *for an implementation* rather than invariant across them. The mean is safe at any precision anyone quotes; the loss count is 300 one way and 301 the other, since guesses with equal expected value need not distribute their losses alike. Both still round to 0.0240%.

The endpoints are compared **leaf by leaf** rather than on the numbers alone, over seven sets of switches and five boards. The page reads the shape of a reply as much as its values, so a field missing or spelled differently breaks it as thoroughly as a wrong score would — which is how four bugs came out at once: the prior toggles read under a key the page never sends, the *Shape of f* menu ignored, `score_label` shortened, and the mark saying which form is the fitted one following the selection instead of the fit. Two things are tolerated and both are named in the output: guesses of equal score, and the order among words of equal weight, where numpy's `argsort` is unstable and the Rust's sort is not.

Three of the five boards are one row of an opening the prior switches can produce — `salet`, `tares` and `trace`. A board starting on the word its setting actually opens with is the only one answered out of `openings.json` rather than solved on the spot, so without them the precomputed turn-two rankings would have nothing checking them at all. Each is deliberately wide enough to stay clear of the endgame, and that is not fastidiousness. The unstable ordering above is tolerable while it only picks between guesses of equal value, but the branch cap turns it into more than that: a different order among tied bounds is a different fifty guesses searched, and one side can come back with an alternative the other never valued. On one narrow position the Rust's second pick scores 2.8077 where the Python's scores 2.8269 — both honest upper bounds under the cap, and not a difference `_drop_tied_words` can excuse.

The messages are checked separately, by driving both implementations into the same nine refusals and comparing the words they refuse in. Every message exists twice, once in `web.py` and once in the Rust the static build answers from, and the editor lists both copies — so this is what notices when only one of them was rewritten. The Python side is driven in process: a handler raises before it writes anything, so a bare instance with the pool on it reaches the message without a server.

**On how close is close enough.** The tolerance sits in the window between what floating point costs and what anything actually turns on. Adding 12,947 positive floats left to right accumulates about `n * eps` ~ 3e-12 of relative error, and the two sides add them in different orders, so ~3.6e-12 turns up in `effective_size` and means nothing. At the other end, over all 12,947 guesses at the opening the closest pair of scores differs by 1.1e-9 under the shipped prior and 6e-11 under the loosest, so a disagreement has to reach 6e-11 before it can reorder anything. The tolerance is three times over the floor and six under the ceiling, and results are reported as a fraction of it.

It is an absolute *and* relative tolerance, which matters more than it sounds. A score of zero is an ordinary answer here: a guess leaving every candidate in one bucket gains exactly no information, and two of the 12,947 do that on a five-candidate set. Dividing by that to get a relative error says nothing, and an earlier version of this file reported a disagreement of 1e+285 for exactly that reason.

## `site_current.py` — the built site is in step with the source it was built from

`site/` is generated by `build_site.sh` and would normally be gitignored. It is committed because Vercel deploys this repository by cloning it, which turns a build artifact into something that can go stale: edit `app.js`, commit, push, and what deploys is the page built last week. Nothing else in the tree reads `site/`, so nothing else would ever notice.

The oracle is the build. `--full` runs `build_site.sh` into a temporary directory and compares every byte of the result against what is committed, which is the only way to catch a `wordle.wasm` left behind by a change to the Rust — the failure with the least chance of being spotted by eye, since the page still loads and still works, just as an older version of itself.

Without `--full` it checks the part that needs no compiler: the nine files `build_site.sh` copies through untouched, and `app.js`, which it compares through `wording.Wording.served` — the same call the build makes rather than a second copy of what it does.

`index.html` is left to `--full` on purpose. The build threads the `wasm-api.js` script tag into it on the way past, and repeating that rule here would be exactly the kind of duplicated knowledge this file exists to catch going stale.

## `wasm_api.test.js` — the static build's shim keeps the page's contract

`app.js` was written against a server, and `web/wasm-api.js` stands in for one. Standing in means more than returning the right JSON. `api()` throws on a reply that is not ok and renders a board on one that is, so a refusal that arrives as 200 is applied as though it were a board — and `applyState` reads `state.rows.map` off a payload with no rows, which takes the page down where it stands. Three gray guesses in the solve view reach a board no word can satisfy, and that was enough to do it on the deployed site.

Nothing else could see it. `rust_parity.py` compares the two implementations' replies field by field and their refusals word for word, but the HTTP status is the shim's own invention on the way past, and the shim is JavaScript that no Python test loads.

So the shim is loaded unmodified through `vm` against a fake worker that answers with whatever the case wants, and the checks are about the contract rather than the content: a board arrives ok, a refusal arrives not-ok and carries its message through what `api()` does with it, a reply that is not JSON at all counts as a refusal, anything outside `/api/` is passed to the real `fetch` untouched, and `/api/game/new` picks up the seed the module has no way to generate.
