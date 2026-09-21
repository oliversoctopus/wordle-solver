# Tests

```
python tests/run.py           a few seconds, needs nothing loaded
python tests/run.py --full    also the parts that load the 167 MB table
```

Eleven checks over ten files here and one at the root. None asserts a number written by hand. Each runs over inputs nobody chose and checks against a second implementation, the answer record, or a property that holds of every board. `anchors.py` is the exception, and pins the chain to facts from outside the repository.

## `board.test.js` — the solve board's editing rules

`web/app.js` is loaded unmodified through `vm` over a fake DOM and a fake server. The anchors name a sequence and the board it should leave, one per rule in `README.md` § *Editing the board*. The properties hold of every board and are checked after every keystroke of a random walk: at most one free row after the last word, the cursor on the board, every row the solver was told about present and never more than six, the shadow in the free row and nowhere else, a board with nothing pending equal to what the solver returned, and every tile holding an allowed letter and a color that belongs to one. The walk plays whole intentions, such as a word or clearing a row by hand, one press at a time. A failure prints the seed and the sequence.

```
node tests/board.test.js [walks] [edits]     default 200 walks of 60
```

## `front_ends_agree.py` — the page and the terminal ask the same question

`wordle_solver.py play` builds a `PriorConfig` from flags and `web.py` builds one from a JSON body. Each parser is the other's oracle. The shared flags start in the same place, all four switch combinations name the same prior, and the solver is built with the same options. `--full` also compares the recommendation over several boards, given as an answer and the words played at it, with patterns from `score_guess`.

## `splits.py` — one word per split, and the words it stands for

The ranking names one word per distinct split of the candidates, and `split_siblings` names the rest. Which words split alike is settled by comparing their table columns with numpy, and what a split is worth by scoring it again or asking `best_guess` for the pick. Every position of a played-out game is checked in both modes. Needs the table, so it runs under `--full`.

```
python tests/splits.py --games 8     four games per mode by default
```

## `scoring.py` — each scoring rule is what it claims to be

Information gain, expected guesses under `f`, and information gain with a price on being a possible answer share one path. The price is recomputed from the prior weights with numpy. Switching it off must leave the mode bit-identical, a pooled score must equal the full pass restricted to that pool, and the ranking's first entry must be the guess `best_guess` plays, at the same value. Needs the table.

```
python tests/scoring.py --games 6     three games per mode by default
```

## `anchors.py` — the facts the rest is measured against

Four things are asserted from definitions outside this repository, worked out by hand. The coloring rule, on seven patterns of which six turn on a duplicate letter: `eerie` against `crane` is `NNYNG`. The pattern encoding: a base-3 numeral read from the left, so `GGGGG` is 242. Information gain in bits: four equally likely candidates told apart are worth 2 bits, an even coloring is worth 0, and a 3:1 split is worth H(3/4, 1/4). What optimal play is worth: one candidate costs 1 guess, two equal ones <!--c-->1.5, two at 3:1 <!--c-->1.25, and three cost 5/3 when a guess separates them. The last two run on a six-word game whose table `score_guess` builds, and each asserts the structure it rests on before the value. Two bounds then run over random positions: no guess is worth more than the prior entropy, and no play beats winning outright on the likeliest word. `--full` adds `score_guess` against `feedback_table.npy` over <!--c-->400 drawn words, half of them with a repeated letter.

```
python tests/anchors.py --full --seed 7 --rounds 200
```

## `fit_provenance.py` — the cached f is a fit of the answer record we have

The estimator cache is keyed on the `PriorConfig` alone, so `solutions.py --write` leaves every fit describing a shorter record. This compares each cache's `fitted_at` stamp, or its `ranked_at` stamp where `form_default.py --promote` kept it as the best fit on a longer record, with the date of the newest answer the solver uses, and names the configuration to refit. The repeat-off priors are exempt, since `priors.build_weights` reads the record only under `PriorConfig.repeat`.

## `doc_refs.py` — every pointer into the documents and the code resolves

A reference to a document names its heading, as `README *Results*`, and matches on a prefix. `SOLVER_EXPLAINED` is also numbered, so `§12` resolves against that numbering. A reference to code names a symbol, as `priors.build_weights` or `source::Columns`. The Python is read with `ast` and the Rust matched against its item declarations, and `EXTERNAL` lists the few names that belong to neither. Whitespace is collapsed first, so a title split across a docstring's line break still matches. It cannot tell that a pointer lands on the right section, only on a real one.

## `doc_numbers.py` — every figure the documents quote is the one its script recorded

At the repository root, since it is also the tool that rewrites the documents. Each script in `analysis/` records its figures to `analysis/results/<script>.json`, and the documents mark each figure with an HTML comment naming the key. Run with no flags, it fails on any figure that differs from its results file. `--write` rewrites them and lists what moved, which is the list of sentences to reread. `--unbound` fails on a figure that carries no marker. It passes the word counts, the pattern count, the shipped defaults it reads out of the modules, section numbers, years and puzzle numbers, and anything in code. The rest is bound, marked `<!--c-->` as a definition or a worked example, or marked `<!--u-->` as a figure measured once with no script here.

## `rust_parity.py` — the Rust core agrees with the Python

Each ported piece runs against the Python on the same inputs. The coloring rule is checked against every entry of `feedback_table.npy`. The prior weights are compared per configuration. `f` is checked over a grid twice per form, on synthetic coefficients and on the cached fit. The training draw is compared exactly, index by index, and the least-squares solve on a tolerance over three datasets built to reach its awkward paths. One round of policy iteration is then run both ways, and the check stops there, since the two fits differ in their last bits and diverge legitimately afterwards. The kernels run over candidate sets Python builds, straddling the `_RESCAN_MAX` and `MAX_PACKED` thresholds. The split helpers are compared in order. The endgame search is checked over positions played out two plies deep, at every turn count and under three arms, and must return the identical guess, value and ranking. The solver dispatch is checked for which branch answers, across six arms. The endpoints are compared leaf by leaf over seven settings and eight boards, six of them one row behind an opening so the precomputed rankings are exercised. The messages are compared word for word across nine refusals. The letter odds are the one field compared on a tolerance, since `web.py` computes them with a BLAS product. `--every-answer` plays all 12,947 words both ways and diffs the lengths. It skips when nothing is built.

## `wasm_api.test.js` — the static build's shim keeps the page's contract

`web/wasm-api.js` is loaded unmodified through `vm` against a fake worker. A board must arrive ok, a refusal must arrive not-ok with its message intact, a reply that is not JSON counts as a refusal, anything outside `/api/` passes to the real `fetch`, and `/api/game/new` picks up the seed the module cannot generate. `app.js` throws on a not-ok reply and renders a board on an ok one, so a refusal returned as 200 would be applied as a board.

## `voice.py` — the voice and comment rules

`CLAUDE.md` § *Voice* and § *Conventions* are checked here. The script scans every document, comment and docstring, the Cargo manifests and the git dotfiles included, for the banned phrases, and flags a file header or a docstring over ten lines, a run of plain comment lines over two, a measurement or an edit date in a source comment, and a `CLAUDE.md` over 1,500 words. It also counts em-dashes, "rather than" and short closing sentences per thousand words of a document's prose, against ceilings that the rewritten documents sit well under, and it fails a document whose sentence lengths vary by less than 0.45 of their mean, a floor under every writer `VOICE.md` names and above prose written to a fixed length. A quoted phrase does not count, so a rule may name what it bans.

```
python tests/voice.py                  the repo
python tests/voice.py PATH...          these files or directories
```
