# Tests

```
python tests/run.py           a few seconds, needs nothing loaded
python tests/run.py --full    also the parts that load the 167 MB table
```

Four files, and each says what it is for at the top. They cover the places where being wrong is *quiet*: the board can be left in a state that looks plausible, the solver can answer under a prior nobody chose, and a list of words can be five spellings of one guess. Nothing here tests whether the page looks right, and nothing here can.

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

## `fit_provenance.py` — the fitted `f` is a fit of the record we have

Not a test of code. `estimator` keys its cache on the `PriorConfig` alone, so it will happily load a fit made against a shorter answer log, and nothing says so. This compares the `fitted_at` stamp each cache already carries against the newest answer the solver is allowed to see, for all four priors the page can name. What it catches is not a wrong function but a stale one.

Refit whatever it names:

```
python wordle_solver.py train --this-config-only
```
