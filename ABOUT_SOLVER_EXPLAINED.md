# About `SOLVER_EXPLAINED.md`

`SOLVER_EXPLAINED.md` explains how this Wordle solver works, derived from nothing, for a reader. This file says what it is for, how to view it so the math renders, and what order to read it in.

## What it is for

It exists so that **you can understand the solver without reading the code**: not what the functions are called or which flag does what, but why the solver is built the way it is and why each piece has to exist.

It is written as a derivation. Every section opens with a problem and closes with the machinery that answers it, and nothing is introduced before the question it exists to answer. Entropy is not presented as a formula to accept. The question "which guess shrinks the set most usefully?" comes first, then counting, then bits, then the chain rule that makes information gain computable. The same goes for the prior, for `f`, and for the endgame search.

It assumes no background in solvers, search, or information theory, and re-derives everything past the mathematics named in *Who it is written for* below.

Two things it is **not**:

* **Not a reference.** It does not enumerate the flags or the file layout. `README.md` says what the program does.
* **Not a summary of `README.md`, and `README.md` is not a summary of it.** They are two different jobs over the same solver, and both are kept current.

## Who it is written for

It has one reader, and the level is calibrated to them. That reader is described here so that a later edit calibrates to the same person and not to whoever is writing.

* A high schooler, with mathematics through **AP Calculus BC** and **AP Statistics**.
* Competition mathematics: **AIME** three times, by the AMC 10 route, and not past it.
* Competitive programming: **USACO Silver**.
* AI competitions: the **USA AI Olympiad** national camp, and a silver medal at the **International AI Olympiad**.

That last line is the one to be careful with, and the reader has said so. A young competition needs less advanced mathematics than its results suggest, and a medal there is not a claim to linear algebra or measure theory. The general rule matters more than any single entry above: **do not infer knowledge of a subject from success at a competition that does not examine that subject.**

What follows for this document:

**Assume no knowledge of this codebase.** The reader has not opened `kernels.py` and is not going to, which is the point of the document. So anything that lives only in the source has to be built in the document before it is leaned on: that a candidate set is an array of column indices into the table, what the 243-slot accumulators are, what a `pool` is, why the memo key is raw bytes. The test is whether a sentence would mean anything to someone who cannot see the function it describes. "The reduce zeroes the bucket on the spot" fails it, and either earns the two lines that make it mean something or comes out. Names are the exception. `priors.build_weights` is a pointer for afterwards, and no argument should need the reader to have followed it.

**Fair to assume.** Logarithms and exponentials. Expected value, mean, standard deviation, the normal distribution, and the difference between a sample and the thing it estimates, all of which AP Statistics covers and the derivation leans on constantly. Fitting a line by least squares, residuals, and correlation as a number between −1 and 1. Derivatives and integrals, though little here needs them. Recursion, sorting, and simple dynamic programming.

**Not fair to assume, so derived in the document.** All of information theory: entropy, bits, information gain (§3). What a prior is and what it means to weight a word list (§4–§5). Value functions, policies, and policy iteration (§7). Game-tree search, expectimax, and pruning a search with bounds (§10). None of that appears in AP Statistics, in AIME, or in USACO Silver. Nor does anything about the machine underneath: how an array sits in memory (§1), cache lines and memory-bound loops and what a compiler is allowed to reorder (§8), packing values into a machine integer and what a hash collision costs (§10), cache eviction policy (§12).

**Needs a sentence of gloss at first use.** Statistics past the AP syllabus, and especially where it resembles something on it. Rank correlation is not the `r` taught there, and RMSE needs spelling out as the root-mean-square of the residuals. A standard error written `± 0.004` should be explained once. So should notation that is ordinary in code and unusual in a classroom, such as `argmin`, `Σ` over buckets, and `O(n)`.

## How to view it

The file is Markdown with LaTeX display equations (`$$ ... $$`). Tables, code blocks and headings render in any Markdown viewer. The equations need a viewer with math support. Pick one:

### Option A — GitHub Gist (recommended)

1. Go to <https://gist.github.com> (sign in; a gist is the only step that needs an account).
2. Set the filename to `SOLVER_EXPLAINED.md`. The `.md` extension turns on rendering.
3. Paste the entire file into the body.
4. Click **Create secret gist**. Secret means unlisted, not private, so do not treat the URL as a secret.

GitHub renders `$$ ... $$` math natively, along with the tables and code blocks. This is the option the document is read in, and the result is a stable link.

It also decides one rule about how the math may be written. **No backslash-punctuation macros in an equation**: no `\,`, `\;`, `\!`, `\:`, and no `\\` for a line break. Markdown treats a backslash before any ASCII punctuation as an escape, so GitHub resolves `\;` to a bare `;` before the math renderer sees it, and a semicolon is typeset in the middle of the formula. Macros named with letters, such as `\frac`, `\sum`, `\qquad` and `\big(`, are not escapes and come through untouched. Nothing is lost by the rule, since TeX spaces binary operators correctly on its own and those macros only add decoration. Writing `\\;` to get one through works on GitHub and breaks the other three viewers, where KaTeX reads the `\\` as a line break.

### Option B — StackEdit (no account at all)

1. Go to <https://stackedit.io> and click **Start writing**.
2. Select all of the sample text in the left pane and paste the file over it.
3. The right pane is the rendered view. KaTeX is on by default.

The `</>` button in the toolbar exports a standalone HTML file if you want one.

### Option C — HackMD

1. Go to <https://hackmd.io> and start a new guest note.
2. Paste. Split view renders live, with KaTeX and slightly nicer typography than B.

### Backup — locally, in VS Code

Open the file and press `Ctrl+Shift+V`. VS Code ships a built-in *Markdown Math* extension, so KaTeX works with nothing installed. `Ctrl+K V` opens the preview side by side with the source instead.

> If a viewer shows raw `$$` symbols, it lacks math support. Every formula in the document is also stated in words in the surrounding prose, and the code references are plain text.

## The route through it

Read start to finish the first time. Each section builds on the one before.

| § | what it establishes |
| --- | --- |
| 0 | the one question the whole program answers, and the two word lists |
| 1 | the feedback rule, why duplicate letters make it subtle, why 167 MB of table is worth it, and the shape in memory that §8 and §10 depend on |
| 2 | the reframe everything rests on: **a guess is a partition of the candidate set** |
| 3 | entropy and information gain, built up from counting |
| 4 | the answer is not drawn uniformly: the editor model, the regime that justifies its two rates, and the "count" of a weighted set |
| 5 | how the mass is spread inside each group: the repeat histogram, WordleBot's weights, and the fallback that needs no scrape |
| 6 | the real objective, `E(g, S)`, and **why minimizing it is not maximizing information** |
| 7 | where `f` comes from, since it is defined in terms of the play it decides: policy iteration, and why the fit is per shape |
| 8 | the scoring loop: the rearrangement that makes it cheap, and the two cost assumptions that turned out to be wrong |
| 9 | the six-guess deadline, and what losing should cost |
| 10 | the endgame search, the four ideas that make brute force affordable, and the shallower search above it |
| 11 | why the loss penalty is 5 when 1 is the principled answer |
| 12 | why positions are memoized, and why the cache evicts rather than refuses |
| 13 | one full turn of play, with every piece above in its place |
| 14 | how to measure a prior without fooling yourself, one question about the priors the data settles, and one about their shapes it cannot |
| 15 | a better map, and next to no better score: the five measurements behind it |
| 16 | the five ideas that make any of it finish, collected |
| 17 | what the same project costs with none of them: a year to fit `f` once |

If you have ten minutes: **§2, §6, §15**. §2 is the idea the solver is built on, §6 is where it stops being a standard entropy solver, and §15 is the main finding, a better map and next to no better score, which held five separate times.

§15 reads cold and points back to the parts it depends on. §16 is the same lesson from the engineering side and reads cold too, and §17 is what the project costs without §16's ideas.

## How it sits next to the other documents

| document | the question it answers |
| --- | --- |
| `ABOUT_SOLVER_EXPLAINED.md` | this file: who the long one is for, how to view it, what order to read it in |
| `SOLVER_EXPLAINED.md` | **why** the solver is built this way, from the ground up |
| `README.md` | **what** it does: the flags, the priors, every measured result, the tables |
| `analysis/README.md` | **how the numbers were measured**, and which script reproduces each |
| `VOICE.md` | how the documents are written, and where a new sentence goes |
| `CLAUDE.md` | conventions for changing the code |

The two long documents are kept in step. When behavior changes, whichever one documents it is updated, and where they quote the same measurement they quote the same number.

## What keeps it true

The figures in it come from the same runs as `README.md`'s, so a refit of the estimator or an update to `data/solutions.json` moves both. `analysis/` holds the scripts behind them.
