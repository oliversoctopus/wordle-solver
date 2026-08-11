# About `SOLVER_EXPLAINED.md`

`SOLVER_EXPLAINED.md` is the long one: how this Wordle solver works, derived
from nothing, for a person rather than for a compiler. This file says what it is
for, how to view it so the math renders, and what order to read it in.

## What it is for

It exists so that **you can understand the solver without reading the code**.
Not what the functions are called or which flag does what — why the thing is
built the way it is, and why each piece had to exist at all.

It is written as a derivation, not a description. Every section opens with a
problem and closes with the machinery that answers it, and nothing is introduced
before the question it exists to answer. Entropy is not presented as a formula to
accept; you get the "which guess shrinks the set most usefully?" question first,
then counting, then bits, then the chain rule that makes information gain
computable. The same for the prior, for `f`, and for the endgame search.

It assumes no background in solvers, search, or information theory. It assumes
you are comfortable with logarithms and averages, and it re-derives everything
past that. Twelve display equations appear, and every one of them is also stated
in words in the surrounding prose, so a viewer without math support costs you
nothing but elegance.

Two things it deliberately is **not**:

* **Not a reference.** It does not enumerate the flags or the file layout. When
  you want *what the program does*, that is `README.md`.
* **Not a summary of `README.md`, and `README.md` is not a summary of it.** They
  are two different jobs over the same solver, and both are kept current.

## How to view it

The file is Markdown with twelve LaTeX display equations (`$$ ... $$`). Tables,
code blocks and headings render in any Markdown viewer; the equations need a
viewer with math support. Pick one:

### Option A — GitHub Gist (recommended)

1. Go to <https://gist.github.com> (sign in; a gist is the only step that needs an account).
2. Set the filename to `SOLVER_EXPLAINED.md` — the `.md` extension is what turns on rendering.
3. Paste the entire file into the body.
4. Click **Create secret gist**. Secret means unlisted, not private, so don't treat the URL as a secret.

GitHub renders `$$ ... $$` math natively, along with the tables and code blocks.
This is the highest-fidelity option and the result is a stable link you can share.

### Option B — StackEdit (no account at all)

1. Go to <https://stackedit.io> and click **Start writing**.
2. Select all of the sample text in the left pane and paste the file over it.
3. The right pane is the rendered view. KaTeX is on by default.

The `</>` button in the toolbar exports a standalone HTML file if you want one.

### Option C — HackMD

1. Go to <https://hackmd.io> and start a new guest note.
2. Paste. Split view renders live, with KaTeX and slightly nicer typography than B.

### Backup — locally, in VS Code

Open the file and press `Ctrl+Shift+V`. VS Code ships a built-in *Markdown Math*
extension, so KaTeX works with nothing installed. `Ctrl+K V` opens the preview
side by side with the source instead.

> If a viewer shows raw `$$` symbols, it simply lacks math support. Nothing is
> lost: every formula in the document is also stated in words in the surrounding
> prose, and the code references are plain text.

## The route through it

Read start to finish the first time. The order is the argument.

| § | what it establishes |
| --- | --- |
| 0 | the one question the whole program answers, and the two word lists |
| 1 | the feedback rule, why duplicate letters make it subtle, and why 167 MB of table is worth it |
| 2 | the reframe everything rests on: **a guess is a partition of the candidate set** |
| 3 | entropy and information gain, built up from counting rather than quoted |
| 4 | the answer is not drawn uniformly — the repeat and curation priors, their shapes, and (4.6) how to measure a prior without fooling yourself |
| 5 | effective size: giving a weighted set an honest "count", and the rearrangement that makes it cheap |
| 6 | the real objective, `E(g, S)`, and **why minimizing it is not entropy in disguise** |
| 7 | where `f` comes from, since it is defined in terms of the play it decides — policy iteration |
| 8 | the six-guess deadline, and what losing should cost |
| 9 | the endgame search, and the four ideas that make brute force affordable |
| 10 | why the loss penalty is 5 when 1 is the principled answer |
| 11 | one full turn of play, with every piece above in its place |
| 12 | the result worth framing |

If you have ten minutes rather than an evening: **§2, §6, §12**. §2 is the idea
the solver is built on, §6 is where it stops being a standard entropy solver, and
§12 is the finding worth carrying away — *better valuation is not better
decisions*, which held four separate times here and cost real work to learn.

If you want the punchline first, §12 reads fine cold and will send you back for
the parts you want.

## How it sits next to the other documents

| document | the question it answers |
| --- | --- |
| `SOLVER_EXPLAINED.md` | **why** the solver is built this way, from the ground up |
| `README.md` | **what** it does: the flags, the priors, every measured result, the tables |
| `analysis/README.md` | **how the numbers were measured**, and which script reproduces each |
| `CLAUDE.md` | conventions for changing the code |

The overlap is deliberate and the two long documents are kept in step: when
behavior changes, whichever one documents it gets updated, and where they quote
the same measurement they quote the same number.

## What keeps it true

The figures in it come from the same runs as `README.md`'s, so a refit of the
estimator or an update to `data/solutions.json` moves both. `analysis/` holds the
scripts behind them.

It does carry its own history on purpose — "under the flat shapes the same
decomposition read 0.462", and so on. That is not staleness. A derivation is
partly a record of what was tried, and several of the results here are only
convincing next to the thing they replaced.
