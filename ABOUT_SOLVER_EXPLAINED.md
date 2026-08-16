# About `SOLVER_EXPLAINED.md`

`SOLVER_EXPLAINED.md` is the long one: how this Wordle solver works, derived from nothing, for a person rather than for a compiler. This file says what it is for, how to view it so the math renders, and what order to read it in.

## What it is for

It exists so that **you can understand the solver without reading the code**. Not what the functions are called or which flag does what — why the thing is built the way it is, and why each piece had to exist at all.

It is written as a derivation, not a description. Every section opens with a problem and closes with the machinery that answers it, and nothing is introduced before the question it exists to answer. Entropy is not presented as a formula to accept; you get the "which guess shrinks the set most usefully?" question first, then counting, then bits, then the chain rule that makes information gain computable. The same for the prior, for `f`, and for the endgame search.

It assumes no background in solvers, search, or information theory, and re-derives everything past the mathematics named in *Who it is written for* below. Every display equation is also stated in words in the surrounding prose, so a viewer without math support costs you nothing but elegance.

Two things it deliberately is **not**:

* **Not a reference.** It does not enumerate the flags or the file layout. When you want *what the program does*, that is `README.md`.
* **Not a summary of `README.md`, and `README.md` is not a summary of it.** They are two different jobs over the same solver, and both are kept current.

## Who it is written for

One reader, and the level is calibrated to them. It is written down here so that a later edit calibrates to the same person rather than to whoever is writing.

* A high schooler, with mathematics through **AP Calculus BC** and **AP Statistics**.
* Competition mathematics: **AIME** three times, by the AMC 10 route, and not past it.
* Competitive programming: **USACO Silver**.
* AI competitions: the **USA AI Olympiad** national camp, and a silver medal at the **International AI Olympiad**.

That last line is the one to be careful with, and the reader has said so plainly: a young competition needs less advanced mathematics than its results suggest. A medal there is not a claim to linear algebra or measure theory. The general form of it, which matters more than any single entry above: **do not infer knowledge of a subject from success at a competition that does not examine that subject.**

What follows for this document:

**Assume no knowledge of this codebase at all.** The reader has not opened `kernels.py` and is not going to — that is the entire point of the document. So anything that lives only in the source has to be built here before it is leaned on: that a candidate set is an array of column indices into the table, what the 243-slot accumulators are, what a `pool` is, why the memo key is raw bytes. The test is whether a sentence would mean anything to someone who cannot see the function it describes. "The reduce zeroes the bucket on the spot" fails it, and either earns the two lines that make it mean something or comes out. Names are the exception, and the point: `priors.build_weights` is a pointer for afterwards, and no argument should need the reader to have followed it.

**Fair to assume.** Logarithms and exponentials. Expected value, mean, standard deviation, the normal distribution, and the difference between a sample and the thing it estimates — AP Statistics covers these and the derivation leans on them constantly. Fitting a line by least squares, residuals, and correlation as a number between −1 and 1. Derivatives and integrals, though little here needs them. Recursion, sorting, and simple dynamic programming.

**Not fair to assume, so derived here.** Information theory entire: entropy, bits, information gain (§3). What a prior is and what it means to weight a word list (§4–§5). Value functions, policies, and policy iteration (§10). Game-tree search — expectimax, and pruning a search with bounds (§12). None of that appears in AP Statistics, in AIME, or in USACO Silver. Nor does anything about the machine underneath: how an array sits in memory (§1), cache lines and memory-bound loops and what a compiler is allowed to reorder (§8), packing values into a machine integer and what a hash collision costs (§12), cache eviction policy (§13).

**Needs a sentence of gloss at first use.** Statistics past the AP syllabus, and especially where it resembles something on it: rank correlation is not the `r` taught there, and RMSE wants saying out loud as the root-mean-square of the residuals. A standard error written `± 0.004` should be explained once. So should notation that is ordinary in code and unusual in a classroom — `argmin`, `Σ` over buckets, `O(n)`.

## How to view it

The file is Markdown with LaTeX display equations (`$$ ... $$`). Tables, code blocks and headings render in any Markdown viewer; the equations need a viewer with math support. Pick one:

### Option A — GitHub Gist (recommended)

1. Go to <https://gist.github.com> (sign in; a gist is the only step that needs an account).
2. Set the filename to `SOLVER_EXPLAINED.md` — the `.md` extension is what turns on rendering.
3. Paste the entire file into the body.
4. Click **Create secret gist**. Secret means unlisted, not private, so don't treat the URL as a secret.

GitHub renders `$$ ... $$` math natively, along with the tables and code blocks. This is the option the document is read in, and the result is a stable link you can share.

It also decides one rule about how the math may be written. **No backslash-punctuation macros in an equation** — no `\,`, `\;`, `\!`, `\:`, and no `\\` for a line break. Markdown treats a backslash before any ASCII punctuation as an escape, so GitHub resolves `\;` to a bare `;` before the math renderer ever sees it, and a semicolon is duly typeset in the middle of the formula. Macros named with letters — `\frac`, `\sum`, `\qquad`, `\big(` — are not escapes and come through untouched. Nothing is lost by the rule: TeX spaces binary operators correctly on its own, and those macros only ever added decoration on top of spacing that was already right. The trick of writing `\\;` to smuggle one through works on GitHub and breaks the other three viewers, where KaTeX reads the `\\` as a line break.

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

> If a viewer shows raw `$$` symbols, it simply lacks math support. Nothing is lost: every formula in the document is also stated in words in the surrounding prose, and the code references are plain text.

## The route through it

Read start to finish the first time. The order is the argument.

| § | what it establishes |
| --- | --- |
| 0 | the one question the whole program answers, and the two word lists |
| 1 | the feedback rule, why duplicate letters make it subtle, why 167 MB of table is worth it, and the shape in memory that §7, §8 and §12 all turn out to be arguing about |
| 2 | the reframe everything rests on: **a guess is a partition of the candidate set** |
| 3 | entropy and information gain, built up from counting rather than quoted |
| 4 | the answer is not drawn uniformly — the editor model, and the regime that justifies its two rates |
| 5 | how the mass is spread *inside* each group: the repeat histogram, WordleBot's weights, and the fallback that needs no scrape |
| 6 | how to measure a prior without fooling yourself, one question about the priors the data settles emphatically, and one about their shapes it cannot settle at all |
| 7 | effective size: giving a weighted set an honest "count", and the rearrangement that makes it cheap |
| 8 | the scoring loop, and the two cost assumptions that turned out to be wrong |
| 9 | the real objective, `E(g, S)`, and **why minimizing it is not entropy in disguise** |
| 10 | where `f` comes from, since it is defined in terms of the play it decides — policy iteration |
| 11 | the six-guess deadline, and what losing should cost |
| 12 | the endgame search, and the four ideas that make brute force affordable |
| 13 | why positions are memoized, and why the cache evicts rather than jams |
| 14 | why the loss penalty is 5 when 1 is the principled answer |
| 15 | one full turn of play, with every piece above in its place |
| 16 | the result worth framing |
| 17 | the five ideas that make any of it finish, collected |

If you have ten minutes rather than an evening: **§2, §9, §16**. §2 is the idea the solver is built on, §9 is where it stops being a standard entropy solver, and §16 is the finding worth carrying away — *a better map, but no better score*, which held four separate times here and cost real work to learn.

If you want the punchline first, §16 reads fine cold and will send you back for the parts you want. §17 is the same lesson from the engineering side and reads cold too.

## How it sits next to the other documents

| document | the question it answers |
| --- | --- |
| `ABOUT_SOLVER_EXPLAINED.md` | this file: who the long one is for, how to view it, what order to read it in |
| `SOLVER_EXPLAINED.md` | **why** the solver is built this way, from the ground up |
| `README.md` | **what** it does: the flags, the priors, every measured result, the tables |
| `analysis/README.md` | **how the numbers were measured**, and which script reproduces each |
| `CLAUDE.md` | conventions for changing the code |

The overlap is deliberate and the two long documents are kept in step: when behavior changes, whichever one documents it gets updated, and where they quote the same measurement they quote the same number.

## What keeps it true

The figures in it come from the same runs as `README.md`'s, so a refit of the estimator or an update to `data/solutions.json` moves both. `analysis/` holds the scripts behind them.

## If you edit it

This document decays in a specific way, and it has done it once already. Findings get appended rather than absorbed: a paragraph states a result, a later paragraph opens "under the shipped shapes the same decomposition comes out…", and a third explains which of the two you should believe. Each addition is individually reasonable. The sum is a changelog wearing the clothes of an explanation, and it reads like one — the reader ends up reconstructing the project's history in order to extract its conclusions.

It was rewritten end to end to undo that. These are the rules that undid it, and following them is what keeps a rewrite from being needed again.

**Say what is true now, in the place it belongs.** When a number moves, edit the sentence that carries it. Do not add a paragraph saying it moved. If two paragraphs disagree, one of them is a patch and should not exist.

**Keep superseded numbers only where the contrast is the lesson.** The possible-answer bonus growing 0.46 → 0.56 as the prior sharpens is evidence for §16's thesis, so it stays — as a clause, inside the current statement. "An earlier version of this table reported −8.08" teaches nothing and goes. The test: does the old number explain something about the solver, or only about the repo?

**Never let the document talk about itself.** No "the claim this section used to make is dead", no "this section has been updated", no dates of revision. Where the project was wrong and the error is instructive — §6.3's superadditivity, §6.1's `sqrt` trap, the `fastmath` bug in §8 — tell it as something that happened to the *project*, in the past tense, as part of the argument. The reader should never be made aware they are reading a file with a git history.

**Put a new finding where the argument needs it, not at the end.** Hard mode earns its place in §9 because it is the cleanest test of the entropy identity; it appears again in §12 and §15 only as a consequence, and in §16 as an example of the thesis. It is not a section of its own. Ask what a new result is *evidence for*, and put it next to the claim it supports. The one deliberate exception is §17, which collects patterns the reader has already met in place — it adds no new material, and if it ever starts to, that material belongs upstream.

**Say each thing once.** The warning that the §4 table is not comparable across rows belongs at that table and nowhere else. A caveat repeated in three sections is three places to forget to update.

**Every join is bridged, from one side or the other.** Often a section ends by naming the question the next one answers: §0 lists what §1–§3 must supply, §2 ends "now we need to make 'many small ones' precise", §14 points at the order §15 puts everything in. Just as often the bridge sits at the far end instead — §7 opens on what unequal weights break, §8 on what removing the logarithms exposed — and those sections want no closing sentence above them saying it a second time. What matters is that the seam is carried once, not that every section has a last line pointing forward. The order is the argument, and a section that could be moved without anyone noticing is not carrying its weight; but a sentence appended to satisfy the rule where the next section already bridges is worse than the join it papers over. Where a section opens cold on its own thesis, as §16 does, nothing reaches back and the bridge has to come from the line above it — which is why §15 ends by pointing at the split §16 makes.

**Point at code by name, never by line number** — `priors.build_weights`, not `priors.py:221`. One pass found seven references that had silently drifted.

**Section numbers are load-bearing, which does not make them fixed.** The map above and some eighty-odd cross-references are keyed to them, so renumbering is a whole-repository edit — and a whole-repository edit is allowed, provided every reference moves with it in the same change. They live in three files, and only three: `SOLVER_EXPLAINED.md` itself, this one, and `analysis/README.md`, which pairs each measured claim with the script behind it and so has to say where the claim lives. Nothing else references a section number, and nothing else should: `CLAUDE.md` and the source point at this document by name alone, which needs no maintenance. `git grep -n "§[0-9]"` finds every reference — the digit matters, since `§ *Editing the board*` and its kind point at a title rather than a number and survive any renumbering untouched — and the job is done when each hit reads correctly, not when the numbers have been shifted. Renumber when the argument genuinely wants a new section in the middle; where a subsection at the end of an existing section does the same work, that is still the cheaper move.

A quick way to check yourself: read the section you edited from its first line, out loud if you can stand it. If you hear the seam where your paragraph joins, the reader will too.
