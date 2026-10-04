# The voice

Who is writing the documents, who is reading them, what the writing does at each kind of paragraph, what the check does not catch, and four pairs that show it. CLAUDE.md *Voice* holds the floor `tests/voice.py` checks. This file says what the writing sounds like once it clears that floor.

## Who is speaking, and to whom

The person who built the solver, some months in, explaining it at a whiteboard to one reader. The reader is the one `ABOUT_SOLVER_EXPLAINED.md` *Who it is written for* describes. The speaker knows the material cold, has been wrong about it several times, and is still a little surprised by §15. They are trying to get one person to see what they see, and they have as long as that takes.

The speaker says "I" for what they did, what they don't know and what surprised them, and "we" when the reader and the writer are working a step together. They use contractions, a colon to introduce the thing just named, and a bullet list when the content is a list. A new idea starts a new paragraph, and a paragraph can be one sentence long.

The habits below are drawn from four writers doing this well: Norvig's spelling corrector essay, Luu's branch prediction talk, Nielsen's first chapter on neural networks, and Hamming's Bell Labs talk. Each habit names the one it comes from.

## Reporting a result

The exact figure appears once, in the table or in the sentence that reports it. Every later sentence rounds it and says what it means. Norvig reports 75% on the development set and 68% on the test set, and the next sentence says he met his goals for brevity, development time and speed, and not for accuracy.

When a result is surprising, the sentence says what one expected and why. Luu writes that the natural next step is a wider saturating counter, and that it turns out more bits buy little. The reader learns the result and the intuition it corrects in the same breath.

When the cause is not measured, the sentence offers the candidates and stops. Norvig: perhaps the test set was hard, or perhaps the simple model cannot reach 80%. Hamming, mid-story: he cannot prove the cause and effect, because one could equally say the closed door is a symptom.

Concreteness comes from the example and never from a figure of speech. Norvig's examples are the words `thew`, `the` and `thaw`. This repo's are `salet`, `crane`, the 243 patterns and the eleven turn-two substitutions. A sentence reaching for pays, charges, prices, handicaps or verdicts is a sentence whose example is missing.

A paragraph ends when its idea is finished. Luu's paragraphs end on the cost in cycles, and Norvig's on the next thing to try. Neither ends on a line built to be remembered.

## Deriving something

The reader's question is asked in the reader's words before it is answered. Norvig asks why `adres` should become `address` and not `acres`. Nielsen asks what to make of the algebraic form of the sigmoid, and answers that its shape matters and its formula does not.

A hard step gets the room it needs. Nielsen spends three paragraphs on the sigmoid, works the two limiting cases before the general claim, and tells the reader not to panic at the partial derivatives. One sentence per idea is the right rule for easy ideas. The rule for a hard one is as many sentences as the reader needs.

The speaker says which part matters and which is a detail, so the reader knows where to spend attention.

## Telling what went wrong

A story is told in order, with the moment of being wrong left in. Hamming stormed into Bode's office, got his answer, and slunk out. The `fastmath` bug, the mis-paired `anchored2` and the superadditivity error are stories of that kind, and each has such a moment. What stays banned is the document talking about its own edits. The first is the project's history. The second is the file's.

## The documentation files

`README.md`, `analysis/README.md`, `tests/README.md` and `ABOUT_SOLVER_EXPLAINED.md` are references. Their reader arrives at one section from a search or a link, wants one thing, and leaves. Everything under *Reporting a result* carries over: the figure once and then what it means, what one would expect and what happened, the candidate causes offered when none was measured, and an example in place of a figure of speech. The whiteboard does not carry over. A reference has no room for the speaker's surprise, so "I" appears only where a choice needs an owner, "we" only where reader and writer are looking at the same table, and a contraction only where the sentence is stiff without one.

Two habits belong to the reference alone. A list is formatted as a list, whether of scripts, rules or fixed points, and a paragraph naming four things in a row is a list that has not been formatted yet. A grid of figures is a table, and a sentence strung with figures is usually a table row that has not been laid out. The procedural sections, on the page, the board and the editor, already read as a calm person pointing at one thing per sentence, and they are left as they are.

## What clears the check and still sounds performed

`tests/voice.py` bans a short list of phrases, and prose that clears it can still read as written for effect. Read each paragraph back for these:

* A short line closing a paragraph, built to be remembered.
* A word stressed in italics.
* A dash aside.
* "X, not Y" where Y was never live.
* A run of short sentences where one sentence would carry the relation.
* A sentence that announces the next one.

## Four pairs

Each pair shows a paragraph as it stands and as the speaker above would write it. The rewrites drop the bound figures where the words carry them, and the document versions carry the markers `doc_numbers.py` needs.

### A result under a table

Before, from §15.1, under the table of the eleven substitutions scored against six weightings:

> "Read the real-answer rows as differences, not levels. Both arms replay them under today's prior, which prices every one of them as a word already used, so the level is that handicap and the delta is the comparison. The enumerated rows disagree among themselves, since a flat weighting of the curated list charges for what the default prior pays for. The real answers are too few to settle it either way. The reversal does not need the whole distance: blended toward that flat weighting, the default prior stops paying for the substitution 0.30 of the way there."

After:

Before reading the two real-answer rows, know that their levels mean nothing. Both arms were replayed under today's prior, and today's prior knows that every one of those answers has already been used, so it marks all of them down. That is why they sit near four guesses while the enumerated rows sit near three. The deltas are the only part to compare.

The enumerated rows are the interesting part, because they don't agree with each other. Under the default prior the eleven substitutions help. Under a flat weighting of the curated list they hurt. I think the reason is straightforward once you remember what the default prior does: it puts most of its weight on words the editor hasn't used yet, so the substitutions it picked lean toward those words. A flat weighting gives the used words just as much, and then leaning toward unused words costs you.

The real answers can't settle it. Their error bars are wider than the effect on both rows, and shrinking them would take far more puzzles than have been played.

What I find most telling is how little it takes to flip the sign. Blend the default prior toward the flat weighting, and the gain is gone well before a third of the way there.

### A README results paragraph

Before, from README *Which form is the default*, after the table showing the top six forms inside the fit's own noise:

> "A refit therefore chooses between fits rather than forms. `form_default --promote` ranks the fit already in play alongside the new archives and keeps it when nothing beats it, so a refit can improve a prior and cannot demote one. For the same reason `compare` fits four of the nine by default, `anchored2`, `invonly`, `saturating`, `spline`, one per family (the anchored pair, the inverse alone, the line with a concave bend, the free-form), each the member that has won a prior. `--forms all` fits the rest. On this refit the four held every winner, so the set gave up 0.0000 against fitting all nine. What it could give up is bounded by the spread above."

After:

So a refit is choosing among fits that all do about equally well, and the winner is whichever one happened to train luckiest. Two things follow for how the tooling behaves:

* `form_default --promote` ranks the fit already in play alongside the new archives and keeps it unless something beats it. A refit can improve a prior but can't make one worse, which matters when the differences are this close to noise.
* `compare` fits only four of the nine forms by default: `anchored2`, `invonly`, `saturating` and `spline`, one from each family, each a form that has won under some prior. `--forms all` fits the other five.

On this refit the four included every winner, so restricting to them cost nothing against fitting all nine. On another refit it could cost at most the spread inside the top group, because that's as far as any of the top forms sits from the best one.

### A derivation step

Before, from §7, after the policy iteration loop:

> "That is **policy iteration with function approximation**. Evaluate the current policy by rollout, then improve it by acting greedily on the fitted value function. Every form of `f` is linear in its parameters. In `a + b·x + c·√x` the terms are nonlinear and the parameters are not, so each refit is one closed-form least-squares solve."

After:

That loop has a name in reinforcement learning: **policy iteration with function approximation**. We evaluate the current policy by playing it out, improve it by acting greedily on the fitted value function, and repeat.

The refit step is cheaper than it looks. Every form of `f` is linear in its parameters, even the ones that curve in `x`. Take `a + b·x + c·√x`. The `√x` bends the curve, but look at what the parameters do: `a`, `b` and `c` each multiply some fixed function of `x`, and the results are added. That is exactly the setup for fitting a line through points, just with three columns instead of two, and the same closed-form least-squares solution works. Nothing in training needs an iterative optimizer.

### A mistake

Before, from §8 *The optimization that had to be taken back out*:

> "Numba hands each kernel to LLVM, and LLVM offers a `fastmath` flag: permission to reorder floating-point arithmetic for speed, as if addition were associative. In floating point it is not quite, and the reordering is usually harmless. Here it was not harmless, and the damage was not to the arithmetic. The reduce reads a bucket and then writes a zero into it for the next guess. Restructuring the loop swapped those two memory operations, so **every bucket maximum read back as the zero its own loop had just written**. The flag licensed reordering the additions. What moved was a store. It surfaced as one scorer returning exactly 3.0 for every guess and the search preferring nonsense words. A wrong answer that obvious gets found. The other two scorers survived only because a logarithm or a call to the estimator sits between their read and their write, which the compiler cannot reorder across. None of the kernels carry the flag now. Removing it costs nothing on the widest positions and up to about a tenth of the time in the middle of the range."

After:

Numba hands each kernel to LLVM, and LLVM offers a `fastmath` flag: permission to reorder floating-point arithmetic for speed, treating addition as associative when in floating point it's only nearly so. The reordering is usually harmless, and for a while I had every kernel carrying the flag.

Then one scorer started returning exactly 3.0 for every guess, and the search began preferring nonsense words. If you know what `fastmath` does, rounding is the obvious suspect, since reordering additions is the thing it permits. It was nothing of the kind. The reduce reads a bucket and then writes a zero into it, ready for the next guess. LLVM had restructured the loop so the write came first, and every bucket maximum read back as the zero its own loop had just written. I had licensed it to reorder the additions. What moved was a store.

The part I still find unnerving is that the other two scorers carried the same flag and never broke. A logarithm or a call to the estimator happens to sit between their read and their write, and the compiler can't move a store across either, so they were safe by accident. I took the flag out of all of them. It costs nothing on the widest positions and up to about a tenth of the time in the middle of the range.
