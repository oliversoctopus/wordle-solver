# Wordle solver

```
python wordle_solver.py play         # interactive solving
python wordle_solver.py train        # fit + cache the estimator, every prior combination
python wordle_solver.py benchmark    # self-play: greedy entropy vs. expected guesses
python wordle_solver.py compare      # policy iteration once per form of f, then tabulate
python wordle_solver.py info --forms # what is cached, and which forms exist
```

The prior toggles, `--mode` and `--form` go after the subcommand:

```
python wordle_solver.py play --no-curation-prior --mode entropy
python wordle_solver.py train --this-config-only --no-repeat-prior --form saturating
```

## Playing it yourself

```
python wordle_game.py                  menu: type an answer, or draw one from a prior
python wordle_game.py --answer crane   play against a known answer
python wordle_game.py --random         draw the answer from the prior
```

During a game, `?` shows the solver's recommendation (Enter accepts it) and
`quit` gives up. `--assist` shows it every turn, `--auto` hands the game over,
and `--seed` makes a `--random` answer reproducible. The game only reads the
word lists — the 167 MB feedback table is loaded lazily, and only for a hint.

The board is recapped at the end as colored blocks. `--share` adds the emoji
grid underneath it, which is the one worth pasting somewhere else.

## In a browser

```
python web.py                 serve on http://127.0.0.1:8000 and open a browser
python web.py --port 9000     somewhere else
python web.py --no-browser    do not open one
```

Both terminal front ends are there, plus one thing that has no terminal
equivalent. The group beside *Settings* says which of the three the page is
doing. It marks where you are, not where the button will take you.

**Solve** is `wordle_solver.py play`. You enter the feedback by coloring tiles
instead of typing `NYGNN`, and the prior toggles, `--mode`, `--form` and
`--hard` all become switches. *Shape of f* leaves the choice to the cache by
default and marks which form that is, since the winning form is fitted per prior
and moves with the toggles. Alongside the board is a panel of alternatives and
still-possible candidates, with the prior, the objective and the fitted f under
*Advanced*. Each of those comes with an explanation of what it is doing, and the
rates quoted in it are measured from `solutions.json` at the moment the board is
built. Nothing is written into the page, so the explanation can't fall behind the
answer record the way a sentence quoting "about 10%" would. The count runs one
puzzle behind the table in *Both defaults describe a regime six months old*
below: the server drops the last entry of `solutions.json` (`--mask-last`,
default 1) to keep today's answer out of the solver's prior while you are solving
today's puzzle.

**Play** is `wordle_game.py`. The *Answer* box takes a word of your own, which is
the quickest way to set the solver a position, and it is the only setting that
deals a new game — an answer already dealt cannot be given a different one.
*Hint* is the terminal game's `?` and presses again to close, the *Assist* switch
is `--assist` (Enter then plays its pick, which is `--auto`), *Reveal answer* is
`quit`, and the emoji grid at the end is `--share`. The answer stays on the
server until the game is over, so the network traffic does not spoil it.

The panel beside the board holds three cards, and all three fold away. They start
closed.

*Solver's pick* holds the word the solver would play here and its score.
*Alternatives* holds the next four behind it; clicking one of those words puts it
in the row you are typing. Both cards open when you turn *Assist* on, and both
open when you press *Hint*. Neither is ever closed for you, so a card you opened
with *Assist* off is still open on your next turn.

*Letter frequency* sits above those two and *Assist* doesn't touch it. You open
and close it yourself, and it stays as you left it for the rest of the game. Each
letter carries the chance that the answer contains it somewhere: the prior's
weight on the remaining candidates that hold the letter, over the prior's weight
on all of them. On an empty board E is at the top with about 40%. Vowels are
listed above the other letters.

The prior doing the weighting is the one the answer was drawn from, not the one
the switches currently name, so turning a prior off mid-game rethinks the hint
but leaves these numbers alone. A letter that no remaining candidate holds drops
off the list, since 0% says nothing the gray keys have not said already. Letters
at 100% stay and sort to the top, so late in a game the list opens with the ones
the answer is known to contain.

**Find** is the page's own, and it runs backwards from everything else here. The
answer is known and the guess is the unknown. Type an answer, color the pattern
that a guess came back as, and you get every word in the list that would have
produced that pattern against that answer. It is the question you ask of somebody
else's share grid. Neither row describes the other. That is why there is no ghost
word and nothing is grayed out, and why a gray tile in the pattern is a color
rather than an empty tile waiting for a letter.

The match itself is exact and immediate, since the answer is one column of the
feedback table and finding the words is a comparison over it instead of a
scoring pass. The three numbers each word carries take more work:

* **Words left** is how many candidates would survive that guess *given this
  pattern*, so it ranks the words with hindsight — smallest first is what would
  have told you the most about this particular answer.
* **Bits** is expected information gain, averaged over every answer the guess
  could have faced. It does not depend on the pattern at all, so it ranks the
  words as openings instead.
* **Chance** is the prior's own probability that the guess is itself the answer.

The last two move with the prior switches. Nothing else on the page reaches the
finder at all, which is why the mode, the shape of f and hard mode are hidden
while it is showing. The finder never asks for a guess, needs no fitted `f`, and
works under any prior.

| key | solve | play | find |
| --- | --- | --- | --- |
| A–Z | write a letter into the selected tile | write a letter | write a letter of the answer |
| ← → | move along the row | move along the row | move along the row |
| ↑ ↓ | move between rows, never past the row being played | — | move between the answer and the pattern |
| Space | change the selected tile: gray → yellow → green | — | change the selected pattern tile |
| Enter | on an empty row take the suggestion, otherwise submit | guess, or play the solver's pick when it is showing | find every word that fits |

Going back up to an earlier row and correcting it is the point of ↑. Every submit
sends the whole board, so a tile fixed four rows up re-runs the solver from
there. You never have to delete the rows under it first.

### Editing the board

The page keeps its own copy of every row, and typing and coloring are local until
Enter sends the whole board back. The solver only ever sees complete boards,
which is what makes correcting an earlier row the same operation as playing a new
one. Rows the solver has not been told about yet are drawn with a dashed border.

Taking a word off is the exception, and goes back on its own. Writing one is a
proposal, which is what Enter is for; removing one is not, since a board cannot
half-forget a word it is no longer showing. However the word goes — *Delete row*,
or backspaced away letter by letter — what remains has to be the board those
words alone would have made, down to the shadow and every number in the panel.

Five rules fall out of that, and they are the answers to the things that used to
go wrong:

**A color belongs to the letter under it.** Clearing a letter clears its color,
and typing a different letter into a tile resets that tile to gray. Otherwise a
row typed over an old one comes back wearing the old one's feedback — a letter
turning yellow for no reason because the word before it had a yellow there.
Space on an *empty* tile does nothing for the same reason: the color would not
show, and would then belong to whatever was typed there next.

**Typing at the end of a full word replaces the last letter**, as it does in any
other box that is already full. A stray keypress costs one letter, not the word.
Retyping a rejected guess from scratch therefore means clearing the row first.

**Backspace only ever deletes leftwards** — the tile under the cursor, then the
one before it. Letters to the right of the cursor are left alone, so the first
few letters of a row can go without disturbing the last ones. The backspace that
empties the row is the one that takes the word off the board, and it counts as a
deletion: a row cleared by hand is a row deleted, and the two cannot look
different afterwards.

**Delete row empties the row in place and leaves the gap.** The rows under it
stay where they were played, so the word that should have been there goes
straight into the hole. Submit with the gap still empty and you get *Row 2 is
empty* back. Nothing closes up and nothing slides.

The refusal is deliberate. A word only reaches the board because it was played,
which makes a row that needs deleting nearly always a row that needs
*replacing*; closing the gap would quietly claim you played those later words
earlier than you did. The position is not free to move either. Both the endgame
search and the estimator are fed `guesses_left = MAX_GUESSES - len(rows)`, so
shifting rows up would change what the solver recommends.

An empty row with nothing under it is a different thing — it is the row waiting
to be played, so it is simply not sent. That is what makes *Delete row* on the
last guess an undo, and what clears the board back to the opening
recommendation. Only that gap-free case goes back on its own; a hole higher up
waits for the word that fills it, because the board it makes is one the solver
refuses anyway.

**The suggestion is a shadow, not a row.** The solver's pick is drawn faint in
the row after the last real word, and it follows that word rather than standing
at a fixed height. Take the last guess off and the board goes back, so what comes
up into the vacated row is the pick for the shorter board — the board is a row
shorter and says so. *Delete row* cannot touch the shadow itself, there being
nothing there to delete. Take it — click it, or press Enter on an empty row — and
it stops being a shadow: it is then a word in a row, and deletes like any other.

There is exactly one free row, and the shadow is in it. Otherwise a row emptied
by hand leaves a second free row under it, still selectable and sitting past the
end of the board the shadow describes, and typing in that one makes the shadow
vanish from a row you are not even in. The board is trimmed back to the last word
and its free row.

The rules are one promise between them. Three words with the third taken off
leaves exactly what two words played from the start would leave, and nothing on
the page can tell you which of the two happened.

One request is in flight at a time, since the solver is behind a single lock
anyway. A second one made meanwhile is *held*, not dropped. Dropping it used to
leave the page showing something nobody had asked for: a game view with no game
dealt into it, every panel reading "—", and it would stay that way. The loading
veil stays up until there is a board behind it for the same reason, retrying if
the first request fails.

**A setting throws nothing away.** The three views share one set of switches, and
changing any of them — a prior, the mode, the shape of f, hard mode — asks the
solver again about the position you are already in. The solve board is re-run
over the rows the server has already accepted, so a word half typed into the
open row is neither submitted nor lost; the game keeps its answer and re-asks the
hint; the finder searches again. Only the *Answer* box deals a new game.

The terminal menu works the other way round, with four combinations to choose
between before the deal, because there the prior decides what the answer is drawn
from. Here the answer was drawn under whatever was on at the time and it stays
what it is. Turning a prior off mid-game rethinks the advice instead of dealing a
new word, and that is the only reading under which you can ask what the *other*
model of the editor would have played from here.

Nothing is rebuilt behind your back either. A setting changed while a view is
hidden marks that view for a re-run, and the re-run happens when you next look
at it. The alternative would be a view quietly claiming a setting it was never
built under.

### Rewriting what it says

`/editor` lists every string the site can say, each beside a box to rewrite it,
with the page itself alongside. Save, and the page says it — no restart.

Nothing is registered by hand. `wording.py` reads `index.html`, `app.js` and
`web.py` and picks the text out by how it is used (between tags, assigned to
`textContent`, passed to `say()` or `SolverError()`, and so on), so a string
written the ordinary way is editable the moment it exists and the list cannot
fall behind the site. Each is shown with the file, the line and where on the
page it sits.

A choice between two strings counts as one of those uses, and what marks it as
text can be lines above the arm it is looking at — `why:` over a ternary laid
out one arm per line. Reading only the arm's own line missed the longest prose
on the page, the two explanations of the priors, until 2026-08-10.

Strings are found at a character offset, so the same sentence written twice is
two entries and rewriting one leaves the other alone. Edits are applied to the
served bytes of `index.html` and `app.js`, and saved in `wording.json` keyed by
`<file>:<hash of the text>:<n>` — which survives the code being worked on, since
neither line numbers nor offsets do. What the *server* says is matched on the way
out instead, values and all: `Row {index + 1} contradicts the rows above it`
becomes a pattern, and whatever it captured goes back into the rewritten
version. That is by wording rather than by position, so a message written twice
in `web.py` is one entry.

A value baked into a string is part of it: an edit may move `${count}` but not
drop it, and saving is refused if one goes missing. `--no-editor` turns the whole
thing off, and it is served only when the server is listening to this machine
alone.

**Save** leaves the source alone. The rewrites live in `wording.json` and are
applied to the served bytes on the way out. **Write into the code** puts them in
the files for good and clears `wording.json`. That is the end state worth
reaching, because an override is matched to the code by wording and stays fragile
for exactly as long as it sits outside it. Prose is re-wrapped to the indentation
it lands in, and both phrases are appended to `wording_history.json`. That record
is the only one kept. The bake edits the source in place with no undo, so it is
worth reading the source afterwards. The server keeps running the code it started
with, and its own messages need a restart before they read the new way.

Nothing to install: the server is `http.server` and the page is three static
files in `web/`, with three more behind `/editor`. Startup warms the feedback
table and the numba kernels in the background while the page waits. That makes
the first load the slow one. A single lock covers the solver. Scoring ~13k
guesses already uses every core, so two requests at once would only make both
slower.

## Layout

| file | contents |
| --- | --- |
| `wordle_data.py` | word lists, the feedback table, past solutions, tile/pattern conversion |
| `priors.py` | `PriorConfig` and the prior over answers it produces |
| `kernels.py` | numba scorers over all ~13k guesses: information gain, E(g, S), two-turn value |
| `estimator.py` | the forms of f(n), their least-squares fit, and the parameter cache |
| `exact.py` | exact endgame search, used wherever it is affordable |
| `solver.py` | `WordleSolver`: pick a guess, apply feedback, play a game out |
| `train.py` | policy iteration, self-play evaluation, form comparison |
| `wordle_solver.py` | command line front end |
| `wordle_game.py` | play Wordle yourself, against a chosen or a sampled answer |
| `web.py`, `web/` | both of the above in a browser, plus the word finder: stdlib HTTP server and a static page |
| `wording.py` | finds every string the site says, so `/editor` can rewrite any of it |
| `wordle_solver.ipynb` | the analysis behind the prior probabilities |
| `data/`, `data/derived/` | the word lists and answer record; everything `feedback_table.py` regenerates |
| `feedback_table.py` | one-off: builds `feedback_table.npy` from the word lists |
| `answer_like.py` | one-off: builds `answer_like.json`, the score behind `--curation-shape frequency` |
| `solutions.py` | brings `solutions.json` up to date from the NYT, one request per day since the last |
| `word_lists.py` | one-off: refetches `wordles.json` and `nonwordles.json` upstream |
| `analysis/` | the measurement scripts behind every number in the prior sections below |

The last three verify against the file they would write and do nothing unless
passed `--write`, since each one rewrites something the solver reads.
`feedback_table.py` predates that rule and rebuilds in place as soon as it is
run — or imported.

## The two priors

The answer is not drawn uniformly, and the two departures from uniform can be
switched on and off independently:

* **repeat prior** (`--no-repeat-prior`) — the editor reuses an already-played
  answer, so used words share `--repeat-prob` of the mass instead of their
  proportional share.
* **curation prior** (`--no-curation-prior`) — some answers come from outside
  the curated 2022 list, so those words share `--noncurated-prob` of what is
  left.

The model is sequential — a repeat first, then curated vs. non-curated on the
remainder — so `--noncurated-prob` is *conditional on not being a repeat*, and
must be measured with that denominator.

Each group also has a **shape**, which is how its mass is spread over its
members. The defaults are the sharpened shapes, worth −0.064 ± 0.032 guesses
over the flat spread the two priors used on their own.

* `--repeat-shape binned` (default) weights a past answer by where it first
  appeared, binned 200 wide and Laplace-smoothed — reuse concentrates near the
  start of the run, not uniformly over everything played.
* `--curation-shape wordlebot` (default) weights an off-list word by WordleBot's
  own published weight, keeping 5% of the mass outside that list as a hedge;
  `--curation-shape frequency` weights it by how answer-like the word is
  instead, which is slightly weaker but needs no vendored list.
* `--repeat-shape uniform`, `--curation-shape uniform` give the flat spread back.

Each shape is worth having on its own terms, and the two are **additive within
noise**. They were adopted on a superadditivity that current data does not show;
see *Shaping the two priors* below. The shape is part of the cache key because
the best form of `f` depends on it. `sqrt` is right for the uniform shapes, and
it picks the wrong opening word for the sharpened ones.

### Both defaults describe a regime six months old

Measured over the 1,879 answers in `solutions.json`, through Wordle #1879
(2026-08-10). The puzzle number is `days_since_launch + 1`; the `id` field is
an NYT internal key and stops matching after #506.

| window | puzzles | repeat rate | off-list \| not a repeat |
| --- | --- | --- | --- |
| 2021–2025 | 1,657 | 0.00% | 1.75% |
| **since Wordle #1690 (2026-02-02)** | **190** | **11.1%** | **8.9%** |
| last 50 | 50 | 12.0% | 11.4% |
| all time | 1,879 | 1.1% | 2.5% |

Both defaults of 0.10 are right for the current regime and roughly 10× and 4×
too high against the full record. Nothing was ever off-list before 2023, and
`cigar` on 2026-02-02 was the first repeat in Wordle's history; every repeat
since falls in the last 190 puzzles, a little over one in nine.

The two moves started together because the curated pool is running out: 1,812
of the 2,309 curated words are used, 497 remain, and fresh ones are consumed at
~334/year — about 1.5 years left. Going off-list and recycling old answers are
the same editorial response to the same constraint. Extrapolating both upward is
defensible on that basis, and 0.10 is a forward-looking number rather than a
historical one.

### Sharpening the repeat prior

By default `build_weights` spreads the repeat mass uniformly over all 1,858 used
words, but the reuse is not uniform: 19 of the 21 repeats came from Wordle
#1–#297 (median #116), with only `sandy` (#1227) and `smile` (#831) outside — a
10% outlier rate, too high to zero out.

Scored prequentially — each model predicts repeat *i* from repeats 1…*i−1*
only, so no parameter ever sees the point it is scored on:

| model | logL over 21 events | vs. uniform |
| --- | --- | --- |
| uniform over used (current) | −157.1 | — |
| binned, width 200, α = 0.5 | −133.1 | **+24.0** |
| binned, width 200, α = 1.0 | −135.9 | +21.2 |
| best of 16 window variants (last K ∈ 3/5/8/all, ±25…±200) | −143.6 | +13.5 |

The winner is a Laplace-smoothed histogram over fixed bins, at +1.14 ± 0.21 nats
per event (5.4σ, winning 17 of 21). Scored in-sample the same comparison claims
+1.50, so overfitting inflates it ~31% without manufacturing it.

The window row is the point of the exercise. A window has a boundary, a boundary
is a parameter, and it has to be tuned on the same events it is then scored
against — so even the best of sixteen tuned windows lands 10 nats behind a
histogram with nothing to tune. (A window needs a third knob besides `K` and the
half-width: how much mass to leave outside. With a hard cutoff a single repeat
landing outside costs ~7 nats, which is `sandy` and `smile` all over again.)

The obvious rival is an age-based shape, with weight rising by time since last
use instead of binning by where in the answer order the word first appeared. It
does not beat this. Refitting its rate prequentially scores −150.1, worse than
binned and barely better than uniform; the rate is unstable on twenty-odd
events. A rate fixed at 2/year reaches −131.6, but only when chosen with
hindsight on the same events, and it wins by piling mass on the oldest words and
then paying for it on `sandy` (−11.93 against binned's −8.43) and `smile`. The
Laplace floor is what makes binned survive its outliers.

On its own it changes nothing that matters. Replaying all 190 real puzzles of
the current regime, rebuilding the prior from strictly earlier puzzles each day,
the binned prior scores 3.2211 against uniform's 3.2474 — a paired **−0.0263 ±
0.0263**, with 19 games changing length, 11 better and 8 worse. The subgroups
nearly cancel: −0.143 ± 0.173 on the 21 repeat answers against −0.012 ± 0.021 on
the other 169, and repeats are only a ninth of games. (Measured when a fit for
the uniform shapes was still on hand; `analysis/replay.py` replays the shapes
that ship now, so it no longer produces this pair.)

The paired sd is 0.36 guesses, so resolving a 0.02-guess effect at 2σ needs
~1,300 regime games at one per day, which is why this sat unshipped.

Whether the source window is drifting forward is likewise unresolved, and the
evidence has weakened. The drift rate is how far the source moves per puzzle, so
**0 is a static window and 1.0 is a fixed replay lag** — one puzzle a day means a
constant lag drags the source forward one position a day. Over the first 18
events it was +0.55 ± 1.56, an interval covering both; over all 21 it is
−0.10 ± 1.15, which has collapsed onto the static reading while still excluding
nothing. The rank correlation between repeat order and source index fell the
same way, +0.46 (p = 0.04) to +0.30 (p = 0.18), because all three of the newest
repeats came from early in the answer order. The replay lag averages 4.36 years
with an sd of 0.83, and 18 of 21 events sit between 4.4 and 5.1 years.

## Shaping the two priors

Both shapes are worth having and both ship. They were originally adopted because
they looked *superadditive*, worth much more together than apart. On current data
they no longer look that way; see the note under the table.

Weighting off-list words instead of flatly is a large modeling gain on its own.
Scored prequentially over the off-list answers, uniform is −9.27 nats per event;
an answer-like score fitted against the curated list as positives is −8.14 at the
shipped `t` = 0.35; and WordleBot's published weights, validated against a 2023
snapshot that predates 42 of the 46 events, are −6.46, with the effective
support falling from 10,638 words to 911.

That does not convert into guesses by itself. Measured on three arms — the 21
real repeats and 46 off-list answers with the prior rebuilt as of each puzzle,
plus the 497 still-unused curated words — blended at the regime's own mixture of
11.1% / 7.9% / 81.1%:

| shapes | repeat | off-list | fresh | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.143 ± 0.078 | −0.978 ± 0.105 | +0.022 ± 0.028 | −0.044 ± 0.025 |
| repeat `binned` alone | −0.143 ± 0.173 | −0.022 ± 0.049 | −0.006 ± 0.016 | −0.022 ± 0.024 |
| `binned` + `frequency` | −0.286 ± 0.184 | −0.522 ± 0.145 | −0.002 ± 0.018 | **−0.074 ± 0.028** |
| `binned` + `wordlebot` | +0.095 ± 0.228 | −0.826 ± 0.150 | −0.012 ± 0.018 | **−0.064 ± 0.032** |

Concentrating the off-list mass on ~850 words gives each about 1e-4 against a
used word's 5.4e-5, so past answers stop competing for the solver's attention.
That is where the residual penalty `wordlebot` alone carries on the repeat arm
comes from.

**The superadditivity is gone.** These shapes were adopted because `wordlebot`
alone blended to +0.009 (a wash), `binned` alone to −0.021, and the pair to
−0.091 — far more than the sum. Recomputed here they are −0.044, −0.022 and
−0.064, against a sum of −0.066: additive, within noise. The pair is still the
best row and still worth shipping; the *reason* given for shipping it no longer
holds.

The change is worth two caveats. These numbers come from a solver whose estimator
was refitted afterwards, and every configuration's winning form moved, so they
are not comparable like-for-like with the published ones: the policy changed, not
just the data. `frequency` (−0.074 ± 0.028) and `wordlebot` (−0.064 ± 0.032) also
sit well inside each other's intervals now. The shipped default of `wordlebot`
is no longer measurably the better of the two.

**The first three rows are history now.** They need fits for the uniform shapes,
which the repo no longer carries, and since 2026-08-10 a prior with no fit
raises rather than playing an invented `f`. `analysis/replay.py` keeps the one
contrast both remaining fits support, `frequency` against `wordlebot` head to
head: **−0.010 ± 0.025** blended, which is exactly the distance between the last
two rows, and +0.005 ± 0.049 over the 190 regime puzzles — a wash either way.
The two trade rather than one dominating: `frequency` takes −0.381 ± 0.161 on
the repeat arm and gives back +0.304 ± 0.112 on the off-list arm.

One thing this measurement still needs: the form of `f` must be refitted per
shape, or `sqrt` flips the opening word to `raise`, which is worse by
0.034 ± 0.028 on its own.

## Choosing a guess

`--mode entropy` is the greedy scorer: maximize expected information gain,
breaking near-ties in favor of a word that could be the answer.

`--mode expected` (the default) instead minimizes an estimate of how many
guesses the game will take in total:

```
E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)
```

where the candidates in state `S` are bucketed by the feedback pattern `r` that
guess `g` would produce, `p_r` is the bucket's prior mass, and `n_r` is its
*effective size* `2 ** H` (the plain count when the prior is flat over it).
Buckets of one or two candidates are worth exactly 1 and `2 - p_max`, so those
are used directly rather than going through f. Dropping the `GGGGG` term is
what makes a guess that could be the answer beat an equally informative guess
that could not.

Both are only fallbacks, used where the game is too wide to search.

### Hard mode (`--hard`, off by default)

Wordle's own hard mode: every guess must be a word that could still be the
answer. It is a flag on `play` and `benchmark` and a switch on the page.

Mechanically it is one substitution. The pool of guesses to score becomes the
candidate set instead of all 12,947 words (`WordleSolver.guess_pool`), which the
kernels already take as a parameter, and the exact search narrows the same way.
In the search that also removes the partition dedupe: only a candidate itself can
come back `GGGGG`, so no two candidates split S identically and there is nothing
left to merge.

It is deliberately **not** a `train` flag. The fit is cached per prior and the
restriction is not part of the slug, so training under it would overwrite the
ordinary fit. `f` is therefore always the one fitted for unrestricted play, and
its scores read a little low under the restriction: the position it is estimating
is one where the solver would be allowed a probe that it is not in fact allowed.
The restriction costs guesses for that same reason. A word that cannot win is
often the sharper split.

What it costs, every word weighted by the row's own prior, the same words in both
arms, failures counted as 7 (`analysis/hard_mode.py`). Both arms are exact, so
the cost is a difference rather than an estimate of one:

| priors | free | hard | cost | solve rate, free → hard |
| --- | --- | --- | --- | --- |
| none | 4.0718 | 4.4512 | +0.3794 | 99.93% → 92.28% |
| curation only | 3.4903 | 3.5903 | +0.1000 | 99.99% → 99.58% |
| repeat only | 4.0111 | 4.3723 | +0.3612 | 99.92% → 92.68% |
| **both (default)** | **3.0770** | **3.1039** | **+0.0269** | **99.98% → 99.90%** |
| both, `frequency` | 3.1214 | 3.1480 | +0.0266 | 99.96% → 99.70% |

Under the shipped prior it costs **+0.0269** guesses, about a twentieth of what
the estimator is worth over greedy entropy. What sets the size is how flat the
prior is: with no priors at all the cost is fourteen times larger and the solve
rate falls to **92.28%**, because a wide candidate set with no probe available is
exactly the position the restriction forbids you to fix. Concentrate the mass and
the candidate set collapses fast enough that the likely answers are decent probes
anyway.

The losses are where it really shows. Even in the default row, where the mean
barely moves, the loss probability roughly quadruples. Hard mode arrives in the
one-letter families of *The penalty* below and then cannot spend a turn
separating them, since the separating word is the one it is not allowed to play.

The opening word never changes, in any row. Every one of the 12,947 words carries
prior mass, so at the first guess every word is still a possible answer and the
restriction binds nothing at all.

Under `--mode entropy` the restriction **helps**, by −0.1813 guesses on the
default prior (3.6815 → 3.5002). That is the identity under *Results* read
backwards: what greedy entropy is missing is the bonus for guessing a word that
could be the answer, and hard mode supplies it by force. It pays in losses, its
solve rate falling from 100.00% to 99.86%, and it still does not reach the
expected-guesses policy's 3.0770 — which gets the same bonus for free, and keeps
the probes.

## The exact endgame

Once the candidate set is small, or few turns remain, `exact.py` computes the
value outright:

```
V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
V(S, 0) = loss_penalty
```

with `t` the guesses left including this one. A loss penalty of 1 means running
out costs one unit more than the six turns already spent, so a lost game scores
7. The deadline then needs no separate objective and no risk weighting.

V is non-*decreasing* in t here: with more turns left you are obliged to finish
rather than pay the comparatively cheap penalty. That is what makes the
two-turn value a valid lower bound, which the search prunes with.

Four things make it affordable:

* **Partition dedupe.** Guesses that split S identically are interchangeable, so
  the branching factor is the number of distinct splits rather than 12,947. A
  guess that is itself a candidate always splits distinctly (only it produces
  `GGGGG`), so a possible answer is never merged with a probe. It only helps
  while S is tiny: by 12 candidates ~12.7k of the guesses already split
  differently.
* **Closed-form leaves.** One or two turns left, and one or two candidates left,
  are closed forms — `V(g, 2) = 1 + (A + (A − B)·penalty) / total` over the
  non-winning buckets, A their mass and B the sum of their maxima. One pass over
  the guesses, no estimator, exactly the cost of scoring them heuristically, so
  **two turns or fewer is always solved exactly** at any candidate count.
* **Memoization.** A candidate set is solved once and reused, since the same
  position is reached along many different orders of guesses. Two-turn positions
  are where this pays: four such calls in five are repeats of one already
  answered, and one position in a 300-game sample came up 373 times. Caching them
  instead of re-running the closed form is worth about 30% of the search's total
  time, which is why the memo is consulted *before* the two-turn branch and not
  after it.
* **Branch and bound.** Guesses are tried in order of their two-turn value —
  computed only for the distinct splits, which are about a quarter of the pool —
  and within a guess each bucket's bound is replaced by its exact value one at a
  time, abandoning as soon as the running total passes the incumbent.
* **Branch cap** (`--exact-branch`, default 50, `0` to disable). Only the best 50
  splits by that bound are tried. Branch and bound finds the best guess early and
  then spends nearly all its time *proving* nothing beats it; the cap buys the
  answer and skips the proof. Over 1,275 real positions, caps of 25 and 50 pick
  the uncapped-optimal guess **every time** (a cap of 10 misses 5), while wall
  clock falls from 322 s to 8 s and the worst single position from 30 s to 1.9 s.

Deeper than two turns the search is gated on `--exact-max` (default 60), with a
node budget as a backstop that falls back to the heuristic. A typical search
visits 1–20 nodes and costs 1–30 ms. The value `solve` returns is therefore an
upper bound on the true one, and the search is optimal-within-the-cap rather
than provably optimal. Correctness is checked against an independent
brute-force implementation of the recursion over random candidate sets.

### What it is worth

From the first state the search takes over, the whole continuation is optimal,
so it cannot be worse than the estimator in expectation. It is also barely
better. 30,000 unseen answers, paired:

| policy | mean | paired Δ | solve rate | wall clock |
| --- | --- | --- | --- | --- |
| greedy entropy | 3.6879 | — | 99.930% | 8 s |
| expected, no exact search | 3.1827 | — | 99.833% | 9 s |
| exact for t ≤ 2 only | 3.1826 | −0.0001 ± 0.0002 | 99.847% | 9 s |
| exact for m ≤ 10 | 3.1835 | +0.0008 ± 0.0029 | 99.853% | 41 s |
| exact for m ≤ 24 | 3.1834 | +0.0007 ± 0.0032 | 99.870% | 793 s |

Only the free part (t ≤ 2) shows a consistent sign. Deeper search moves the mean
by less than the noise floor at up to 88× the wall clock; what it does move is
the loss rate, 50 → 39.

Per decision it is sharper, and this is the one place a valuation can be checked
against the truth rather than against a better guess at it. Over 1,200 positions
the search takes over, weighted by how often play reaches them
(`analysis/exact_decisions.py`), the estimator picks a different guess **38.8%**
of the time and gives up **0.0053 guesses** by it. What that costs depends
entirely on how much is left:

| candidates | positions | the two differ | estimator regret | of which are ties |
| --- | --- | --- | --- | --- |
| 3–4 | 296 | 32.3% | 0.0000 | all of them |
| 5–10 | 406 | 44.5% | 0.0034 | 39.4% |
| 11–60 | 498 | 38.1% | 0.0086 | 11.2% |

At three and four candidates every disagreement is a tie — a different word of
equal value, where `f` cannot be improved on because there is nothing to improve.
The ties thin out as the set grows, and by 11–60 most disagreements are real,
which is where the 0.0053 comes from.

The search's own regret reads 0.0004 rather than zero, and that is the loss
penalty rather than any limit of the search: it optimizes a penalized score while
this column measures mean guesses. At `--loss-penalty 1` it is 0.0000 exactly,
which also says the branch cap costs nothing at these sizes.

Refitting `f` under the search moves the curve by under 0.03 guesses everywhere
(fit RMSE 0.63) and does not change measured play, so `train` keeps only the free
`t ≤ 2` part by default; `--exact-max 24` fits under the full policy.

## Losing less

### Greedy entropy loses less, not more

Every word played as the answer, under both priors:

| policy | mean | loss probability | solve rate |
| --- | --- | --- | --- |
| greedy entropy | 3.6815 | 0.0034% | 99.9966% |
| expected, no exact search, penalty 1 | 3.0773 | 0.0671% | 99.9329% |
| expected, exact-max 24, penalty 5 | 3.0772 | 0.0250% | 99.9750% |
| **expected, exact-max 60, penalty 5 (default)** | **3.0770** | **0.0240%** | **99.9760%** |

Entropy is not lossless — 73 words beat it — but it loses seven times less often
than the estimator, and the two are not failing on the same words: they share 27.
What protects entropy is that it cannot see the prior. Separating candidates is
what maximizes information, and separating is also what never gambles; the
estimator backs the prior instead, and pays for it when the prior is confidently
wrong, which *The penalty* below takes apart.

The six tenths of a guess the estimator wins are therefore not free. They cost
about seven times the loss rate, and `--loss-penalty` is the dial that buys some
of it back. Neither rate is large enough for the default 200-game `benchmark` to
notice: it finds zero losses 99.3% of the time under entropy and 95.3% under the
default, which is why both look lossless there.

Wordle is provably solvable in five guesses, but that theorem is about a solver
that knows the answer is one of the curated 2,309. This one hedges over all
12,947, and 9% of the prior mass sits on the 10,638 never-curated ones, which is
where the losses collect: of the 300 words the shipped prior can lose on, 290 are
never-curated — `aahed`, `anele`, `avyze`, `bizzy`. Playing all 2,309 curated
words as the answer instead:

| policy | mean | worst | losses |
| --- | --- | --- | --- |
| greedy entropy | 3.8281 | 6 | 0 / 2309 |
| expected (defaults) | 3.7523 | 7 | 10 / 2309 |

The estimator wins the mean here too, and takes ten losses doing it. Every one of
the ten — `foggy`, `hater`, `water`, `brook`, `mushy`, `wound`, `rider`, `ratty`,
`hound`, `catty` — is a word that has already been an answer, which is the repeat
prior working rather than failing. The 497 still-unused curated words hold about
81% of the mass while the 1,812 already used share 10.2% of it, so sweeping the
curated list uniformly asks the solver to do well on a distribution its own prior
says is unlikely. Greedy entropy dodges that by ignoring the prior altogether —
the same blindness that costs it six tenths of a guess everywhere else.

The six-guess games in that table are wins rather than near-misses: they are what
hedging over 12,947 words costs, and closing them would take a full-depth
optimizer over the restricted set.

### The penalty

`--loss-penalty` defaults to **5** rather than the risk-neutral 1, and the reason
starts with what a loss actually is. It is not an endgame blunder: at the sizes
these positions reach, `exact.ExactEndgame` solves them outright, so the guess
played is the optimal one. The solver is backing its prior. Holding
`batty`/`catty`/`natty`/`patty`/`ratty`/`tatty` with four turns left it plays
`tatty`, which carries 75% of that family's mass, and it declines `bronc` — a word
splitting the six of them five ways — because separating spends a turn to insure
against a 0.4% case. A loss is the prior being confidently wrong, and the penalty
is what makes the insurance worth buying anyway.

Every word played as the answer, under each of the four prior combinations,
blended equally:

| `--loss-penalty` | loss probability | mean (solved) | mean (loss = 7) |
| --- | --- | --- | --- |
| 1 | 0.1168% | 3.6585 | 3.6621 |
| 2 | 0.0633% | 3.6603 | 3.6622 |
| **5 (default)** | **0.0458%** | **3.6611** | **3.6626** |
| 10 | 0.0386% | 3.6618 | 3.6630 |
| 30 | 0.0357% | 3.6624 | 3.6635 |

Three fifths of the loss probability disappears between 1 and 5, for **+0.0005
guesses** on the penalized mean. Opening words are untouched, since the root never
reaches the exact search.

That last column cannot settle which penalty is right, and it is worth seeing why:
scoring a loss at 7 is exactly the objective `--loss-penalty 1` optimizes, so
ranking settings by it can only favor risk neutrality. Name the price of a loss
instead — L guesses — and each setting becomes a line in L, so the choice falls
out of where the lines cross:

| a loss is worth | best penalty |
| --- | --- |
| under 7.1 | 1 |
| 7.1 – 9.0 | 2 |
| **9.0 – 13.7** | **5** |
| 13.7 – 22.1 | 10 |
| over 22.1 | 30 |

5 is right if a lost game is worth between nine and fourteen guesses, three to
eight worse than scraping a win at the wire. Little rides on the choice: anywhere
in that range the best setting beats the next by under 0.0002 guesses.

### Widening the search's reach

The second lever is `--exact-max`, how large a candidate set the exact search
will take on. At the default penalty, again over every word and every prior:

| `--exact-max` | loss probability | mean (loss = 7) |
| --- | --- | --- |
| 24 | 0.0609% | 3.6647 |
| 40 | 0.0518% | 3.6635 |
| **60 (default)** | **0.0458%** | **3.6626** |
| 80 | 0.0417% | 3.6618 |
| 100 | 0.0417% | 3.6613 |

This lever costs nothing in guesses: reaching further replaces an estimate with a
true value rather than trading one thing for another, so the mean improves all
the way down the column. What it buys falls off sharply, though. Going 24 → 60
takes a quarter of the loss mass; 60 → 100 takes a further ninth of it and
0.0013 guesses.

**Under the shipped prior it buys nothing at all.** Held at 60, 80 and 100 the
loss probability is 0.0240%, 0.0240% and 0.0239% and the mean is 3.0770
throughout; reaching to 100 rescues two words out of 12,947 and not one of the
ten curated losses. Every gain in the table above comes from the flat and
curation-only rows. The families that beat the solver — `water`/`hater`,
`catty`/`ratty` — are not positions a deeper search can crack, because nothing is
wrong with how they are being played; see *The penalty*.

What stops it at 60 is time, and only the flat prior pays that. Reaching to 100
runs 3.5× slower there, and the node budget starts firing — six bailouts in four
hundred games, each one a position that falls back to the heuristic instead of
being solved, so past 60 the "exact" search is quietly less exact. Under the
shipped prior the curve is nearly flat, 100 costing 1.13× what 60 does, because
concentrated mass means the candidate set rarely gets that big. The penalty
drives none of this: 1, 5 and 30 run within 7% of each other at either reach,
which is worth saying because the bound is capped at `min(penalty, 1)` while true
values scale with the penalty, and that asymmetry might have been expected to
show up as cost. It does not.

Between the penalty and the wider reach, the loss probability falls from 0.1198%
to 0.0458% — **a 62% reduction** — and the mean *improves* by 0.0019, the reach
gaining more than the penalty spends.

## Lookahead (`--lookahead`, off by default)

Above the exact search sits an optional width-limited expectimax. Only the guess
is pruned, never the feedback — dropping a bucket would leave the sum something
other than an expectation:

```
V_d(S, t) = min over the best `--beam-width` guesses of
            [ 1 + sum_{r != GGGGG} p_r · V_{d-1}(S_r, t-1) ]
V_0(S, t) = min_g E(g, S)
```

with the exact search taking any sub-position it can reach. `--beam-width` is a
fan-out at each node rather than a frontier held to a constant size, which makes
this an expectimax with its guesses pruned and not a beam search: a second ply
multiplies the work by the width again, and that is where the 205 s below comes
from.

Over 20,000 paired games (5,000 per prior, identical answers,
`analysis/lookahead.py`), one ply at width 10 is worth **−0.0034 ± 0.0041**
guesses — 0.9σ, which is to say nothing at all, for roughly 10× the wall clock.
What little there is sits on the flat no-priors case at −0.0094 ± 0.0080; under
the shipped prior it reads +0.0012 ± 0.0050, slightly the wrong way. Width 25
measured no better, the opening word stays `salet` at every width from 5 to 40,
and two plies costs 205 s for the opening position alone.

Its value is not a fixed property of the solver: what lookahead measures is the
room the base policy leaves above itself, so refitting `f` takes it away. This
was worth a real −0.0158 ± 0.0038 under the forms that shipped before the
2026-08-10 refit.

The interesting part is that it barely matters *despite* valuing positions far
better. The policy is deterministic, so the true expected guesses from a position
can be computed rather than estimated — play it out once per candidate answer,
weighted by the prior. Each valuation is scored against what its own play
achieves, over 222 positions from real games, weighted by how often play reaches
them (`analysis/lookahead_accuracy.py`):

| valuation | true value | bias | mean abs error | RMSE |
| --- | --- | --- | --- | --- |
| `min_g E(g, S)` (what the solver scores positions at) | 2.7634 | +0.0392 | 0.0868 | 0.1122 |
| one-ply lookahead | 2.7526 | +0.0010 | 0.0031 | **0.0111** |

Ten times more accurate, and near enough unbiased. Those are the positions where
the heuristic is the one deciding, which means turns two and three; by the fourth
the candidate set is under `--exact-max` and both valuations return the same true
value.

Accuracy in the level is not what drives play. A valuation wrong by a constant at
a position still orders the guesses perfectly and costs nothing, so what matters
is the ranking *within* a position. Playing out the true value of the ten guesses
lookahead re-ranks, at those same positions:

| | blended | none | curation | repeat | shipped |
| --- | --- | --- | --- | --- | --- |
| spread of true value, best to worst | 0.0941 | 0.0925 | 0.0778 | 0.0927 | 0.1131 |
| `min_g E(g, S)` regret | 0.0115 | 0.0194 | 0.0082 | 0.0153 | 0.0033 |
| one-ply lookahead regret | 0.0013 | 0.0012 | 0.0035 | 0.0003 | 0.0000 |
| the estimator picks the best | 52.5% | 38.1% | 64.1% | 42.3% | 65.4% |
| lookahead picks a different guess | 50.1% | 64.2% | 47.5% | 53.7% | 35.2% |
| rank correlation with the truth | +0.541 | +0.478 | +0.526 | +0.471 | +0.687 |

The estimator orders those ten barely better than a coin flip, and its pick is
the best one just over half the time. Lookahead disagrees in half of all games,
changes the length of a quarter of them (2,644 better against 2,597 worse), and
is right to disagree — 0.0013 given up against 0.0115. Better valuation does buy
better decisions here, under every prior.

What it does not buy is guesses, and the first row is why. Those ten spread
0.0941 between best and worst, and the estimator's pick gives up an eighth of
that. The top of the ranking is a plateau: the estimator finds it reliably and
cannot tell one spot on it from another, which costs a hundredth of a guess
because there is little there to tell apart. Recovering 0.0102 of it per decision
still reads as −0.0034 over a whole game, a game only meeting such a position
where the exact search has not already taken over.

**This is the recurring result in this project**: a valuation improvement that is
real, large, and correctly converted into decisions still buys almost nothing. It
holds for the deep exact search, for one-ply lookahead, for turn-conditioned `f`,
and for the sharpened repeat prior above — though only lookahead's has been taken
apart at the level of the decision, the other three measured at the outcome.

### Conditioning f on turns remaining (tried, reverted)

`f(n)` estimates guesses-to-finish with no turn limit, so a position with four
turns left is scored as one with two. Fitting `f` once per turn count produces
curves with a large turn effect (a 2,000-candidate bucket worth 4.97 with three
turns left against 3.04 with five) and changes nothing: over 80,000 paired games,
losses 81 → 79 and mean +0.0005 ± 0.0007. By the time the deadline binds the
exact search has already taken the decision — over 16,000 games, with three turns
left `f` decides 10 positions against the search's 2,764, and with two or fewer
it decides none. The rows where turn-conditioning would matter are the rows
nobody reads.

## Forms of f

Any basis linear in its parameters works, so every fit is a single least-squares
solve. `estimator.FORMS` holds them, all written as
`f(n) = base + sum_i params[i] * term_i(n)` with `x = log2(n)` and `base` either
0 or 1 (pinning `f(1) = 1`); the result is floored at 1.

f must not decrease anywhere in `[1, 12947]` — a value function that rewards a
larger candidate set would let the search prefer uninformative guesses — so
terms are dropped from the end (most flexible first) until it is monotone.

## Policy iteration

`train` alternates:

1. **evaluate** — play 2,000 sampled games with the current policy, recording
   for every position its effective size and how many guesses were actually
   needed from there;
2. **improve** — refit `f` to those pairs; acting greedily on the new `f` is the
   improved policy.

Round 0 is seeded with greedy entropy. Training games are played with no
deadline, so targets are unbiased estimates of expected guesses and `f` keeps
that meaning; the six-guess limit is a constraint layered on top.

The loop stops when the parameters settle, or when the measured mean stops
improving (a greedy policy can cycle), and caches the parameters whose policy
measured best — the form, opening word, fit history and holdout score, one file
per prior combination in `estimator_cache/params_<slug>.json` (with `compare`
archiving to `estimator_cache/forms/<slug>__<form>.json`). The slug names the
shapes as well as the toggles, so a fit made under one shape can never be read
back under another. A cached file is refused if it was fitted with different
prior probabilities, or predates the form registry.

**A prior with no fit does not play.** `load_active` raises `NotFitted` and names
the `train` command that fixes it. The alternative was to bend a fixed reference
curve onto the chosen form and play that, and the invented curve was never a
measurement of any prior: under the shipped one it read f(2000) ≈ 3.69 where the
fitted `f` says 2.55. It flipped the opening word from `salet` to `raise`, cost
0.034 ± 0.004 guesses paired over 20,000 games, and was never saved, so every run
paid for it again. Entropy mode, which never consults `f`, still runs on any
prior.

The refusal carries the `train` command that ends it, so both front ends can hand
it over as it stands: the terminal prints it and exits 2 rather than a traceback,
and the page reports it as a 503 with the same command in it. Startup shows it on
the veil instead of a stack trace, since there is nothing to debug.

## Results

Both priors on with the shipped shapes, six-guess limit, failures counted as 7.
Every row is the fit archived for that form, replayed rather than refitted, by
`analysis/results_tables.py`. There is no sample here and so no error bar: the
solver is deterministic, so every one of the 12,947 words is played once and
weighted by the prior, which *computes* each mean rather than estimating it.

| form | expression | terms | opening | mean | vs. best | loss probability |
| --- | --- | --- | --- | --- | --- | --- |
| spline | piecewise linear, knots at x = 2/4/6/8 | 5/6 | salet | 3.0770 | — | 0.0240% |
| saturating | a + b·x + c·log2(1+x) | 3/3 | salet | 3.0786 | +0.0016 | 0.0253% |
| invonly | a + b·(1−1/n) | 2/2 | salet | 3.0799 | +0.0029 | 0.0233% |
| anchored2 | 1 + a·x + b·(1−1/n) | 2/2 | salet | 3.0876 | +0.0106 | 0.0230% |
| anchored | 1 + a·x + b·(1−1/n) + c·x² | 2/3 | raise | 3.1165 | +0.0395 | 0.0278% |
| sqrt | a + b·x + c·√x | 3/3 | raise | 3.1165 | +0.0395 | 0.0275% |
| cubic | a + b·x + c·x² + d·x³ | 2/4 | raise | 3.1179 | +0.0409 | 0.0335% |
| loglinear | a + b·log2(n) | 2/2 | raise | 3.1179 | +0.0409 | 0.0335% |
| quadratic | a + b·x + c·x² | 2/3 | raise | 3.1179 | +0.0409 | 0.0335% |
| greedy entropy | — | — | raise | 3.6815 | +0.6045 | 0.0034% |

The table divides where the opening word does. The top three are separated by
0.0029 guesses, and since these are exact figures that ordering is a fact about
this prior rather than a coin landing — though 0.0029 is still far too little to
choose a *default* on, which is what the next section is for. Everything from
`anchored` down gives up ~0.04 guesses, more than ten times the spread among the
leaders. `anchored` and `sqrt` land on the same value, and `cubic`, `loglinear`
and `quadratic` are identical because the monotonicity guard cuts all three back
to the same `a + b·x`.

The loss column is the one place greedy entropy wins, and it wins by a lot: see
*Losing less* for why the estimator's six tenths of a guess costs seven times the
loss rate.

What separates the groups is **concavity in x**, not fit quality: `anchored2`
has the worst fit RMSE of any form (0.70 against `invonly`'s 0.49) and still
beats the five below it. With `f` exactly linear in x,

```
E(g, S) = 1 + a·(1 − p_win) + b·(H(S) − IG(g))
```

so minimizing E is the same as maximizing `b·IG + a·p_win`: greedy entropy plus a
bonus for possible answers. That is why `loglinear` picks entropy's opening word
and gains the least — of the 0.60 guesses between greedy entropy and the best
form, ~0.56 is that bonus and ~0.04 is curvature.

Saturation is no longer the liability it was under the flat prior. `invonly`
flattens out completely (f(200) ≈ f(2000)) and used to rank 6th; with the mass
concentrated it comes third, because the effective size rarely reaches the range
where the curve has gone flat.

### Which form is the default

`--form` defaults to **`saturating`**: it is the family `train` fits when you do
not name one. It was `sqrt` until 2026-08-10, on a ranking the table above no
longer supports.

The default is what gets fitted to a prior nobody has tried yet, so
`analysis/form_default.py` replays every archived fit over every word the answer
could be, weighted by each prior in turn. Nothing is held out because nothing
needs to be: the evaluation covers the whole space, and the gaps are exact.

| form | terms | mean gap to the prior's best | worst gap | worst rank of 9 |
| --- | --- | --- | --- | --- |
| **saturating** | 3 | **+0.0015** | +0.0045 | **6** |
| invonly | 2 | +0.0017 | **+0.0029** | 9 |
| spline | 5–6 | +0.0020 | +0.0062 | 8 |
| anchored2 | 2 | +0.0046 | +0.0106 | 9 |
| cubic | 2–4 | +0.0169 | +0.0409 | 7 |
| sqrt | 3 | +0.0173 | +0.0409 | 9 |

The top three span 0.0005 guesses on the mean and 0.0033 on the worst case, which
is not enough to separate them on performance: `saturating` takes the best mean
gap and `invonly` the best worst gap. The rank column discriminates less than it
looks, too — on the repeat-only prior all nine forms sit within 0.0022 of each
other, so `invonly` being 9th there costs almost nothing.

What keeps `saturating` the default is that it is never the outlier. It is the
best form outright under `frequency`, second under `wordlebot` by 0.0016, and
never worse than 6th anywhere — the best worst case of the nine. `invonly` gives
up 0.0029 under `wordlebot`, and `spline` carries six parameters into priors
whose rollouts have never been seen, dropping to 8th on the repeat-only one.

That `sqrt` is not in this group is the firm part of the result. Its mean gap is
eleven times the leaders' and its worst is 0.041 guesses.

Read this in preference to ranking the archives by their own stored holdout,
which is the score `compare` picked the winners on: that ordering puts
`anchored2` first and `saturating` third, where measuring over every answer puts
`anchored2` fourth.

Nothing shipped changes: all five prior combinations have a cached fit, and
`load_active` prefers it. What moves is `train` with no `--form`, and any
combination whose cache is missing.

Per prior combination, shipped shapes, each with its own fitted form. Every word
is weighted by *the solver's own prior*, so each row asks "if this model of the
editor is right, how well does the solver do?" The rows are **not** comparable to
each other, since a concentrated prior makes the game genuinely easier. The
opening word and the solve rate are the expected-guesses mode's:

| priors | form | opening | entropy mean | expected mean | solve rate |
| --- | --- | --- | --- | --- | --- |
| none | saturating | tares | 4.0794 | 4.0718 | 99.93% |
| curation only | invonly | trace | 3.7802 | 3.4903 | 99.99% |
| repeat only | loglinear | tares | 4.0639 | 4.0111 | 99.92% |
| both | spline | salet | 3.6815 | 3.0770 | 99.98% |

The gain comes from the curation prior. When the prior is nearly flat, expected
guesses and information gain rank guesses almost identically — with no priors the
estimator is worth **0.0076** guesses against entropy, next to **0.6045** with
both priors on. Eighty times less for the same machinery, as the identity above
predicts. (It was reported as nothing at all before 2026-08-10; it is small, but
it is there, and being exact it is no longer a question of resolving it.)

Because `f` estimates expected guesses, the policy is risk-neutral away from the
endgame, where greedy entropy is implicitly cautious. `--loss-penalty 5`
compensates, closing about two thirds of the gap in loss probability, but not for
nothing and not the whole way: the estimator still loses seven times as often as
entropy does (see [Losing less](#losing-less)). `--loss-penalty 1` restores risk
neutrality and gives that ground back.
