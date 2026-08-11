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

Both terminal front ends are there, and one thing that has no terminal
equivalent. The group beside *Settings* says which of the three the page is
doing — it marks where you are, rather than where the button takes you.

**Solve** is `wordle_solver.py play`: the feedback is entered by coloring tiles
rather than typing `NYGNN`, and the prior toggles, `--mode`, `--form` and
`--hard` are switches rather than flags. *Shape of f* leaves the choice to the
cache by default and marks which form that is, since the winning form is fitted
per prior and moves with the toggles. The panel alongside the board carries the
alternatives, the candidates that are still possible, and — under *Advanced* —
the prior with the rates it is set from, the objective and the fitted f, each
with an explanation of what it is doing. Those rates are measured from
`solutions.json` when the board is built rather than written into the page, so
the explanation cannot fall behind the answer record the way a sentence quoting
"about 10%" does. They count one puzzle fewer than the table in *Both defaults
describe a regime six months old* below: the server drops the last entry of
`solutions.json` (`--mask-last`, default 1) so that today's answer is not sitting
in the solver's prior while you are solving today's puzzle.

**Play** is `wordle_game.py`. The *Answer* box takes a word of your own, which is
the quickest way to set the solver a position, and it is the only setting that
deals a new game — an answer already dealt cannot be given a different one.
*Hint* is the terminal game's `?` and presses again to close, the *Assist* switch
is `--assist` (Enter then plays its pick, which is `--auto`), *Reveal answer* is
`quit`, and the emoji grid at the end is `--share`. Under the recommendation is
an *Alternatives* card holding the rest of the ranking, opened by *Assist* and
never closed behind you; clicking a word puts it in the row you are typing. The
answer stays on the server until the game is over, so the network traffic does
not spoil it.

**Find** is the page's own, and runs the other way round from everything else:
the answer is known and the guess is what is being looked for. Type an answer,
color the pattern a guess came back as, and it lists every word in the list that
would have produced that pattern against that answer — the question you ask of
somebody else's share grid. The two rows do not describe each other, so there is
no ghost word and nothing is grayed out; a gray tile there is a color the pattern
can be, not an empty tile waiting for a letter.

The match itself is exact and immediate — the answer is a column of the feedback
table, so finding the words is a comparison over it rather than a scoring pass.
The three numbers each word carries are not:

* **Words left** is how many candidates would survive that guess *given this
  pattern*, so it ranks the words with hindsight — smallest first is what would
  have told you the most about this particular answer.
* **Bits** is expected information gain, averaged over every answer the guess
  could have faced. It does not depend on the pattern at all, so it ranks the
  words as openings instead.
* **Chance** is the prior's own probability that the guess is itself the answer.

Both of the last two move with the prior switches and nothing else does, which is
why the mode, the shape of f and hard mode are hidden while the finder is
showing: none of them reaches it. It never asks for a guess, so it needs no
fitted `f` and works under any prior.

| key | solve | play | find |
| --- | --- | --- | --- |
| A–Z | write a letter into the selected tile | write a letter | write a letter of the answer |
| ← → | move along the row | move along the row | move along the row |
| ↑ ↓ | move between rows, never past the row being played | — | move between the answer and the pattern |
| Space | change the selected tile: gray → yellow → green | — | change the selected pattern tile |
| Enter | on an empty row take the suggestion, otherwise submit | guess, or play the solver's pick when it is showing | find every word that fits |

Going back up to an earlier row and correcting it is the point of ↑: the whole
board is sent on every submit, so a tile fixed four rows up re-runs the solver
from there rather than needing the rows under it deleted first.

### Editing the board

The page keeps its own copy of every row, and every change to it — typing,
coloring, deleting — is local until Enter sends the whole board back. The solver
only ever sees complete boards, which is what makes correcting an earlier row
the same operation as playing a new one. Rows the solver has not been told about
yet are drawn with a dashed border.

Four rules fall out of that, and they are the answers to the things that used to
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
few letters of a row can go without disturbing the last ones.

**Delete row empties the row in place and leaves the gap.** The rows under it
stay where they were played, so the word that should have been there goes
straight into the hole. Submitting with the gap still empty is refused — *Row 2
is empty* — rather than closing it up and sliding the rows below into new
positions.

That is a deliberate refusal rather than a missing feature. A word only reaches
the board because it was played, so a row that needs deleting is nearly always
one that needs *replacing*; closing the gap would silently claim you played
those later words earlier than you did. And the position is not free to move:
`guesses_left = MAX_GUESSES - len(rows)` feeds the endgame search and the
estimator, so shifting rows up changes what the solver recommends.

An empty row with nothing under it is a different thing — it is the row waiting
to be played, so it is simply not sent. That is what makes *Delete row* on the
last guess an undo, and what clears the board back to the opening
recommendation.

One request is in flight at a time, since the solver is behind a single lock
anyway. A second made meanwhile is *held* rather than dropped: dropping it left
the page showing something nobody asked for — a game view with no game dealt
into it, every panel reading "—" — and stay that way. For the same reason the
loading veil stays up until there is a board behind it, retrying if the first
request fails.

**A setting throws nothing away.** The three views share one set of switches, and
changing any of them — a prior, the mode, the shape of f, hard mode — asks the
solver again about the position you are already in. The solve board is re-run
over the rows the server has already accepted, so a word half typed into the
open row is neither submitted nor lost; the game keeps its answer and re-asks the
hint; the finder searches again. Only the *Answer* box deals a new game.

This is a change from the four combinations the terminal menu offers, where the
prior decides what the answer is drawn from and so has to be chosen before the
deal. Here the answer was drawn under whatever was on at the time and stays what
it is: turning a prior off mid-game rethinks the advice rather than dealing a new
word, which is the only reading that lets you ask what the *other* model of the
editor would have played from here.

Nothing is rebuilt behind your back either. A setting changed while a view is not
showing marks it, and it is re-run when you next look at it — the alternative
being a view that quietly claims a setting it was never built under.

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

**Save** leaves the source alone — the rewrites live in `wording.json` and are
applied to the served bytes on the way out. **Write into the code** puts them in
the files for good and clears `wording.json`, which is the end state worth
reaching: an override is matched to the code by wording, so it is only fragile
for as long as it sits outside it. Prose is re-wrapped to the indentation it
lands in, and the old and new phrases are appended to `wording_history.json`.
That record is the only one kept — the bake edits the source in place and there
is no undo, so it is worth a look at the source afterwards. The server keeps
running the code it started with, so its own messages need a restart before they
read the new way.

Nothing to install: the server is `http.server` and the page is three static
files in `web/`, with three more behind `/editor`. The feedback table and the
numba kernels are warmed up in the background at startup and the page waits for
them, so the first load is the slow one. The solver is behind a single lock —
scoring ~13k guesses already uses every core, so two requests at once would only
make both slower.

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

The shapes are only worth having **together**: see *Shaping the two priors*
below. The shape is part of the cache key, because the best form of `f` depends
on it — `sqrt` is right for the uniform shapes and picks the wrong opening word
for the sharpened ones.

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
the same editorial response to the same constraint, so extrapolating both
upward is defensible, and 0.10 is the forward-looking number rather than the
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

An age-based shape — weight rising with time since last use, rather than binned
by where in the answer order the word first appeared — is the obvious rival and
does not beat this. Refitting its rate prequentially scores −143.1, worse than
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

Both shapes are worth having, and are shipped rather than left out. They were
originally adopted because they looked *superadditive* — worth much more
together than apart. On current data they no longer do; see the note under the
table.

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
used word's 5.4e-5, so past answers stop competing for the solver's attention —
hence the residual penalty `wordlebot` alone still carries on the repeat arm.

**The superadditivity is gone.** These shapes were adopted because `wordlebot`
alone blended to +0.009 (a wash), `binned` alone to −0.021, and the pair to
−0.091 — far more than the sum. Recomputed here they are −0.044, −0.022 and
−0.064, against a sum of −0.066: additive, within noise. The pair is still the
best row and still worth shipping; the *reason* given for shipping it no longer
holds.

Two caveats before reading much into the change. These numbers come from a
solver whose estimator was refitted afterwards, and every configuration's
winning form moved, so they are not comparable like-for-like with the published
ones — the policy changed, not just the data. And `frequency` (−0.074 ± 0.028)
and `wordlebot` (−0.064 ± 0.032) now sit well inside each other's intervals, so
the shipped default of `wordlebot` is no longer measurably the better of the two.

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

Mechanically it is one substitution — the pool of guesses to score becomes the
candidate set rather than all 12,947 words (`WordleSolver.guess_pool`), which the
kernels already take as a parameter, and the exact search narrows the same way.
In the search that also removes the partition dedupe: only a candidate itself can
come back `GGGGG`, so no two candidates split S identically and there is nothing
left to merge.

It is deliberately **not** a `train` flag. The fit is cached per prior and the
restriction is not part of the slug, so training under it would overwrite the
ordinary fit. So `f` is always the one fitted for unrestricted play, and under
the restriction its scores read a little low — the estimate is of a position from
which the solver would be allowed a probe it is not in fact allowed. That is the
same reason the restriction costs guesses at all: a word that cannot win is often
the sharper split.

What it costs, over 20,000 games per row, answers drawn from the row's own prior,
the same answers in both arms, failures counted as 7 (`analysis/hard_mode.py`):

| priors | free | hard | paired cost | losses, free → hard |
| --- | --- | --- | --- | --- |
| none | 4.0629 | 4.4467 | +0.3839 ± 0.0073 | 12 → 1561 |
| curation only | 3.4846 | 3.5928 | +0.1081 ± 0.0046 | 2 → 87 |
| repeat only | 4.0133 | 4.3688 | +0.3554 ± 0.0072 | 9 → 1501 |
| **both (default)** | **3.0770** | **3.1059** | **+0.0290 ± 0.0028** | **2 → 16** |
| both, `frequency` | 3.1193 | 3.1490 | +0.0297 ± 0.0027 | 11 → 65 |

Under the shipped prior it costs **+0.029 ± 0.003** guesses — real at 10σ, and
about a twentieth of what the estimator is worth over greedy entropy. What sets
the size is how flat the prior is: with no priors at all the cost is thirteen
times larger and the solve rate falls from 99.94% to **92.20%**, because a wide
candidate set with no probe available is exactly the position the restriction
forbids you to fix. Concentrate the mass and the candidate set collapses fast
enough that the likely answers are decent probes anyway.

The losses are where it really shows. Even in the default row, where the mean
barely moves, they go 2 → 16 over the same games. Hard mode arrives in the
one-letter families of *The penalty* below and then cannot spend a turn
separating them, since the separating word is the one it is not allowed to play.

The opening word never changes, in any row. Every one of the 12,947 words carries
prior mass, so at the first guess every word is still a possible answer and the
restriction binds nothing at all.

Under `--mode entropy` the restriction **helps**, by −0.176 ± 0.005 guesses on the
default prior (3.6785 → 3.5021, over the same 20,000 games). That is the identity
under *Results* read backwards: what greedy entropy is missing is the bonus for
guessing a word that could be the answer, and hard mode supplies it by force. It
pays in losses, 0 → 30, and it still does not reach the expected-guesses policy's
3.0770 — which gets the same bonus for free, and keeps the probes.

## The exact endgame

Once the candidate set is small, or few turns remain, `exact.py` computes the
value outright:

```
V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
V(S, 0) = loss_penalty
```

with `t` the guesses left including this one. A loss penalty of 1 means running
out costs one unit more than the six turns already spent — a lost game scores 7
— so the deadline needs no separate objective and no risk weighting.

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
* **Branch and bound.** Guesses are tried in order of their two-turn value, and
  within a guess each bucket's bound is replaced by its exact value one at a
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
the loss rate, 50 → 39. Per decision over 3,000 games the estimator differs from
the search on 18.9% of the states it reaches, at a mean regret of **0.0026
guesses** — and at 3–4 candidates it differs 16.4% of the time with *exactly
zero* regret, which are genuine ties.

Refitting `f` under the search moves the curve by under 0.03 guesses everywhere
(fit RMSE 0.63) and does not change measured play, so `train` keeps only the free
`t ≤ 2` part by default; `--exact-max 24` fits under the full policy.

## Losing less

### Greedy entropy is not a lossless baseline

10,000 games under both priors, identical answers throughout:

| policy | mean | losses | solve rate |
| --- | --- | --- | --- |
| greedy entropy | 3.6895 | 9 | 99.910% |
| expected, no exact search, penalty 1 | 3.1828 | 8 | 99.920% |
| expected, exact-max 24, penalty 5 | 3.1816 | 2 | 99.980% |
| **expected, exact-max 60, penalty 5 (default)** | **3.1815** | **4** | **99.960%** |

Entropy loses about 1 game in 1,000, so the default 200-game `benchmark` finds
zero losses ~83% of the time — which is why it looks lossless. The losses are
disjoint (8 entropy-only, 4 expected-only, none shared) and 9 against 4 is not
significant (McNemar p ≈ 0.39). The two lose at indistinguishable rates, so the
estimator's half-guess advantage in the mean is free.

Wordle is provably solvable in five guesses, but that theorem is about the
curated 2,309 words. Here every one of the 12,947 words carries prior mass, and
under both priors 9% of it sits on the 10,638 never-curated ones. Every loss in
every run above was a non-curated word — `karks`, `jiber`, `fents`, `faffy`,
`sixes`/`wives`/`nines`/`doves`. Playing all 2,309 curated words as the answer:

| policy | mean | worst | losses |
| --- | --- | --- | --- |
| greedy entropy | 3.8094 | 6 | 0 / 2309 |
| expected (defaults) | 3.6154 | 6 | 0 / 2309 |

Neither ever fails on the real answer list. The six-guess games are wins; they
exist because the solver hedges over 12,947 possible answers instead of knowing
the answer is one of 2,309, and the five-guess guarantee needs a full-depth
optimizer over the restricted set.

### The penalty

Losing is cheap to avoid, which is why `--loss-penalty` defaults to **5** rather
than the risk-neutral 1. The last guess is already played optimally — over 20,000
games per prior, expected losses given the positions reached (Σ 1−p_max) match
the actual count closely (42.2 against 46, 23.5 against 19, 49.6 against 52, 19.1
against 20). The solver is not blundering at the end; it is *arriving* somewhere
it cannot win, with a one-letter family it cannot separate in the turns left —
`vills`/`jills`/`zills`, `jakes`/`wawes`/`waxes`. The fix is to spend a turn
separating rather than gamble, and that is what the penalty buys.

80,000 games (20,000 per prior), same answers throughout:

| `--loss-penalty` | losses | loss rate | mean (solved) | mean (loss = 7) |
| --- | --- | --- | --- | --- |
| 1 | 137 | 0.171% | 3.7104 | 3.7160 |
| 2 | 93 | 0.116% | 3.7121 | 3.7159 |
| **5 (default)** | **81** | **0.101%** | 3.7128 | 3.7161 |
| 10 | 79 | 0.099% | 3.7129 | 3.7162 |
| 30 | 74 | 0.092% | 3.7145 | 3.7175 |

Two fifths of all losses disappear for **+0.0001 guesses** on the penalized mean.
5 sits at the knee: past it the losses stop falling and the mean starts to rise.
Opening words are untouched, since the root never reaches the exact search.

### Widening the search's reach

The floor at ~79 is where the exact search stops reaching, so the second lever
is `--exact-max`. Losses are rare and the games are identical, so this counts
games lost by exactly one setting ("fixed" = the baseline lost it, the wider
search did not).

| `--exact-max` | losses | fixed | broken | mean vs baseline | wall clock |
| --- | --- | --- | --- | --- | --- |
| 24 | 81 | — | — | — | 46 s |
| 40 | 71 | 18 | 8 | −0.0004 ± 0.0006 | 73 s |
| **60 (default)** | **60** | **27** | **6** | **−0.0014 ± 0.0007** | 103 s |

A quarter of the remaining losses go at 2.2× the wall clock, and the mean gets
*better* — 27 fixed against 6 broken is decisive (McNemar χ² ≈ 13, p ≈ 0.0002).
Cost climbs steeply past this: the pruning bound is capped at `min(penalty, 1)`
while true values scale with the penalty, so the bound loosens as the penalty
rises. `--exact-max 100` was abandoned unfinished after more than an order of
magnitude longer than 60.

Between the penalty and the wider reach, losses go from 137 to 60 — **a 56%
reduction** — with the mean unchanged.

## Lookahead (`--lookahead`, off by default)

Above the exact search sits an optional width-limited expectimax. Only the guess
is pruned, never the feedback — dropping a bucket would leave the sum something
other than an expectation:

```
V_d(S, t) = min over the best `--beam-width` guesses of
            [ 1 + sum_{r != GGGGG} p_r · V_{d-1}(S_r, t-1) ]
V_0(S, t) = min_g E(g, S)
```

with the exact search taking any sub-position it can reach. Over 20,000 paired
games (5,000 per prior, identical answers), one ply at width 10 is worth
**−0.0158 ± 0.0038** guesses — real at 4.2σ, but for roughly 10× the wall clock,
and most of it comes from the flat no-priors case (−0.0378 ± 0.0084) rather than
the default one (−0.0080 ± 0.0064). Width 25 measured no better, the opening word
stays `salet` at every width from 5 to 40, and two plies costs 205 s for the
opening position alone.

The interesting part is that it barely matters *despite* valuing positions far
better. The policy is deterministic, so the true expected guesses from a position
can be computed exactly — play it out once per candidate answer, weighted by the
prior. Against that truth, over 55 real positions:

| estimator | bias | mean abs error | RMSE |
| --- | --- | --- | --- |
| `min_g E(g, S)` (what the solver scores positions at) | −0.0064 | 0.0435 | 0.0663 |
| one-ply lookahead | −0.0116 | 0.0118 | **0.0216** |

Three times more accurate, and it changes the guess in ~17% of games — but those
changes split almost evenly between better and worse (2,414 vs 2,167), because
the estimator's errors are close to common-mode across the guesses being compared
at any one position and so cancel in the `argmin`.

**This is the recurring result in this project**: better *valuation* is not
better *decisions*. It holds for the deep exact search, for one-ply lookahead,
for turn-conditioned `f`, and for the sharpened repeat prior above. The solver
has been decision-limited, not valuation-limited, for some time.

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
the `train` command that fixes it, rather than bending a fixed reference curve
onto the chosen form and playing that. The invented curve was never a
measurement of any prior — under the shipped one it read f(2000) ≈ 3.69 where
the fitted `f` says 2.55 — it flipped the opening word from `salet` to `raise`,
it cost 0.034 ± 0.004 guesses paired over 20,000 games, and it was never saved,
so every run paid for it again. Entropy mode, which never consults `f`, still
runs on any prior.

The refusal carries the `train` command that ends it, so both front ends can hand
it over as it stands: the terminal prints it and exits 2 rather than a traceback,
and the page reports it as a 503 with the same command in it. Startup shows it on
the veil instead of a stack trace, since there is nothing to debug.

## Results

Both priors on with the shipped shapes, 30,000 unseen answers (6 blocks of
5,000, identical answers for every row), six-guess limit, failures counted as 7.
Every row is the fit archived for that form, replayed rather than refitted, by
`analysis/results_tables.py`. *Block sd* is the spread of that row's gap over
the six blocks:

| form | expression | terms | opening | mean | vs. best | block sd |
| --- | --- | --- | --- | --- | --- | --- |
| spline | piecewise linear, knots at x = 2/4/6/8 | 5/6 | salet | 3.0789 | — | — |
| saturating | a + b·x + c·log2(1+x) | 3/3 | salet | 3.0807 | +0.0018 | 0.0014 |
| invonly | a + b·(1−1/n) | 2/2 | salet | 3.0816 | +0.0027 | 0.0019 |
| anchored2 | 1 + a·x + b·(1−1/n) | 2/2 | salet | 3.0891 | +0.0102 | 0.0044 |
| anchored | 1 + a·x + b·(1−1/n) + c·x² | 2/3 | raise | 3.1169 | +0.0380 | 0.0105 |
| sqrt | a + b·x + c·√x | 3/3 | raise | 3.1170 | +0.0381 | 0.0104 |
| cubic | a + b·x + c·x² + d·x³ | 2/4 | raise | 3.1190 | +0.0401 | 0.0092 |
| loglinear | a + b·log2(n) | 2/2 | raise | 3.1190 | +0.0401 | 0.0092 |
| quadratic | a + b·x + c·x² | 2/3 | raise | 3.1190 | +0.0401 | 0.0092 |
| greedy entropy | — | — | raise | 3.6802 | +0.6013 | 0.0107 |

The table divides where the opening word does. The top three are one result —
0.0027 apart, on paired per-block sds of 0.0014–0.0019 — and which of them comes
first moves with the seed; `analysis/form_default.py` on a different seed puts
`saturating` ahead of `spline`. Everything from `anchored` down gives up ~0.04
guesses, several times its own block-to-block spread. `cubic`, `loglinear` and
`quadratic` are identical because the monotonicity guard cuts all three back to
the same `a + b·x`.

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
`analysis/form_default.py` replays every archived fit against a holdout none of
them were selected on — a fresh seed, 3,000 answers per prior, the same answers
for every form, so the gaps are paired:

| form | terms | mean gap to the prior's best | worst gap | worst rank of 9 |
| --- | --- | --- | --- | --- |
| spline | 5–6 | +0.0045 | +0.0083 | 7 |
| invonly | 2 | +0.0045 | +0.0113 | 9 |
| **saturating** | 3 | **+0.0046** | +0.0097 | 7 |
| anchored2 | 2 | +0.0095 | +0.0203 | 6 |
| cubic | 2–4 | +0.0139 | +0.0353 | 7 |
| sqrt | 3 | +0.0173 | +0.0323 | 9 |

The top three are one result, not three: 0.0001 guesses apart against standard
errors of 0.002–0.014. What separates them is what happens when they are wrong.
`invonly` is 9th of 9 on the repeat-only prior; `spline` matches `saturating`
everywhere but carries six parameters into a prior whose rollouts have never
been seen. `saturating` is never worse than 7th, and it is the *best* form on
both of the priors that actually ship — 3.0750 with `wordlebot`, 3.1123 with
`frequency`, where `sqrt` is 6th and 9th and opens `raise` instead of `salet`.

That `sqrt` is not in this group is the firm part of the result. Its mean gap is
four times the leaders' and its worst is 0.032 guesses, well outside noise.

Read this in preference to ranking the archives by their own stored holdout,
which is the score `compare` picked the winners on: that ordering puts
`anchored2` first and `saturating` third, and a fresh seed puts `anchored2`
fourth.

Nothing shipped changes: all five prior combinations have a cached fit, and
`load_active` prefers it. What moves is `train` with no `--form`, and any
combination whose cache is missing.

Per prior combination, shipped shapes, each with its own fitted form, 20,000
answers. Answers are drawn from *the solver's own prior*, so each row asks "if
this model of the editor is right, how well does the solver do?" — exact within
a row (the two modes see byte-identical answers), but the rows are **not**
comparable to each other, since a concentrated prior makes the game genuinely
easier. The opening word and the solve rate are the expected-guesses mode's:

| priors | form | opening | entropy mean | expected mean | solve rate |
| --- | --- | --- | --- | --- | --- |
| none | saturating | tares | 4.0786 | 4.0668 | 99.94% |
| curation only | invonly | trace | 3.7826 | 3.4952 | 100.00% |
| repeat only | loglinear | tares | 4.0732 | 4.0168 | 99.91% |
| both | spline | salet | 3.6805 | 3.0791 | 99.98% |

The gain comes from the curation prior. When the prior is nearly flat, expected
guesses and information gain rank guesses almost identically — with no priors the
estimator is worth **−0.012 ± 0.004** guesses against entropy, paired over the
same 20,000 games, next to **−0.601 ± 0.005** with both priors on. Fifty times
less for the same machinery, as the identity above predicts. (It was reported as
nothing at all before 2026-08-10; it is small, but it is there.)

Because `f` estimates expected guesses, the policy is risk-neutral away from the
endgame, where greedy entropy is implicitly cautious; `--loss-penalty 5`
compensates and recovers most of the gap for free (see
[Losing less](#losing-less)), and `--loss-penalty 1` restores risk neutrality.
