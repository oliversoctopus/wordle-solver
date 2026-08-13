# How to read this document

This file is Markdown with twelve LaTeX display equations (`$$ ... $$`), so it
wants a viewer with math support — a GitHub gist, StackEdit, HackMD, or
`Ctrl+Shift+V` in VS Code. **`ABOUT_SOLVER_EXPLAINED.md` has the steps for each**,
along with what this document is for and what order to read it in.

> If your viewer shows raw `$$` symbols, it simply lacks math support. Nothing is
> lost: every formula here is also stated in words in the surrounding prose, and
> the code references are plain text.

Read it start to finish the first time — the order is the argument. In ten
minutes instead: §2, §6, §12.

---

# The Wordle solver, from the ground up

## 0. What the program actually has to do

Every turn, the solver holds a set of words that are *still possible answers*. It
picks a guess, sees five colored tiles, and that shrinks the set. The whole
project is one question:

> **Which guess shrinks the set the most usefully?**

Everything else — entropy, the fitted curve `f`, the endgame search — is
machinery for answering that.

Two word lists matter:

- **12,947 words** the solver is allowed to guess, and which it treats as
  *possibly* being the answer (`ind2word.json`)
- **2,309 words** on the original curated 2022 answer list (`wordles.json`) — the
  words the NYT editor mostly draws from

We need three things before that question can even be asked precisely: a way to
compute feedback (§1), a way to describe what a guess *does* to the candidate set
(§2), and a way to measure how good that is (§3).

## 1. The feedback rule, and why there's a 167 MB file

`wordle_data.score_guess` implements the coloring rule, including the
fiddly duplicate-letter case. The rule is two passes: **greens are assigned
first**, and then yellows consume whatever letters are left over, at most one
tile per copy of the letter in the answer.

Two examples against the answer `abide`, both verified by running `score_guess`:

```
guess  speed   ->  N N Y N Y      guess  geese   ->  N N N N G
       s p e e d                         g e e s e
```

- `speed` has two E's and `abide` has one, so the **first** E takes the yellow
  and the second gets nothing. The D is also yellow — it's in the answer, just
  not in position 4. So the pattern is `NNYNY`.
- `geese` has three E's. The one in position 5 matches `abide`'s E exactly, so it
  goes green — and because greens are assigned first, it consumes the answer's
  only E and the two *earlier* E's get nothing despite coming first in reading
  order. Hence `NNNNG`.

Each pattern is 5 tiles × 3 colors, so encode it in base 3 with N=0, Y=1, G=2.
That gives an integer in `[0, 243)`; `NNYNY` = 10, `NNNNG` = 2, and
`GGGGG` = 242 (`WIN_PATTERN`).

Now the key engineering move. The feedback for a (guess, answer) pair never
changes, so compute all of them once:

```
feedback_table[guess_index, answer_index] = pattern number   # one uint8
```

That's 12,947² ≈ 167 million bytes — the `feedback_table.npy` file. Building it
takes a while; after that, "what would I see?" is a single array lookup instead
of a string comparison. This is the same instinct as a prefix-sum table: pay
O(n²) once so every query is O(1). Every hot loop in the project is a scan over
one row of this table.

## 2. A guess is a partition

Fix the candidate set `S` and a guess `g`. Every candidate in `S` produces *some*
feedback pattern against `g`. Group them by that pattern:

```
guess = "crane", S = {every 5-letter word}
  pattern NNNNN  -> 4,283 candidates
  pattern NNNNG  ->   612 candidates
  pattern GNNNN  ->   371 candidates
  ...            (about 150 non-empty buckets)
  pattern GGGGG  ->     1 candidate  (crane itself)
```

**You are going to land in exactly one bucket, and that bucket becomes your new
candidate set.** So a guess isn't really "a word" — it's a *partition of `S` into
at most 243 buckets*, and the only thing that matters about a guess is the shape
of that partition.

Two words with the same partition are the same move. That sounds like a
throwaway observation; it is what makes the endgame search affordable in §9.

Bad guesses leave one huge bucket. Good guesses chop `S` into many small ones.
Now we need to make "many small ones" precise.

## 3. Entropy, from scratch

This is the one piece of genuinely new math, so I'll build it rather than cite
it.

**Question:** I'm thinking of one of `n` equally likely words. You ask yes/no
questions. How many do you need?

Binary search: each question halves the set, so you need **log₂(n)** questions.
Call that "log₂(n) bits of uncertainty." 1024 words = 10 bits.

**Now the unequal case.** Suppose one word has probability 1/2, another 1/4, and
four more have 1/16 each. Uniform reasoning would say log₂(6) ≈ 2.58 bits, but
that's wrong — you'd ask "is it the first one?" and half the time you're *done in
one question*. Weight each outcome by how likely it is:

- an outcome of probability `p` "feels like" a set of size `1/p`, i.e. log₂(1/p) = −log₂(p) bits
- average over outcomes:

$$H = -\sum_i p_i \log_2 p_i$$

For that example: ½(1) + ¼(2) + 4·[1/16 · 4] = 0.5 + 0.5 + 1 = **2 bits**, less
than 2.58. Concentrated probability = less uncertainty. That's the whole idea.
`H` is maximized at log₂(n) when everything is equally likely, and is 0 when you
already know the answer.

**Information gain.** Before guessing, uncertainty is `H(S)`. After, you're in
bucket `r` with probability `p_r`, and still uncertain by `H(S_r)`. So

$$\mathrm{IG}(g) = H(S) - \sum_r p_r H(S_r)$$

Here's the pretty part, and it's why the code is fast. Knowing which word it is =
knowing *which bucket* + knowing *which word within that bucket*. Uncertainty
adds the same way:

$$H(S) = \underbrace{H(p_1, \ldots, p_k)}_{\text{which bucket}} + \sum_r p_r H(S_r)$$

Substituting, **the information gained is exactly the entropy of the bucket-mass
distribution.** You never need to look inside the buckets. That's
`kernels._ig` — accumulate mass per pattern, then `h -= p*log2(p)`.

That gives **`--mode entropy`**: pick the guess whose partition has the highest
bucket entropy. It's the standard 3Blue1Brown-style solver, and it averages
**3.68 guesses** here.

That is the baseline, and the rest of this document is about beating it. There
are two ways to try. One is to search harder — look further ahead, solve the
endgame exactly. The other is to notice that the question in §0 was never quite
right: it assumed every word is equally likely to be the answer, and no word ever
was. The second turns out to be worth far more than the first, which is §12's
punchline and the reason §4 is the longest section here.

## 4. Not all words are equally likely (the priors)

`--mode entropy` implicitly assumes any of the 12,947 words is equally likely.
That's badly wrong, and fixing it turns out to be worth more than *any* search
improvement in the project.

`priors.build_weights` models the editor as a two-step process:

1. With probability **0.10**, reuse a word that has already been an answer
   (among the ~1,858 used ones).
2. Otherwise pick an unused word, which is **off** the curated list with
   probability **0.10**.

So the weight vector has three groups: used words, unused-and-curated, and
unused-and-non-curated. Every word gets strictly positive weight, so nothing is
ever ruled out entirely.

Note the conditioning: the off-list probability is *conditional on not being a
repeat*, because the model is sequential. Computing it against all puzzles rather
than non-repeat puzzles understates it.

Those two 0.10s look aggressive against Wordle's full history, and the reason
they aren't is a regime change: measured over the **current editorial regime
(Wordle #1690 onward, from 2026-02-02)**, the repeat rate is 11.1% and the
off-list rate is 8.9%. Over all 1,879 puzzles they'd be 1.1% and 2.5% — roughly
10× and 4× lower. What drives it is mechanical: 1,812 of 2,309 curated words are
used up and ~334 are consumed a year, so the pool empties in about 1.5 years.
Repeats and off-list picks are both the editor's response to that.

**The prior is where the wins are.** Every word as the answer, weighted by that
row's own prior, each row playing the form of `f` fitted for it:

| priors on | entropy | expected-guess mode |
| --- | --- | --- |
| none | 4.0794 | 4.0718 |
| repeat only | 4.0639 | 4.0111 |
| curation only | 3.7802 | **3.4903** |
| both | 3.6815 | **3.0770** |

Read *down* a column, not across a row: the weights come from each row's own
prior, so a row is exact under the model it assumes, but a concentrated prior
makes the game genuinely easier and the rows are not comparable to each other. What the shapes
below are worth needs a different measurement entirely, which is §4.6.

The column to look at is the gap between the two. With both priors on, the
expected-guess objective beats entropy by 0.60 guesses. With no prior at all it
beats it by 0.0076 — eighty times less, for exactly the same machinery. §6 shows
why that collapse is not a coincidence but an identity.

Splitting the words into three groups says *how much* mass each group gets. It
says nothing about how that mass is spread within a group, and that is the rest
of §4.

### 4.1 The shape problem

The obvious answer — spread it flat — is wrong twice over.

Flat inside the used group says `cigar` (Wordle #1) and `sandy` (#1227) are
equally likely to come round again. Flat inside the non-curated group says
`squid` and `xylyl` are equally likely to be tomorrow's answer. Neither is close
to true, and the second is the expensive one: the non-curated group is 10,638 of
the 12,947 words, so it dominates the candidate *count* at almost every position
even while holding only 9% of the mass.

Call the within-group spread the group's **shape**. Both shapes were uniform in
the original design; both are now sharpened, and the next two sections are how.

### 4.2 Sharpening the repeat group

There are 21 repeats in Wordle's history, and they are not drawn from the used
pool evenly. Nineteen come from Wordle #1–#297 (median #116); only `sandy`
(#1227) and `smile` (#831) sit outside that window. The editor appears to be
replaying the *beginning* of the run.

An 11% outlier rate is too high to just zero out everything past #297, so the
shape is a **Laplace-smoothed histogram**. Bin the used words by the puzzle
number where they first appeared, 200 wide, and give bin *b* the weight

$$w_b = \frac{r_b + \alpha}{n_b}, \qquad \alpha = 0.5$$

where `r_b` is how many repeats have come out of that bin and `n_b` is how many
used words live in it; then spread each bin's share evenly over its members. The
`α` is what keeps a bin that has never produced a repeat from being written off
entirely — it is the "maybe next time" term, and it is why no word's weight ever
reaches zero.

Fixed bins rather than a sliding window is deliberate, and the reason is a small
lesson in overfitting. A window needs a boundary; a boundary is a parameter; and
that parameter has to be tuned on the same 21 events you are trying to predict.
Scored prequentially — each repeat predicted from the ones before it only, so no
parameter ever sees the event it is scored on — flat gives a log-likelihood of
**−157.1** over the 21 events and the binned histogram gives **−133.1**, about
**+1.14 nats per event**. The best of sixteen tuned windows reaches only −143.6,
and an age-based shape whose rate is refit as the events arrive lands at −150.1.
Twenty-odd events cannot pin a rate down, and the histogram has nothing to pin.

In practice `cigar` ends up with about **6×** the flat repeat weight, and words
that first appeared late fall below it.

### 4.3 Sharpening the off-list group: WordleBot's own weights

The off-list group is the interesting one, because "which words could be
off-list answers" turns out to be *knowable* rather than guessable.

The first clue is in `solutions.json` itself. Its `id` field is not an opaque
key — it is the **1-based index into `wordles.json`**. Every off-list answer has
an id past the curated list's 2,309, and 44 of the 46 land in **2310–2397**.
Sorted by id they come out in alphabetical runs: twelve consecutive at 2318–2331
(`beaut laser mommy muggy nervy pious primp shred spate taupe tinge tizzy`),
seven more at 2343–2352. A run of twelve in order by chance is 1/12! ≈ 2×10⁻⁹.
So the NYT appended words to the same internal list in alphabetically-sorted
batches, and off-list answers are drawn from a block of roughly ninety — not
from 10,638.

Knowing the block is small does not name its members. But somebody has already
published a list that does: **WordleBot**, the NYT's own solver, carries a
per-word weight, scraped and published at
`wordletools.azurewebsites.net/weightedbottles`. It is 3,209 words with weights
from 0.226% to 58% (2,110 of them pinned at the 58% cap). Of those, 3,194 are
words this solver can guess: 2,302 curated, **892 not**, and all 46 off-list
answers to date among them.

That last fact proves nothing on its own — a list that absorbed answers after
they were used would look identical. The test that matters is whether an *old*
copy predicts *later* answers. The Wayback Machine has a snapshot from
**2023-10-12** — 3,158 words, 841 of them non-curated, frozen almost three
years ago:

| snapshot 2023-10-12 | |
| --- | --- |
| off-list answers used after it | 41 |
| ...already in the snapshot | **39** |
| the misses | `loris`, `kefir` |
| words added to the list since | 84 |
| ...of those, ever an answer | **2** (the same two) |

So it predicts forward at about 95%, and its growth is not backfilling.

The shape follows directly: give **95%** of the off-list mass to the listed
words in proportion to their published weight, and spread the remaining **5%**
over everything else. The hedge is not decoration — the snapshot missed 2 of 41,
so 5% is the measured miss rate, and it is what stops a `loris` from being a
blindside.

Scored prequentially on the 42 post-snapshot events, in nats per event, each
shape at the constant that actually ships:

| shape | per event | effective support |
| --- | --- | --- |
| uniform | −9.27 | 10,638 |
| answer-like frequency (§4.4), `t` = 0.35 | −8.13 | 4,857 |
| **WordleBot, 5% hedge** | **−6.46** | **911** |

"Effective support" is `2^H` from §5 applied to the prior itself: how many
equally likely words the shape is really spreading over; `answer_like.py` prints
the middle row's on every run. Restricting to
post-snapshot events barely matters — over all 46 off-list events WordleBot also
scores −6.46, so it is not winning by having quietly absorbed the early ones.
Score each shape at its *fitted* constant instead of its shipped one and both
middle and bottom rows improve, which is why the table specifies: a shape has to
be scored as it will actually run.

**The 5% hedge stays flat, and that is deliberate.** Weighting it by
answer-likeness (§4.4) is the obvious refinement and it fails on the population
that matters. Scored over all 46 off-list answers as though the hedge had had to
cover them it looks strong, +1.32 nats/event at a median 3.95× the flat share —
but those are words WordleBot *does* list. The hedge only fires on a word the
list is missing, and a word is missing because it is odd: `loris` sits at the
53rd percentile of answer-likeness and would get **0.32×** the flat share,
`kefir` at the 87th and would get 2.36×. Over the only two events that have ever
needed it, shaping the hedge is worth −0.13 nats. Flat is the honest expression
of not knowing which word the list is short of; `analysis/curation_shapes.py`
prints the comparison.

### 4.4 The fallback: guessing what is answer-like

The WordleBot shape depends on a third-party scrape of somebody else's site.
That is a fine thing to vendor and a bad thing to *need*, so there is a fallback
that depends on nobody's scrape — built from the word lists already in the repo
and a published frequency corpus (`wordfreq`).

The trick is that the curated list is 2,309 labeled examples of "a word an
editor considered answerable", and it is completely independent of the 46
off-list events. So fit a logistic regression with the curated words as
positives and the non-curated words as negatives, and use its log-odds to score
the words the editor has not used yet. The features are word frequency (Zipf,
from `wordfreq`), a cross-lingual "is this actually a name" lift, suffix flags
and duplicate letters. Four of the thirteen weights carry the model, and those
four are legible:

| feature | weight | reading |
| --- | --- | --- |
| Zipf frequency | **+4.00** | common words get picked |
| ends in `s` | **−5.87** | 37% of the pool is a plural; 9% of off-list answers are |
| ends in `ed` | −5.09 | past tenses do not get picked either |
| foreign lift | −1.16 | as common abroad as at home ⇒ a name, not a word |

The other nine are a Zipf square, a zero-frequency flag, the maximum foreign
Zipf, the suffixes `es`/`er`/`y`/`ly` and a duplicate-letter count, and three of
them sit near zero. They are not decoration: `answer_like.py` is the definition
of this score, and a fit that keeps only the four legible features still
correlates at r ≈ 0.98 with the shipped table while reproducing none of it.

The score is a log-odds running −19.8 to +4.6, and the weight is
`exp(t × score)`. The temperature `t` matters because the model answers a
slightly different question than the one asked: it predicts "would have been
*curated*", not "will be tomorrow's *off-list* answer". Taking it at face value
(`t = 1`) collapses the effective support to 554 words, over-committing to a
question nobody asked. Fitting `t` against the actual off-list events gives
≈0.5; end to end, `t` of 0.20, 0.35 and 0.50 land within **0.004 guesses** of
each other. It is a flat optimum, and **0.35** is simply the middle of it.

At `t = 0.35`, relative to the flat weight it replaces:

| promoted | | demoted | |
| --- | --- | --- | --- |
| `squid` | 8.5× | `karks` | 0.014× |
| `maven` | 4.4× | `fents` | 0.014× |
| `aloha` | 4.0× | `xylyl` | 0.10× |
| `kazoo` | 2.4× | `faffy` | 0.096× |

Every word on the left is a real off-list answer; every word on the right is one
of the words behind a recorded loss in §10.

### 4.5 Do the two shapes need each other?

These two shapes arrived together, and the argument for shipping them as a pair
was that they looked *superadditive*: the WordleBot shape measured a wash on its
own, the binned repeat shape close to nothing on its own, and the pair far more
than the sum of the two. That is a strong reason to ship two things at once.

On current data it is gone. The three configurations blend to −0.044, −0.022 and
−0.064 (§4.6), against a sum of −0.066: **additive, within noise.** So the
question in the heading has to be asked again from scratch.

What survives is a mechanism, and it is worth following because the arithmetic
behind it is checkable even where the guess counts are not. Concentrating the
off-list mass onto its ~870 effective words gives each about **1.0×10⁻⁴**, where
a used word under flat repeat mass has 0.10/1,858 = **5.4×10⁻⁵**. Sharpening the
curation side alone therefore promotes any off-list word above any past answer,
nearly two to one. On average that is not far wrong — the observed rates justify
≈9.1×10⁻⁵ against ≈6.0×10⁻⁵, a ratio of 1.5 — but it is wrong *word by word*,
because flat repeat mass spreads evenly over 1,858 words when 19 of the 21
repeats came from the first 300. The solver stops attending to `cigar` and
attends to ~870 off-list candidates instead, and when the answer *is* `cigar`, it
pays.

That predicts two things, and the data treats them very differently:

* **Sharpening curation alone should cost something on repeats.** It does:
  `wordlebot` alone is **+0.143 ± 0.078** on the repeat arm, and the two rows
  containing `wordlebot` are the only ones that lose ground there at all. About
  1.8σ — suggestive, not settled.
* **Adding the binned repeat shape should relieve that cost.** Here the data has
  nothing to say. The pair scores **+0.095 ± 0.228** on the same arm: better than
  +0.143, by a fifth of its own error bar.

That second interval is the whole problem, and no amount of cleverness fixes it.
Twenty-one repeat events carry ±0.17–0.23 however they are analyzed. Reaching
±0.05, enough to call a 0.1-guess repair, needs about sixteen times the events —
some 340 repeats, which at the regime's 11.1% rate is roughly 3,000 puzzles.
Eight years of Wordle.

So the honest answer is: **a reason to expect interference, not evidence of it.**
Sharpening one side does make the other's flatness the larger error in the
weights; that part is arithmetic. Whether it costs measurable guesses, and
whether the second shape buys them back, is a question twenty-one events cannot
answer.

None of which argues against shipping both, because the pair never needed the
interaction to justify it. Each shape independently beats flat at its own job, by
a margin the prequential scoring *can* resolve: +1.14 ± 0.21 nats per event for
`binned` (§4.2), −6.46 against −9.27 for `wordlebot` (§4.3). Two separately
justified improvements, adding up. That is a smaller claim than the one they were
adopted on, and it is the one the data supports.

### 4.6 Measuring any of this without fooling yourself

Two traps, both of which hid the effect at first.

**The form of `f` has to be refitted per shape.** `f` (§7) is queried at its
largest arguments exactly once per game — at the opening, where the bucket is
the whole word list, which is the extrapolated tail of the fit. Sharpening the
prior shifts every `n_eff` into that tail, and the ranking of opening words
flips: under `sqrt`, the form chosen for the *uniform* shapes, a sharpened prior
picks `raise` over `salet`. `raise` is genuinely worse — forcing it under the
unchanged uniform prior costs **+0.034 ± 0.028** — so this is estimator error,
not a real preference. The three best-fitting forms under the sharpened prior all
pick `salet` (`invonly`, `saturating` and `spline`, at fit RMSE 0.486–0.490
against `sqrt`'s 0.505), and so does one-ply lookahead. This is why the shape is
part of the cache key.

Fit quality is not the criterion, though, and that is worth seeing plainly:
`anchored2` has the worst RMSE of any form at 0.70 and picks `salet` anyway,
while `anchored` fits a hair *better* than `sqrt` (0.5041 against 0.5047) and
still picks `raise`. What decides the opening word is concavity in `x` (§6), not
how closely the curve tracks the pairs it was fitted to.

**The evaluation needs three arms, one of them repeats.** A shape can only be
judged on answers that actually happened, never on answers sampled from its own
belief — a prior scored against its own prior is grading its own homework, and
that swing is larger than the effect being measured. The three arms are the **21
real repeats** and the **46 off-list answers**, each replayed with the prior
rebuilt from strictly earlier puzzles, plus the **497 curated words still
unused**, blended at the current regime's own mixture of 11.1% / 7.9% / 81.1%:

| shapes | repeat (21) | off-list (46) | fresh (497) | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.143 ± 0.078 | −0.978 ± 0.105 | +0.022 ± 0.028 | −0.044 ± 0.025 |
| repeat `binned` alone | −0.143 ± 0.173 | −0.022 ± 0.049 | −0.006 ± 0.016 | −0.022 ± 0.024 |
| `binned` + `frequency` | −0.286 ± 0.184 | −0.522 ± 0.145 | −0.002 ± 0.018 | **−0.074 ± 0.028** |
| `binned` + `wordlebot` | +0.095 ± 0.228 | −0.826 ± 0.150 | −0.012 ± 0.018 | **−0.064 ± 0.032** |

Unlike the table in §4, **these columns are comparable across rows**: every row
is scored against the same real answers, so the blend is a genuine paired
comparison rather than each prior being asked how it fares in a world of its own
devising.

The table's provenance limits what it can be set beside. It comes from a solver
whose estimator has since been refitted, and every configuration's winning form
moved in that refit, so the policy changed underneath the measurement as well as
the data: these numbers are comparable to each other and to nothing else. The top
three rows can no longer be recomputed from what the repo carries either, since
they need fits for the uniform shapes and a prior with no fit refuses to play
(§7). What `analysis/replay.py` measures now is the one contrast
both surviving fits support, `frequency` against `wordlebot` head to head:
**−0.010 ± 0.025** blended, the same distance the bottom two rows sit apart,
reached without the baseline in between.

The repeat column is also where §4.5 lives, and it has softened considerably
since it was first measured: `wordlebot` alone carried a **+0.611** penalty
there, against +0.143 now, and blended to a wash at +0.009. That penalty is what
made the pair look worth so much more than its parts.

Which leaves the honest caveat, the width of that arm. Twenty-one events carry
±0.17–0.23, making it the loosest term in every blend and leaving
`binned + frequency` and `binned + wordlebot` well inside one sigma of each
other. The choice between them rests more on whether you want a vendored scrape
in the repo than on the measurement.

## 5. Effective size: making a weighted set have a "count"

Unequal weights break something the last section quietly relied on. "This bucket
has 40 candidates" is misleading — 40 candidates with one of them at 90%
probability is much easier than 40 equally likely ones. Everything from here on
needs a size that knows the difference.

The fix (`estimator.effective_size`):

$$n_{\text{eff}} = 2^{H}$$

where `H` is the entropy of that bucket's *normalized* weights. If the weights
are flat, `H = log₂(count)` and `n_eff = count` exactly. As probability
concentrates, `n_eff` slides toward 1. It's "how many equally likely words would
be this hard."

### The rearrangement that makes this cheap

There's an implementation trick here worth working through, because it's the
single biggest optimization in the scoring loop.

**The problem.** `H` is defined on the *normalized* weights `p_i = w_i / m`. But
`m` — the bucket's total mass — isn't known until you've seen every member of the
bucket. And which bucket a candidate lands in depends on the guess. So the naive
reading is: for each of the 12,947 guesses, pass over the candidates to find the
bucket totals, then pass over them *again* taking a logarithm per candidate to
build each bucket's entropy. That's one `log` per (guess, candidate) pair.

**The fix.** Expand the definition and split the logarithm. Write `m = Σ w_i` over
the bucket's members:

$$H = -\sum_i p_i \log_2 p_i = -\sum_i \frac{w_i}{m} \log_2\!\left(\frac{w_i}{m}\right)$$

Now use `log₂(w/m) = log₂ w − log₂ m` to break the coupled term apart:

$$H = -\frac{1}{m}\sum_i w_i \big(\log_2 w_i - \log_2 m\big)
    = -\frac{1}{m}\sum_i w_i \log_2 w_i \;+\; \frac{\log_2 m}{m}\sum_i w_i$$

The second sum is just `m` again, so `(log₂ m / m) · m = log₂ m`, leaving

$$H = \log_2 m - \frac{\sum_i w_i \log_2 w_i}{m}$$

**Why that's a win.** Look at what each piece depends on. `m` and the sum both
range over "the members of this bucket," which does depend on the guess. But the
*summand* `w_i log₂ w_i` depends only on candidate `i` — on its raw prior weight,
with no reference to the bucket, the guess, or the normalization. Normalizing is
what coupled a candidate's contribution to a global property of its bucket, and
factoring the logarithm decouples it again.

So `w_i log₂ w_i` is computed **once per candidate**, up front, for all guesses
(`kernels._w_log_w`). The inner accumulate loop then just adds a
precomputed float into a bucket:

```
w_log_w[r] += wlw[i]        # instead of a log() call here
```

and the reduce recovers the entropy from two accumulators it already has:

```python
x = np.log2(mass) - wlw / mass      # in _bucket_guesses — this x *is* H
```

Cost, for a position with `m` candidates scored against all 12,947 guesses: the
naive version needs `12,947 × m` logarithms, the split version needs `m`. At a
wide position (`m` ≈ 12,947) that's 167 million logs against thirteen thousand —
which is why the module docstring says this alone was most of the cost of scoring
a wide position.

**One more consequence.** Since `n_eff = 2^H`, that `x` is exactly `log₂(n_eff)` —
already in the units `f` wants. That's why `f_eval_x` is parameterized by `x`
rather than by `n` (`estimator.f_eval_x`): every caller reaches `f` through an
entropy, so it has `x` in hand, and going via `n` would mean computing `2^x` only
for `f` to take `log₂` straight back off.

Sanity check on random weights, direct definition against the rearranged one:

```
direct : 2.331157535638251
split  : 2.3311575356382495     # agree to 1.3e-15
```

With a size that respects the prior, the objective can finally be stated in the
units the game actually scores.

## 6. The real objective: minimize expected guesses

Entropy answers "which guess is most *informative*." But the game doesn't score
you on bits — it scores you on **guesses**. Those aren't the same thing, and the
gap is where the half-guess improvement lives.

Suppose you had a function

> **`f(n)`** = expected number of further guesses needed from a position of
> effective size `n`

Then the value of guess `g` is immediate:

$$E(g, S) = 1 + \sum_{r \ne \texttt{GGGGG}} p_r \, f(n_r)$$

"One guess for this turn, plus the average cost of wherever I land." Two details
carry real weight:

- **Excluding `GGGGG` is the whole reason a possible answer beats an equally
  informative non-answer.** That bucket costs nothing more — you already won. So
  a guess that might just *be* the answer gets a free discount equal to its
  probability.
- **Buckets of 1 and 2 skip `f` entirely** (`kernels._bucket_guesses`), because their
  values are known in closed form: 1 candidate is worth exactly `1`, and 2
  candidates are worth `2 − p_max` (guess the likelier one; if wrong, you now
  know the other).

Where `f` comes from is §7's problem. What it has to *look like* is settled here,
and the answer is sharper than it sounds.

### Why this isn't just entropy in disguise

Suppose `f` is **linear**: `f(n) = a + b·x` with `x = log₂(n)`. Substituting, and
using the chain rule from §3:

$$E(g,S) = 1 + a(1 - p_{\text{win}}) + b\big(H(S) - \mathrm{IG}(g)\big)$$

`H(S)` doesn't depend on `g`, so minimizing `E` is *identical* to maximizing
`b·IG(g) + a·p_win` — greedy entropy, plus the possible-answer bonus. **A linear
`f` buys you nothing but the bonus.** All the remaining gain comes from `f` being
*concave* in `x`.

That identity is testable, because `loglinear` is a form of `f` that *is* linear
in `x`, so by the argument above it should score exactly "entropy plus the
bonus" — and it does. Under the shipped shapes greedy entropy scores 3.6815,
`loglinear` 3.1179, and the best form 3.0770. So the gap splits:

| | worth |
| --- | --- |
| the possible-answer bonus | **0.564** |
| curvature in `f` | **0.041** |
| total, entropy → best | 0.605 |

Nearly all of the improvement over a standard entropy solver is the free discount
for guessing a word that might be the answer. The fitted curve's shape — the
thing §7 works hardest for — contributes about a fifteenth of it. Under the flat
shapes the same split read 0.462 and 0.041: the bonus grows as the prior sharpens
and curvature is worth what it always was, which is the first hint of §12.

Wordle's own **hard mode** is the cleanest test of the identity, because it
isolates the bonus from everything else. Hard mode requires every guess to be a
word that could still be the answer, so restricting greedy entropy to it forces
`p_win` up without touching entropy's scoring rule at all. It gains
**0.1813 guesses** (3.6815 → 3.5002, `analysis/hard_mode.py`) — a third of what
the objective gets by *pricing* the
same bonus rather than imposing it. The objective does better because it takes
the bonus **and** keeps the probes: it plays a non-candidate whenever the split
is worth more than the win chance given up, which is exactly the trade hard mode
forbids.

So `f` needs to be concave in `x`, and the fitted curve for the default prior is
a piecewise-linear spline that obliges:

```
f(n) = 0.955 + 0.372·x − 0.112·(x−2)₊ − 0.138·(x−4)₊ − 0.104·(x−6)₊
                                            (x = log₂ n, (u)₊ = max(u, 0))
```

Every knot bends it flatter and none bends it back — the sixth, at `x = 8`, was
dropped altogether by the monotonicity guard §7 describes. Concretely:
`f(2)=1.33`, `f(3)=1.54`, `f(10)=2.04`, `f(100)=2.47`, `f(1000)=2.53`,
`f(12947)=2.60`. Going from 100 candidates to 1,000 costs **0.06 guesses**, and
from 1,000 to the whole word list another 0.07. That is the honest shape of the
game, and precisely what a linear `f` cannot express.

It is also much flatter than the curve fitted under the flat shapes
(`0.61 + 0.77·√x`, running 2.58 → 3.02 → 3.42 over those last three points),
which is the prior changing what `f` even has to model: with the mass
concentrated, the effective size rarely reaches the range where a saturating
curve has gone flat, so saturation stopped being the liability it used to be.

One caveat that matters when reading any of this: `f` is fitted **per prior**, so
the curve above is the default prior's and no other. Turn a prior off and a
different curve, in a different functional form, is what scores the guesses.

## 7. Where does `f` come from? Policy iteration

Circular problem: to play well you need `f`, but `f` is defined as the cost
*under good play*. This is the same chicken-and-egg as value functions in
reinforcement learning, and it's solved the same way — bootstrap and iterate
(`train.policy_iteration`).

```
round 0:  play 2,000 games with greedy entropy (needs no f)
          -> for every position visited, record
             (n_eff, guesses actually needed from there)
          -> least-squares fit f to those pairs

round k:  play 2,000 games acting greedily on the current f
          -> record the same pairs
          -> refit
```

This is **policy iteration with function approximation**: *evaluate* the current
policy by rollout, *improve* by acting greedily on the fitted value function.
Because every form is linear in its parameters (`a + b·x + c·√x` — the *terms*
are nonlinear, the *parameters* aren't), each refit is a single closed-form
least-squares solve, not an optimization.

You can watch it converge in
`estimator_cache/params_repeat1_curation1_r-binned_c-wordlebot.json`:

| round | mean guesses |
| --- | --- |
| entropy seed | 3.654 |
| iter 1 | 3.150 |
| iter 2–4 | 3.100 |

Essentially all the gain arrives in one iteration; the rest is a small limit
cycle, so the loop caches the parameters whose *measured play* was best rather
than the last fit (a greedy policy can oscillate).

Three constraints worth flagging:

- **Training games are played with no 6-turn deadline.** Otherwise a game that
  runs long gets truncated at 6, biasing the targets downward. The deadline is a
  constraint layered *on top* of `f`, not part of it — which is §8.
- **`f` must be non-decreasing** on `[1, 12947]` (`estimator._is_monotone`). If a
  bigger candidate set could score *cheaper*, the search would prefer
  uninformative guesses that dump everything into one bucket. When the full basis
  violates that, terms are dropped from the end until it holds — which is why
  `quadratic` scores identically to `loglinear` in the results table: its `x²`
  term gets dropped every time.
- **A prior with no cached fit does not play.** Rather than bending some
  reference curve onto the chosen form and playing a policy nobody measured, the
  estimator raises and names the `train` command that fixes it. Entropy mode
  never consults `f` and runs on anything.

## 8. The deadline, and what losing costs

Everything so far has quietly ignored the fact that real Wordle gives you only
**six turns**. `f` estimates guesses-to-finish with no deadline at all, which §7
explained is deliberate — but the next section's formulas need a cost for running
out, so define it here before using it.

The obvious approach would be a second objective: minimize guesses, subject to
some separate constraint about not losing. That's awkward, because you'd then need
a weight trading the two off. Instead the deadline is folded into the *same units*
as everything else. Define one number:

> **`loss_penalty`** — what it costs to run out of turns, charged on top of the
> six guesses already spent.

So a lost game scores `6 + loss_penalty`. Setting `loss_penalty = 1` makes a loss
score exactly **7**, i.e. "as if you'd needed one more guess than you had." That
is the **risk-neutral** setting: minimizing expected score then means exactly what
it says, and losing carries no special horror beyond the turn it cost you.

In the value recursion this appears as the base case: with zero turns left, the
position is worth `loss_penalty` (`exact.py`). No separate deadline objective, no
risk weighting — one recursion covers both "win fast" and "don't lose."

Raising the penalty above 1 buys solve rate at the expense of the mean, by making
the search willing to spend a turn on safety. **The default is 5**, and §10
explains why that number and not 1 — but the value is just a knob, and every
formula below holds for any setting of it. Note that it is a *decision-time*
price only: measured results always score a lost game as 7, whatever the penalty
was set to, or rows run at different penalties could not be compared.

## 9. The endgame: stop estimating, start solving

`f` is a fitted approximation. But when few candidates remain, you don't need
approximation — you can compute the true answer by brute force (`exact.py`).

$$V(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r \, V(S_r, t-1)\Big], \qquad V(S, 0) = \text{penalty}$$

`t` is turns remaining. This is **expectimax**: you *minimize* over your guess
(you choose it) but take an *expectation* over the feedback (you don't). Not
minimax — there's no adversary, just chance.

Naively this is hopeless: 12,947 guesses × 243 patterns × recursion. Four things
make it affordable.

**(a) Partition dedupe.** This is where §2 pays off: a guess *is* its partition,
so two guesses that split `S` identically are perfectly interchangeable, and the
branching factor is the number of *distinct partitions* rather than 12,947. With
≤7 candidates the whole feedback vector packs into one int64 (8 bits per
candidate) and equal keys are provably equal partitions; above that it's an FNV
hash with an explicit collision check, so a collision costs a redundant
representative but never a missing one (`kernels.distinct_partitions`). Nice safety property:
a guess that could itself be the answer *always* splits distinctly, since only it
produces `GGGGG` — so a real candidate can never get merged away into a probe.
(Which is also why hard mode skips this step entirely: if every guess must be a
candidate, no two of them can split `S` the same way, and there is nothing to
merge.)

This only bites while `S` is tiny, though: at 12 candidates, 12.7k of the 13k
guesses already split differently, so beyond that the branching factor is
effectively the whole word list. That, not the node count, is what makes the
occasional deep position slow.

**(b) Closed-form leaves.** With two turns left, no search is needed at all. On
the last turn the best you can do in a bucket is play its heaviest member, worth
`1 + (1−p_max)·penalty`, so

$$V(g,2) = 1 + \frac{A + (A - B)\cdot\text{penalty}}{\text{total}}$$

with `A` the mass and `B` the sum of bucket maxima over non-winning buckets.
That's *one pass over the guesses* — exactly what heuristic scoring costs. So
**two turns or fewer is always solved exactly, for free, at any candidate
count.**

**(c) Branch and bound.** That two-turn value is also a valid **lower bound** on
`V(S,t)` for larger `t`. (Why: under this convention `V` is non-*decreasing* in
`t` — with more turns left you're obliged to actually finish rather than pay the
comparatively cheap penalty.) So order guesses by that bound, and inside a guess
replace bucket bounds with exact values one at a time, abandoning the moment the
running total exceeds the incumbent — biggest-mass buckets first, since they
tighten fastest. Standard search pruning, on a well-chosen admissible bound.

**(d) Branch cap** (default 50). Only the 50 best splits by that bound are tried
at all. The justification is sharp: branch and bound finds the best guess early
and then spends nearly all its time *proving* nothing beats it. The cap buys the
answer and skips the proof. Over 1,275 real positions, caps of 25 and 50 pick the
uncapped-optimal guess **every single time**, while wall clock drops 322 s → 8 s.
The honest caveat, which the code states: the returned value is an upper bound on
the true one, so this is optimal-within-the-cap, not provably optimal.

## 10. Why the penalty is 5 and not 1

Risk neutrality (`penalty = 1`) is the principled default, so it takes evidence to
move off it. Start with what the losses are. They aren't endgame blunders: at the
sizes these positions reach, `exact.ExactEndgame` solves them outright, so the
guess played is optimal. The solver is backing its prior. Holding
`batty`/`catty`/`natty`/`patty`/`ratty`/`tatty` with four turns left it plays
`tatty`, which carries 75% of the family's mass, and it turns down `bronc` — a
word that splits the six of them five ways — because separating spends a turn to
insure against a 0.4% case. A loss is the prior being confidently wrong, and the
penalty is what makes that insurance worth buying:

| penalty | loss probability | mean (loss = 7) |
| --- | --- | --- |
| 1 | 0.1168% | 3.6621 |
| **5** | **0.0458%** | **3.6626** |
| 30 | 0.0357% | 3.6635 |

Three fifths of the loss probability vanishes for **+0.0005 guesses**. Widening
`--exact-max` from 24 to 60 adds the rest, and the two together — risk-neutral
play at the narrow reach against the shipped pair — run **0.1198% → 0.0458%, a
62% reduction, with the mean improving by 0.0019** on the way, because reaching
deeper replaces an estimate with a true value instead of trading one good for
another.

That last column can't pick the penalty, though, and the reason is instructive.
Scoring a loss at 7 is precisely the objective `penalty = 1` optimizes, so ranking
settings by it can only ever favor risk neutrality — the comparison is circular.
Name the price of a loss instead, `L` guesses, and every setting becomes a line in
`L`. Penalty 1 wins below `L` = 7.1, 2 up to 9.0, **5 from 9.0 to 13.7**, 10 to
22.1, and 30 above that. So 5 is right if a lost game costs between nine and
fourteen guesses, and little rides on it: across that whole range the best setting
beats the next by under 0.0002 guesses.

Losses collect on the never-curated words — 290 of the 300 the shipped prior can
lose on. Wordle is provably solvable in five guesses, but that theorem is about a
solver that knows the answer is one of the 2,309 curated words, and this one
spreads 9% of its mass over the other 10,638. Swept uniformly over the curated
list it loses 10 of 2,309, every one of them a word that has already been an
answer — which is §4.2's repeat prior calling them unlikely, by construction.

Every piece of the guess rule is now on the table; §11 puts them in the order the
solver applies them.

## 11. Putting it together — one turn of play

```
candidates = all 12,947 words
each turn:
    if only 1-2 candidates          -> closed form
    elif exact search can reach it  -> exact.solve()          (true optimum)
         (turns <= 2 always; or <= 60 candidates deeper)
    elif last turn                  -> heaviest candidate
    else                            -> argmin over 12,947 of E(g,S)
    show guess, read feedback, filter candidates
```

Every line that says 12,947 is the free choice of guess, and hard mode is the one
variation on this dispatch: it replaces that pool with the candidate set itself
and changes nothing else. `f` stays the one fitted for free play, so its scores
read slightly optimistic under the restriction — it is pricing a probe the solver
is no longer allowed to make.

The heuristic `E(g,S)` is genuinely a **fallback**: it only runs where the game is
too wide to solve outright. With both priors on and the shapes that ship, over
every word the answer could be, the whole thing averages **3.0770 guesses** and
opens with `salet`.

## 12. The result worth framing

This is the thread running through the whole project, and it's the most
interesting thing in it:

> **Better decisions, worth almost nothing.**

One-ply lookahead values positions **ten times more accurately** than
`min_g E(g,S)` — RMSE 0.0111 vs 0.1122, measured against ground truth computed by
playing positions out exactly. That accuracy converts: it disagrees about the
guess in **half** of all games, and it is right to, giving up 0.0013 guesses
where the estimator gives up 0.0115. It moves the outcome in a quarter of games,
those changes splitting **2,644 better vs 2,597 worse**, and nets −0.0034 ±
0.0041 guesses — nothing, at 0.9σ — for ~10× the wall clock.

The tempting explanation is that you're taking an `argmin`, so the estimator's
errors are **common-mode** across the guesses at one position and cancel in the
comparison, leaving the ranking intact. They don't. The estimator ranks the ten
guesses lookahead re-ranks at a correlation of **+0.54** with their true values
and picks the best of them just over half the time — a valuation whose errors
cancelled would not be overruled half the time by an accurate one.

What is true instead is that the top of the ranking is a **plateau**. Those ten
guesses spread 0.0941 guesses between best and worst, and the estimator's pick
gives up 0.0115 of that, an eighth of the range, while ordering the middle of it
almost at random. It finds the plateau reliably and can't tell one spot on it
from another, which costs a hundredth of a guess because there's little there to
tell apart. `analysis/lookahead_accuracy.py --decisions` is where those come from.

The same outcome holds four separate times here: deep exact search (+0.0007 ±
0.0029, i.e. nothing), one-ply lookahead, turn-conditioned `f` (+0.0005 ±
0.0007), and the sharpened repeat prior *on its own* (+0.0056 ± 0.0294, despite
being 4.4σ better as a prior). Only lookahead's has been taken apart at the level
of the decision; the other three are measured at the outcome. What the solver has
been short of is **somewhere better to go**, not a better map.

The exceptions are what sharpen the rule, and they have a common shape: every one
of them changes *what the solver is choosing between* rather than how well it
scores the choices.

The shapes of §4 are the first. Together they are worth −0.064 ± 0.032, which is
small but is more than deeper search and better lookahead managed between them. A
prior changes the quantity being optimized — which words are plausible answers at
all — so it is a decision-side change wearing valuation-side clothes.

Hard mode is the same thing in its bluntest form, and it makes the distinction
hard to argue with. It improves no valuation whatsoever; it deletes words from
the choice set. Under greedy entropy that is worth **−0.1813** guesses,
where one-ply lookahead buys nothing at all by valuing positions ten times
better. Take a choice away from a policy that was pricing it wrong and you
gain more than by pricing everything else right. Under the expected-guesses
objective, which already prices it correctly, the identical restriction *costs*
+0.0269 — same intervention, opposite sign, entirely because of what the
objective was already getting right.

Which is the honest reading of the headline number too. The 0.60-guess gain over
greedy entropy is 0.56 from the possible-answer bonus and 0.04 from curvature in
`f`. And the bonus is the *prior* showing up inside the objective — `p_win` is
nothing but the mass the prior puts on the word being guessed, which is why the
same split read 0.46 under the flat shapes and 0.56 under the sharpened ones. The
search contributes the 0.04.

Modeling the editor beat improving the search, every time.

---

## Where to look in the code

In reading order:

| where | what it establishes |
| --- | --- |
| `wordle_data.score_guess` | the feedback rule and the precomputed table (§1) |
| `priors.build_weights` | the weight vector over answers (§4) |
| `priors._repeat_shape` | the repeat shape, binned by first appearance (§4.2) |
| `priors._curation_shape` | the curation shapes, WordleBot and frequency (§4.3, §4.4) |
| `kernels._ig` | the entropy scorer (§3) |
| `estimator.effective_size` | effective size, and the split-logarithm rearrangement (§5) |
| `kernels._expected_guesses` | the expected-guesses scorer (§6) |
| `estimator.FORMS`, `estimator.fit` | `f`, its functional forms, and the monotonicity guard (§6, §7) |
| `train.policy_iteration` | policy iteration (§7) |
| `exact.ExactEndgame` | the endgame search, and the loss penalty (§8, §9) |
| `solver.WordleSolver._best_guess_uncached` | the dispatch deciding which of the above runs (§11) |

Named rather than given as line numbers, which drift with every edit. Every
entry is greppable: `grep -rn "def score_guess" .`
