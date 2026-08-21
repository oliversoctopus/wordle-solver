# How to read this document

This file is Markdown with LaTeX display equations (`$$ ... $$`), so it wants a viewer with math support — a GitHub gist, StackEdit, HackMD, or `Ctrl+Shift+V` in VS Code. **`ABOUT_SOLVER_EXPLAINED.md` has the steps for each**, along with what this document is for and what order to read it in.

> If your viewer shows raw `$$` symbols, it simply lacks math support. Nothing is lost: every formula here is also stated in words in the surrounding prose, and the code references are plain text.

Read it start to finish the first time — the order is the argument. In ten minutes instead: §2, §9, §16.

---

# The Wordle solver, from the ground up

## 0. What the program actually has to do

Every turn, the solver holds a set of words that are *still possible answers*. It picks a guess, sees five colored tiles, and that shrinks the set. The whole project is one question:

> **Which guess shrinks the set the most usefully?**

Everything else — entropy, the fitted curve `f`, the endgame search — is machinery for answering that.

Two word lists matter:

- **12,947 words** the solver is allowed to guess, and which it treats as *possibly* being the answer (`ind2word.json`)
- **2,309 words** on the original curated 2022 answer list (`wordles.json`) — the words the NYT editor mostly draws from

We need three things before that question can even be asked precisely: a way to compute feedback (§1), a way to describe what a guess *does* to the candidate set (§2), and a way to measure how good that is (§3).

## 1. The feedback rule, and why there's a 167 MB file

`wordle_data.score_guess` implements the coloring rule, including the fiddly duplicate-letter case. The rule is two passes: **greens are assigned first**, and then yellows consume whatever letters are left over, at most one tile per copy of the letter in the answer.

Two examples against the answer `abide`, both verified by running `score_guess`:

```
guess  speed   ->  N N Y N Y      guess  geese   ->  N N N N G
       s p e e d                         g e e s e
```

- `speed` has two E's and `abide` has one, so the **first** E takes the yellow and the second gets nothing. The D is also yellow — it's in the answer, just not in position 4. So the pattern is `NNYNY`.
- `geese` has three E's. The one in position 5 matches `abide`'s E exactly, so it goes green — and because greens are assigned first, it consumes the answer's only E and the two *earlier* E's get nothing despite coming first in reading order. Hence `NNNNG`.

Each pattern is 5 tiles × 3 colors, so encode it in base 3 with N=0, Y=1, G=2. That gives an integer in `[0, 243)`; `NNYNY` = 10, `NNNNG` = 2, and `GGGGG` = 242 (`WIN_PATTERN`).

Now the key engineering move. The feedback for a (guess, answer) pair never changes, so compute all of them once:

```
feedback_table[guess_index, answer_index] = pattern number   # one uint8
```

That's 12,947² ≈ 167 million bytes — the `feedback_table.npy` file. Building it takes a while; after that, "what would I see?" is a single array lookup instead of a string comparison. This is the same instinct as a prefix-sum table: pay O(n²) once so every query is O(1). Almost every inner loop in the project is a scan along one row of this table.

Almost, because the table is a speed choice rather than a necessity, and the choice is closer than 167 MB suggests. Scoring the rule afresh on every pair costs about 3.5× a lookup — 0.7 ms against 0.2 ms on a median turn-two position, and 18.5 ms against 5.3 ms on the worst of them. On a machine that can hold the file that ratio settles the matter, which is why the file exists. It settles the other way in a browser tab, running the same solver compiled to WebAssembly, where 167 MB can be neither downloaded nor rebuilt and 18.5 ms is a comfortable price for one turn.

That build does not give up tabulating; it tabulates the sliver it needs. The endgame search of §12 visits hundreds of nodes for a single position and rescans the same thousands of guesses at every one of them, and every candidate it will ever see belongs to the set it started with — so scoring that block once, 440 KB rather than 167 MB, turns all the rest of the search back into lookups. Everything below assumes the full table, which is what every measurement in this document was taken on.

### What the program is actually holding

Under all of that is a shape in memory, and three later arguments — the rearrangement in §7, the memory bill in §8, the packed keys in §12 — are arguments about that shape rather than about Wordle, so it is worth setting down before any of them.

After startup, words are never compared as text. Each of the 12,947 is assigned an index once, and from then on a word *is* its index (`ind2word.json` is that numbering, and `word2ind.json` is it read backwards). So:

- the table is a 12,947 × 12,947 grid of single bytes, where **the row is the guess and the column is the answer**, and the entry is the pattern number above;
- a guess is a row number, so "score every guess" means visiting all 12,947 rows;
- the candidate set is an array of *column* numbers in ascending order — `[3, 17, 964, …]` — starting as all 12,947 and getting shorter every turn as the feedback rules words out (`kernels.filter_candidates`).

Filtering keeps the order but not the spacing. By turn three the candidates are forty-odd columns scattered the width of a row of thirteen thousand, and that spread is what §8 turns out to be paying for.

Two things follow from the shape, and both come back later.

**The table is read along rows, and that is not an accident.** Row `guess` is laid out contiguously in memory, so scanning it for a *whole* candidate set walks straight through memory. Doing it for that scattered handful does not, and §8 is where the bill arrives.

**The loops that scan it are compiled, not interpreted.** `kernels.py` is decorated with numba's `@njit`, which translates each function to machine code the first time it runs — Python's own loops would be a hundred times too slow for 13,000 guesses × 13,000 candidates. One such compiled scoring loop is a **kernel**, and `kernels.py` is nothing but a handful of them. `cache=True` writes the compiled code to disk, so only the first run of a fresh checkout pays for it. That compile, plus loading the 167 MB table, is the startup wait the web page hides behind its veil.

## 2. A guess is a partition

Fix the candidate set `S` and a guess `g`. Every candidate in `S` produces *some* feedback pattern against `g`. Group them by that pattern:

```
guess = "crane", S = {every 5-letter word}
  pattern NNNNN  -> 4,283 candidates
  pattern NNNNG  ->   612 candidates
  pattern GNNNN  ->   371 candidates
  ...            (about 150 non-empty buckets)
  pattern GGGGG  ->     1 candidate  (crane itself)
```

**You are going to land in exactly one bucket, and that bucket becomes your new candidate set.** So a guess isn't really "a word" — it's a *partition of `S` into at most 243 buckets*, and the only thing that matters about a guess is the shape of that partition.

Two words with the same partition are the same move. That sounds like a throwaway observation; it is what makes the endgame search affordable in §12.

Bad guesses leave one huge bucket. Good guesses chop `S` into many small ones. Now we need to make "many small ones" precise.

## 3. Entropy, from scratch

This is the one piece of genuinely new math, so I'll build it rather than cite it.

**Question:** I'm thinking of one of `n` equally likely words. You ask yes/no questions. How many do you need?

Binary search: each question halves the set, so you need **log₂(n)** questions. Call that "log₂(n) bits of uncertainty." 1024 words = 10 bits.

**Now the unequal case.** Suppose one word has probability 1/2, another 1/4, and four more have 1/16 each. Uniform reasoning would say log₂(6) ≈ 2.58 bits, but that's wrong — you'd ask "is it the first one?" and half the time you're *done in one question*. Weight each outcome by how likely it is:

- an outcome of probability `p` "feels like" a set of size `1/p`, i.e. log₂(1/p) = −log₂(p) bits
- average over outcomes:

$$H = -\sum_i p_i \log_2 p_i$$

For that example: ½(1) + ¼(2) + 4·[1/16 · 4] = 0.5 + 0.5 + 1 = **2 bits**, less than 2.58. Concentrated probability = less uncertainty. That's the whole idea. `H` is maximized at log₂(n) when everything is equally likely, and is 0 when you already know the answer.

**Information gain.** Before guessing, uncertainty is `H(S)`. After, you're in bucket `r` with probability `p_r`, and still uncertain by `H(S_r)`. So

$$\mathrm{IG}(g) = H(S) - \sum_r p_r H(S_r)$$

Here's the pretty part, and it's why the code is fast. Knowing which word it is = knowing *which bucket* + knowing *which word within that bucket*. Uncertainty adds the same way:

$$H(S) = \underbrace{H(p_1, \ldots, p_k)}_{\text{which bucket}} + \sum_r p_r H(S_r)$$

Substituting, **the information gained is exactly the entropy of the bucket-mass distribution.** You never need to look inside the buckets. That's `kernels._ig`: total up the **mass** of each pattern — a bucket's mass being the combined weight of the candidates that land in it, probability before it is divided through by the total — and then `h -= p*log2(p)` over the buckets.

That gives **`--mode entropy`**: pick the guess whose partition has the highest bucket entropy. It's the standard 3Blue1Brown-style solver, and it averages **3.75 guesses** here — a figure measured under the prior §4 is about to build, as every measured number in this document is. Scored under the uniform assumption this section has been making instead, the same rule reads 4.12, and §4's table is where the reason those two cannot be set against each other belongs.

That is the baseline, and the rest of this document is about beating it. There are two ways to try. One is to search harder — look further ahead, solve the endgame exactly. The other is to notice that the question in §0 was never quite right: it assumed every word is equally likely to be the answer, and no word ever was. The second turns out to be worth far more than the first, which is §16's punchline.

## 4. Not all words are equally likely (the priors)

`--mode entropy` implicitly assumes any of the 12,947 words is equally likely. That's badly wrong, and fixing it turns out to be worth more than *any* search improvement in the project.

`priors.build_weights` models the editor as a two-step process: with probability **0.10** reuse a word that has already been an answer (among the ~1,858 used ones), otherwise pick an unused word, which is **off** the curated list with probability **0.10**. That gives three groups — used, unused-and-curated, unused-and-non-curated — each word keeping strictly positive weight, so nothing is ever ruled out. Note the conditioning: the off-list probability is *conditional on not being a repeat*, the model being sequential, so computing it against all puzzles rather than non-repeat ones understates it.

Those two 0.10s look aggressive against Wordle's full history, and the reason they aren't is a regime change. Over the **current editorial regime (Wordle #1690 onward, from 2026-02-02)** the repeat rate is 11.1% and the off-list rate 8.9%; over all 1,879 puzzles they'd be 1.1% and 2.5%, roughly 10× and 4× lower. What drives it is mechanical: 1,812 of 2,309 curated words are used up and ~334 are consumed a year, so the pool empties in about 1.5 years, and both rates are the editor's response to that.

**The prior is where the wins are.** Every word as the answer, weighted by that row's own prior, each row playing the form of `f` fitted for it:

| priors on | entropy | expected-guess mode |
| --- | --- | --- |
| none | 4.1223 | 4.0718 |
| repeat only | 4.1033 | 4.0111 |
| curation only | 3.8455 | **3.4903** |
| both | 3.7507 | **3.0770** |

Read *down* a column, never across a row: each row's weights are its own prior's, so a concentrated prior makes the game genuinely easier and no two rows are comparable. What the columns show is the gap between the two objectives — 0.67 guesses with both priors on, 0.0505 with none, thirteen times less for exactly the same machinery. §9 shows why that collapse is an identity rather than a coincidence.

Three groups says how much mass each one gets, not how it spreads *within* one, and the obvious answer — flat — is wrong twice over. It says `cigar` (Wordle #1) and `sandy` (#1227) are equally likely to come round again, and that `squid` and `xylyl` are equally likely to be tomorrow's answer. The second is the expensive one: that group is 10,638 of the 12,947 words, so it dominates the candidate *count* at almost every position while holding only 9% of the mass. Call the within-group spread the group's **shape**; both were uniform in the original design and both are now sharpened, which is §5.

## 5. Shaping the mass inside each group

### 5.1 The repeat group: a smoothed histogram

There are 21 repeats in Wordle's history, and they are not drawn from the used pool evenly. Nineteen come from Wordle #1–#297 (median #116); only `sandy` (#1227) and `smile` (#831) sit outside that window. The editor appears to be replaying the *beginning* of the run.

An 11% outlier rate is too high to just zero out everything past #297, so the shape is a **Laplace-smoothed histogram**. Bin the used words by the puzzle number where they first appeared, 200 wide, and give bin *b* the weight

$$w_b = \frac{r_b + \alpha}{n_b}, \qquad \alpha = 0.5$$

where `r_b` is how many repeats came out of that bin and `n_b` how many used words live in it; then spread each bin's share evenly over its members. The `α` keeps a bin that has never produced a repeat from being written off entirely — the "maybe next time" term, and the reason no word's weight reaches zero. `cigar` ends up with about **6×** the flat repeat weight, and late words fall below it.

Fixed bins rather than a sliding window is deliberate, and the reason is a small lesson in overfitting: a window needs a boundary, a boundary is a parameter, and that parameter has to be tuned on the same 21 events you are trying to predict. Scored prequentially — each repeat predicted from the ones before it only, so no parameter ever sees the event it is scored on — flat gives a log-likelihood of **−157.1** against the histogram's **−133.1**, about **+1.14 ± 0.21 nats per event**. That `±` is a standard error: how far the estimate itself would wander if the same 21 events were drawn again, so a gap several times its own `±` is one to believe and a smaller one is not. Here it is five times. The best of sixteen tuned windows reaches only −143.6 and an age-based shape refit as events arrive lands at −150.1: twenty-odd events cannot pin a rate down, and the histogram has nothing to pin.

### 5.2 The off-list group: WordleBot's own weights

This is the interesting group, because "which words could be off-list answers" turns out to be *knowable* rather than guessable.

The first clue is in `solutions.json` itself. Its `id` field is not an opaque key — it is the **1-based index into `wordles.json`**. Every off-list answer has an id past the curated list's 2,309, and 44 of the 46 land in **2310–2397**. Sorted by id they come out in alphabetical runs: twelve consecutive at 2318–2331 (`beaut laser mommy muggy nervy pious primp shred spate taupe tinge tizzy`), seven more at 2343–2352. A run of twelve in order by chance is 1/12! ≈ 2×10⁻⁹. So the NYT appended words to the same internal list in alphabetically-sorted batches, and off-list answers are drawn from a block of roughly ninety — not from 10,638.

Knowing the block is small does not name its members, but somebody has already published a list that does. **WordleBot**, the NYT's own solver, carries a per-word weight, scraped and published at `wordletools.azurewebsites.net/weightedbottles`: 3,209 words with weights from 0.226% to 58%, of which 3,194 are guessable here — 2,302 curated, **892 not**, and all 46 off-list answers to date among them. That last fact proves nothing on its own, since a list that absorbed answers after they were used would look identical. The test that matters is whether an *old* copy predicts *later* answers, and the Wayback Machine has a snapshot from **2023-10-12**, frozen almost three years ago:

| snapshot 2023-10-12 | |
| --- | --- |
| off-list answers used after it | 41 |
| ...already in the snapshot | **39** |
| the misses | `loris`, `kefir` |
| words added to the list since | 84 |
| ...of those, ever an answer | **2** (the same two) |

So it predicts forward at about 95%, and its growth is not backfilling.

The shape follows directly: **95%** of the off-list mass goes to the listed words in proportion to their published weight, and the remaining **5%** spreads over everything else — a hedge set at the measured miss rate, and what stops a `loris` from being a blindside.

Scored prequentially on the 42 post-snapshot events, each shape at the constant that actually ships:

| shape | nats per event | effective support |
| --- | --- | --- |
| uniform | −9.27 | 10,638 |
| answer-like frequency (§5.3), `t` = 0.35 | −8.13 | 4,857 |
| **WordleBot, 5% hedge** | **−6.46** | **911** |

"Effective support" is `2^H` from §7 applied to the prior itself: how many equally likely words the shape is really spreading over. WordleBot scores −6.46 over all 46 events too, so it is not winning by having quietly absorbed the early ones. Score each shape at its *fitted* constant instead and the lower two rows both improve, which is why the table specifies: a shape has to be scored as it will actually run.

**The 5% hedge stays flat, and that is deliberate.** Weighting it by answer-likeness is the obvious refinement and it fails on the population that matters, because the hedge only fires on a word the list is *missing*, and a word is missing because it is odd. `loris` sits at the 53rd percentile of answer-likeness and would get **0.32×** the flat share. Over the only two events that have ever needed the hedge, shaping it is worth −0.13 nats — flat being the honest expression of not knowing which word the list is short of.

### 5.3 The fallback that needs no scrape

Depending on somebody else's site is a fine thing to vendor and a bad thing to *need*, so there is an alternative shape built only from what the repo already has. The curated list is 2,309 labeled examples of "a word an editor considered answerable", independent of the 46 off-list events, so `answer_like.py` fits a logistic regression with those as positives and the non-curated words as negatives, then scores the unused words by its log-odds. Four of its thirteen weights carry the model, and those four are legible:

| feature | weight | reading |
| --- | --- | --- |
| Zipf frequency | **+4.00** | common words get picked |
| ends in `s` | **−5.87** | 37% of the pool is a plural; 9% of off-list answers are |
| ends in `ed` | −5.09 | past tenses do not get picked either |
| foreign lift | −1.16 | as common abroad as at home ⇒ a name, not a word |

The other nine still matter: a fit keeping only these four correlates at r ≈ 0.98 with the shipped table while reproducing none of it. Against the flat weight the shape replaces, at the temperature that ships:

| promoted | | demoted | |
| --- | --- | --- | --- |
| `squid` | 8.5× | `karks` | 0.014× |
| `maven` | 4.4× | `fents` | 0.014× |
| `aloha` | 4.0× | `xylyl` | 0.10× |
| `kazoo` | 2.4× | `faffy` | 0.096× |

Every word on the left is a real off-list answer, every one on the right behind a recorded loss in §14. The shape scores −8.13 against WordleBot's −6.46 above: worse, and the price of owing nobody a scrape.

## 6. Measuring a prior without fooling yourself

Two traps, then a question the data cannot settle.

### 6.1 The form of `f` has to be refitted per shape

`f` (§10) is queried at its largest arguments exactly once per game — at the opening, where the bucket is the whole word list, which is the extrapolated tail of the fit. Sharpening the prior shifts every effective size into that tail, and the ranking of opening words flips: under `sqrt`, the form chosen for the *uniform* shapes, a sharpened prior picks `raise` over `salet`. `raise` is genuinely worse — forcing it under the unchanged uniform prior costs **+0.034 ± 0.028** — so this is estimator error, not a real preference. The three best-fitting forms under the sharpened prior all pick `salet` (`invonly`, `saturating` and `spline`), and so does one-ply lookahead — playing each guess, then scoring the positions it could lead to, one *ply* being one turn of search. This is why the shape is part of the cache key.

Fit quality is not the criterion, though. Rank the forms by **RMSE** — the root-mean-square of the residuals, the typical distance between the fitted curve and the points it was fitted to, so smaller is closer — and the ranking does not predict the opening at all: `anchored2` has the worst RMSE of any form at 0.70 and picks `salet` anyway, while `anchored` fits a hair *better* than `sqrt` (0.5041 against 0.5047) and still picks `raise`. What decides the opening word is concavity in `x` (§9), not how closely the curve tracks its own fitting data.

### 6.2 The evaluation needs three arms, one of them repeats

A shape can only be judged on answers that actually happened, never on answers sampled from its own belief — a prior scored against its own prior is grading its own homework, and that swing is larger than the effect being measured. The arms are the **21 real repeats** and the **46 off-list answers**, each replayed with the prior rebuilt from strictly earlier puzzles, plus the **497 curated words still unused**, blended at the regime's own 11.1% / 7.9% / 81.1%:

| shapes | repeat (21) | off-list (46) | fresh (497) | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.143 ± 0.078 | −0.978 ± 0.105 | +0.022 ± 0.028 | −0.044 ± 0.025 |
| repeat `binned` alone | −0.143 ± 0.173 | −0.022 ± 0.049 | −0.006 ± 0.016 | −0.022 ± 0.024 |
| `binned` + `frequency` | −0.286 ± 0.184 | −0.522 ± 0.145 | −0.002 ± 0.018 | **−0.074 ± 0.028** |
| `binned` + `wordlebot` | +0.095 ± 0.228 | −0.826 ± 0.150 | −0.012 ± 0.018 | **−0.064 ± 0.032** |

Unlike §4's table, **these columns are comparable across rows**: every row is scored against the same real answers, so the blend is a genuine paired comparison rather than each prior being asked how it fares in a world of its own devising. They are comparable to nothing outside it, though, the estimator having been refitted since. What `analysis/replay.py` measures now is the one contrast both surviving fits support: `frequency` against `wordlebot` head to head, at **−0.010 ± 0.025** blended.

The repeat arm is the loosest term in every blend at ±0.17–0.23, leaving the bottom two rows well inside one standard error of each other. Choosing between them rests more on whether you want a vendored scrape in the repo than on the measurement.

### 6.3 Do the two shapes need each other?

No: they are **additive, within noise**. The three configurations blend to −0.044, −0.022 and −0.064 against a sum of −0.066, and each shape independently beats flat at its own job by a margin the prequential scoring *can* resolve — +1.14 ± 0.21 nats per event for `binned` (§5.1), −6.46 against −9.27 for `wordlebot` (§5.2). Both ship because each earns its place alone.

There is a mechanism that predicts otherwise, and it is worth knowing about even though these events cannot see it. Sharpening one side makes the other's flatness the larger error in the weights: concentrating the off-list mass onto its ~870 effective words gives each about **1.0×10⁻⁴** where a used word under flat repeat mass has **5.4×10⁻⁵**, so curation alone promotes every off-list word above every past answer, and when the answer is a `cigar` that costs. Whether the repeat shape buys it back is not a question 21 events can answer — that arm carries ±0.17–0.23 however it is analyzed, and resolving a 0.1-guess effect on it needs some 340 repeats, eight years of Wordle. So the mechanism is unmeasured here rather than absent. That interference of this general kind is real gets settled one level up: §6.4 finds the two *priors* these shapes sit inside strongly entangled, though by a different route — each miscalibrating the rate the other governs, rather than one group outranking another.

### 6.4 Do the two *priors* need each other?

A shape only ever redistributes mass inside a group its prior has already carved out. The prior decides whether the group is there to spread anything over: switch curation off and there is no curated-versus-not split at all, just 12,947 words sharing the mass equally (§4), non-answers included. That is the larger question, and it has a much larger answer. `analysis/replay.py --all-priors` replays the same 190 regime puzzles under all four on/off combinations, every arm paired on the same answer, for means of **3.9421** with neither prior, **3.5368** with curation alone, **3.9000** with repeat alone, and **3.1789** with both.

Those four numbers do not add up, and the way they fail to is the finding. Turning the repeat prior on is worth **+0.042 ± 0.042** when curation is off — nothing — and **+0.358 ± 0.059** when it is on. The gap between those two, the *interaction*, is **−0.316 ± 0.072**, and the three-arm blend of §6.2 agrees at −0.294 ± 0.046 — a wider view of the same data rather than a second opinion, its repeat arm being those same 21 events. The priors are strongly superadditive even though their shapes are not.

Read from the other side it is the same number, as an interaction always is: the curation prior is worth +0.405 ± 0.055 with the repeat prior off and +0.721 ± 0.060 with it on, and 0.721 − 0.405 is that 0.316 again. Neither prior is the one that "needs" the other. What they need is each other.

The reason is calibration, and it shows plainest if you ask what rates each prior implies — against the two the regime really shows, an 11.1% repeat rate and 8.9% off-list among the answers that are not repeats (§4):

| prior | implied repeat | implied off-list \| not a repeat |
| --- | --- | --- |
| none | 14.4% | 95.5% |
| curation only | 71.3% | 32.4% |
| repeat only | 10.0% | 95.5% |
| both | 10.0% | 10.0% |
| *observed* | *11.1%* | *8.9%* |

Every prior but the last is wildly wrong about the rate the *other* one governs, and that is the whole of the interaction. Uniform lands near enough on repeats by pure accident — 1,858 used words out of 12,947 is 14.4% — so adding the repeat prior to it corrects a column that was not much wrong while leaving the 95.5% exactly where it was, which is why it buys nothing. Curation alone pulls that second column down to 32.4% and *creates* the repeat error in the act: confining the field to a 2,309-word list of which 1,812 are spent is implicitly a claim that the editor repeats himself two times in three. Only with both switched on is either rate right.

Per word, that correction reads as a used word falling from **3.8×10⁻⁴** to **5.4×10⁻⁵** while a fresh curated word climbs from **3.9×10⁻⁴** to **1.6×10⁻³** — the 497 words the editor can still reach without repeating himself or leaving the list.

It also prices the repeat arm, where curation alone beats the pair by **0.571 ± 0.213** on the 21 answers that really were repeats. That is what a calibrated prior costs rather than a fault in it: move a class from 71.3% down to 10.0% and you pay on the occasions it turns up, in exchange for the fresh arm's **+0.519 ± 0.032** across 81.1% of the regime. Any honest prior makes that trade.

One caveat rides along with all four numbers. Each arm carries the `f` fitted for its own prior, as §6.1 says it must, so what is being compared is four *solvers* rather than four beliefs held by one solver. That is the honest comparison — a prior the estimator has not been fitted to is not a configuration anyone would ship — but it does mean a refit moves every row, and the four have to be refitted together or not at all.

## 7. Effective size: making a weighted set have a "count"

Unequal weights break something the last three sections quietly relied on. "This bucket has 40 candidates" is misleading — 40 candidates with one of them at 90% probability is much easier than 40 equally likely ones. Everything from here on needs a size that knows the difference.

The fix (`estimator.effective_size`):

$$n_{\text{eff}} = 2^{H}$$

where `H` is the entropy of that bucket's *normalized* weights. If the weights are flat, `H = log₂(count)` and `n_eff = count` exactly. As probability concentrates, `n_eff` slides toward 1. It's "how many equally likely words would be this hard."

### The rearrangement that makes this cheap

There's an implementation trick here worth working through, because it's the single biggest optimization in the scoring loop.

**The problem.** `H` is defined on the *normalized* weights `p_i = w_i / m`. But `m` — the bucket's total mass — isn't known until you've seen every member of the bucket. And which bucket a candidate lands in depends on the guess. So the naive reading is: for each of the 12,947 guesses, pass over the candidates to find the bucket totals, then pass over them *again* taking a logarithm per candidate to build each bucket's entropy. That's one `log` per (guess, candidate) pair.

**The fix.** Expand the definition and split the logarithm. Write `m = Σ w_i` over the bucket's members:

$$H = -\sum_i p_i \log_2 p_i = -\sum_i \frac{w_i}{m} \log_2\left(\frac{w_i}{m}\right)$$

Now use `log₂(w/m) = log₂ w − log₂ m` to break the coupled term apart:

$$H = -\frac{1}{m}\sum_i w_i \big(\log_2 w_i - \log_2 m\big)
    = -\frac{1}{m}\sum_i w_i \log_2 w_i + \frac{\log_2 m}{m}\sum_i w_i$$

The second sum is just `m` again, so `(log₂ m / m) · m = log₂ m`, leaving

$$H = \log_2 m - \frac{\sum_i w_i \log_2 w_i}{m}$$

**Why that's a win.** Look at what each piece depends on. `m` and the sum both range over "the members of this bucket," which does depend on the guess. But the *summand* `w_i log₂ w_i` depends only on candidate `i` — on its raw prior weight, with no reference to the bucket, the guess, or the normalization. Normalizing is what coupled a candidate's contribution to a global property of its bucket, and factoring the logarithm decouples it again.

To see where that pays, it helps to have the loop it pays inside. Every scorer in `kernels.py` has the same two-phase shape, and it is worth seeing once, since §8 takes it apart:

```
for each guess:
    accumulate:  one pass over the candidates, adding each one's
                 contribution into a slot picked out by its pattern
    reduce:      turn those 243 slots into one number, the guess's score
```

The slots are the **accumulators**: four arrays of 243 floats, one entry per possible pattern, holding that bucket's running mass, its candidate count, its running total of `w log₂ w`, and its heaviest single weight. The accumulate pass is the hot one, running once per (guess, candidate) pair — 167 million times at the opening — so everything expensive is kept out of it.

`w_i log₂ w_i` is therefore computed **once per candidate**, before any guess is scored (`kernels._w_log_w`), and the accumulate pass adds a precomputed float into a slot where the naive version would call a logarithm. The reduce then reads two of the slots it already has — the bucket's mass and its total `w log₂ w` — and recovers `H = log₂(mass) − (Σ w_i log₂ w_i) / mass` with one logarithm per *bucket* rather than one per candidate (`kernels._bucket_guesses`).

Cost, for a position with `c` candidates scored against all 12,947 guesses: the naive version needs `12,947 × c` logarithms, the split version needs `c`. At a wide position (`c` ≈ 12,947) that's 167 million logs against thirteen thousand.

**One more consequence.** Since `n_eff = 2^H`, the `H` the reduce just recovered is exactly `log₂(n_eff)` — already in the units `f` wants. That's why `f_eval_x` is parameterized by `x = log₂ n` rather than by `n` (`estimator.f_eval_x`): every caller reaches `f` through an entropy, so it has `x` in hand, and going via `n` would mean computing `2^x` only for `f` to take `log₂` straight back off.

Sanity check on random weights, direct definition against the rearranged one:

```
direct : 2.331157535638251
split  : 2.3311575356382495     # agree to 1.3e-15
```

## 8. The scoring loop, and where the time actually goes

Removing those logarithms exposed what was underneath, and it was not what the arithmetic suggested. This section is the rest of what `kernels.py` does, and it is worth its own space because the reasoning generalizes: the obvious cost model was wrong twice.

**First wrong assumption: that wide positions are the expensive ones.** They are the expensive *individual* calls, but small candidate sets are scored far more often — every node of the endgame search and every late turn of every training rollout scans a handful of candidates against all ~13k guesses. The aggregate cost lives in the small cases, so that is where to optimize.

Start with the accumulators, since what they cost decides how the reduce has to be written. Four fresh 243-element arrays per guess would mean allocating and zeroing some fifty thousand floats thirteen thousand times over, so instead they are allocated once per block of at most `kernels._MAX_BLOCK` = 64 guesses and reused across the block. That reuse puts them under an obligation: whatever one guess writes into them has to be gone before the next guess starts.

So the reduce carries two jobs — total the buckets up, and leave them zeroed — and the obvious way to do both is a walk over all 243 patterns, which costs 243 whether there were 12,000 candidates or five. With five candidates at most five buckets are non-empty, and the other 238 checks are pure waste, repeated for every one of 13,000 guesses.

Below `kernels._RESCAN_MAX` (128 candidates), then, the reduce **walks the candidates a second time instead**, taking each bucket the first time it is reached and zeroing it on the spot. Zeroing there is what makes the second walk correct as well as cheap: it clears the accumulator for the next guess, and it stops the bucket being counted twice when a later candidate lands in it. With five candidates the sweep does 243 reads and 243 writes per guess where the second pass does five of each.

The two branches total the buckets in different orders, so a 127-candidate position and a 129-candidate one disagree in their last bits. That is harmless, and measurably so: over all 12,947 guesses at the opening, the closest pair of scores differs by 1.1e-9 under the shipped prior and 6e-11 under the loosest, against an ulp of 4e-16. Nothing here is decided by differences anywhere near rounding — which is also why the branch on candidate count is safe to take at all.

| candidates | reduce over patterns | reduce over candidates |
| --- | --- | --- |
| 5 | 243 | 5 |
| 20 | 243 | 20 |
| 128 | 243 | 128 |
| 12,947 | 243 | (not used — the sweep is cheaper) |

**Second wrong assumption: that the cost is arithmetic.** It is memory, and seeing why means looking at what the loop actually touches. It fixes one guess, walks the candidate set in order, and reads a single byte per candidate — `feedback_table[guess, candidates[i]]`. All of those reads come from one row of the table, the row belonging to that guess, at the column numbers §1 gave the surviving words. The accumulators they feed are 243 floats, two kilobytes, small enough to sit in the cache nearest the core for the whole block. So the table is the only thing being streamed in, and the only thing that differs between one position and the next is how far apart the columns are.

That distance is the cost. Processors do not fetch single bytes; they fetch a **cache line**, a contiguous block of 64 bytes, and keep it in fast memory near the core. Sixty-four adjacent columns therefore arrive on one fetch and the other 63 reads are free, which is the early game: the candidates are most of the word list and pack the row densely. Late in the game the survivors are a few words scattered anywhere in an alphabetical list of 12,947, so consecutive columns sit thousands of bytes apart, every read lands on a line of its own, and each one spends 64 bytes to use a single byte. That is the real bill at every small position, and it is also why the second pass over the candidates is nearly free: it re-reads the very lines the first pass has already pulled in.

How far apart the two cases can get is bounded, and the arithmetic is worth doing. A row is 12,947 bytes, or 203 cache lines. A dense run of `k` candidates fetches `k/64` of them; a scattered set of `k` fetches one per candidate, up to the point where it runs out of distinct lines to land on and stops at 203. Scatter costs the gap between those two, which widens until `k` reaches a couple of hundred and narrows after, since the dense figure keeps climbing while the scattered one is already pinned at its ceiling. By the time both sets are the whole word list each is reading all 203 lines and there is nothing between them. At the other end, under about twenty-five candidates the whole scan is a couple of dozen lines and too small to matter either way. The penalty lives in the middle of the range — and so, it turns out, does everything the alternative layout below can win.

**Those blocks are also the unit of parallelism.** The guesses to be scored are cut into `4 × get_num_threads()` blocks and handed to numba's `prange`, a parallel loop that spreads its iterations across cores, and each block carries accumulators of its own, so no two threads are ever writing into the same array. Several blocks per thread rather than exactly one means a thread that draws slow work does not hold everyone up at the end.

### Storing the table the other way round

Indexing it `[answer, guess]` instead is the obvious alternative, and it does fix the reads: a kernel written for that layout takes the same time on scattered candidates as on adjacent ones, which the shipped one plainly does not. What it does not do is make the scatter go away. Fixing a guess and walking the candidates finishes one guess before starting the next, so a single set of 243 buckets serves a whole block of 64; fixing a candidate and walking the guesses leaves every guess in the block partway through at once, so each needs a set of its own. Those buckets are read and written on every (guess, candidate) pair — exactly as often as the table is — so the cost moves rather than disappearing, from the table to the accumulator.

The trade can be dodged rather than taken. For a block of G guesses, read the G × m tile out of an answer-major copy — contiguous, G bytes per candidate — transposing it into a small buffer on the way in. Each guess's feedback vector is then a contiguous row of that tile, so the reduce is the one above, over one accumulator that stays in L1. That is what `fused_kernels` does, and against the candidate sets real play produces it is worth 1.40× between 200 and 400 candidates and 1.20× between 401 and 600, falling through parity by about 600 as the tile stops fitting in cache.

Above that the shipped layout cannot be beaten, and the reason is structural rather than a matter of tuning. Reducing a guess needs that guess's whole feedback vector, which an answer-major table holds as a strided column, so the alternative is forced either to keep every guess in the block accumulating at once or to rebuild the vector by transposing — and at the widest positions the shipped scan is already reading the table a full cache line at a time, leaving nothing for either to recover. The opening scan is where that bites hardest, being the most expensive single call in the game.

The two layouts are therefore complementary rather than ranked, which is why `fused_kernels` is a dispatch below 600 candidates and not a replacement, and why it is off unless asked for. Turned on it is worth 1.09× over whole games under expected guesses and 1.63× under entropy; the distance between those and the per-kernel figures is simply how much of a game is not scoring at all, the endgame search's own recursion above all. It costs 168 MB held in memory, built in about 0.3 s from the table already loaded, and there is no second file.

### The optimization that had to be taken back out

Numba hands each kernel to LLVM, the compiler that turns it into machine code, and offers a `fastmath` flag that grants LLVM license to reorder floating-point arithmetic for speed, on the reasoning that addition is associative. It is not quite associative in floating point, but the reordering is usually harmless.

Here it was not, and the damage was not to the arithmetic at all. The reduce reads a bucket and then writes a zero into it for the next guess; restructuring the loop swapped those two memory operations, so **every bucket maximum read back as the zero its own loop had just written**. The license granted was over the order of the additions, and what moved was a store. It surfaced as `two_turn_scores` returning exactly 3.0 for every guess and the search confidently preferring nonsense words. The other two scorers happened to survive, but only because a logarithm or a call to the estimator sits between their read and their write and gives the compiler something it cannot reorder across — which is luck, not protection, and not a thing to rely on. None of the kernels carry the flag now. It costs nothing on the widest positions and up to about 10% in the middle, which is a small price for scores that are what they claim to be.

### Scoring fewer guesses

Every kernel takes a **pool** — the set of table rows it is being asked to score, standing in for all 12,947 — and returns one score per entry, `pool=None` meaning every row. That parameter is what keeps deeper search affordable: a lookahead exploring the replies to a bucket of ninety candidates does not need all 13k of them, and hard mode (§9) is the same mechanism with the pool set to the candidate set itself.

With a size that respects the prior and a loop that computes it quickly, the objective can finally be stated in the units the game actually scores.

## 9. The real objective: minimize expected guesses

Entropy answers "which guess is most *informative*." But the game doesn't score you on bits — it scores you on **guesses**. Those aren't the same thing, and the gap is where the half-guess improvement lives.

Suppose you had a function

> **`f(n)`** = expected number of further guesses needed from a position of effective size `n`

Then the value of guess `g` is immediate:

$$E(g, S) = 1 + \sum_{r \ne \texttt{GGGGG}} p_r f(n_r)$$

"One guess for this turn, plus the average cost of wherever I land." Two details carry real weight:

- **Excluding `GGGGG` is the whole reason a possible answer beats an equally informative non-answer.** That bucket costs nothing more — you already won. So a guess that might just *be* the answer gets a free discount equal to its probability.
- **Buckets of 1 and 2 skip `f` entirely** (`kernels._bucket_guesses`), because their values are known in closed form: 1 candidate is worth exactly `1`, and 2 candidates are worth `2 − p_max` (guess the likelier one; if wrong, you now know the other).

Where `f` comes from is §10's problem. What it has to *look like* is settled here, and the answer is sharper than it sounds.

### Why this isn't just entropy in disguise

Suppose `f` is **linear**: `f(n) = a + b·x` with `x = log₂(n)`. Substituting, and using the chain rule from §3:

$$E(g,S) = 1 + a(1 - p_{\text{win}}) + b\big(H(S) - \mathrm{IG}(g)\big)$$

`H(S)` doesn't depend on `g`, so minimizing `E` is *identical* to maximizing `b·IG(g) + a·p_win` — greedy entropy, plus the possible-answer bonus. **A linear `f` buys you nothing but the bonus.** All the remaining gain comes from `f` being *concave* in `x`.

That identity is testable, because `loglinear` is a form of `f` that *is* linear in `x`, so by the argument above it should score exactly "entropy plus the bonus" — and it does. Under the shipped shapes greedy entropy scores 3.7507, `loglinear` 3.1179, and the best form 3.0770. So the gap splits:

| | worth |
| --- | --- |
| the possible-answer bonus | **0.633** |
| curvature in `f` | **0.041** |
| total, entropy → best | 0.674 |

None of that bonus needs fitting, which is worth saying plainly given how much of the gain it is. Written in its own terms the rule is `IG(g) + a·p_win(g)`, no `f` anywhere, and two exact facts fix `a` without touching the answer log: one candidate takes one guess and two take 1.5, so a bit of information is worth half a guess and a win outright saves the whole one, which is two bits. Playing `IG + 2·p_win` scores **3.1187**, collecting the same 0.63 the table above books to the bonus with no estimator, no fit and no answer log behind it. README *Results* carries it as a second reference rather than as the baseline, since it pays for those guesses with twelve times greedy entropy's loss rate.

Nearly all of the improvement over a standard entropy solver is the free discount for guessing a word that might be the answer. The fitted curve's shape — the thing §10 works hardest for — contributes about a sixteenth of it. Splitting the same gap under each prior (`analysis/results_tables.py --part priors`) shows both halves are the prior's doing rather than the objective's:

| priors on | entropy | linear `f` | best `f` | bonus | curvature |
| --- | --- | --- | --- | --- | --- |
| none | 4.1223 | 4.0718 | 4.0718 | +0.051 | −0.000 |
| repeat only | 4.1033 | 4.0111 | 4.0111 | +0.092 | +0.000 |
| curation only | 3.8455 | 3.5347 | 3.4903 | +0.311 | +0.044 |
| both | 3.7507 | 3.1179 | 3.0770 | +0.633 | +0.041 |

The bonus grows with how sharp the prior is, which is what §16 is about. Curvature is not a constant ~0.04 either: it is worth that much when the **curation** prior is on and nothing at all when it is off, where the best form of `f` is a straight line and the fitted curve has no shape left to exploit — under `repeat only` policy iteration picks `loglinear` outright. A bent `f` pays only where the prior has made the candidate sets uneven enough to bend it for.

Wordle's own **hard mode** is the cleanest test of the identity, because it isolates the bonus from everything else. Hard mode requires every guess to be a word that could still be the answer, so restricting greedy entropy to it forces `p_win` up without touching entropy's scoring rule at all. It gains **0.2505 guesses** (3.7507 → 3.5002, `analysis/hard_mode.py`) — two fifths of what the objective gets by *pricing* the same bonus rather than imposing it. The objective does better because it takes the bonus **and** keeps the probes: it plays a non-candidate whenever the split is worth more than the win chance given up, which is exactly the trade hard mode forbids.

So `f` needs to be concave in `x`, and the fitted curve for the default prior is a piecewise-linear spline that obliges:

```
f(n) = 0.955 + 0.372·x − 0.112·(x−2)₊ − 0.138·(x−4)₊ − 0.104·(x−6)₊
                                            (x = log₂ n, (u)₊ = max(u, 0))
```

Every knot — a knot being a value of `x` where the curve is allowed to change slope — bends it flatter and none bends it back, and the sixth, at `x = 8`, was dropped altogether by the monotonicity guard §10 describes. Concretely, beside the curve fitted under the flat shapes (`0.61 + 0.77·√x`):

| n | 2 | 3 | 10 | 100 | 1,000 | 12,947 |
| --- | --- | --- | --- | --- | --- | --- |
| `f(n)`, shipped shapes | 1.33 | 1.54 | 2.04 | **2.47** | **2.53** | **2.60** |
| `f(n)`, flat shapes | | | | 2.58 | 3.02 | 3.42 |

Going from 100 candidates to 1,000 costs **0.06 guesses** on the shipped curve, and from 1,000 to the whole word list another 0.07. That is the honest shape of the game, and precisely what a linear `f` cannot express. Over that whole stretch the shipped curve climbs 0.13 where the flat-shape one climbs 0.84, which is more than a rescaling: it is the prior changing what `f` even has to model. With the mass concentrated, the effective size rarely reaches the range where a saturating curve has gone flat, so saturation stopped being the liability it used to be.

One caveat that matters when reading any of this: `f` is fitted **per prior**, so the curve above is the default prior's and no other. Turn a prior off and a different curve, in a different functional form, is what scores the guesses.

## 10. Where does `f` come from? Policy iteration

Circular problem: to play well you need `f`, but `f` is defined as the cost *under good play*. This is the same chicken-and-egg as value functions in reinforcement learning, and it's solved the same way — bootstrap and iterate (`train.policy_iteration`).

```
round 0:  play 2,000 games with greedy entropy (needs no f)
          -> for every position visited, record
             (n_eff, guesses actually needed from there)
          -> least-squares fit f to those pairs

round k:  play 2,000 games acting greedily on the current f
          -> record the same pairs
          -> refit
```

This is **policy iteration with function approximation**: *evaluate* the current policy by rollout, *improve* by acting greedily on the fitted value function. Because every form is linear in its parameters (`a + b·x + c·√x` — the *terms* are nonlinear, the *parameters* aren't), each refit is a single closed-form least-squares solve, not an optimization.

You can watch it converge in `estimator_cache/params_repeat1_curation1_r-binned_c-wordlebot.json`:

| round | mean guesses |
| --- | --- |
| entropy seed | 3.654 |
| iter 1 | 3.150 |
| iter 2–4 | 3.100 |

Essentially all the gain arrives in one iteration; the rest is a small limit cycle, so the loop caches the parameters whose *measured play* was best rather than the last fit (a greedy policy can oscillate). Those cached parameters are themselves an optimization worth naming — fitting takes seconds and there is no reason to repeat it per run, so the fit for each prior combination is written to `estimator_cache/` and loaded thereafter.

Three constraints worth flagging:

- **Training games are played with no 6-turn deadline.** Otherwise a game that runs long gets truncated at 6, biasing the targets downward. The deadline is a constraint layered *on top* of `f`, not part of it — which is §11.
- **`f` must be non-decreasing** on `[1, 12947]` (`estimator._is_monotone`). If a bigger candidate set could score *cheaper*, the search would prefer uninformative guesses that dump everything into one bucket. When the full basis — the ordered list of terms a form is built from — violates that, terms are dropped from the end until it holds, which is why `quadratic` and `loglinear` score identically in `README.md`'s table of the forms: `quadratic`'s `x²` term gets dropped every time, leaving exactly `loglinear` behind.
- **A prior with no cached fit does not play.** Rather than bending some reference curve onto the chosen form and playing a policy nobody measured, the estimator raises and names the `train` command that fixes it. Entropy mode never consults `f` and runs on anything.

## 11. The deadline, and what losing costs

Everything so far has quietly ignored the fact that real Wordle gives you only **six turns**. `f` estimates guesses-to-finish with no deadline at all, which §10 explained is deliberate — but the next section's formulas need a cost for running out, so define it here before using it.

The obvious approach would be a second objective: minimize guesses, subject to some separate constraint about not losing. That's awkward, because you'd then need a weight trading the two off. Instead the deadline is folded into the *same units* as everything else. Define one number:

> **`loss_penalty`** — what it costs to run out of turns, charged on top of the six guesses already spent.

So a lost game scores `6 + loss_penalty`. Setting `loss_penalty = 1` makes a loss score exactly **7**, i.e. "as if you'd needed one more guess than you had." That is the **risk-neutral** setting: minimizing expected score then means exactly what it says, and losing carries no special horror beyond the turn it cost you.

In the value recursion this appears as the base case: with zero turns left, the position is worth `loss_penalty` (`exact.py`). No separate deadline objective, no risk weighting — one recursion covers both "win fast" and "don't lose."

Raising the penalty above 1 buys solve rate at the expense of the mean, by making the search willing to spend a turn on safety. **The default is 5**, and §14 explains why that number and not 1 — but the value is just a knob, and every formula below holds for any setting of it. Note that it is a *decision-time* price only: measured results always score a lost game as 7, whatever the penalty was set to, or rows run at different penalties could not be compared.

## 12. The endgame: stop estimating, start solving

`f` is a fitted approximation. But when few candidates remain, you don't need approximation — you can compute the true answer by brute force (`exact.py`).

$$V(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r V(S_r, t-1)\Big], \qquad V(S, 0) = \text{penalty}$$

`t` is turns remaining. This is **expectimax**: you *minimize* over your guess (you choose it) but take an *expectation* over the feedback (you don't). Not minimax — there's no adversary, just chance.

Naively this is hopeless: 12,947 guesses × 243 patterns × recursion. Four things make it affordable.

**(a) Partition dedupe.** This is where §2 pays off: a guess *is* its partition, so two guesses that split `S` identically are perfectly interchangeable, and the branching factor is the number of *distinct partitions* rather than 12,947.

Grouping guesses that way means comparing, for each pair, the whole list of patterns they produce over the candidates — the guess's **feedback vector**, one number in `[0, 243)` per candidate, which is just the piece of the guess's table row that the candidate columns pick out. Comparing those directly, 13,000 of them against each other, is too slow. So each vector is squeezed down to a single 64-bit integer and the integers are grouped instead.

With seven candidates or fewer the squeeze loses nothing. A pattern fits in 8 bits, seven of them fit in the 56 bits available, and the packed integer therefore *is* the vector: equal integers are identical partitions, and nothing needs checking. Above seven the vector no longer fits and the integer becomes a hash — FNV-1a, a cheap mixing function that folds an input of any length down to 64 bits. Identical vectors always hash alike; different ones almost never collide, but "almost" is not a proof, so every guess is compared once against its group's representative and promoted to a group of its own if it really does split differently (`kernels.distinct_partitions`). A collision costs one redundant representative and can never lose a distinct partition.

The hash is worth defending, since the direct approach is right there: `np.unique` can group the feedback *vectors* themselves, which needs no collision check at all. It is a lexicographic sort, one pass per candidate, and it costs an order of magnitude more than hashing each vector to 64 bits and grouping those. The collision check buys the exactness back for one extra comparison per guess.

Nice safety property: a guess that could itself be the answer *always* splits distinctly, since only it produces `GGGGG` — so a real candidate can never get merged away into a probe. (Which is also why hard mode skips this step entirely: if every guess must be a candidate, no two of them can split `S` the same way, and there is nothing to merge.)

This only bites while `S` is tiny, though: at 12 candidates, 12.7k of the 13k guesses already split differently, so beyond that the branching factor is effectively the whole word list. That, not the node count, is what makes the occasional deep position slow.

**(b) Closed-form leaves.** With two turns left, no search is needed at all. On the last turn the best you can do in a bucket is play its heaviest member, worth `1 + (1−p_max)·penalty`, so

$$V(g,2) = 1 + \frac{A + (A - B)\cdot\text{penalty}}{\text{total}}$$

with `A` the mass and `B` the sum of bucket maxima over non-winning buckets. That's *one pass over the guesses* — exactly what heuristic scoring costs. So **two turns or fewer is always solved exactly, for free, at any candidate count.** The same two-turn position is reached along many different guess orders, about four calls in five being a repeat, so even this closed form is cached rather than recomputed (§13). Note that the penalty in it is a parameter rather than a constant. Called as a leaf it is handed the real one and returns the true value of the position, which is the whole point of a leaf; the next idea calls the same code with a different one, for a different purpose.

**(c) Branch and bound.** That closed form, evaluated at a *lower* penalty, doubles as a **lower bound** on `V(S,t)` for every larger `t`. Write `V₁` for the value computed with the penalty set to 1 and `V₅` for the shipped setting. What the search relies on is

$$V_1(S, 2) \le V_5(S, t) \qquad \text{for every } t \ge 2,$$

which splits into two steps, and both of them need that 1.

*Charging less for a miss cannot raise the value.* Fix any policy whatsoever and score its play twice, once charging 1 for a loss and once charging 5. The second charge is larger on the paths that lose and identical everywhere else, so no policy scores worse under the cheaper penalty, and neither does the best of them: `V₁(S,t) ≤ V₅(S,t)`.

*At a penalty of 1, turns in hand cannot make a position cheaper.* This one reads backwards, since more turns ought to be worth having. They are not here, because running out of them is an *exit*. A position two turns from the deadline may take its last shot, miss, pay 1, and stop; a position with four turns in hand has to carry on until it actually wins, and no continuation finishes for less than the single guess that exit charges. Being allowed to leave is therefore never worse than playing on, and `V₁(S,2) ≤ V₁(S,t)`.

Chain the two and the two-turn value at penalty 1 sits under every deeper value at the penalty actually in force, which is what makes it safe to prune with.

**The cap is not a safety margin — without it the bound is simply false.** At a penalty of 5 the second step reverses outright, because that exit now costs five guesses rather than one and a position near the deadline would much rather have the turns. Take the ten words ending in `-atch`, a real position and a familiar trap. Under the shipped penalty it is worth 1.474 guesses with four turns in hand and 1.556 with three — low numbers, because the prior puts 78% of the family's mass on `hatch` alone. With only two turns it is worth **1.986**, since the last turn is a guess at whatever the first one failed to resolve and every miss is charged at 5. A search that pruned the three-turn position because that 1.986 already beat its incumbent would be discarding a subtree whose true value is 1.556. Recompute the same three numbers with the penalty at 1 and they come out 1.371, 1.459 and 1.462 — rising with `t`, which is the order the bound requires. Hence `ExactEndgame.bound_penalty`, which is `min(loss_penalty, 1)`, and hence its use in the bound and nowhere else; `analysis/loss_penalty.py --bound` prints both rows.

The search then does the standard thing with it. Guesses are ordered by their bound and walked in that order, holding on to the best complete value found so far — the **incumbent**. Within a guess, the total starts as the sum of its buckets' bounds and each bucket is replaced by its exact value one at a time, so the running total only ever rises; the moment it passes the incumbent, the guess is abandoned unevaluated. Biggest-mass buckets go first, since they move the total fastest. All of it is sound because the bound is **admissible** — it never overestimates, so nothing it prunes could have turned out to be the best.

**(d) Branch cap** (default 50). Only the 50 best splits by that bound are tried at all. The justification is sharp: branch and bound finds the best guess early and then spends nearly all its time *proving* nothing beats it. The cap buys the answer and skips the proof. Over 1,275 real positions, caps of 25 and 50 pick the uncapped-optimal guess **every single time**, while wall clock drops 322 s → 8 s and the worst single position from 30 s to 1.9 s. The honest caveat, which the code states: the returned value is an upper bound on the true one, so this is optimal-within-the-cap, not provably optimal.

Two details of the cap are easy to miss and both bite. The first is that it is decided by a *sort*, and the bounds tie in bulk — one real position holds 1,176 of its 1,177 splits at exactly 2.0 — so which fifty are kept is settled almost entirely by the tie-break rather than by the bound. Order those ties arbitrarily and two runs of the same search examine two different sets of guesses. `exact.py` and `exact.rs` therefore both sort stably, which puts a tie in word order, and the whole reason the Python and the Rust return identical answers rests on that plus a matched summation order in the weights the bounds are built from.

The second is that a *ranking* wants a wider cap than a *pick*. The page shows five guesses, and while the winner sits comfortably clear of the cap the runners-up sit right against it. Measured against the uncapped search over two independent samples of 80 positions each, a cap of 50 gets every winner right and still names a worse runner-up in three of the first sample and one of the second, giving up 0.0055 and 0.012 guesses. Eighty is the smallest cap that reproduces the uncapped list exactly on both, so `ExactEndgame.root_branch` is 150 — the same doubling that put the ordinary cap at 50 — and it applies only at the node being ranked, since every node beneath it needs its winner alone.

## 13. Remembering positions

The four ideas above cut the cost of solving a position. This one is about not solving it twice, and it is the largest single speedup in the project.

The reason it works is a fact about the game rather than about the code. Fix the opening word, and the answer decides the feedback — but feedback only comes in 243 varieties, so **all 12,947 possible answers lead to at most 243 distinct positions** at turn two. Playing every word as the answer to measure an opening therefore looks like a 12,947-wide computation and is really a ~240-wide one, repeated. The same collapse happens at every depth: many different paths arrive at the same candidate set.

So `memo.Memo` caches results keyed on the position. The key is `candidates.tobytes()` — the raw bytes of that array of column numbers from §1, used because an array cannot be a dictionary key and its bytes can, and because they are exact where a hash of them would not be — together with the number of turns left, since the right guess genuinely depends on how many are remaining and a value found with four turns left is wrong for two.

There are three of them, and they do not all expire together. `WordleSolver` keeps two, one for the guess it chose and one for lookahead values, and both are dropped whenever `f` is refitted. `ExactEndgame.memo` is not, and that asymmetry is worth noticing: the search returns true values, which depend only on the prior weights and never on the estimator, so a refit leaves every one of them correct.

### Why it evicts rather than jams

A cache needs a cap or it grows without limit. The obvious way to write one is to stop accepting entries once full:

```python
if len(memo) < limit:       # the tempting version
    memo[key] = value
```

That is the wrong way round, and the wrongness is not gradual. What such a cache declines to store are the *most recent* positions, which are exactly the ones the run in front of it is about to reuse; it ends up holding a frozen souvenir of whatever it happened to see first. Cross the limit and the ~50× collapse above disappears all at once.

`memo.Memo` instead drops the **least recently used** entry — LRU, the standard policy of evicting whatever has gone longest without being touched, on the reasoning that it is the least likely to be wanted next. It is a dictionary that remembers the order its entries were last touched: a read moves that entry to the newest end, and an insertion past the cap drops whatever sits at the oldest. That a *read* counts as a use is the whole difference between LRU and simply dropping the oldest insertion, and it is what keeps a position that is looked up constantly resident however long ago it was first computed.

The bookkeeping is not free — every read moves an entry, which is a couple of pointer writes — but the comparison is lopsided. Moving it costs on the order of 100 ns, while the miss it prevents costs a scoring pass over ~13k guesses, several milliseconds. The overhead is roughly ten thousand times smaller than the thing it buys, which is not a close call.

One consequence for anything that sweeps many configurations: a finished run's positions are dead weight, and under LRU they are evicted only after competing with live ones for a while. That is why `WordleSolver.reset_memo` exists, and why it clears only the two memos that depend on the estimator — the search's entries stay valid across everything and stay worth sharing.

## 14. Why the penalty is 5 and not 1

Risk neutrality (`penalty = 1`) is the principled default, so it takes evidence to move off it. Start with what the losses are. They aren't endgame blunders: at the sizes these positions reach, `exact.ExactEndgame` solves them outright, so the guess played is optimal. The solver is backing its prior. Holding `batty`/`catty`/`natty`/`patty`/`ratty`/`tatty` with four turns left it plays `tatty`, which carries 75% of the family's mass, and it turns down `bronc` — a word that splits the six of them five ways — because separating spends a turn to insure against a 0.4% case. A loss is the prior being confidently wrong, and the penalty is what makes that insurance worth buying:

| penalty | loss probability | mean (loss = 7) |
| --- | --- | --- |
| 1 | 0.1168% | 3.6621 |
| **5** | **0.0458%** | **3.6626** |
| 30 | 0.0357% | 3.6635 |

Three fifths of the loss probability vanishes for **+0.0005 guesses**. Widening `--exact-max` from 24 to 60 adds the rest, and the two together — risk-neutral play at the narrow reach against the shipped pair — run **0.1198% → 0.0458%, a 62% reduction, with the mean improving by 0.0019** on the way, because reaching deeper replaces an estimate with a true value instead of trading one good for another.

That last column can't pick the penalty, though, and the reason is instructive. Scoring a loss at 7 is precisely the objective `penalty = 1` optimizes, so ranking settings by it can only ever favor risk neutrality — the comparison is circular. Name the price of a loss instead, `L` guesses, and every setting becomes a line in `L`: penalty 1 wins below `L` = 7.1, **5 from 9.0 to 13.7**, and 30 above 22.1. So 5 is right if a lost game costs between nine and fourteen guesses, and little rides on it, since across that whole range the best setting beats the next by under 0.0002 guesses.

Losses collect on the never-curated words — 290 of the 300 the shipped prior can lose on. Wordle is provably solvable in five guesses, but that theorem is about a solver that knows the answer is one of the 2,309 curated words, and this one spreads 9% of its mass over the other 10,638. Swept uniformly over the curated list it loses 10 of 2,309, every one of them a word that has already been an answer — which is §5.1's repeat prior calling them unlikely, by construction.

Every piece of the guess rule is now on the table; §15 puts them in the order the solver applies them.

## 15. Putting it together — one turn of play

```
candidates = all 12,947 words
each turn:
    if only 1-2 candidates          -> closed form
    elif exact search can reach it  -> exact.solve()          (true optimum)
         (turns <= 2 always; or <= 60 candidates deeper)
    elif last turn                  -> heaviest candidate
    elif lookahead is on            -> value the replies first (§16, off by default)
    else                            -> argmin over 12,947 of E(g,S)
    show guess, read feedback, filter candidates
```

`argmin` is "the argument that minimizes": not the smallest score but the *word* that achieves it, since what the solver needs back is a guess. It is one call to `kernels.expected_guesses_all` and one `.argmin()` on the array it returns.

Every line that says 12,947 is the free choice of guess, and hard mode is the one variation on this dispatch: it replaces that pool with the candidate set itself, in the exact search as much as in the `argmin`, and changes nothing else about which branch runs. `f` stays the one fitted for free play, so its scores read slightly optimistic under the restriction — it is pricing a probe the solver is no longer allowed to make.

Under that dispatch sits most of this document. The lookup that reads the feedback is §1's table; the `argmin` line is §7's rearranged entropy inside §8's blocked, parallel, twice-scanned loop; `exact.solve` is §12's four ideas; and every branch consults §13's memo before doing any of it.

The heuristic `E(g,S)` is genuinely a **fallback**: it only runs where the game is too wide to solve outright. With both priors on and the shapes that ship, over every word the answer could be, the whole thing averages **3.0770 guesses** and opens with `salet`.

That average is what every part above adds up to, and it says nothing about which of them earned it. Splitting it is §16's job, and the split does not fall where the work went.

## 16. The result worth framing

This is the thread running through the whole project, and it's the most interesting thing in it:

> **A better map, but no better score.**

One-ply lookahead values positions **ten times more accurately** than `min_g E(g,S)` — RMSE 0.0111 vs 0.1122, measured against ground truth computed by playing positions out exactly. The accuracy is real and it does reach the decisions: lookahead disagrees about the guess in **half** of all games, and it is right to disagree, since its picks give up 0.0013 guesses against the estimator's 0.0115. Then the games are played out and none of it arrives. The outcome moves in a quarter of them, **2,644 better against 2,597 worse**, for a net of −0.0034 ± 0.0041 guesses at ~10× the wall clock. That is 0.9σ: the difference measured is smaller than the error bar on it, so nothing here is distinguishable from zero.

The tempting explanation is that you're taking an `argmin`, so the estimator's errors are **common-mode** across the guesses at one position and cancel in the comparison, leaving the ranking intact. They don't. The estimator ranks the ten guesses lookahead re-ranks at a **rank correlation** of **+0.54** with their true values, and picks the best of them just over half the time. (Rank correlation asks whether two orderings agree, not whether two quantities lie on a line, so it is not the `r` of a least-squares fit: it is computed from positions in the sorted order, and +1 means identical rankings, 0 none at all. It is the right question here because the solver only ever uses `f` to *sort* guesses.) A valuation whose errors canceled would not be overruled half the time by an accurate one.

What is true instead is that the top of the ranking is a **plateau**. Those ten guesses spread 0.0941 guesses between best and worst, and the estimator's pick gives up 0.0115 of that, an eighth of the range, while ordering the middle of it almost at random. It finds the plateau reliably and can't tell one spot on it from another, which costs a hundredth of a guess because there's little there to tell apart. `analysis/lookahead_accuracy.py --decisions` is where those come from.

The endgame search makes the same point from the other end and makes it more sharply, because where the search reaches, its value is not a better estimate but the truth: a disagreement there is the estimator being wrong, full stop. Over the positions it takes over, the estimator picks a different guess **38.8%** of the time and gives up **0.0053 guesses** doing it. At three and four candidates every single disagreement is a tie: a different word of equal value, nothing to improve on. The ties thin as the set grows, and by 11–60 candidates most disagreements are real, which is the whole of that 0.0053. The same plateau, found from the other side.

The opening is the third case and the starkest, because `f` decides that one alone — no search reaches a position of 12,947 candidates. Played out, `salet` is the best of the two hundred openings `f` likes best, at **zero regret**, where regret is what a choice costs against the best available: the true value of the pick minus the true value of the best option, so zero means nothing on the shortlist would have been better. Yet `f`'s ranking of the twelve leaders correlates with the truth at only **+0.371**, against +0.724 across the full two hundred. That is the shape of the whole result. `f` sorts the field coarsely and then cannot sort the top of it, and every decision it gets right is a decision where coarse was enough.

Turn-conditioned `f` is the fourth case, and it never stood a chance. A curve conditioned on turns remaining differs from the flat one only where the deadline binds, and `f` is consulted **once in 11,338** positions with three or fewer turns left — the search has taken every other one. The +0.0005 ± 0.0007 it measured is a fact about an implementation since removed; the reason it could not have come out otherwise is structural, and still holds.

What the solver has been short of is **somewhere better to go**, not a better map.

The exceptions are what sharpen the rule, and they have a common shape: every one of them changes *what the solver is choosing between* rather than how well it scores the choices.

The shapes of §5 are the first. Together they are worth −0.064 ± 0.032, small in itself and still more than deeper search and better lookahead managed between them. A prior changes the quantity being optimized — which words are plausible answers at all — so it is a decision-side change wearing valuation-side clothes.

The sharpened repeat prior taken alone (−0.022 ± 0.024, despite being 5.4σ better as a prior) belongs here rather than above, and the reason is worth being precise about. Changing the weights changes what a position is *worth*, so the two arms have no common ground truth and no scale on which one guess is better than the other. A better model of the editor is not a better estimate of a fixed quantity; it is a different quantity.

Hard mode is the same thing in its bluntest form, and it makes the distinction hard to argue with. It improves no valuation whatsoever; it deletes words from the choice set. Under greedy entropy that is worth **−0.2505** guesses, against the nothing at all that one-ply lookahead buys by valuing positions ten times better. Take a choice away from a policy that was pricing it wrong and you gain more than by pricing everything else right. Under the expected-guesses objective, which already prices it correctly, the identical restriction *costs* +0.0269 — same intervention, opposite sign, entirely because of what the objective was already getting right.

That is the honest reading of the headline number too. The 0.67-guess gain over greedy entropy is 0.63 from the possible-answer bonus and 0.04 from curvature in `f`. And the bonus is the *prior* showing up inside the objective — `p_win` is nothing but the mass the prior puts on the word being guessed, which is why the split moves with how sharp the prior is. So is the curvature, which §9's table shows falling to nothing wherever the curation prior is off. The search proper is barely in that 0.67 at all: switching the whole of `exact.py` off costs 0.0003 guesses on the mean, since what it moves is the loss probability instead (§12).

Modeling the editor beat improving the search, every time.

## 17. Why any of this finishes

One thread has run underneath all of the above without being named. Choosing a guess means scoring ~13k words against a candidate set, the training loop does that a few million times, and `analysis/` sweeps run it over every word in the list — none of which would terminate on the naive reading of any formula here. Five ideas do the work, and each shows up more than once.

**Precompute whatever does not vary.** The feedback for a (guess, answer) pair is fixed forever, so it becomes a 167 MB table — or, where that cannot be held, the 440 KB of it one search will ever read (§1). `w log w` depends on the candidate but not the guess, so it leaves the inner loop entirely (§7). `f` is always reached through an entropy, so it takes `x = log₂ n` rather than `n` and saves a round trip through exponentiation (§7). In each case the trick is noticing that a quantity inside a doubly-nested loop only depends on the outer index.

**Cache whatever recurs.** Positions repeat constantly — at most 243 of them behind a fixed opening — so results are memoized on the candidate set, which is worth about 50× and is the difference between an overnight sweep and a coffee break (§13). The compiled kernels are cached to disk (§1), and the fitted `f` to `estimator_cache/` (§10), for the same reason at longer timescales.

**Bound before you compute exactly.** The two-turn value is exact, costs one pass, and — evaluated at a capped penalty — is a valid lower bound on every deeper value, which makes it both a leaf and the ordering that prunes the search above it (§12). The branch cap then takes the observation one step further, cheerfully abandoning the *proof* of optimality while keeping the answer.

**Collapse things that are secretly the same.** §2's reframe — a guess is nothing but its partition — is what lets the search treat thousands of words as one move (§12). Recognizing an equivalence is usually cheaper than exploiting a numerical shortcut, and it never costs accuracy.

**Measure where the work actually is.** Two assumptions had to be given up here: that wide positions dominate (small ones are scored far more often), and that the cost is arithmetic (it is memory traffic, one cache line per scattered lookup). Both changed what the right optimization was, and the second is what stopped the search for further ones (§8).

The through-line is that none of these made the solver cleverer. They made a fixed amount of thinking affordable enough to do at the scale the measurements in this document needed — which is its own version of §16's lesson, from the engineering side.

---

## Where to look in the code

In reading order:

| where | what it establishes |
| --- | --- |
| `wordle_data.score_guess` | the feedback rule and the precomputed table (§1) |
| `source::Columns` | the tab-sized table, where the full one cannot be held (§1) |
| `priors.build_weights` | the weight vector over answers (§4) |
| `priors._repeat_shape` | the repeat shape, binned by first appearance (§5.1) |
| `priors._curation_shape` | the curation shapes, WordleBot and frequency (§5.2, §5.3) |
| `kernels._ig` | the entropy scorer (§3) |
| `estimator.effective_size` | effective size, and the split-logarithm rearrangement (§7) |
| `kernels._expected_guesses` | the expected-guesses scorer, and the scoring loop (§8, §9) |
| `estimator.FORMS`, `estimator.fit` | `f`, its functional forms, and the monotonicity guard (§9, §10) |
| `train.policy_iteration` | policy iteration (§10) |
| `exact.ExactEndgame` | the endgame search, and the loss penalty (§11, §12) |
| `memo.Memo` | the bounded LRU every position cache uses (§13) |
| `solver.WordleSolver._best_guess_uncached` | the dispatch deciding which of the above runs (§15) |

Named rather than given as line numbers, which drift with every edit. Every entry is greppable: `grep -rn "def score_guess" .`
