# How to read this document

This file is Markdown with LaTeX display equations, so it needs a viewer with math support. `ABOUT_SOLVER_EXPLAINED.md` says how to get one, who this is written for, and what order to read it in.

Read it start to finish the first time. Each section builds on the one before. If you have ten minutes: §2, §6, §15.

---

# The Wordle solver, from the ground up

## 0. What the program has to do

Every turn, the solver holds a set of words that could still be the answer. It plays a guess, sees five colored tiles, and removes every word the tiles have ruled out. The whole program answers one question:

> **Which guess shrinks the set the most usefully?**

Entropy, the fitted curve `f` and the endgame search are all machinery for answering that question.

Two word lists matter. The solver may play any of **12,947** words, and it treats every one of them as a possible answer (`ind2word.json`). Inside that list is the **2,309**-word answer list Wordle shipped with in 2022 (`wordles.json`), which is where the New York Times editor draws from most days. Most of this document is about the gap between a word being playable and a word being a likely answer.

Three things have to exist before the question can be asked precisely: a way to compute the tiles (§1), a way to describe what a guess does to the candidate set (§2), and a way to measure how good that is (§3).

## 1. The feedback rule, and why there's a 167 MB file

Wordle's coloring rule runs in two passes. Greens go out first, one for each letter in the right place. Yellows are then handed out from the letters the answer has left over, at most one per remaining copy.

The order matters when a guess repeats a letter. Two examples against the answer `abide`, both checked with `wordle_data.score_guess`:

```
guess  speed   ->  N N Y N Y      guess  geese   ->  N N N N G
       s p e e d                         g e e s e
```

`speed` has two E's and `abide` has one. The first E takes the yellow and the second gets nothing, because the answer has no second E. The D is yellow because it is in the answer but not in position four.

`geese` has three E's. The E in position five is an exact match, so it goes green. Greens are assigned first, so it consumes the answer's only E, and the two earlier E's get nothing.

A pattern is five tiles of three colors, so write it in base 3 with N=0, Y=1, G=2. Every pattern is then an integer in `[0, 243)`: `NNYNY` is 10, `NNNNG` is 2, and `GGGGG` is 242 (`WIN_PATTERN`).

The feedback for a (guess, answer) pair depends only on the two words, so it can be computed once and stored:

```
feedback_table[guess_index, answer_index] = pattern number   # one uint8
```

That is 12,947² bytes, about 167 million, which is the `feedback_table.npy` file. Building it takes a while. After that, finding what a guess would show is one array lookup. The trade is the same as a prefix-sum table's: a quadratic cost paid once, then a constant-time query. Nearly every inner loop in this project walks along one row of that table.

The table is a speed choice. Applying the rule to a pair costs a few times what a lookup does. On a machine that can hold the file, the file wins. In a browser tab running the solver compiled to WebAssembly, 167 MB can be neither downloaded nor rebuilt, so the rule is run instead.

### What the program is actually holding

Three later arguments depend on the shape of the table in memory: the rearrangement and the memory bill in §8, and the packed keys in §10. This section sets that shape down.

After startup, words are never compared as text. Each of the 12,947 is given an index once, and from then on the program refers to a word by its index. `ind2word.json` is that numbering and `word2ind.json` is its inverse. So:

- the table is a 12,947 × 12,947 grid of single bytes, where **the row is the guess and the column is the answer**;
- a guess is a row number, and scoring every guess means visiting all 12,947 rows;
- the candidate set is an array of column numbers in ascending order, such as `[3, 17, 964, …]`. It starts as all 12,947 and gets shorter every turn as feedback rules words out (`kernels.filter_candidates`).

Filtering keeps the order but not the spacing. By turn three the survivors are forty-odd columns spread across a row of thirteen thousand.

Two consequences follow.

**The table is read along rows.** A row is contiguous in memory, so scanning one for a whole candidate set is a straight walk. Scanning it for a scattered handful is not, and §8 measures the difference.

**The loops that scan it are compiled.** Every function in `kernels.py` carries numba's `@njit`, which compiles it to machine code the first time it runs. Python's own loops would be a hundred times too slow for thirteen thousand guesses against thirteen thousand candidates. This document calls one such compiled scoring loop a **kernel**, and `kernels.py` is a handful of them. The compiled code is cached to disk, so only the first run of a fresh checkout pays for compilation. That compile, plus loading the table, is the startup wait the web page hides behind its veil.

## 2. A guess is a partition

Fix the candidate set `S` and a guess `g`. Every candidate in `S` produces some pattern against `g`. Group the candidates by pattern. With `crane` as the guess and every word a candidate:

<!--table results_tables.crane.table-->
| pattern | candidates |
| --- | --- |
| `NNNNN` | 1,573 |
| `NNNNY` | 1,295 |
| `NNYNN` | 1,197 |
| … | (168 non-empty buckets) |
| `GGGGG` | 1 (`crane` itself) |
<!--/table-->

The pattern that comes back picks one of those buckets, and that bucket becomes the new candidate set. **A guess is a partition of `S` into at most 243 buckets**, and the only thing that matters about it is the shape of that partition.

Two words that partition `S` identically are the same move, and §10 relies on that to keep the endgame search affordable.

A bad guess leaves one large bucket, so most of `S` survives whichever pattern comes back. A good guess splits `S` into many small buckets, so every pattern rules most of it out. The next section makes "many small ones" precise.

## 3. Entropy, from scratch

This is the one piece of new mathematics in the document. This section builds it from a simpler question.

Suppose I am thinking of one of `n` equally likely words and you may ask yes/no questions. How many do you need?

With binary search each question cuts the set in half, so **log₂(n)** questions do it. Call that log₂(n) bits of uncertainty. A thousand words is about ten bits. The phrasing converts the size of a set into the number of questions it costs.

Now make the words unequal. Suppose one has probability ½, another ¼, and four more have 1/16 each. Counting says log₂(6) ≈ <!--c-->2.58 bits, which overstates it. Half the time the first question, "is it the first one?", finishes the game, and only the rare words cost several questions.

So weight each outcome by how often you pay for it. An outcome of probability `p` behaves like a set of size `1/p`, which is log₂(1/p) = −log₂(p) bits. Averaging over the outcomes gives

$$H = -\sum_i p_i \log_2 p_i$$

In the example the half-probability word costs 1 bit, the quarter-probability word 2, and the four rare ones 4 each. That comes to ½·1 + ¼·2 + 4·(1/16·4) = **2 bits**, under the <!--c-->2.58 that counting claimed. Concentrated probability means less uncertainty. `H` is largest, at log₂(n), when everything is equally likely. It falls to 0 when the answer is already known.

**Information gain** is the change a guess makes to that number. Before the guess the uncertainty is `H(S)`. After it you are in bucket `r` with probability `p_r`, with `H(S_r)` of uncertainty left. The gain is the difference:

$$\mathrm{IG}(g) = H(S) - \sum_r p_r H(S_r)$$

Knowing which word it is amounts to knowing which bucket plus knowing which word inside that bucket. Uncertainty splits the same way:

$$H(S) = \underbrace{H(p_1, \ldots, p_k)}_{\text{which bucket}} + \sum_r p_r H(S_r)$$

Substitute that into the line above and the bucket terms cancel. **The information a guess gains is the entropy of the bucket masses.** Nothing inside a bucket has to be examined, so the computation is cheap. `kernels._ig` computes exactly this. It totals each pattern's mass, where a bucket's mass is the combined weight of the candidates landing in it, then accumulates `h -= p*log2(p)` over the buckets.

`--mode entropy` plays the guess whose partition has the highest bucket entropy. It is the standard solver, the one most people build, and here it averages **<!--n results_tables.both.entropy .2f-->3.76 guesses**. Every measured figure in this document, that one included, is scored under the prior §4 constructs. The same rule scores worse under a flat prior, and §4 explains why the two scores cannot be compared.

That <!--n results_tables.both.entropy .2f-->3.76 is the baseline, and the rest of the document is about beating it. There are two ways to try. One is to search harder: look further ahead, solve the endgame exactly. The other is to correct an assumption in §0's question, that every word is equally likely to be the answer. §15 measures both, and the second is worth far more.

## 4. Not all words are equally likely (the priors)

Greedy entropy treats all 12,947 words as equally plausible answers. Correcting that buys more than any search improvement in the project.

`priors.build_weights` models the editor as making two decisions in sequence. With probability **0.10** she reuses a word that has already been an answer. Otherwise she picks an unused word, and with probability **0.10** that word is **off** the curated list. That gives three groups: used, unused-and-curated, unused-and-non-curated. Every word keeps strictly positive weight, so nothing is ever ruled out entirely. The second rate is conditional on the word not being a repeat, because the model is sequential. Measured against all puzzles instead of against non-repeats, it comes out lower than it should.

Both rates look too high against Wordle's full history, which contains a regime change. Over the current editorial regime, Wordle #1690 onward from 2026-02-02, repeats run at **<!--n regime.since.repeat .1f-->10.5%** and off-list answers at **<!--n regime.since.offlist .1f-->13.2%**. Over all <!--n regime.answers-->1,917 puzzles ever published both rates would be a fraction of that, because for years neither happened at all.

The change has a mechanical cause, which is that the curated list is nearly spent. About <!--n regime.curated_used-->1,835 of its 2,309 words have been used, and roughly <!--n regime.fresh_per_year .0f-->328 more go every year, which leaves about a year and a half of runway. Repeating old answers and reaching outside the list are two responses to the same shortage, and they started together. The 0.10 describes the year ahead, and the full history would put it far lower.

**The prior is where the wins are.** Playing every word as the answer, each weighted by that row's own prior:

<!--table results_tables.entropy_table-->
| priors on | greedy entropy |
| --- | --- |
| none | 4.12 |
| repeat only | 4.10 |
| curation only | 3.85 |
| both | 3.76 |
<!--/table-->

The column is not a scoreboard. Each row weights the words by its own prior, so a sharper prior scores an easier game, and no two rows describe the same population. The column shows how much of the difficulty was in the belief and how much in the board. There is one column because greedy entropy is the only rule built so far. What the prior is worth to a solver that minimizes guesses is shown in §6 and checked in §14 against the puzzles the NYT set.

One reading survives that caveat, because it compares a prior with itself. Switching the repeat prior on moves <!--n results_tables.none.entropy .2f-->4.12 to <!--n results_tables.repeat_only.entropy .2f-->4.10, two hundredths of a guess on a game whose difficulty that same prior set. A belief that barely eases its own examination is doing little work. §14.3 finds that the repeat prior alone is nearly idle and that its job is to correct the other one.

The three groups say how much mass each group gets. They say nothing about how the mass spreads within a group. A flat spread is wrong in both directions. Flat says `cigar`, the first Wordle, is no likelier to come round again than a word played last month. It also says `squid` and `xylyl` are equally plausible as tomorrow's answer. The second mistake costs more. The non-curated group holds 10,638 of the 12,947 words, so it dominates the candidate count at nearly every position while carrying under a tenth of the mass. Call the within-group spread a group's **shape**. §5 builds one for each group.

### How big is a weighted set?

The weights break an assumption §2 and §3 relied on. Once the words are not equally likely, a bucket's count says little about it. Forty candidates with one at 90% is a much easier position than forty even ones. Everything from here on needs a size that knows the difference.

§3's machinery supplies one. Call it the **effective size** of a set (`estimator.effective_size`):

$$n_{\text{eff}} = 2^{H}$$

where `H` is the entropy of the bucket's normalized weights. If the weights are flat, `H` is log₂ of the count and `n_eff` is the count exactly. As probability concentrates, `n_eff` falls toward 1. Read it as the number of equally likely words that would be this hard.

From here on, the size of a position means its effective size. The same measure gives the size of a prior, which §5 uses first.

## 5. Shaping the mass inside each group

### 5.1 The repeat group: a smoothed histogram

There are <!--n drift.events-->24 repeats in Wordle's history so far. They are not drawn evenly from the used words. <!--n drift.from_first_300-->22 of them come from the first three hundred puzzles, with a median around #<!--n drift.median_source .0f-->129. Two sit outside that window: <!--s drift.outliers-->`sandy` (#1227), `smile` (#831)<!--/s-->. The editor appears to be replaying the beginning of the run.

Two in <!--n drift.events-->24 is too high an outlier rate to write off everything past that window. So the shape is a **Laplace-smoothed histogram**. Bin the used words by the puzzle where they first appeared, two hundred wide, and give bin `b` the weight

$$w_b = \frac{r_b + \alpha}{n_b}, \qquad \alpha = 0.5$$

where `r_b` is the number of repeats that came from that bin and `n_b` the number of used words in it. Each bin's share is spread evenly over its members. The `α` keeps a bin that has never produced a repeat from being written off. It is why `sandy` does not break the model, and why no word's weight reaches zero. `cigar` ends up with nearly seven times the flat repeat weight, and the late words fall below it.

The bins are fixed. A sliding window over the recent past sounds more natural, but it needs a boundary, and a boundary is a parameter that <!--n drift.events-->24 events cannot both set and be predicted by.

**Prequential scoring** settles it. Predict each repeat using only the repeats before it, so no parameter sees the event it is graded on. Scored that way, the histogram beats a flat spread by about **<!--n repeat_shapes.gain +.2f-->+1.13 ± <!--n repeat_shapes.gain_se .2f-->0.20 nats per event**. The `±` is a standard error, the spread the estimate would show if the same <!--n repeat_shapes.events-->24 events were drawn again. The gap is <!--n repeat_shapes.sigma .1f-->5.5 times its own uncertainty.

The sliding windows lose. Scored the same way over the same <!--n repeat_shapes.events-->24 events (`analysis/repeat_shapes.py`), in total nats:

<!--table repeat_shapes.field_table-->
| model | log-likelihood | vs. a flat spread |
| --- | --- | --- |
| flat over the used words | −179.7 | — |
| **binned by first appearance, α = 0.5** | **−152.6** | **+27.1** |
| best of sixteen sliding windows | −161.5 | +18.2 |
| worst of those same sixteen | −276.9 | −97.2 |
| a repeat rate refitted as events arrive | −168.9 | +10.8 |
<!--/table-->

The sixteen windows are four choices of how many recent repeats fix the centre against four widths. The row above is the best of them, picked after seeing how each did. The histogram gets no such hindsight, and the best window still trails it by <!--n repeat_shapes.window_behind .0f-->9 nats across the <!--n repeat_shapes.events-->24 events, over a third of a nat apiece. A badly chosen boundary scores far below knowing nothing at all, so the parameter has to be right, and twenty-odd events cannot set it. A histogram has no such parameter.

### 5.2 The off-list group: WordleBot's own weights

Which words could be off-list answers turns out to be knowable, in two steps.

The first clue is in `solutions.json`. Its `id` field is a position in the list the editor draws from. For the first <!--n id_eras.exact_positions-->284 puzzles it is the word's place in `wordles.json`, which is stored in that order. Nearly every off-list answer carries an id just past the end of the curated 2,309. Sorted by id, those words come out in alphabetical runs: <!--n id_eras.run1.words-->12 at <!--n id_eras.run1.lo d-->2318–<!--n id_eras.run1.hi d-->2331 holding `<!--s id_eras.run1.list-->beaut laser mommy muggy nervy pious primp shred spate taupe tinge tizzy<!--/s-->`, then <!--n id_eras.run2.words-->8 more at <!--n id_eras.run2.lo d-->2343–<!--n id_eras.run2.hi d-->2352. Twelve words landing in alphabetical order by chance is a one-in-a-billion event.

A run is one batch. The NYT sorted a set of new candidate words and appended it whole, so the sort order survives in the numbering. Where a run ends and a word lower in the alphabet begins the next, that is the seam between two appends. The gaps inside a run matter too. <!--s id_eras.run1.gap-->2319 and 2320<!--/s--> sit between <!--s id_eras.run1.gap_words-->`beaut` and `laser`<!--/s--> and hold words nobody has seen, because a slot becomes visible only when its word is used. Three seams inside <!--n id_eras.block_lo d-->2310–<!--n id_eras.block_hi d-->2397 put the appended region at roughly ninety words. Off-list answers are drawn from those ninety, not from all 10,638.

The trail ends there, because the block cannot be enumerated. The NYT serves a puzzle by date, so a slot's number appears only once its word has been the answer. The unused half of the block is the half that matters, and it is invisible. Bracketing a gap alphabetically leaves four thousand candidates between `beaut` and `laser`. The test also no longer works on current ids. On 2026-02-02, the day the first repeat appeared and §4's regime begins, the NYT re-sorted the list alphabetically and continued numbering from its end. An id now places a word in the alphabet, not on one list or the other. `analysis/id_eras.py` reconstructs both numberings. List membership alone marks a word as off-list.

A list that names them already exists. **WordleBot**, the NYT's own solver, holds <!--n curation_shapes.wordlebot_words ,d-->3,194 words it treats as plausible solutions, each with a probability drawn from how often the word appears in the Times. Two things about its origin matter here. It is built by the paper's Upshot desk, not by the Games desk that picks the answers, and its probabilities are a frequency count over old articles. So the weights are evidence about English usage, not about the puzzle schedule. The NYT publishes neither the list nor the weights. `wordletools.azurewebsites.net/weightedbottles` reconstructs both from what the bot discloses as it plays, and `wordlebot.py` scrapes that reconstruction.

It covers a few thousand words, of which <!--n curation_shapes.listed_offlist-->892 are outside the curated list. All <!--n curation_shapes.events-->58 off-list answers to date are among them.

That fact proves nothing on its own. A list that absorbed answers after they were used would look the same. The test is whether an old copy predicts later answers. The Wayback Machine has a snapshot from **2023-10-12**, nearly three years old. That copy is vendored beside the live one, `wordlebot.py` fetches both, and `analysis/curation_shapes.py` scores them:

| the 2023-10-12 snapshot | |
| --- | --- |
| off-list answers used after it | <!--n curation_shapes.post-->54 |
| ...already in the snapshot | **<!--n curation_shapes.in_snapshot-->51** |
| the misses | <!--s curation_shapes.misses-->`loris`, `kefir`, `capon`<!--/s--> |
| words added to the list since | <!--n curation_shapes.added-->83 |
| ...of those, ever an answer | **<!--n curation_shapes.added_ever-->3** (the same <!--n curation_shapes.n_misses-->3) |

The old copy predicts forward at about <!--n curation_shapes.forward_pct .0f-->94%. The shape below is built on that number. Whether a miss was later backfilled is a question the archive settles for one of the three misses and not the other two. `capon` is settled. The live list vendored here on 2026-08-10 already carried it, and it was played on 2026-08-26, so the bot held it sixteen days before it was the answer. `loris` and `kefir` are not settled. Both are absent from the capture of 2025-05-25 and present in the one of 2025-09-16, with nothing archived in between, and both were used in that window. A fresher copy would not have helped with those two, since the 2025-05-25 list also lacks them. Scoring each answer against the closest snapshot that predates it gives the same <!--n curation_shapes.in_snapshot-->51 of <!--n curation_shapes.post-->54.

The likelier reading is that the bot held those two before they were played as well. It carries no hedge, since a word outside its <!--n curation_shapes.wordlebot_words ,d-->3,194 has probability zero, and the NYT says it has never lost a game. `capon` is one instance of that. For the two the archive misses, the question turns on what the bot does once its candidate set empties. Nothing here can measure that, so the reading stays a conjecture.

The shape follows from that number. **95%** of the off-list mass goes to the listed words in proportion to their weight. The remaining **5%** spreads over everything else, a hedge set at about the measured miss rate for the next `loris`.

Scored prequentially against the off-list answers, as §5.1 scored the repeat shape, with each shape set as it ships (WordleBot's hedge at 5%, the frequency shape at its shipped temperature):

<!--table curation_shapes.table-->
| shape | nats per event | effective support |
| --- | --- | --- |
| uniform | −9.27 | 10,638 |
| answer-like frequency (§5.3) | −8.06 | 4,857 |
| **WordleBot, 5% hedge** | **−6.45** | **911** |
<!--/table-->

Effective support is §4's effective size applied to the prior itself, the number of equally likely words the shape spreads over. WordleBot's shape is concentrated on about a thousand words where a flat spread covers ten thousand. The middle row is the fallback §5.3 builds, listed here so that all three can be compared. Scored at its own best-fitting setting, each of the lower two improves. The table uses the shipped settings because a shape has to be scored as it will run.

**The 5% hedge stays flat.** Weighting it by how answer-like a word looks is the obvious refinement, and it fails on the population that matters. The hedge fires only on a word the list is missing, and the list tends to miss odd words. `loris` sits mid-pack on answer-likeness and would have received <!--n curation_shapes.hedge.loris .2f-->0.32 of its flat share. `kefir` and `capon` would have taken <!--n curation_shapes.hedge.kefir .1f-->2.4 and <!--n curation_shapes.hedge.capon .1f-->3.3 times theirs. Over the <!--n curation_shapes.n_misses-->3 events that have needed the hedge, that nets to <!--n curation_shapes.hedge.net +.2f-->+0.31 nats apiece, which three events cannot tell from zero. The one that went the other way, `loris`, is the kind of word a list misses. A flat hedge expresses not knowing which word the list is short of.

### 5.3 The fallback that needs no scrape

The repo should not depend on somebody else's website, so there is a second shape built only from what the repo already holds.

The curated list is 2,309 labeled examples of a word an editor thought answerable, and those labels are independent of the <!--n curation_shapes.events-->58 off-list events. `answer_like.py` fits a logistic regression with the curated words as positives and the rest of the list as negatives, then scores the unused words by its log-odds. Thirteen features go in. Four of them carry the model:

| feature | weight | reading |
| --- | --- | --- |
| Zipf frequency | **+4.00** | common words get picked |
| ends in `s` | **−5.87** | a third of the pool is a plural; under a tenth of the off-list answers are |
| ends in `ed` | −5.09 | past tenses do not get picked either |
| foreign lift | −1.16 | as common abroad as at home means a name, not a word |

The other nine still matter. A model keeping only these four correlates at about r = <!--u-->0.98 with the shipped scores and reproduces none of them.

`squid` is promoted several times over the flat weight it replaces, along with `maven`, `aloha` and `kazoo`. All four are real off-list answers. `karks`, `fents`, `xylyl` and `faffy` are demoted by an order of magnitude. They are the kind of never-curated oddity §11's losses collect on. As a model it lands between the flat spread and WordleBot's list, in the middle row of §5.2's table. That gap is the price of needing no scrape.

With the weights built and a size that respects them, §0's question can be put in the units the game scores.

## 6. The real objective: minimize expected guesses

Entropy answers which guess is most informative. The game scores guesses, not bits. The gap between the two is worth about two thirds of a guess.

Suppose you had a function

> **`f(n)`** = the expected number of further guesses needed from a position of effective size `n`

Then the value of a guess is:

$$E(g, S) = 1 + \sum_{r \ne \texttt{GGGGG}} p_r f(n_r)$$

One guess for this turn, plus the average cost of wherever you land. Two details in that formula matter.

**Excluding the `GGGGG` bucket is why a possible answer beats an equally informative non-answer.** That bucket costs nothing more, because the game is already won. So a guess that might be the answer collects a discount equal to its probability of being it.

**Buckets of one and two skip `f`**, because their values are known exactly. One candidate is worth 1. Two are worth `2 − p_max`: guess the likelier one, and if it is wrong the other is known.

§7 says where `f` comes from. This section says what shape `f` has to have.

With both priors on, minimizing `E` beats maximizing information by about two thirds of a guess, from §3's <!--n results_tables.both.entropy .2f-->3.76 to <!--n results_tables.both.expected .2f-->3.06. With both priors off the same machinery buys a thirteenth of that. An identity explains why a better objective does so little under a flat prior.

### A straight-line `f` is entropy plus a bonus

Suppose `f` is **linear**: `f(n) = a + b·x`, with `x = log₂(n)`. Substitute it into `E` and apply the chain rule from §3:

$$E(g,S) = 1 + a(1 - p_{\text{win}}) + b\big(H(S) - \mathrm{IG}(g)\big)$$

`H(S)` does not depend on `g`, so minimizing `E` is the same as maximizing `b·IG(g) + a·p_win`. That is greedy entropy with a bonus. **A linear `f` buys nothing but the bonus.** Every remaining gain comes from `f` being concave in `x`.

The identity is testable. One of the fitted forms is linear in `x`, so it should score exactly as entropy plus the bonus. It does, so the gain over greedy entropy splits in two, shown here under every prior (`analysis/results_tables.py --part priors`):

<!--table results_tables.bonus_table-->
| priors on | possible-answer bonus | curvature in `f` |
| --- | --- | --- |
| none | +0.051 | +0.002 |
| repeat only | +0.090 | +0.003 |
| curation only | +0.317 | +0.039 |
| **both (the default)** | **+0.652** | **+0.040** |
<!--/table-->

In the bottom row, nearly all of the improvement over a standard entropy solver is the discount for guessing a word that might be the answer. The fitted curve's shape, which §7 works hardest for, is about a sixteenth of it.

Down the columns, both halves depend on the prior. The bonus climbs <!--n results_tables.bonus_ratio .0f-->13-fold as the prior sharpens, and §15 is about that. Curvature is worth about <!--n results_tables.both.curvature .2f-->0.04 wherever the **curation** prior is on and next to nothing where it is off. The fitted curve below shows why a prior decides whether a curve has anything to offer.

None of that bonus needs fitting. Written in its own terms the rule is `IG(g) + a·p_win(g)`, with no `f` in it. Two exact facts fix `a` without data. One candidate takes one guess and two take one and a half, so a bit of information is worth half a guess. Winning outright saves a whole guess, which is two bits' worth. A solver playing that rule collects nearly the same bonus with no estimator, no fit and no answer log. `README.md` carries it as a second reference and not as the baseline, because it pays for those guesses with a much higher loss rate.

The solver's **hard mode** is the cleanest test of the identity, because it isolates the bonus. It requires every guess to be a word that could still be the answer. That is narrower than the NYT's rule of the same name, which only requires the revealed letters to be reused. Restricting greedy entropy to hard mode forces `p_win` up without touching entropy's scoring rule. It gains about a quarter of a guess, roughly two fifths of what the objective gets by pricing the same bonus instead of imposing it.

The objective does better because it takes the bonus and keeps the probes. It plays a non-candidate whenever the split is worth more than the win chance it gives up, the trade hard mode forbids. The wider pool the NYT's rule allows shows the same thing from the other side. The objective declines those probes across all 12,947 answers, while greedy entropy takes <!--n hard_mode_rules.entropy.off_pool ,d-->3,099 of them and is worse for it.

Curvature is what is left once that bonus is priced, and some of it is expected before any fitting. A guess cuts a candidate set by a factor, and the factor available depends on how much there is to cut. Two candidates admit one bit at most, while a few dozen admit four or five, since one guess can send them to as many as 243 patterns. Bits per guess rise with the size of the set, guesses per bit fall, and `f` flattens as `x` grows. The piecewise-linear spline fitted for the default prior shows this. It is one of the forms `compare` archives, within a few thousandths of the fit that ships, and its shape can be read off:

```
f(n) = <!--s results_tables.spline.equation-->0.931 + 0.457·x − 0.248·(x−2)₊ − 0.096·(x−4)₊<!--/s-->
                                            (x = log₂ n, (u)₊ = max(u, 0))
```

A knot is a value of `x` where the curve may change slope. Both knots bend it flatter, taking the cost per doubling through <!--s results_tables.spline.slopes-->0.457, 0.209, 0.113<!--/s--> of a guess. Both bends sit below sixteen candidates, and above that the fit is a straight line. Beside it, the same form fitted under flat shapes:

<!--table results_tables.fn_table-->
| n | 2 | 4 | 16 | 100 | 1,000 | 12,947 |
| --- | --- | --- | --- | --- | --- | --- |
| `f(n)`, shipped shapes | **1.39** | **1.85** | **2.26** | **2.56** | **2.94** | **3.35** |
| `f(n)`, flat shapes | 1.33 | 1.79 | 2.28 | 2.58 | 6.38 | 13.35 |
<!--/table-->

The two halves of that table say different things. On the left both curves bend, and land within a rounding of each other. This is the part a linear `f` cannot express, and it is bought entirely in positions of a handful of candidates. The right half is a warning. The flat-shape fit kept a hinge of <!--n results_tables.flat_spline.top_hinge +.2f-->+1.82 at <!--c-->256 candidates, so it reads <!--n results_tables.flat_spline.f1000 .1f-->6.4 guesses at a thousand and climbs from there. A knot that far out is set by the few training positions beyond it, so the fit is extrapolating. It plays as well as any other form under that prior because the solver almost never asks about a set that wide after the opening. Concentrated mass keeps effective sizes small, so the positions the solver meets sit where the curve bends and where the data can hold it.

That answers the curvature column. A flatter prior does not leave `f` without a shape. Every fitted `f` in this repo is concave, including the ones fitted with the curation prior off. The one fitted with no prior saturates hardest, flat to three decimals by a thousand candidates. What those configurations lack is a use for the shape. A bend is worth only the guesses it reorders. Where the candidate sets are large and alike, a bend and a straight line rank the same guesses the same way. That last step is a reading of the fitted curves, not a measurement.

One caveat applies to all of this. `f` is fitted **per prior**, so the curve above is the default prior's. With a prior turned off, a different curve in a different functional form scores the guesses.

## 7. Where does `f` come from? Policy iteration

There is a circularity. Playing well needs `f`, and `f` is defined as the cost under good play. A value function in reinforcement learning has the same problem, and the same solution: bootstrap, then iterate (`train.policy_iteration`):

```
round 0:  play 2,000 games with greedy entropy (which needs no f)
          -> for every position visited, record
             (n_eff, guesses actually needed from there)
          -> least-squares fit f to those pairs

round k:  play 2,000 games acting greedily on the current f
          -> record the same pairs
          -> refit
```

That is **policy iteration with function approximation**. Evaluate the current policy by rollout, then improve it by acting greedily on the fitted value function. Every form of `f` is linear in its parameters. In `a + b·x + c·√x` the terms are nonlinear and the parameters are not, so each refit is one closed-form least-squares solve.

The cached parameters show it converging. The entropy seed measures about <!--n results_tables.rounds.seed .2f-->3.73 guesses. The first iteration takes it to <!--n results_tables.rounds.iter1 .2f-->3.08, the second to <!--n results_tables.rounds.iter2 .2f-->3.07, and the rest of the rounds wander in a small limit cycle around that. Nearly all of the gain arrives in one iteration. A greedy policy can oscillate, so the loop keeps the parameters whose measured play was best, not the ones it stopped on. Fitting takes seconds, so each prior combination's fit is written to `estimator_cache/` once and loaded thereafter.

Three constraints apply.

**Training games are played with no six-turn deadline.** Otherwise a game that runs long is truncated at six and the targets are biased downward. The deadline is a constraint on top of `f` and not part of it, which §9 covers.

**`f` must be non-decreasing** across the whole range (`estimator._is_monotone`). If a bigger candidate set could score cheaper, the search would look for uninformative guesses that dump everything into one bucket. When the full basis violates that, terms are dropped from the end until it holds. Two of the forms in `README.md`'s table score identically because the guard cuts one of them back until it is the other.

**A prior with no cached fit does not play.** The estimator raises and names the `train` command that makes one, instead of playing a policy nobody measured. Entropy mode never consults `f` and runs on anything.

### The form of `f` has to be refitted per shape

The cache is keyed per prior, and it has to be keyed per shape as well, since the shapes are the weights and `f` is fitted against the weights. A mis-paired `f` fails badly.

`f` is queried at its largest argument once per game, at the opening, where the candidate set is the whole word list. That is the far tail of the fit, where there is little data. Changing the shapes of §5 moves every effective size out there, so a curve fitted for one pair of shapes can rank opening words wrongly under another.

`spline` is the form that ships. Under the sharpened shapes it wins and opens `salet`. Under two flat shapes it wins again and opens `salet` again. Paired with a sharpened repeat shape and a flat curation shape, the same family opens `raise` and falls to seventh of nine, behind every form that stayed concave. Nothing about the board changed, only which curve was fitted to which weights. This is estimator error, and it is why the shapes are part of the fit's cache key.

The failure has a direction. `raise` is what greedy entropy opens with, since by §6 a straight-line `f` reduces to entropy plus a bonus. A curve that extrapolates badly goes flat in the tail, and a flat curve is a straight one. So a mis-fitted `f` fails by collapsing into the policy it was meant to improve on.

Ranking the forms by fit quality does not catch this. Under the shipped shapes, the two forms that open `raise` are also the two that fit worst, which looks like fit quality deciding the opening. Both facts have one cause. The monotonicity guard cuts those forms back to a straight line, which fits worse and also collapses by §6's identity. Under other shapes the two come apart.

In one configuration the worst-fitting form of the nine opens `salet` while two much better-fitting ones open `raise`. `sqrt` shows the same from the other end. Under one pairing of shapes it has the best fit of all nine forms and still places fifth on true mean, while `spline` fits three places worse and flips the opening. Concavity in `x` decides the opening word. How closely a curve tracks the points it was fitted to is a different question.

## 8. The scoring loop, and where the time actually goes

§3's information gain and §6's `E(g, S)` are the same loop with a different last line. That loop runs more often than anything else here. This section covers a rearrangement that removes nearly all of its arithmetic, and then what the arithmetic was hiding.

### The rearrangement that makes this cheap

**The problem.** `H` is defined on normalized weights, `p_i = w_i / m`. The bucket's total mass `m` is not known until every member of the bucket has been seen, and which bucket a candidate lands in depends on the guess. Done directly, that means one pass over the candidates per guess to find the bucket totals, then a second pass taking a logarithm per candidate to build each bucket's entropy. That is one logarithm for every (guess, candidate) pair.

**The fix.** Expand the definition and split the logarithm apart. Writing `m` for the sum of the bucket's weights:

$$H = -\sum_i p_i \log_2 p_i = -\sum_i \frac{w_i}{m} \log_2\left(\frac{w_i}{m}\right)$$

Use `log₂(w/m) = log₂ w − log₂ m` to break the coupled term in two:

$$H = -\frac{1}{m}\sum_i w_i \big(\log_2 w_i - \log_2 m\big)
    = -\frac{1}{m}\sum_i w_i \log_2 w_i + \frac{\log_2 m}{m}\sum_i w_i$$

The second sum is `m` again, so that term collapses to `log₂ m`, leaving

$$H = \log_2 m - \frac{\sum_i w_i \log_2 w_i}{m}$$

**Why that helps.** The mass `m` and the sum both range over the members of the bucket, which depends on the guess. The summand `w_i log₂ w_i` depends only on candidate `i`, on its raw prior weight, with no reference to the bucket, the guess or the normalization. Normalizing coupled a candidate's contribution to a property of its bucket. Splitting the logarithm decouples it.

Every scorer in `kernels.py` has the same two-phase shape:

```
for each guess:
    accumulate:  one pass over the candidates, adding each one's
                 contribution into a slot picked out by its pattern
    reduce:      turn those 243 slots into one number, the guess's score
```

The slots are the **accumulators**: four arrays of 243 floats, one entry per possible pattern, holding the bucket's running mass, its candidate count, its running total of `w log₂ w`, and its heaviest single weight. The accumulate pass runs once per (guess, candidate) pair, which at the opening is 167 million times. Anything expensive has to stay out of it.

So `w_i log₂ w_i` is computed **once per candidate**, before any guess is scored (`kernels._w_log_w`). The accumulate pass adds a precomputed float into a slot where the direct version would have called a logarithm. The reduce reads back the bucket's mass and its total of `w log₂ w` and recovers `H` with one logarithm per bucket instead of one per candidate.

For a position with `c` candidates scored against all 12,947 guesses, the direct version takes `12,947 × c` logarithms and the split version takes `c`. At a wide position that is 167 million against thirteen thousand.

There is a second consequence. Since `n_eff = 2^H`, the `H` the reduce recovers is `log₂(n_eff)`, the unit `f` takes. That is why `estimator.f_eval_x` is parameterized by `x = log₂ n` and not by `n`. Every caller reaches `f` through an entropy, so every caller has `x`. Going by way of `n` would compute `2^x` only for `f` to take the logarithm back off.

A sanity check on random weights, direct definition against the rearranged one:

```
direct : 2.331157535638251
split  : 2.3311575356382495     # agree to 1.3e-15
```

### Where the time actually goes

With the logarithms gone, the remaining cost was not where the arithmetic suggested. The obvious cost model was wrong twice.

**First wrong assumption: that the wide positions are the expensive ones.** They are the expensive individual calls. Small candidate sets are scored far more often: every node of the endgame search and every late turn of every training rollout scans a handful of candidates against all thirteen thousand guesses. Most of the time goes to the small cases, so that is where the optimization has to go.

The cost of the accumulators decides how the reduce is written. Four fresh 243-element arrays per guess would mean allocating and zeroing some fifty thousand floats thirteen thousand times over. So they are allocated once per block of at most 64 guesses and reused across the block. Whatever one guess writes has to be cleared before the next guess starts.

The reduce therefore has two jobs: total the buckets, and leave them zeroed. The obvious way to do both is a walk over all 243 patterns, which costs 243 whether there were twelve thousand candidates or five. With five candidates, at most five buckets are non-empty, and the other <!--c-->238 checks are wasted for every one of thirteen thousand guesses.

So below a threshold the reduce **walks the candidates a second time instead**, taking each bucket the first time it is reached and zeroing it on the spot. The zeroing clears the slot for the next guess, and it stops a bucket being counted twice when a later candidate lands in it. Five candidates then cost five reads and five writes instead of 243 of each.

The two branches total the buckets in different orders, so a position just under the threshold and one just over disagree in their last bits. That is harmless, because at the opening the closest pair of scores among all 12,947 guesses sits millions of ulps apart. Nothing in this loop is decided by differences near rounding, so branching on the candidate count is safe.

**Second wrong assumption: that the cost is arithmetic.** It is memory. The loop fixes a guess, walks the candidate set in order, and reads a single byte per candidate. All of those reads come from that guess's row of the table, at the column numbers §1 gave the surviving words. The accumulators they feed are 243 floats, two kilobytes, small enough to stay in the cache nearest the core for the whole block. So the table is the only thing streamed in, and the only thing that differs between one position and the next is how far apart the columns are.

That distance is the cost. A processor fetches a **cache line**, a contiguous block of 64 bytes, and holds it near the core. Sixty-four adjacent columns arrive on one fetch and the other 63 reads are free. That is the early game, where the candidates are most of the word list and pack the row densely. Late in the game the survivors are a few words scattered through an alphabetical list of thirteen thousand. Consecutive columns sit thousands of bytes apart, every read lands on a line of its own, and each one fetches 64 bytes to use one. That is the bill at every small position. It is also why the second pass over the candidates is nearly free: it re-reads the lines the first pass pulled in.

The gap between the two cases is bounded. A row is 12,947 bytes, which is <!--c-->203 cache lines. A dense run of `k` candidates costs `k/64` of them. A scattered set costs one line each, until it runs out of distinct lines and stops at <!--c-->203.

| candidates | packed together | scattered |
| --- | --- | --- |
| 20 | under a line | 20 lines |
| 200 | 3 lines | 200 lines |
| the whole list | 203 lines | 203 lines |

So the gap opens up to a couple of hundred candidates and closes again, as the dense figure keeps climbing after the scattered one has reached its ceiling. Under a couple of dozen the whole scan is too small to matter either way. The cost is in the middle of the range, and so is everything the alternative layout below can win.

**The blocks are also the unit of parallelism.** The guesses are cut into several blocks per core and handed to numba's `prange`, which spreads the iterations across threads. Each block has its own accumulators, so no two threads write into the same array. With several blocks per thread, a thread that draws slow work does not hold the others up at the end.

### Storing the table the other way round

Indexing the table `[answer, guess]` is the obvious alternative, and it does fix the reads. A kernel written for that layout takes the same time on scattered candidates as on adjacent ones.

It does not make the scatter go away. Fixing a guess and walking the candidates finishes one guess before starting the next, so a single set of 243 buckets serves a whole block. Fixing a candidate and walking the guesses leaves every guess in the block partway through at once, so each one needs its own set. Those buckets are touched on every (guess, candidate) pair, as often as the table is. The cost moves from the table to the accumulator.

The trade can be avoided. Read a whole contiguous tile out of an answer-major copy, transposing it into a small buffer on the way in. Each guess's feedback vector is then a contiguous row of that tile, and the reduce is the one above, over an accumulator that never leaves L1.

That is `fused_kernels`. Scan for scan it is about twice as fast as the shipped kernel around a hundred candidates, and falls to parity a few hundred later as the tile stops fitting in cache. Above that there is nothing to recover, because the shipped scan already uses a full cache line per fetch. Each layout wins on part of the range, so `fused_kernels` is a dispatch for mid-sized positions, not a replacement.

Over whole games the dispatch is worth **<!--n fused_gain.entropy.whole .2f-->1.58×** under entropy and **<!--n fused_gain.expected.whole .2f-->1.04×** under expected guesses. The second figure is loose. An expected-guesses scan carries the estimator and the endgame search on top of the table read, and only the read gets faster, so the gain is a small share of a large number and moves with which answers are drawn. It does not move with the fit.

### The optimization that had to be taken back out

Numba hands each kernel to LLVM, and LLVM offers a `fastmath` flag: permission to reorder floating-point arithmetic for speed, as if addition were associative. In floating point it is not quite, and the reordering is usually harmless.

Here it was not harmless, and the damage was not to the arithmetic. The reduce reads a bucket and then writes a zero into it for the next guess. Restructuring the loop swapped those two memory operations, so **every bucket maximum read back as the zero its own loop had just written**. The flag licensed reordering the additions. What moved was a store.

It surfaced as one scorer returning exactly 3.0 for every guess and the search preferring nonsense words. A wrong answer that obvious gets found. The other two scorers survived only because a logarithm or a call to the estimator sits between their read and their write, which the compiler cannot reorder across. None of the kernels carry the flag now. Removing it costs nothing on the widest positions and up to about a tenth of the time in the middle of the range.

### Scoring fewer guesses

Every kernel takes a **pool**, the set of table rows it is asked to score in place of all 12,947, and returns one score per entry. That keeps deeper search affordable. A search exploring the replies to a bucket of ninety candidates does not need all thirteen thousand guesses. Hard mode is the same mechanism with the pool set to the candidate set.

That is the cost of scoring a guess. None of it prices running out of turns.

## 9. The deadline, and what losing costs

Everything so far has ignored that Wordle gives six turns. `f` estimates guesses-to-finish with no deadline, as §7 explained. §10's formulas need a cost for running out, so it is defined here.

The obvious approach is a second objective: minimize guesses, subject to a separate constraint about not losing. That needs a weight trading the two against each other. Instead the deadline is folded into the same units as everything else. Define one number:

> **`loss_penalty`** — what it costs to run out of turns, charged on top of the six guesses already spent.

A lost game scores `6 + loss_penalty`. A penalty of 1 makes a loss score exactly 7, as if one more guess had been needed. That is the **risk-neutral** setting, where minimizing expected score means what it says and losing costs nothing beyond the turn.

In the value recursion this is the base case: with zero turns left, a position is worth `loss_penalty`. One recursion covers both winning fast and not losing.

Raising the penalty above 1 buys solve rate at the expense of the mean, by making the search willing to spend a turn on safety. **The default is 5**, and §11 explains why. Everything below holds for any setting. The penalty is a decision-time price only. Measured results always score a lost game as 7, whatever the penalty was, so that rows run at different penalties can be compared.

## 10. The endgame: stop estimating, start solving

`f` is a fitted approximation. Once few candidates remain, the true value is computable by brute force (`exact.py`):

$$V(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r V(S_r, t-1)\Big], \qquad V(S, 0) = \text{penalty}$$

with `t` the turns remaining. This is **expectimax**: a minimum over the guess, which you choose, and an expectation over the feedback, which you do not. There is no adversary, so it is not minimax.

Done directly the recursion is hopeless: thirteen thousand guesses times 243 patterns, all the way down. Four ideas make it affordable.

**(a) Partition dedupe.** By §2 a guess is its partition, so two guesses that split `S` identically are interchangeable. The branching factor becomes the number of distinct partitions instead of the number of words.

Grouping guesses that way means comparing the patterns they produce over the candidates. That list is the guess's **feedback vector**, one number in `[0, 243)` per candidate, the slice of the guess's table row that the candidate columns pick out. Comparing thirteen thousand such vectors directly is too slow. So each vector is packed into one 64-bit integer, and the integers are grouped.

With seven candidates or fewer the packing loses nothing. A pattern fits in 8 bits, seven of them fit in the 56 bits available, and the packed integer is the vector. Equal integers mean identical partitions.

Above seven the vector no longer fits and the integer becomes a hash. FNV-1a is a cheap mixing function that folds an input of any length down to 64 bits. Identical vectors always hash alike, and different ones almost never collide. Almost never is not a proof, so every guess is compared once against its group's representative and promoted to a group of its own if it splits differently. A collision then costs one redundant representative and never loses a distinct partition.

`np.unique` would group the feedback vectors themselves with no collision check. It is a lexicographic sort, one pass per candidate, and it costs an order of magnitude more than hashing each vector and grouping the hashes. One extra comparison per guess buys the exactness back.

A guess that could itself be the answer always splits distinctly, since only it produces `GGGGG`. So a candidate is never merged into a probe. Hard mode skips this step for the same reason: if every guess must be a candidate, no two of them split `S` the same way.

The dedupe only helps while `S` is tiny. By a dozen candidates almost every guess in the list splits differently, so beyond that the branching factor is the whole word list. That is what makes the occasional deep position slow, not the node count.

**(b) Closed-form leaves.** With two turns left, no search is needed. Play `g`, and every reply but `GGGGG` leaves you in some bucket with one turn to spend. On a last turn you name the heaviest word in the bucket and are right with probability `M_r / m_r`, writing `m_r` for bucket `r`'s mass and `M_r` for its largest single weight. So

$$V(S_r, 1) = 1 + \Big(1 - \frac{M_r}{m_r}\Big)\cdot\text{penalty}$$

Now average that over the buckets, each reached with probability `m_r / total`:

$$V(g,2) = 1 + \sum_{r \ne \texttt{GGGGG}} \frac{m_r}{\text{total}}\left[1 + \Big(1 - \frac{M_r}{m_r}\Big)\text{penalty}\right]$$

The `m_r` outside the bracket cancels the one inside it. That turns a sum of ratios into a sum of two quantities a scan can accumulate:

$$V(g,2) = 1 + \frac{A + (A - B)\cdot\text{penalty}}{\text{total}}$$

with `A` the mass and `B` the sum of bucket maxima over the non-winning buckets. Those two totals are all a guess's 243 buckets reduce to, so `kernels._two_turn` keeps two accumulators. That is one pass over the guesses, the same as heuristic scoring. So **two turns or fewer is always solved exactly, at any candidate count, for the price of a heuristic scan.** The same two-turn position is reached along many orders of guesses, with about four calls in five a repeat, so even this closed form is cached (§12).

The penalty in that formula is a parameter. Called as a leaf, the code is handed the real one and returns the true value of the position. The next idea calls the same code with a different penalty for a different purpose.

**(c) Branch and bound.** That closed form, evaluated at a lower penalty, is a **lower bound** on `V(S,t)` for every larger `t`. Write `V₁` for the value computed with the penalty set to 1 and `V₅` for the shipped setting. The search relies on

$$V_1(S, 2) \le V_5(S, t) \qquad \text{for every } t \ge 2$$

which splits into two steps, and both need the 1.

First, at a penalty of 1, turns in hand cannot make a position cheaper, so `V₁(S,2) ≤ V₁(S,t)`. More turns sound worth having, but here they are not, because running out is an exit. A position two turns from the deadline can take its last shot, miss, pay 1, and stop. A position with four turns in hand has to carry on until it wins, and no continuation finishes for less than the one guess the exit charges.

Second, charging less for a miss cannot raise the value, so `V₁(S,t) ≤ V₅(S,t)`. Fix any policy and score its play twice, once charging 1 for a loss and once charging 5. The second charge is larger on the paths that lose and identical everywhere else, so no policy scores worse under the cheaper penalty, and neither does the best of them.

Together, the two-turn value at penalty 1 sits under every deeper value at the penalty in force, so pruning on it is safe.

**Without the cap at 1 the bound is false.** At a penalty of 5 the first step reverses, because the exit now costs five guesses and a position near the deadline would rather have the turns. The ten words ending in `-atch` are a real position that shows it:

<!--table loss_penalty.bound.table-->
| turns in hand `t` | 2 | 3 | 4 |
| --- | --- | --- | --- |
| value at penalty 5 | **1.91** | 1.57 | 1.50 |
| value at penalty 1 | 1.37 | 1.45 | 1.49 |
<!--/table-->

The bound needs those values to rise with `t`, and only the second row does. The family is cheap to begin with because the prior puts most of its mass on `hatch`. With two turns and a miss charged at 5, the last turn is a guess at whatever the first failed to resolve, and the position prices at <!--n loss_penalty.bound.two_turn_at5 .2f-->1.91. A search that pruned the three-turn position because <!--n loss_penalty.bound.two_turn_at5 .2f-->1.91 already beat its incumbent would discard a subtree whose true value is <!--n loss_penalty.bound.three_turn_at5 .2f-->1.57. So `ExactEndgame.bound_penalty` is `min(loss_penalty, 1)`, and it is used in the bound and nowhere else.

The search uses the bound in the standard way. Guesses are ordered by it and walked in that order, keeping the best complete value found so far, the **incumbent**. Within a guess, the running total starts as the sum of its buckets' bounds, and each bucket is replaced by its exact value one at a time, so the total only rises. Once it passes the incumbent, the guess is abandoned. Biggest-mass buckets go first, since they move the total fastest.

This is sound because the bound is **admissible**, meaning it never overestimates, so nothing it prunes could have been the best.

**(d) Branch cap.** Only the fifty best splits by that bound are tried, and this is the one idea of the four that gives something up. The other three are exact: dedupe merges guesses that are interchangeable, the closed form is a value, and an admissible bound prunes only what could not have won. The cap has no such guarantee. A position holding <!--n branch_cap.tied.splits ,d-->1,578 distinct splits has <!--n branch_cap.tied.discarded ,d-->1,528 of them discarded unexamined, and if the best guess is among those, the search never knows. What comes back is an upper bound on the true value, optimal within the cap.

It pays because the bound is a good ranking near the top, even where it is a loose estimate of the value. The optimum reliably sorts into the first few dozen, so the cap keeps it and discards the tail, where the bound's looseness costs nothing. Measured over more than a thousand real positions, caps of 25 and 50 pick the uncapped-optimal guess every time, and the wall clock for the sweep falls from around five minutes to eight seconds. That is an empirical record and not a proof, since nothing rules out a position whose best guess sorts fiftieth.

Two details matter.

**The cap is decided by a sort, and the bounds tie in bulk.** One real position holds <!--n branch_cap.tied.tied ,d-->962 of its <!--n branch_cap.tied.splits ,d-->1,578 splits at the bound the cap's edge falls on, <!--n branch_cap.tied.at .1f-->2.0, so which fifty are kept is settled almost entirely by the tie-break. With ties ordered arbitrarily, two runs of the same search examine two different sets of guesses. Both implementations therefore sort stably, which puts a tie in word order. A tie is decided on the bound rounded first, coarsely enough that two summation orders of the same bound cannot fall on opposite sides of it.

The general rule is that a threshold, a cap or a top-k standing on a key that ties in bulk is brittle whoever evaluates it. Make the comparison as coarse as the quantity is meaningful, and let a fixed rule break the ties.

**A ranking wants a wider cap than a pick.** The page shows five guesses. The winner sits clear of the cap, but the runners-up sit against it. Measured against the uncapped search, a cap of 50 gets every winner right and occasionally names a worse fifth word. So `ExactEndgame.root_branch` is wider, and it applies only at the node being ranked. Every node beneath it needs its winner alone.

### When there is no table to read

§8 priced reading the table. The browser build has no table, since 167 MB can be neither downloaded nor rebuilt, and the search is where that matters.

For choosing a guess, applying the rule afresh costs a few times a lookup, which is affordable. The search visits hundreds of nodes for a single position and rescans the whole guess list at every one, so a set of thirty-odd candidates has the coloring rule run over the same half-million pairs several hundred times. In the browser that made one turn take <!--u-->5.6 seconds, long enough that a tab looks dead.

The fix is §1's table at a much smaller scope. Every candidate the search will see belongs to the set it started from, since each bucket is a subset of the one above it. So the pairs it can ask about are known before it starts. Score every guess against that starting set once, up front, and every node below reads a byte instead of coloring a word. That is <!--u-->440 KB instead of 167 MB, one scan instead of hundreds, and the same turn drops from <!--u-->5.6 seconds to <!--u-->1.1. Nothing outside those columns is ever asked for, so no exactness is lost.

### Lookahead: the same recursion, one ply up

The search above waits until the game is nearly over. The same recursion runs at a position of any size if it stops early and estimates the rest. That is `--lookahead`. Instead of scoring a guess with `E(g, S)`, it values each of that guess's replies by scoring those positions, and picks the guess by what comes back.

$$V_d(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r V_{d-1}(S_r, t-1)\Big], \qquad V_0(S, t) = \min_g E(g, S)$$

It is the expectimax above with two changes. The leaves are the heuristic instead of the truth, so it runs at a width the exact search cannot afford, and the exact search still takes any sub-position it can reach. Only the guess is capped, never the feedback, because dropping a bucket would leave the sum something other than an expectation.

The cost is a fan-out. Every guess kept expands into every one of its buckets, so each further ply multiplies the work by the cap again. One ply at a cap of ten runs about ten times slower than the estimator alone. It is off by default, and §15 measures what it is worth.

## 11. Why the penalty is 5 and not 1

Risk neutrality is the principled default, so moving off it takes evidence. Start with what the losses are.

They are not endgame blunders. At the sizes these positions reach, `exact.ExactEndgame` solves them outright, so the guess played is optimal. The solver is backing its prior. Holding `batty`, `catty`, `natty`, `patty`, `ratty` and `tatty` with four turns left, it plays `tatty`, which carries three quarters of that family's mass. It turns down `bronc`, which splits the six of them five ways, because separating them spends a turn to insure against a case the prior puts under half a percent on. A loss is the prior being confidently wrong, and the penalty makes that insurance worth buying.

| penalty | loss probability | mean (loss scored as 7) |
| --- | --- | --- |
| 1 | 0.1185% | 3.6607 |
| **5** | **0.0513%** | **3.6611** |
| 30 | 0.0417% | 3.6620 |

Well over half the loss probability vanishes between 1 and 5 for **four ten-thousandths of a guess**. Widening the exact search's reach adds the rest. The two together cut the loss probability by about three fifths, and the mean improves on the way, because reaching deeper replaces an estimate with a true value.

That last column cannot pick the penalty. Scoring a loss at 7 is the objective a penalty of 1 optimizes, so ranking the settings by that column always favors risk neutrality. The comparison is circular. Name the price of a loss instead, `L` guesses, and every setting becomes a line in `L`, so the choice falls out of where the lines cross. Penalty 1 wins below `L` ≈ <!--n loss_penalty.crossover.1_2 .1f-->7.1, **5 wins between <!--n loss_penalty.crossover.2_5 .1f-->8.6 and <!--n loss_penalty.crossover.5_10 .1f-->12.7**, and 30 only past <!--n loss_penalty.crossover.10_30 .1f-->24.8 (`analysis/loss_penalty.py` prints the crossings; README *The penalty* carries them).

That leaves where `L` sits, and one end of the range can be measured. Scoring a loss at 7 says a lost game would have needed exactly one more guess. Lift the deadline and carry each losing game on from the position it died in, and the solver takes **<!--n loss_cost.penalty5.extra .1f-->1.2 more guesses** to finish, occasionally <!--n loss_cost.penalty5.worst-->4 (`analysis/loss_cost.py`). So the risk-neutral price of a loss is about **<!--n loss_cost.penalty5.price .1f-->7.2**, past the point where penalty 1 gives way. Risk neutrality does not endorse penalty 1.

It endorses the next setting up, which takes over at that crossover, and nothing near 5. Continuing a lost game prices it in guesses, which is the risk-neutral question, so this fixes the leftmost line and cannot reach the rest. Choosing 5 is a different kind of claim: that a failed puzzle costs more than the guesses it cost, because Wordle scores you out of six and does not count forever. No continuation can measure that, and it is a cheap claim to get wrong. Anywhere in the range where 5 wins, it beats the next setting by a fraction of a thousandth of a guess.

The losses collect almost entirely on the never-curated words. Wordle is provably solvable in five guesses, but that theorem is about a solver that knows the answer is one of the 2,309 curated words. This one spreads a tenth of its mass over the other 10,638. Swept over the curated list uniformly, it loses eleven words out of 2,309, every one a word that has already been an answer. §5.1's repeat prior calls those unlikely by construction.

## 12. Remembering positions

The four ideas of §10 cut the cost of solving a position once. This section is about not solving the same position twice, which is the largest single speedup in the project.

It works because of a fact about the game. Fix the opening word. The answer decides the feedback, but feedback comes in only 243 varieties, so **all 12,947 possible answers lead to at most 243 distinct positions** at turn two. Playing every word as the answer to measure an opening looks like a thirteen-thousand-wide computation and is a two-hundred-wide one, repeated. The same collapse happens at every depth, since many paths arrive at the same candidate set. Storing each answer against its position, instead of recomputing it every time the position recurs, is worth about **a hundredfold** over a sweep of every answer (`analysis/ablations.py`).

A position is the pair the answer depends on: the surviving candidate set, and the number of turns left. `memo.Memo` keys on both. The candidate set is keyed as the raw bytes of the array of column numbers from §1, because an array cannot be a dictionary key and its bytes can, and because bytes are exact where a hash is not. The turn count is in the key because the recursion bottoms out on the deadline, so the same set of words has different values at different depths. A value found with four turns in hand is wrong for two. §10's `-atch` family is the worked example.

There are three such caches, and they do not expire together. `WordleSolver` keeps two, the guess it chose at a position and the lookahead value of one. Both are dropped whenever `f` is refitted, because both are answers the old `f` gave. `ExactEndgame.memo` holds true values, which depend on the prior weights and not on the estimator, so a refit leaves them correct and they are kept.

### Why it evicts rather than refuses

A cache needs a cap or it grows without limit. The obvious way to write one is to stop accepting entries once it is full:

```python
if len(memo) < limit:       # the tempting version
    memo[key] = value
```

That is the wrong way round. Such a cache declines to store the most recent positions, which are the ones the current run is about to reuse. It ends up holding whatever it saw first. Past the limit, the hundredfold saving above disappears at once.

`memo.Memo` instead drops the **least recently used** entry. LRU is the standard policy of evicting whatever has gone longest untouched, as the least likely to be wanted next. The memo is a dictionary that remembers the order its entries were last touched. A read moves the entry to the newest end, and an insertion past the cap drops whatever sits at the oldest. A read counting as a use is the difference between LRU and dropping the oldest insertion. It keeps a position that is looked up constantly resident however long ago it was first computed.

The bookkeeping is not free, since every read moves an entry with a couple of pointer writes. Moving an entry costs on the order of a hundred nanoseconds. The miss it prevents costs a scoring pass over thirteen thousand guesses, which is milliseconds. The overhead is roughly ten thousand times smaller than what it buys.

One worry does not apply. A sweep that plays two hundred openings in one process leaves each finished opening's positions behind, and almost none of them recur, because two openings partition the word list differently. Those stale entries are the ones that have gone longest untouched, so every insertion evicts one of them before it reaches anything the current run is using. LRU sheds them in the right order.

`WordleSolver.reset_memo` therefore buys memory, not hit rate. It clears only the two memos that depend on the estimator. The search's entries stay valid across everything.

Every piece of the guess rule is now defined. §13 puts them in the order the solver applies them.

## 13. Putting it together — one turn of play

```
candidates = all 12,947 words
each turn:
    if only 1-2 candidates          -> closed form
    elif exact search can reach it  -> exact.solve()          (true optimum)
         (turns <= 2 always; or <= 60 candidates deeper)
    elif last turn                  -> heaviest candidate
    elif lookahead is on            -> value the replies first (§10, off by default)
    else                            -> argmin over 12,947 of E(g,S)
    show guess, read feedback, filter candidates
```

`argmin` is the argument that minimizes: the word achieving the smallest score, since a guess is what the solver needs back. It is one call to `kernels.expected_guesses_all` and one `.argmin()` over the array it returns.

Every line that says 12,947 is the free choice of guess, and hard mode is the one variation on this dispatch. It replaces that pool with the candidate set, in the exact search as in the `argmin`, and changes nothing else about which branch runs. `f` stays the curve fitted for free play, so its scores read slightly optimistic under the restriction, since it prices a probe the solver is no longer allowed to make.

Most of this document sits under that dispatch. The lookup that reads the feedback is §1's table. The `argmin` is §8's rearranged entropy inside its blocked, parallel, twice-scanned loop. `exact.solve` is §10's four ideas. Every branch consults §12's memo first.

The heuristic `E(g,S)` is a **fallback**, running only where the game is too wide to solve outright. With both priors on and the shapes that ship, over every word the answer could be, the solver averages **<!--n results_tables.both.expected .4f-->3.0636 guesses** and opens with `<!--s results_tables.both.opening-->salet<!--/s-->`. Against the answers the NYT has set in the current regime it reads about <!--n replay.priors.both.mean .2f-->3.16 (§14.3).

That <!--n results_tables.both.expected .4f-->3.0636 is what every part above adds up to, and it says nothing about which part earned it. §14 asks what the model of the editor was worth, and §15 asks which piece of the machinery earned the number.

## 14. Measuring a prior without fooling yourself

With the solver assembled, the question §4 and §5 deferred can be asked: what was the model of the editor worth? This section covers one trap in the evaluation, then two questions, of which the data settles one.

### 14.1 The evaluation needs three arms

A shape can only be judged on answers that happened. A prior scored against answers sampled from its own belief is grading its own work, and the swing from doing that is larger than the effect being measured.

So the evaluation has three arms: the **<!--n replay.shapes.arms.repeats-->24 real repeats**, the **<!--n replay.shapes.arms.offlist-->58 off-list answers**, and the regime's **<!--n replay.shapes.arms.fresh-->177 fresh curated answers**. Each is replayed with the prior rebuilt from strictly earlier puzzles, and the arms are blended at the rates the regime shows. Every configuration measured has both priors on, against a baseline that also has both on with flat shapes inside them, so the only thing that moves is how the mass spreads.

Every figure below is a difference from that baseline, in guesses, so negative is better:

<!--table replay.shapes.arm_table-->
| shapes | repeat | off-list | fresh | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.042 ± 0.127 | −0.879 ± 0.118 | +0.107 ± 0.038 | −0.016 ± 0.035 |
| repeat `binned` alone | −0.333 ± 0.155 | −0.052 ± 0.124 | +0.130 ± 0.041 | +0.060 ± 0.039 |
| `binned` + `frequency` | −0.417 ± 0.146 | −0.397 ± 0.130 | +0.124 ± 0.044 | +0.006 ± 0.040 |
| `binned` + `wordlebot` | −0.417 ± 0.158 | −0.862 ± 0.117 | +0.090 ± 0.040 | −0.076 ± 0.038 |
<!--/table-->

Each shape moves the arm it was built for: a third of a guess when the answer is a repeat, most of one when it is off-list. Each then pays some of it back across the three quarters of answers that are neither, and most of the gain goes in the paying. The shipped pair nets **<!--n replay.shapes.together.blend-->−0.076 ± 0.038** blended, <!--n replay.shapes.together.blend_sigma .1f-->2.0 standard errors from zero. That is the shapes showing in guesses and not only in nats, and it is the one blend in the last column outside its own error.

Unlike §4's table, these columns are comparable between rows, because every row is scored on the same real answers. The table is comparable to nothing outside itself. Each row carries the `f` fitted for its own shapes, as §7 requires, so a refit moves the whole table and the rows have to be refitted together.

The loosest term in every blend is the repeat arm, at about ±<!--n replay.shapes.repeat_alone.arm.repeat_se .2f-->0.16 on <!--n replay.shapes.arms.repeats-->24 events, which leaves the bottom two rows' blends overlapping. Paired directly, they separate. `wordlebot` beats `frequency` by <!--n replay.frequency.blend-->+0.081 ± 0.025 on the same three arms, nearly all of it on the off-list arm at <!--n replay.frequency.arm.offlist-->+0.466 ± 0.089 (`analysis/replay.py`). That agrees with §5.2, which scored the two curation shapes at their own job: <!--n curation_shapes.wordlebot.nats .2f-->−6.45 nats per event against <!--n curation_shapes.frequency.nats .2f-->−8.06, over <!--n curation_shapes.events-->58 events. The modeling evidence is the sharper of the two instruments, and the vendored scrape earns its place on both.

One mistake in that evaluation is easy to repeat. The fresh arm was first a census: all the curated words still unused, played as if today. That is far more data than <!--n replay.shapes.arms.fresh-->177 real answers, and it looked like the better arm. It was training on the test set. `train.policy_iteration` draws its practice answers from the prior it is fitting, and that prior's fresh-curated group is, at fitting time, exactly those <!--n fresh_arm.census-->474 words. So four fifths of the evaluation's weight sat on material every arm had trained against.

The damage is not that it favors one arm's words over another's. Both solvers reach their own tuned best on ground they know, so the arm carrying four fifths of the grade reports a tie. A tie is not neutral, because it hands the verdict to the two small arms, where a sharpened shape looks its best. The sharper shapes collect the credit of the repeat and off-list columns while the column that should have charged them for it says nothing.

`analysis/fresh_arm.py` prices what was concealed. Sharpening the repeat shape costs about nothing on the census (<!--n fresh_arm.census_today-->−0.004 ± 0.014) and about a tenth of a guess on the answers that happened (<!--n fresh_arm.own.real-->+0.130 ± 0.041). Playing the census on those same days does not close the gap, and forcing both arms onto a single `f` mostly does, which shows it was the fit that differed and not the words.

The arm is the <!--n replay.shapes.arms.fresh-->177 real answers, which costs precision where the variance lives. What it keeps is the one advantage this evaluation has over a plain replay: <!--n replay.shapes.arms.offlist-->58 off-list events instead of the <!--n replay.shapes.n_offlist-->27 inside the regime. The two agree, with the plain replay putting the shipped pair at <!--n replay.shapes.together.vs-->−0.0789 ± 0.0462 and the blend at <!--n replay.shapes.together.blend-->−0.076 ± 0.038, both a small gain at the edge of what a season of puzzles can resolve.

The census made the evaluation kinder to itself than reality. The opposite worry is that it is stricter. Every arm is replayed with the prior rebuilt from strictly earlier puzzles one day at a time, as the prequential scores in §5.1 and §5.2 are. The shipped solver is not that current, since `data/solutions.json` is refreshed by hand and stops at yesterday when it is. The table therefore describes a solver whose record is fresher than the one anybody plays against.

It describes it closely enough. `analysis/refit_lag.py` replays the regime with the record truncated to `t − lag`. A week of staleness is worth <!--n refit_lag.every7-->−0.0081 ± 0.0084 guesses averaged over the refresh cycle it implies. A month's lag leaves <!--n refit_lag.lag30.changed-->30 games changed with the mean inside its error of nothing. Refitting every k days costs no more than a few thousandths for every k out to eight. The reason is that a stale record barely moves the weights. This is not the flat basin README *The prior sits in a flat basin* describes for other perturbations of the prior. A lag of k days leaves k words carrying fresh-curated mass instead of used mass and changes nothing else. For a week that is <!--n refit_lag.lag7.mass .2f-->1.11% of the prior's mass, against the <!--n refit_lag.yardstick .2f-->6.01% that flattening the repeat shape moves. None of it involves refitting `f`. The cache is keyed on the prior alone, so one fit serves every day of a replay and every day of a deployment.

### 14.2 Do the two shapes need each other?

The guesses cannot say. Read the blended column of §14.1's table as if it added up, and the pair does better than the two apart would predict: <!--n replay.shapes.together.blend_delta +.3f-->−0.076 against a sum of <!--n replay.shapes.apart +.3f-->+0.043. That margin is a tenth against an error of about half of it, and two of the three figures behind it are inside their own error of zero.

Each shape does have a margin the prequential scoring can resolve at its own job, and both ship on that modeling evidence: the histogram's <!--n repeat_shapes.gain +.2f-->+1.13 ± <!--n repeat_shapes.gain_se .2f-->0.20 nats per event in §5.1, and WordleBot's list against a flat spread in §5.2. WordleBot's shows in guesses as well (§14.1). The histogram's does not, together or apart.

A mechanism predicts they should interfere, though these events cannot see it. Sharpening one side makes the other side's flatness the larger error in the weights. Concentrating the off-list mass onto its thousand-odd effective words makes each of them heavier than a used word under a flat repeat spread. So the curation shape on its own promotes every off-list word above every past answer. When the answer is a `cigar`, that costs.

Whether the repeat shape buys it back is not a question <!--n replay.shapes.arms.repeats-->24 events can answer. Resolving an effect that size on that arm would take several hundred repeats, which at the current rate is eight years of Wordle. So the mechanism goes unmeasured here, which is not evidence that it is absent.

Interference of this kind is settled one level up. §14.3 finds the two priors these shapes live inside strongly entangled, by a different route: each miscalibrates the rate the other governs.

### 14.3 Do the two *priors* need each other?

A shape only redistributes mass inside a group its prior has carved out. The prior decides whether the group exists. With curation off there is no curated-versus-not split to spread anything over, only 12,947 words sharing the mass equally, non-answers included. That is the larger question, and it has a much larger answer.

`analysis/replay.py --all-priors` replays the <!--n replay.priors.puzzles-->228 puzzles of the current regime under all four on/off combinations, pairing every arm on the same answer:

<!--table replay.priors.grid_table-->
| mean guesses | curation off | curation on | on − off |
| --- | --- | --- | --- |
| **repeat off** | 3.9912 | 3.5351 | −0.456 ± 0.053 |
| **repeat on** | 3.9737 | 3.1579 | −0.816 ± 0.052 |
| **on − off** | −0.018 ± 0.037 | −0.377 ± 0.052 | **−0.360 ± 0.065** |
<!--/table-->

Those four means do not add up. Each margin is a subtraction along a row or a column, so the corner can be reached two ways and comes out the same both times. That corner is the **interaction**, and it is larger than either prior buys on its own. The priors are strongly superadditive, even though §14.1 found their shapes are not. The repeat prior alone is worth almost nothing, and what it buys is the calibration of the other one.

One caveat applies to the four cells. Each carries the `f` fitted for its own prior, as §7 requires, so the comparison is between four solvers, not four beliefs held by one solver. That is the right comparison, since a prior the estimator was never fitted to is not a configuration anyone would ship. It means a refit moves every cell, and the four have to be refitted together.

The two are entangled through calibration, which shows in the rates each prior implies. Those follow from the weights alone, with no estimator in them, so the caveat above does not reach them. Set against the two rates the regime shows:

<!--table regime.implied_table-->
| prior | implied repeat rate | implied off-list rate, given not a repeat |
| --- | --- | --- |
| none | 14.6% | 95.7% |
| curation only | 72.4% | 33.1% |
| repeat only | 10.0% | 95.7% |
| both | 10.0% | 10.0% |
| *observed* | *10.5%* | *13.2%* |
<!--/table-->

Every prior but the last is far off on the rate the other one governs, and that is the whole of the interaction.

Take them in turn. A uniform prior lands near the repeat rate by accident: <!--n regime.used_words-->1,893 used words out of 12,947 happens to be <!--n regime.used_share .1f-->14.6%. Adding the repeat prior to it corrects a column that was barely wrong and leaves the <!--n regime.implied.none.offlist .1f-->95.7% where it was. So that prior buys almost nothing. Curation alone pulls the second column down to <!--n regime.implied.curation_only.offlist .1f-->33.1% and creates the repeat error in doing it. Confining the field to a 2,309-word list of which <!--n regime.curated_used-->1,835 are spent is a claim that the editor repeats herself seven times in ten. Only with both on is either rate right.

Per word, that correction is a used word falling sevenfold while a fresh curated word climbs fourfold. The fresh curated words are the <!--n regime.curated_unused-->474 the editor can still reach without repeating herself or leaving the list.

The table also prices calibration on the repeats themselves, where curation alone beats the calibrated pair on the <!--n replay.priors.n_repeats-->24 answers that were repeats. That is the price of calibration. Moving a class from an implied <!--n regime.implied.curation_only.repeat .1f-->72.4% down to 10.0% costs on the occasions it turns up, in exchange for doing better across the four fifths of the regime that is neither a repeat nor off-list. Any calibrated prior makes that trade.

Getting the model of the editor roughly right is worth more than half a guess. §15 measures what every other kind of improvement in this project was worth.

## 15. A better map, and next to no better score

Five separate attempts to value positions better have all come out the same way:

| the improvement | how much better the valuation got | what it bought on the mean |
| --- | --- | --- |
| one-ply lookahead | RMSE **0.0125** against the estimator's **0.1609** | −0.0080 ± 0.0036 over 20,000 games; −0.0018 ± 0.0036 under the shipped prior |
| the endgame search | exact — the true value, not a better estimate | under a hundredth of a guess, over the positions it reaches |
| the opening word | `f` ranks the top twelve at **+0.35** against truth, the full two hundred at **+0.74** | nothing left to buy: `salet` is already optimal, at zero regret |
| turn-conditioned `f` | — | nothing: `f` is never consulted where the deadline binds |
| the second guess | true values for all <!--n turn2_regret.decided-->51 positions `f` still decides | **<!--n turn2_regret.paid .4f-->0.0034** guesses, and the sign flips under a flat weighting of the curated list |

Every row buys a real improvement in valuation and none of them buys a score. The rest of this section is why.

### 15.1 The top of the ranking is a plateau

**One-ply lookahead** values the replies to a guess instead of estimating them. RMSE there is the root-mean-square of the errors against ground truth, the true expected guesses from a position. That is computable because the policy is deterministic, so a position can be played out once per candidate answer. The accuracy reaches the decisions. Lookahead disagrees about which guess to play in two games in five, and it is right to disagree, giving up about a tenth as much value per decision. Then the games are played out and almost none of it arrives. The outcome moves in a fifth of them, slightly more often for the better than for the worse. The net needs twenty thousand games to separate from zero, at roughly ten times the wall clock, and on the one configuration that ships it does not separate.

The tempting explanation is that the `argmin` makes the estimator's errors **common-mode** across the guesses at one position, so they cancel in the comparison and leave the ranking intact. They do not. The estimator ranks the ten guesses lookahead re-ranks at a rank correlation of about **<!--n lookahead_accuracy.decisions.rho +.2f-->+0.55** with their true values, on a scale where +1 is identical and 0 is no agreement. It picks the best of them a little over half the time. A valuation whose errors cancelled would not be overruled two games in five by an accurate one.

The top of the ranking is a **plateau**. Those ten guesses spread about a tenth of a guess between best and worst. The estimator's pick gives up about a ninth of that spread, and it orders the middle of the list almost at random. It finds the plateau reliably and cannot tell one spot on it from another. That costs little, because there is little to tell apart.

**The endgame search** makes the same point more sharply. Where it reaches, its value is the truth, so a disagreement there is the estimator being wrong. Over the positions the search claims, the estimator picks a different guess about a third of the time and gives up under a hundredth of a guess doing it. At three and four candidates every disagreement is a tie, a different word of exactly equal value. The ties thin out as the set grows, and at the largest sets the search reaches, most disagreements are real. That is the same plateau, found from the other side.

**The opening** is decided by `f` alone, since no search reaches a position of 12,947 candidates. Played out, `salet` is the best of the two hundred openings `f` likes best. Its regret, what the choice costs against the best available, is zero. Yet the two correlations in the table say `f` sorts the whole field far better than it sorts the top of it. `f` sorts the field coarsely and cannot sort the top of it, and every decision it gets right is one where coarse was enough.

**Turn-conditioned `f`** could not have helped. A curve conditioned on turns remaining differs from a flat one only where the deadline binds, and `f` is consulted in **none** of the positions with three or fewer turns left, because the search takes every one of them. The small gain measured for it belonged to an implementation that is gone, and the structural reason holds regardless.

**The second guess** puts a number on the ceiling. Behind a fixed opening, the whole of turn 2 is one decision per feedback pattern: <!--n turn2_regret.positions-->193 reachable positions, of which the endgame search solves <!--n turn2_regret.solved-->142 outright, leaving <!--n turn2_regret.decided-->51 that `f` decides alone. Play out the true value of the fifty best guesses at each, weight each position by the probability play reaches it, and the estimator gives up **<!--n turn2_regret.census .4f-->0.0034 guesses**. That figure is exact, because substituting the <!--n turn2_regret.moved-->11 decisions it gets wrong and sweeping every answer moves the mean by that amount. Turn 1 is already optimal and the search owns everything from turn 4, so that <!--n turn2_regret.paid .4f-->0.0034 is the whole of what any valuation improvement could be worth anywhere in the game.

The number does not hold up elsewhere. Score those same <!--n turn2_regret.moved-->11 substitutions against differently weighted fields, the four enumerated ones exactly and the real answers as a sample:

<!--table turn2_regret.table-->
| distribution | shipped | turn-2 optimum | delta |
| --- | --- | --- | --- |
| the default prior (what chose them) | 3.0636 | 3.0602 | **−0.0034** |
| uniform over all 12,947 | 4.4876 | 4.4834 | **−0.0042** |
| uniform over the curated list | 3.7731 | 3.7809 | **+0.0078** |
| uniform over the still unused | 2.9367 | 2.9304 | **−0.0063** |
| the 228 real regime answers | 4.0658 | 4.0614 | −0.0044 ± 0.0211 |
| all 1,917 real answers ever | 3.9849 | 3.9963 | +0.0115 ± 0.0074 |
<!--/table-->

Read the real-answer rows as differences, not levels. Both arms replay them under today's prior, which prices every one of them as a word already used, so the level is that handicap and the delta is the comparison. The enumerated rows disagree among themselves, since a flat weighting of the curated list charges for what the default prior pays for. The real answers are too few to settle it either way. The reversal does not need the whole distance: blended toward that flat weighting, the default prior stops paying for the substitution <!--n turn2_regret.blend_crossing .2f-->0.30 of the way there.

The gain is the residue of a reshuffle in which over a thousand answers change length. Slightly more of the prior's mass improves than worsens, and <!--n turn2_regret.paid .4f-->0.0034 is what is left after that near-cancellation, a couple of thousandths of the standard deviation of the changes it rides on. Which side wins is decided by the prior's weights, and the prior is only a model of the editor.

That is sharper than the four cases above it, because it does not depend on any valuation being imperfect. **Playing the last position the estimator decides perfectly is worth less than the error in the prior it would be playing perfectly for.** No depth of search, further ply or better-fitted `f` gets past that.

Improving how well the solver values its options is worth almost nothing here. What is left is the options themselves.

### 15.2 The exceptions all change the choice set

Every exception changes what the solver is choosing between, not how well it scores the choices.

**The shapes of §5** are worth **<!--n replay.shapes.together.blend-->−0.076 ± 0.038** guesses blended (§14.1). That is small, and still more than deeper search and better lookahead managed between them. A prior changes which words are plausible answers at all, which is the quantity being optimized, so it is a decision-side change.

**The sharpened repeat prior** belongs here, not among the five. On real answers it reads as a small cost despite being much better as a model of the editor. The two figures are not in competition, because changing the weights changes what a position is worth, so the two arms share no ground truth on which guess is better. §14.1 takes that apart.

**Hard mode** is the bluntest form, since it improves no valuation and only deletes words from the choice set. Under greedy entropy that is worth about a quarter of a guess, where a thirteen-times-better valuation buys nothing. Taking a choice away from a policy that was pricing it wrong gains more than pricing everything else right. Under the expected-guesses objective, which already prices that choice correctly, the same restriction costs a little.

The same applies to the headline number. Most of the two thirds of a guess over greedy entropy is the possible-answer bonus, and little of it is curvature in `f`. The bonus depends on the prior, because `p_win` is the mass the prior puts on the guessed word, so the split moves with how sharp the prior is. §6 showed the curvature also falls to nothing when the curation prior is off. The search contributes almost none of the two thirds: turning `exact.py` off changes the mean by a few ten-thousandths and mainly changes the loss probability.

## 16. Why any of this finishes

Choosing a single guess means scoring thirteen thousand words against a candidate set. The training loop does that a few million times, and the analysis sweeps run it over every word in the list. Five ideas make that affordable, and each shows up more than once.

**Precompute whatever does not vary.** The feedback for a (guess, answer) pair is fixed, so it becomes a table (§1), or, where the table cannot be held, the fraction of it one search will read (§10). The quantity `w log w` depends on the candidate and not on the guess, so it leaves the inner loop (§8). `f` is always reached through an entropy, so it takes `x = log₂ n` instead of `n` and saves a round trip through exponentiation (§8). In every case a quantity inside a doubly-nested loop depends only on the outer index.

**Cache whatever recurs.** Positions repeat constantly, with at most 243 of them behind a fixed opening, so results are memoized on the candidate set. That is the largest single speedup here (§12). The compiled kernels are cached to disk (§1) and the fitted `f` to a JSON file (§7), for the same reason at longer timescales.

**Bound before computing exactly.** The two-turn value is exact, costs one pass, and evaluated at a capped penalty is a valid lower bound on every deeper value. That makes it both a leaf and the ordering that prunes the search above it (§10). The branch cap sits on top of that as the one step that gives up a guarantee. The ablation below shows how little it does, because the bound already prunes to well inside the cap.

**Collapse things that are the same.** §2's reframing, that a guess is its partition, lets the search treat thousands of words as one move (§10). Recognizing an equivalence is usually cheaper than a numerical shortcut, and it never costs accuracy.

**Measure where the work is.** Three assumptions had to be given up: that the wide positions dominate, when the small ones are scored far more often; that the cost is arithmetic, when it is memory traffic at one cache line per scattered lookup (§8); and that the branch cap was carrying the endgame, when the bound underneath it was. Each changed what the right optimization was, and the second ended the search for further ones. The value of an optimization that has never been taken out is a guess.

`analysis/ablations.py` takes each idea back out with everything else left alone. Two warnings about reading it. The three groups are different operations, one scan, one search and one sweep, so a factor from one says nothing about another, and the naive whole is **not** the product of these parts. Every figure is a slowdown against the shipped code, so larger means the idea was worth more:

<!--table ablations.table-->
| taking one idea back out | at the opening | at a mid-game position |
| --- | --- | --- |
| `w log w` back inside the inner loop | 4.9× | 1.9× |
| no table — the coloring rule applied per pair | 47× | 11× |
| one core instead of 16 | 8.6× | 7.9× |
| **all three at once, in plain Python** | 12,570× | 3,362× |
<!--/table-->

The other two levels are one figure each. Lifting the branch cap over <!--n ablations.search_positions-->150 real endgame positions costs <!--n ablations.branch_cap .1f-->2.6×. Playing all 12,947 answers with the three position caches held to a single entry costs **<!--n ablations.memo.at12947 .0f-->100×**. That figure climbs with the length of the run, since the longer a memo has been filling, the more of what a later game asks for it already holds. The same ablation over <!--c-->400 answers costs <!--n ablations.memo.at400 .1f-->14.7×.

The bottom row of the table answers the question in the heading. A direct reading of `E(g, S)`, with no table, no rearrangement and no compiled kernel, takes **about seven and a half minutes to choose a single opening guess**, before any endgame search. `train.policy_iteration` asks for a guess a quarter of a million times.

The branch cap's figure misleads, and counting shows it where timing does not. `analysis/branch_cap.py` solves <!--n branch_cap.positions-->150 real positions twice, once keeping the fifty best splits and once keeping every one:

<!--table branch_cap.table-->
| over 150 real positions | cap 50 | no cap |
| --- | --- | --- |
| interior nodes searched | 180 | 180 |
| subtrees evaluated | 355 | 9,086 |
| splits evaluated at the busiest single node | 50 | 8,781 |
<!--/table-->

The cap changes no interior node. The same <!--n branch_cap.nodes-->180 subtrees are searched either way, and all <!--n branch_cap.same_guess-->150 positions come back with the same guess at the same value, which checks the claim §10 makes for it. What it changes is the third row. At <!--n branch_cap.unbound_nodes-->179 of the <!--n branch_cap.nodes-->180 nodes the bound cuts the loop after one or two splits, and fifty is a limit nothing comes near. At <!--n branch_cap.binding_nodes-->1, the bound fails to separate anything, and the uncapped search goes through <!--n branch_cap.busiest.none-->8,781 splits to arrive at the guess it already had.

So the endgame is pruned by the admissible bound rather than by the cap, and thoroughly enough that the cap is idle almost everywhere. The cap insures against the rare node where the bound says nothing.

The <!--n ablations.branch_cap .1f-->2.6× above averages the cap's effect over <!--n branch_cap.positions-->150 positions, which reads as a modest saving taken evenly from all of them. It is nothing at <!--n branch_cap.unbound_nodes-->179 nodes out of <!--n branch_cap.nodes-->180 and a large saving at the one, from <!--n branch_cap.busiest.none-->8,781 splits down to fifty. Timed over forty positions instead of a hundred and fifty, the figure falls to <!--u-->1.0×, because the smaller sample missed that node. Averaging a rare, large effect over the occasions it does not fire makes it look small and steady.

None of this was mechanical. §8's rearrangement is algebra, and a two-turn formula serving as both a leaf and the bound above it is specific to this game.

None of the five makes the solver play better, since it picks the same words without them. The branch cap is the only one that could differ, and the counts above say it does not. What they bought is scale, and §15 is what it was spent on: twenty thousand games, two hundred openings played out in full, every answer swept again under each reweighting.

## 17. A year to fit `f` once

§16 priced each idea by taking it out, and warned that the factors do not compose, since they are three operations against three baselines. One composition is legitimate. Take the one operation that was measured end to end with nothing left in it, an opening scan at seven and a half minutes, and count how many of those the project performs.

Nothing in the right-hand column below was run to completion. Each figure is the one measured scan times a counted number of scans. These are projections where the rest of this document's numbers are measurements (`analysis/ablations.py --projection`):

| what the project does once | shipped | with none of it |
| --- | --- | --- |
| choose the opening guess | <!--s ablations.projection.opening.shipped-->32 ms<!--/s--> | <!--s ablations.projection.opening.naive-->6.9 minutes<!--/s--> |
| play every answer once, all 12,947 games | <!--n timing.sweep.expected.python .1f-->11.0 s | <!--s ablations.projection.sweep.naive-->62 days<!--/s--> |
| `train` — fit `f` for all five shipped priors | <!--n timing.train.python .1f-->40.4 s | <!--s ablations.projection.train.naive-->355 days<!--/s--> |

The count is the same kind of thing in both lower rows because of the memo. Every game in a sweep and every rollout in a fit opens at the same position, the full word list, so the shipped solver scans it once per process and hands the answer to the other 12,946 games. Without the caches, each game pays the seven and a half minutes again. What happens after the opening barely registers. The position left after `salet` holds <!--n ablations.midgame.candidates-->268 candidates, and all the later turns of a game together add a few percent to a cost the first turn has set.

Divide the top row by the work inside it. One opening scan is 12,947 guesses against 12,947 candidates, **167,624,809 (guess, answer) pairs**, each colored and each dropping its candidate's weight into one of 243 buckets. Pure Python spends about **<!--n ablations.projection.us_per_pair_python .1f-->2.5 microseconds** on a pair: a function call, the two-pass rule of §1 over five characters, a dictionary update. The shipped kernels spend about **<!--n ablations.projection.ns_per_pair .1f-->0.2 nanoseconds** across sixteen cores, roughly three nanoseconds on each, which is about what one cache line costs to arrive from memory. The whole <!--n ablations.opening.python ,.0f-->12,570× is the distance between calling a function and fetching one byte from a table. That is why §8 found the loop memory-bound, with nothing else left inside it.

The ratio grows down the table because the levels stack. The first row is the scoring optimizations alone. The second puts the memo on top of them, so the opening scan happens once instead of thirteen thousand times. The third is more of the same: policy iteration plays 2,000 games a round, <!--n results_tables.rounds.min-->4 to <!--n results_tables.rounds.max-->6 rounds per prior, <!--n results_tables.rounds.priors-->5 priors, and every guess any of them asks for is priced by the first row.

That is not this project with a longer wait. §15 rests on five measurements, and every one is a sweep or a fit. At the right-hand column's prices the cheapest is a season's work, so none of them gets made, and the finding they add up to stays a plausible guess.

---

## Where to look in the code

In reading order:

| where | what it establishes |
| --- | --- |
| `wordle_data.score_guess` | the feedback rule and the precomputed table (§1) |
| `kernels._ig` | the entropy scorer (§3) |
| `priors.build_weights` | the weight vector over answers (§4) |
| `estimator.effective_size` | effective size (§4), and the rearrangement that makes it cheap (§8) |
| `priors._repeat_shape` | the repeat shape, binned by first appearance (§5.1) |
| `priors._curation_shape` | the curation shapes, WordleBot and frequency (§5.2, §5.3) |
| `kernels._expected_guesses` | the expected-guesses scorer (§6), inside the scoring loop (§8) |
| `estimator.FORMS`, `estimator.fit` | `f`, its functional forms, and the monotonicity guard (§6, §7) |
| `train.policy_iteration` | policy iteration (§7) |
| `exact.ExactEndgame` | the endgame search, and the loss penalty (§9, §10) |
| `source::Columns` | the same search with no table behind it (§10) |
| `memo.Memo` | the bounded LRU every position cache uses (§12) |
| `solver.WordleSolver._best_guess_uncached` | the dispatch deciding which of the above runs (§13) |

Entries are named because line numbers drift with every edit. Every entry is greppable: `grep -rn "def score_guess" .`
