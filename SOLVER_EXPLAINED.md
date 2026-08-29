# How to read this document

This file is Markdown with LaTeX display equations, so it wants a viewer with math support. `ABOUT_SOLVER_EXPLAINED.md` says how to get one, who this is written for, and what order to read it in.

Read it start to finish the first time. The order is the argument. If you have ten minutes rather than an evening: §2, §6, §15.

---

# The Wordle solver, from the ground up

## 0. What the program has to do

Every turn, the solver is holding a set of words that could still be the answer. It plays a guess, sees five colored tiles, and crosses off everything the tiles have ruled out. That is the game. The whole program is one question:

> **Which guess shrinks the set the most usefully?**

Entropy, the fitted curve `f`, the endgame search — all of it is machinery for answering that one question.

Two word lists matter. The solver may play any of **12,947** words, and it treats every one of them as a word that could turn out to be the answer (`ind2word.json`). Sitting inside that list is the **2,309**-word answer list Wordle shipped with in 2022 (`wordles.json`), which is where the New York Times editor draws from most days. The distance between *may be guessed* and *is likely to be the answer* is where most of this document lives.

Three things have to exist before the question above can even be asked precisely: a way to compute the tiles (§1), a way to describe what a guess *does* to the candidate set (§2), and a way to measure how good that is (§3).

## 1. The feedback rule, and why there's a 167 MB file

Wordle's coloring rule is the kind of thing everyone knows until they have to write it down. It runs in two passes. Greens go out first, one for each letter sitting in the right place. Yellows are then handed out from whatever letters the answer has left over, at most one per remaining copy.

Duplicates are where that ordering starts to bite. Two examples against the answer `abide`, both of them checked by running `wordle_data.score_guess`:

```
guess  speed   ->  N N Y N Y      guess  geese   ->  N N N N G
       s p e e d                         g e e s e
```

`speed` has two E's and `abide` has one. The first E claims the yellow and the second gets nothing — there is no second E to be yellow *about*. The D is yellow too, being in the answer but not in position four.

`geese` has three E's, and this is the case that catches people. The E in position five is an exact match, so it goes green. Because greens are assigned first, it consumes the answer's only E, and the two E's that come *earlier* in reading order get nothing at all.

A pattern is five tiles of three colors, so write it in base 3 with N=0, Y=1, G=2. Every pattern is then an integer in `[0, 243)`: `NNYNY` is 10, `NNNNG` is 2, and the one you want, `GGGGG`, is 242 (`WIN_PATTERN`).

Now the engineering move the rest of the project rests on. The feedback for a given (guess, answer) pair never changes — it is a fact about two words, not about the state of a game — so it can be computed once and looked up forever:

```
feedback_table[guess_index, answer_index] = pattern number   # one uint8
```

That is 12,947² bytes, about 167 million of them, which is the `feedback_table.npy` file. Building it takes a while. After that, "what would I see if I played this?" is one array lookup rather than a pass over two strings. It is the same instinct as a prefix-sum table: pay a quadratic cost once, and every query afterwards is free. Nearly every inner loop in this project is a walk along one row of that table.

Nearly, because the table is a speed choice and not a necessity, and the choice is closer than 167 MB makes it sound. Applying the rule from scratch to a pair costs only a few times what a lookup does, and on a machine that can hold the file that ratio settles it: the file wins. In a browser tab running the same solver compiled to WebAssembly, where 167 MB can be neither downloaded nor rebuilt, it settles the other way and the rule is simply run.

### What the program is actually holding

Underneath all of this is a shape in memory. Three later arguments — the rearrangement and the memory bill in §8, the packed keys in §10 — turn out to be arguments about that shape rather than about Wordle, so it is worth setting down before any of them arrive.

After startup, words are never compared as text. Each of the 12,947 is handed an index once, and from then on a word simply *is* its index. (`ind2word.json` is that numbering; `word2ind.json` is the same thing read backwards.) So:

- the table is a 12,947 × 12,947 grid of single bytes, where **the row is the guess and the column is the answer**;
- a guess is a row number, and "score every guess" means visiting all 12,947 rows;
- the candidate set is an array of *column* numbers in ascending order — `[3, 17, 964, …]` — which starts as all 12,947 and gets shorter every turn as feedback rules words out (`kernels.filter_candidates`).

Filtering keeps the order but not the spacing. By turn three the survivors are forty-odd columns scattered across the width of a row of thirteen thousand, and that spread is what §8 turns out to be paying for.

Two consequences follow, and both come back.

**The table is read along rows, and that is deliberate.** A row is laid out contiguously, so scanning one for a *whole* candidate set walks straight through memory. Scanning it for a scattered handful does not, and §8 is where the bill for that arrives.

**The loops that scan it are compiled, not interpreted.** Everything in `kernels.py` carries numba's `@njit`, which turns the function into machine code the first time it runs; Python's own loops would be a hundred times too slow for thirteen thousand guesses against thirteen thousand candidates. One such compiled scoring loop is what this document calls a **kernel**, and `kernels.py` is nothing but a handful of them. The compiled code is cached to disk, so only the first run of a fresh checkout pays for it. That compile, plus loading the table, is the startup wait the web page hides behind its veil.

## 2. A guess is a partition

Here is the reframe the whole solver is built on.

Fix the candidate set `S` and a guess `g`. Every candidate in `S` produces *some* pattern against `g`. Group them by which one:

```
guess = "crane", S = {every 5-letter word}
  pattern NNNNN  -> 4,283 candidates
  pattern NNNNG  ->   612 candidates
  pattern GNNNN  ->   371 candidates
  ...            (about 150 non-empty buckets)
  pattern GGGGG  ->     1 candidate  (crane itself)
```

You are going to land in exactly one of those buckets, and that bucket becomes your new candidate set. So a guess is not really a word at all. **A guess is a partition of `S` into at most 243 buckets**, and the only thing that matters about it is the shape of that partition.

Two words that partition `S` identically are the same move. That sounds like an idle observation. It is what makes the endgame search of §10 affordable.

A bad guess leaves one enormous bucket, because whatever you learn, you barely narrow anything. A good guess chops `S` into many small ones. Now "many small ones" has to be made precise.

## 3. Entropy, from scratch

This is the one piece of genuinely new mathematics in the document, so I will build it rather than cite it.

Start with a simpler question. I am thinking of one of `n` equally likely words, and you may ask yes/no questions. How many do you need?

Binary search: each question cuts the set in half, so **log₂(n)** questions do it. Call that log₂(n) *bits* of uncertainty. A thousand words is about ten bits, and the useful thing about the phrasing is that it converts "how big is this set" into "how many questions is this set worth".

Now make the words unequal. Suppose one has probability ½, another ¼, and four more have 1/16 each. Counting says log₂(6) ≈ 2.58 bits, and counting is wrong. Half the time you ask "is it the first one?" and you are *finished*. The rare words are the only ones that cost you anything, and they are rare.

So weight each outcome by how often you actually pay for it. An outcome of probability `p` feels like a set of size `1/p`, worth log₂(1/p) = −log₂(p) bits, and averaging over the outcomes gives

$$H = -\sum_i p_i \log_2 p_i$$

Run the example through it — the half-probability word costs 1 bit, the quarter-probability word 2, the four rare ones 4 each — and it comes to ½·1 + ¼·2 + 4·(1/16·4) = **2 bits**, comfortably under the 2.58 that counting claimed. Concentrated probability means less uncertainty. That is the whole idea. `H` is largest, at log₂(n), when everything is equally likely, and falls to 0 when you already know the answer.

**Information gain** is what a guess does to that number. Before you guess, the uncertainty is `H(S)`. Afterwards you are in bucket `r` with probability `p_r` and still uncertain by `H(S_r)`, so what you gained is the difference:

$$\mathrm{IG}(g) = H(S) - \sum_r p_r H(S_r)$$

And now the pretty part, which is also why the code is fast. Knowing which word it is amounts to knowing *which bucket* plus knowing *which word inside that bucket*. Uncertainty splits along exactly the same seam:

$$H(S) = \underbrace{H(p_1, \ldots, p_k)}_{\text{which bucket}} + \sum_r p_r H(S_r)$$

Substitute that into the line above and the bucket terms cancel: **the information a guess gains is exactly the entropy of the bucket masses.** You never have to look inside a bucket. That is `kernels._ig` — total up each pattern's mass, where a bucket's mass is the combined weight of the candidates landing in it, and accumulate `h -= p*log2(p)` over the buckets.

That rule is `--mode entropy`: play the guess whose partition has the highest bucket entropy. It is the standard solver, the one most people build, and here it averages **3.75 guesses**. Every measured figure in this document, that one included, is scored under the prior §4 is about to construct. That matters more than it sounds: the same rule reads worse under a flat prior, and §4 explains why those two scores cannot be set against each other.

So 3.75 is the baseline, and the rest of the document is about beating it. There are two ways to try. One is to search harder: look further ahead, solve the endgame exactly. The other is to notice that §0's question was never quite right. It quietly assumed every word is equally likely to be the answer, and no word ever was. The second turns out to be worth far more than the first, which is §15's punchline.

## 4. Not all words are equally likely (the priors)

Greedy entropy treats all 12,947 words as equally plausible answers. That is badly wrong, and correcting it buys more than any search improvement in the project.

`priors.build_weights` models the editor as making two decisions in sequence. With probability **0.10** she reuses a word that has already been an answer. Otherwise she picks something unused, and with probability **0.10** that word is **off** the curated list. Three groups fall out — used, unused-and-curated, unused-and-non-curated — and every word keeps strictly positive weight, so nothing is ever ruled out entirely. Note where the second rate sits: it is conditional on the word *not* being a repeat, because the model is sequential. Measure it against all puzzles instead of against non-repeats and you will understate it.

Both of those 0.10s look far too aggressive against Wordle's full history, and the reason they are not is that the history contains a regime change. Over the current editorial regime — Wordle #1690 onward, from 2026-02-02 — repeats run at **10.8%** and off-list answers at **9.4%**. Over all 1,892 puzzles ever published they would be a fraction of that, because for years neither happened at all.

What drives the change is mechanical, and it is the kind of thing a model should be built on. The curated list is nearly spent: about 1,822 of its 2,309 words have been used, and roughly 333 more go every year. That leaves something like a year and a half of runway. Repeating old answers and reaching outside the list are the same editorial response to the same shortage, they started together, and 0.10 is a forward-looking rate rather than a historical one.

**The prior is where the wins are.** Playing every word as the answer, each one weighted by that row's own prior:

| priors on | greedy entropy |
| --- | --- |
| none | 4.12 |
| repeat only | 4.10 |
| curation only | 3.85 |
| both | 3.75 |

Resist reading that column as a scoreboard. Each row weights the words by its own prior, so a sharper prior is scoring an easier game, and no two rows describe the same population — what the column shows is how much of the difficulty was in the belief rather than in the board. There is only one column for a related reason: every figure here is greedy entropy, the one rule built so far, and what the prior is worth to a solver *trying to minimize guesses* is §6's to show and §14's to check against the puzzles the NYT actually set.

One reading does survive that caveat, because it compares a prior against itself rather than against another. Switching the repeat prior on moves 4.12 to 4.10 — two hundredths of a guess, on a game that same prior chose the difficulty of. A belief that can barely make its own examination easier is not doing much work, and it is the first sign of something §14.3 takes apart properly: alone, the repeat prior is nearly idle, and what it is really for is correcting the other one.

Three groups is a statement about how much mass each group gets. It says nothing about how the mass spreads *within* a group, and the obvious answer — spread it flat — is wrong in both directions. Flat says `cigar`, the very first Wordle, is no likelier to come round again than a word played last month. It also says `squid` and `xylyl` are equally plausible as tomorrow's answer. The second mistake is the expensive one: the non-curated group holds 10,638 of the 12,947 words, so it dominates the candidate *count* at nearly every position while carrying under a tenth of the mass. Call the within-group spread a group's **shape**. Both shapes started flat and neither is now — but before either can be described, the weights have broken something.

### How big is a weighted set?

The weights just built break something §2 and §3 leaned on without saying so. "This bucket has 40 candidates" stopped being informative the moment the words stopped being equally likely: forty candidates with one of them at 90% is a much easier position than forty even ones. Everything from here on needs a size that knows the difference.

The fix is to reuse §3's machinery. Call it the **effective size** of a set (`estimator.effective_size`):

$$n_{\text{eff}} = 2^{H}$$

where `H` is the entropy of that bucket's *normalized* weights. If the weights are flat, `H` is log₂ of the count and `n_eff` is the count exactly. As probability concentrates, `n_eff` slides toward 1. Read it as "how many equally likely words would be this hard".

That is what this document means from here on whenever it asks how big a position is. It is also the honest way to ask how big a *prior* is, which is the first thing §5 needs.

## 5. Shaping the mass inside each group

### 5.1 The repeat group: a smoothed histogram

There are 22 repeats in Wordle's history so far, and they are not drawn evenly from the pool of used words. Twenty of them come from the first three hundred puzzles, with a median around #120. Only two sit outside that window: `sandy` from #1227 and `smile` from #831. Whatever the editor is doing, it looks like replaying the *beginning* of the run.

Two in twenty-two is too high an outlier rate to write off everything past that window, though. So the shape is a **Laplace-smoothed histogram**. Bin the used words by the puzzle where they first appeared, two hundred wide, and give bin `b` the weight

$$w_b = \frac{r_b + \alpha}{n_b}, \qquad \alpha = 0.5$$

where `r_b` is how many repeats came out of that bin and `n_b` how many used words live there; each bin's share is then spread evenly over its members. That `α` is the whole reason `sandy` does not break the model — it keeps a bin that has never yet produced a repeat from being written off completely. It is the "maybe next time" term, and it is why no word's weight ever reaches zero. In the end `cigar` carries about six times the flat repeat weight and the late words fall below it.

The bins are fixed rather than sliding, and *that* is the part worth learning from. A sliding window over the recent past is the more natural-sounding model, and it needs a boundary. A boundary is a parameter, and 22 events cannot afford to feed a parameter and be predicted by it.

The way to settle that without fooling yourself is **prequential scoring**: predict each repeat using only the repeats before it, so no parameter ever sees the event it is graded on. Scored that way, the histogram beats a flat spread by about **+1.18 ± 0.20 nats per event**. That `±` is a standard error — how far the estimate itself would wander if the same 22 events were drawn again — so the useful reading is that the gap is nearly six times its own uncertainty, which is a gap to believe.

The windows lose, and the whole field is worth seeing at once. Scored the same way over the same 22 events (`analysis/repeat_shapes.py`), in total nats rather than per event:

| model | log-likelihood | vs. a flat spread |
| --- | --- | --- |
| flat over the used words | −164.6 | — |
| **binned by first appearance, α = 0.5** | **−138.8** | **+25.9** |
| best of sixteen sliding windows | −149.6 | +15.1 |
| worst of those same sixteen | −252.6 | −88.0 |
| a repeat rate refitted as events arrive | −156.3 | +8.3 |

How that comparison is set up is what makes losing mean something. The sixteen windows are four choices of how many recent repeats fix the centre against four widths, and the row above is the *best* of them, picked after seeing how each one did. That is hindsight the histogram is never given, and it still trails by eleven nats across the 22 events — about half a nat apiece. Choose the boundary badly instead and the model scores far below knowing nothing at all, which is the other half of the same point: the parameter has to be right, and there is nothing to set it with. Twenty-odd events cannot pin a rate down, and a histogram has no rate to pin.

### 5.2 The off-list group: WordleBot's own weights

This is the interesting group, because "which words could be off-list answers" turns out to be *knowable* rather than guessable. Getting there is a small piece of detective work.

The first clue is sitting in `solutions.json`. Its `id` field is not an opaque key but a position in the list the editor draws from — for the first 284 puzzles it is exactly the word's place in `wordles.json`, which is stored in that same order. Nearly every off-list answer carries an id just past the end of the curated 2,309. Sort those ids and the words come out in alphabetical runs — twelve at 2318–2331 holding `beaut laser mommy muggy nervy pious primp shred spate taupe tinge tizzy`, then seven more at 2343–2352. Twelve words landing in alphabetical order by chance is a one-in-a-billion event.

A run is one batch. The NYT sorted a set of new candidate words, appended it whole, and the sort order survives in the numbering; where a run ends and a word lower in the alphabet begins the next one is the seam between two appends. The gaps *inside* a run matter as much — 2319 and 2320 sit between `beaut` and `laser`, holding words nobody has seen, because a slot becomes visible only when its word is used. Three seams inside 2310–2397 put the appended region at roughly ninety words, so off-list answers are being drawn from that rather than from 10,638.

Which is where the trail ends, because the block cannot be enumerated. The NYT serves a puzzle by date, so a slot's number arrives only once its word has been the answer, and the half of the block that matters is exactly the half still invisible; bracketing a gap alphabetically leaves four thousand candidates between `beaut` and `laser`. Nor is any of it a test to run today — on 2026-02-02, the day the first repeat appeared and §4's regime begins, the NYT re-sorted the list alphabetically and carried on numbering from the end of it, so an id now places a word in the alphabet rather than on one list or the other. `analysis/id_eras.py` reconstructs both numberings. What marks a word off-list is list membership and nothing else.

But somebody has already worked out a list that does name them. **WordleBot**, the NYT's own solver, holds about 3,200 words it treats as plausible solutions, each carrying a probability drawn from how often the word appears in the Times. Two things about where that comes from matter here. It is built by the paper's Upshot desk and not by the Games desk that picks the answers, and its probabilities are a frequency count over old articles, so the weights are evidence about English usage rather than a view of the puzzle schedule. And the NYT publishes neither the list nor the weights: `wordletools.azurewebsites.net/weightedbottles` reconstructs both from what the bot discloses as it plays, and that reconstruction is what `wordlebot.py` scrapes.

It covers a few thousand words, of which 892 are outside the curated list — and all 48 off-list answers to date are among them.

That last fact proves nothing on its own. A list that quietly absorbed answers *after* they were used would look exactly the same. The test that matters is whether an *old* copy predicts *later* answers, and one exists: the Wayback Machine has a snapshot from **2023-10-12**, frozen nearly three years ago. That copy is vendored beside the live one, `wordlebot.py` fetches both, and `analysis/curation_shapes.py` scores them, so this is recomputed rather than remembered:

| the 2023-10-12 snapshot | |
| --- | --- |
| off-list answers used after it | 44 |
| ...already in the snapshot | **42** |
| the misses | `loris`, `kefir` |
| words added to the list since | 83 |
| ...of those, ever an answer | **2** (the same two) |

So the old copy predicts forward at about 95%, which is the number the shape below is built on. Whether the two it missed were later *backfilled* is a question the archive cannot settle: both are absent from the capture of 2025-05-25 and present in the one of 2025-09-16, with nothing archived in between — exactly the window in which both were used. A fresher copy would not have helped either, the 2025-05-25 list missing them too, so scoring each answer against the closest snapshot that predates it gives back the same 42 of 44.

The likelier reading is that the bot did hold them before they were played, since it carries no hedge — a word outside its 3,200 has probability zero rather than a small one — and the NYT says it has never lost a game; but that turns on what the bot does once its candidate set empties, which nothing here can measure, so it stays a conjecture.

The shape follows directly from that number. **95%** of the off-list mass goes to the listed words in proportion to their weight, and the remaining **5%** spreads over everything else — a hedge set at the measured miss rate, which is what stops the next `loris` from being a blindside.

Scored prequentially against the off-list answers, the way §5.1 scored the repeat shape, with each shape set as it actually ships — WordleBot's hedge at 5%, the frequency shape at its shipped temperature:

| shape | nats per event | effective support |
| --- | --- | --- |
| uniform | −9.27 | 10,638 |
| answer-like frequency (§5.3) | −8.15 | 4,857 |
| **WordleBot, 5% hedge** | **−6.46** | **911** |

Effective support is §4's effective size applied to the prior itself: how many equally likely words the shape is really spreading over. It is the honest way to say that a distribution is concentrated on about a thousand words rather than spread over ten thousand. The middle row is the fallback §5.3 builds, listed here so that all three can be read against each other. Score any of them at its own best-fitting setting instead and the lower two both improve, which is why the shipped settings are the ones in the table: a shape has to be scored as it will actually run.

**The 5% hedge stays flat, and that is deliberate.** Weighting it by how answer-like a word looks is the obvious refinement, and it fails on exactly the population that matters. The hedge only ever fires on a word the list is *missing*, and a word goes missing because it is odd. `loris` sits mid-pack on answer-likeness and would have received about a third of its flat share. Over the only two events that have ever needed the hedge, shaping it costs rather than helps. Flat is the honest expression of not knowing which word the list is short of.

### 5.3 The fallback that needs no scrape

Depending on somebody else's website is a fine thing to vendor and a bad thing to *need*, so there is a second shape built only from what the repo already holds.

The trick is to notice that the curated list is itself 2,309 labeled examples of "a word an editor thought answerable", and those labels are independent of the 48 off-list events. So `answer_like.py` fits a logistic regression with the curated words as positives and the rest of the list as negatives, then scores the unused words by its log-odds. Thirteen features go in. Four of them carry the model, and all four are legible:

| feature | weight | reading |
| --- | --- | --- |
| Zipf frequency | **+4.00** | common words get picked |
| ends in `s` | **−5.87** | a third of the pool is a plural; under a tenth of the off-list answers are |
| ends in `ed` | −5.09 | past tenses do not get picked either |
| foreign lift | −1.16 | as common abroad as at home means a name, not a word |

The other nine are not decorative: a model keeping only these four correlates at about r = 0.98 with the shipped scores and still reproduces none of them.

What comes out is recognizable. `squid` is promoted several times over the flat weight it replaces, along with `maven`, `aloha` and `kazoo` — every one of which is a real off-list answer. Demoted by an order of magnitude are `karks`, `fents`, `xylyl` and `faffy`, which are the kind of never-curated oddity §11's losses collect on. As a model it lands between the flat spread and WordleBot's list — the middle row of §5.2's table — and that gap is the price of owing nobody a scrape.

With the weights built and a size that respects them, the question §0 asked can finally be put in the units the game actually scores.

## 6. The real objective: minimize expected guesses

Entropy answers "which guess is most *informative*". The game does not score you on bits. It scores you on **guesses**, and those are not the same thing. The gap between them is where the next two thirds of a guess live.

Suppose you had a function

> **`f(n)`** = the expected number of further guesses needed from a position of effective size `n`

Then the value of a guess writes itself down:

$$E(g, S) = 1 + \sum_{r \ne \texttt{GGGGG}} p_r f(n_r)$$

One guess for this turn, plus the average cost of wherever you land. Two details in that formula carry real weight.

**Excluding the `GGGGG` bucket is the whole reason a possible answer beats an equally informative non-answer.** That bucket costs nothing more, because you have already won. So a guess that might simply *be* the answer collects a free discount equal to its probability of being it.

**Buckets of one and two skip `f` entirely**, because their values are known exactly: one candidate is worth 1, and two are worth `2 − p_max` — guess the likelier one, and if you are wrong you now know the other.

Where `f` comes from is §7's problem. What `f` has to *look like* is settled here, and the answer is sharper than it sounds.

How much is any of it worth? With both priors on, minimizing this beats maximizing information by about two thirds of a guess — §3's 3.75 down to 3.07. Switch both priors off and the same machinery buys a thirteenth of that. A better objective doing almost nothing under a flat prior is a strange enough result to want a reason, and the reason is an identity rather than an accident.

### A straight-line `f` is entropy plus a bonus

Suppose `f` is **linear**: `f(n) = a + b·x`, with `x = log₂(n)`. Substitute it into `E`, use the chain rule from §3, and everything collapses:

$$E(g,S) = 1 + a(1 - p_{\text{win}}) + b\big(H(S) - \mathrm{IG}(g)\big)$$

`H(S)` does not depend on `g`, so minimizing `E` is *identical* to maximizing `b·IG(g) + a·p_win`. That is greedy entropy with a bonus bolted on. **A linear `f` buys you nothing but the bonus**, and every remaining gain comes from `f` being *concave* in `x`.

The identity is testable, because one of the fitted forms is linear in `x` and so ought to score exactly "entropy plus the bonus". It does, which is what lets the gain over greedy entropy be split in two — and the split is worth seeing under every prior at once (`analysis/results_tables.py --part priors`):

| priors on | possible-answer bonus | curvature in `f` |
| --- | --- | --- |
| none | +0.051 | +0.001 |
| repeat only | +0.090 | +0.004 |
| curation only | +0.312 | +0.039 |
| **both (the default)** | **+0.638** | **+0.040** |

Read the bottom row first. Nearly all of the improvement over a standard entropy solver is the free discount for guessing a word that might be the answer, and the fitted curve's shape — the thing §7 works hardest for — is about a sixteenth of it.

Then read the columns, because both halves turn out to be the prior's doing rather than the objective's. The bonus climbs by more than tenfold as the prior sharpens, which is what §15 is about. Curvature is not a constant ~0.04 either: it is worth that much wherever the **curation** prior is on and next to nothing where it is off. Why a prior should decide whether a curve has anything to offer is a question the fitted curve answers below, once there is a curve to look at.

It is worth saying plainly that none of that bonus needs fitting. Written in its own terms the rule is `IG(g) + a·p_win(g)`, with no `f` anywhere, and two exact facts fix `a` with no data at all: one candidate takes one guess and two take one and a half, so a bit of information is worth half a guess, and winning outright saves a whole one — two bits' worth. A solver playing that rule collects essentially the same bonus with no estimator, no fit and no answer log behind it. `README.md` carries it as a second reference rather than as the baseline, since it pays for those guesses with a much higher loss rate.

The solver's **hard mode** is the cleanest test of the identity, because it isolates the bonus from everything else. It requires every guess to be a word that could still be the answer — narrower than the NYT's rule of the same name, which only asks that the letters already revealed be reused. Restricting greedy entropy to it therefore forces `p_win` up without touching entropy's scoring rule at all — and it gains about a quarter of a guess, which is roughly two fifths of what the objective gets by *pricing* the same bonus instead of imposing it.

The objective does better because it takes the bonus **and** keeps the probes. It plays a non-candidate whenever the split is worth more than the win chance it gives up, which is exactly the trade hard mode forbids. Handing it the wider pool the NYT's own rule allows shows the same thing from the other side: the objective declines those probes across all 12,947 answers, where greedy entropy takes 2,922 of them and is worse for it.

Curvature is what is left once that bonus is priced, and some of it is expected before any fitting happens. A guess cuts a candidate set by a *factor* rather than by an amount, and how large a factor is on offer depends on how much there is to cut: two candidates admit one bit and no more, where a few dozen admit four or five, one guess being able to send them to as many as 243 different patterns. Bits per guess therefore rise with the size of the set, guesses per bit fall, and `f` flattens as `x` grows. The curve fitted for the default prior is a piecewise-linear spline that obliges:

```
f(n) = 0.945 + 0.375·x − 0.099·(x−2)₊ − 0.194·(x−4)₊
                                            (x = log₂ n, (u)₊ = max(u, 0))
```

A knot is a value of `x` where the curve is allowed to change slope; both of these bend it flatter and neither bends it back, taking it from 0.375 of a guess per doubling to 0.276 and then to 0.083. Both bends are spent below sixteen candidates, and above that the fit is a straight line. Set it beside the curve fitted under flat shapes:

| n | 2 | 4 | 16 | 100 | 1,000 | 12,947 |
| --- | --- | --- | --- | --- | --- | --- |
| `f(n)`, shipped shapes | **1.32** | **1.70** | **2.25** | **2.47** | **2.74** | **3.05** |
| `f(n)`, flat shapes | 1.33 | 1.75 | 2.17 | 2.58 | 3.10 | 3.67 |

The two halves of that table say different things. On the left both curves bend, and land within a rounding of each other doing it — this is the part a linear `f` cannot express, and it is bought entirely in positions of a handful of candidates. On the right neither bends at all, and what separates them is a slope: 0.083 of a guess per doubling against 0.156, so across the nine-odd doublings from sixteen candidates to the whole list the flat curve climbs about twice as far. That is the prior changing what `f` has to model rather than rescaling it. Concentrate the mass and effective sizes stay small, so the positions the solver actually meets sit where the curve is doing something instead of out along the straight part.

Which is what the curvature column was asking, and the answer is not that a flatter prior leaves `f` with no shape. Every fitted `f` in this repo is concave, the ones fitted with the curation prior off included — and the one fitted with no prior at all saturates hardest of them all, flat to three decimals by the time it reaches a thousand candidates. What those configurations lack is a *use* for the shape. A bend is only ever worth the guesses it reorders, and where the candidate sets are large and alike a bend and a straight line rank the same guesses the same way. That last step is a reading of the fitted curves rather than a measurement of its own.

One caveat matters when reading any of this: `f` is fitted **per prior**, so the curve above is the default prior's and no other. Turn a prior off and a different curve, in a different functional form, is what scores the guesses.

## 7. Where does `f` come from? Policy iteration

There is a circularity to deal with first. To play well you need `f`; but `f` is defined as the cost *under good play*. This is the same chicken and egg as a value function in reinforcement learning, and it is solved the same way — bootstrap, then iterate (`train.policy_iteration`):

```
round 0:  play 2,000 games with greedy entropy (which needs no f)
          -> for every position visited, record
             (n_eff, guesses actually needed from there)
          -> least-squares fit f to those pairs

round k:  play 2,000 games acting greedily on the current f
          -> record the same pairs
          -> refit
```

That is **policy iteration with function approximation**: *evaluate* the current policy by rollout, then *improve* it by acting greedily on the fitted value function. Every form of `f` is linear in its parameters — `a + b·x + c·√x`, where the *terms* are nonlinear but the *parameters* are not — so each refit is a single closed-form least-squares solve rather than an optimization.

You can watch it converge in the cached parameters. The entropy seed measures about 3.77 guesses, the first iteration takes it to 3.16, the second to 3.07, and the rest of the rounds wander in a small limit cycle around that. Essentially all of the gain arrives in one iteration. Because a greedy policy can oscillate, the loop keeps the parameters whose *measured play* was best rather than the ones it happened to stop on. Those cached parameters are themselves an optimization worth naming: fitting takes seconds, there is no reason to repeat it every run, so each prior combination's fit is written to `estimator_cache/` and loaded thereafter.

Three constraints are worth flagging.

**Training games are played with no six-turn deadline.** Otherwise a game that runs long gets truncated at six and the targets are biased downward. The deadline is a constraint layered *on top* of `f`, not part of it, which is §9.

**`f` must be non-decreasing** across the whole range (`estimator._is_monotone`). If a bigger candidate set could somehow score cheaper, the search would go looking for uninformative guesses that dump everything into one bucket. When the full basis violates that, terms are dropped from the end until it holds — which is why two of the forms in `README.md`'s table score identically: the guard cuts one of them back until it *is* the other.

**A prior with no cached fit does not play.** Rather than bending some reference curve onto the chosen form and playing a policy nobody ever measured, the estimator raises and names the `train` command that fixes it. Entropy mode never consults `f` and runs on anything.

### The form of `f` has to be refitted per shape

That cache is keyed per prior, and it has to be keyed per *shape* as well. There is nothing surprising in that on its own — the shapes are the weights, and `f` is fitted against the weights. What earns the subsection is the way a mis-paired `f` fails, because it does not fail by being slightly off.

`f` is queried at its largest argument exactly once per game: at the opening, where the candidate set is the whole word list. That is the far tail of the fit, extrapolated rather than supported by data. Change the shapes of §5 and every effective size moves out there, so a curve fitted for one pair of shapes can rank opening words wrongly under another.

The case to look at is `spline`, since it is the form that ships. Under the sharpened shapes it wins and opens `salet`. Under two flat shapes it wins again and opens `salet` again. Pair a *sharpened* repeat shape with a *flat* curation shape and the same family opens `raise` and falls to seventh of nine, behind every form that stayed concave. Nothing about the board changed — only which curve was fitted to which weights. This is estimator error, not a preference, and it is why the shapes are part of the fit's cache key.

The failure has a direction worth naming. `raise` is what greedy entropy opens with, and §6 explains why: a straight-line `f` reduces exactly to entropy plus a bonus. A curve that extrapolates badly goes flat in the tail, and a flattened curve is a straight one. So a mis-fitted `f` fails by collapsing into the very policy it was supposed to improve on.

The tempting way to catch that is to rank the forms by how well they fit, and it does not work. Under the shipped shapes, the two forms that open `raise` are also the two that fit worst, which looks like fit quality deciding the opening. It is one fact wearing two hats: both are the monotonicity guard above cutting those forms back to a straight line, which fits worse *and* collapses by §6's identity. Change the shapes and the two come apart entirely.

There is a configuration where the worst-fitting form of the nine opens `salet` anyway while two much better-fitting ones open `raise`. And `sqrt` makes the point from the other end: under one pairing of shapes it has the *best* fit of all nine forms and still only places fifth on true mean, while `spline` fits three places worse and is the one that flips the opening. What decides an opening word is concavity in `x`, which is §6's subject. How closely a curve tracks the points it was fitted to is a different question, and answering it does not answer this one.

## 8. The scoring loop, and where the time actually goes

Both scorers — §3's information gain and §6's `E(g, S)` — are the same loop with a different last line, and that loop runs more often than anything else here. Two things are worth knowing about it, and the second is the more useful: a rearrangement that removes nearly all of its arithmetic, and what the arithmetic turned out to have been hiding.

### The rearrangement that makes this cheap

**The problem.** `H` is defined on *normalized* weights, `p_i = w_i / m`. But `m`, the bucket's total mass, is not known until you have seen every member of the bucket — and which bucket a candidate lands in depends on the guess. Read naively, that means: for each of the 12,947 guesses, pass over the candidates to find the bucket totals, then pass over them *again*, taking a logarithm per candidate, to build each bucket's entropy. One logarithm for every (guess, candidate) pair.

**The fix.** Expand the definition and split the logarithm apart. Writing `m` for the sum of the bucket's weights:

$$H = -\sum_i p_i \log_2 p_i = -\sum_i \frac{w_i}{m} \log_2\left(\frac{w_i}{m}\right)$$

Use `log₂(w/m) = log₂ w − log₂ m` to break the coupled term in two:

$$H = -\frac{1}{m}\sum_i w_i \big(\log_2 w_i - \log_2 m\big)
    = -\frac{1}{m}\sum_i w_i \log_2 w_i + \frac{\log_2 m}{m}\sum_i w_i$$

That second sum is `m` all over again, so the whole term collapses to `log₂ m`, leaving

$$H = \log_2 m - \frac{\sum_i w_i \log_2 w_i}{m}$$

**Why that is a win.** Look at what each piece depends on. The mass `m` and the sum both range over the members of this bucket, which does depend on the guess. But the *summand* `w_i log₂ w_i` depends only on candidate `i` — on its raw prior weight, with no reference to the bucket, the guess, or the normalization. Normalizing is what coupled a candidate's contribution to a global property of its bucket. Splitting the logarithm decouples it again.

To see where that saves anything, it helps to have the loop it saves inside. Every scorer in `kernels.py` has the same two-phase shape:

```
for each guess:
    accumulate:  one pass over the candidates, adding each one's
                 contribution into a slot picked out by its pattern
    reduce:      turn those 243 slots into one number, the guess's score
```

The slots are the **accumulators**: four arrays of 243 floats, one entry per possible pattern, holding that bucket's running mass, its candidate count, its running total of `w log₂ w`, and its heaviest single weight. The accumulate pass is the hot one. It runs once per (guess, candidate) pair, which at the opening is 167 million times, so anything expensive has to be kept out of it.

So `w_i log₂ w_i` is computed **once per candidate**, before any guess is scored (`kernels._w_log_w`). The accumulate pass then adds a precomputed float into a slot where the naive version would have called a logarithm. The reduce reads back two slots it already has — the bucket's mass and its total of `w log₂ w` — and recovers `H` with one logarithm per *bucket* instead of one per candidate.

Count the cost for a position with `c` candidates scored against all 12,947 guesses. The naive version wants `12,947 × c` logarithms; the split version wants `c`. At a wide position that is 167 million against thirteen thousand.

There is a second consequence, smaller but tidy. Since `n_eff = 2^H`, the `H` the reduce just recovered *is* `log₂(n_eff)` — already in the units `f` wants. That is why `estimator.f_eval_x` is parameterized by `x = log₂ n` rather than by `n`. Every caller reaches `f` through an entropy, so every caller has `x` in hand, and going by way of `n` would mean computing `2^x` only for `f` to take the logarithm straight back off.

A sanity check on random weights, direct definition against the rearranged one:

```
direct : 2.331157535638251
split  : 2.3311575356382495     # agree to 1.3e-15
```

### Where the time actually goes

With the logarithms gone, what was left was not what the arithmetic had suggested at all. What follows is the rest of what the loop does and what it costs, and the reasoning is worth more than the loop: the obvious cost model was wrong twice, in two different ways.

**First wrong assumption: that the wide positions are the expensive ones.** They are the expensive individual *calls*. They are not where the time goes. Small candidate sets get scored far more often — every node of the endgame search, every late turn of every training rollout, each one scanning a handful of candidates against all thirteen thousand guesses. The aggregate cost lives in the small cases, so that is where the optimization has to go.

Start with the accumulators, because what they cost decides how the reduce gets written. Four fresh 243-element arrays per guess would mean allocating and zeroing some fifty thousand floats thirteen thousand times over. So they are allocated once per block of at most 64 guesses and reused across the block. That reuse puts them under an obligation: whatever one guess writes has to be gone before the next guess starts.

The reduce therefore carries two jobs — total the buckets up, and leave them zeroed. The obvious way to do both is a walk over all 243 patterns, which costs 243 whether there were twelve thousand candidates or five. With five candidates, at most five buckets are non-empty and the other 238 checks are pure waste, repeated for every one of thirteen thousand guesses.

So below a threshold the reduce **walks the candidates a second time instead**, taking each bucket the first time it is reached and zeroing it on the spot. Zeroing there is what makes the second walk correct as well as cheap: it clears the slot for the next guess, and it stops a bucket being counted twice when a later candidate lands in it. Five candidates then cost five reads and five writes rather than 243 of each.

The two branches total the buckets in different orders, so a position just under the threshold and one just over disagree in their last bits. That is harmless here, and measurably so: at the opening, the closest pair of scores among all 12,947 guesses sits millions of ulps apart. Nothing in this loop is decided by differences anywhere near rounding, which is also what makes branching on the candidate count safe in the first place.

**Second wrong assumption: that the cost is arithmetic.** It is memory, and seeing why means looking at what the loop actually touches. It fixes a guess, walks the candidate set in order, and reads a single byte per candidate. All of those reads come from one row of the table — the row belonging to that guess — at the column numbers §1 gave the surviving words. The accumulators they feed are 243 floats, two kilobytes, small enough to live in the cache nearest the core for the whole block. So the table is the only thing being streamed in, and the only thing that differs between one position and the next is how far apart the columns are.

That distance is the cost. Processors do not fetch single bytes. They fetch a **cache line**, a contiguous block of 64 bytes, and hold it near the core. Sixty-four adjacent columns therefore arrive on one fetch and the other 63 reads are free — which is the early game, where the candidates are most of the word list and pack the row densely. Late in the game the survivors are a few words scattered anywhere in an alphabetical list of thirteen thousand. Consecutive columns sit thousands of bytes apart, every read lands on a line of its own, and each one spends 64 bytes to use a single byte. That is the real bill at every small position. It is also why the second pass over the candidates is nearly free: it re-reads the very lines the first pass just pulled in.

How far apart the two cases can get is bounded, and the arithmetic is short enough to do. A row is 12,947 bytes, which is 203 cache lines. A dense run of `k` candidates costs `k/64` of them; a scattered set costs one line each, until it runs out of distinct lines to land on and pins at 203.

| candidates | packed together | scattered |
| --- | --- | --- |
| 20 | under a line | 20 lines |
| 200 | 3 lines | 200 lines |
| the whole list | 203 lines | 203 lines |

So the gap opens as far as a couple of hundred candidates and closes again, the dense figure still climbing while the scattered one is already at its ceiling. Under a couple of dozen the whole scan is too small to matter either way. The cost lives in the middle of the range — and so, it turns out, does everything the alternative layout below can win.

**Those blocks are also the unit of parallelism.** The guesses are cut into several blocks per core and handed to numba's `prange`, which spreads the iterations across threads. Each block carries accumulators of its own, so no two threads ever write into the same array. Several blocks per thread rather than exactly one means a thread that draws slow work does not hold everyone else up at the end.

### Storing the table the other way round

Indexing the table `[answer, guess]` instead is the obvious alternative, and it does fix the reads: a kernel written for that layout takes the same time on scattered candidates as on adjacent ones, which the shipped one plainly does not.

What it does not do is make the scatter go away. Fixing a guess and walking the candidates finishes one guess before starting the next, so a single set of 243 buckets serves a whole block. Fixing a candidate and walking the guesses leaves every guess in the block partway through at once, so each one needs a set of its own. Those buckets are touched on every (guess, candidate) pair — exactly as often as the table is — so the cost moves rather than disappearing, from the table to the accumulator.

The trade can be dodged instead of taken. Read a whole tile out of an answer-major copy, contiguous, transposing it into a small buffer on the way in, so each guess's feedback vector ends up a contiguous row of that tile and the reduce is the one above, over an accumulator that never leaves L1.

That is `fused_kernels`, and one scan at a time it is worth roughly twice the shipped kernel around a hundred candidates, falling through parity a few hundred later as the tile stops fitting in cache. Above that there is nothing left to recover — the shipped scan is already fetching a full cache line's worth of useful bytes per fetch. The two layouts are complementary rather than ranked, which is why `fused_kernels` is a dispatch for mid-sized positions and not a replacement.

Over whole games the same dispatch is worth **1.77×** under entropy and **1.02×** under expected guesses. Read the second as the loose figure it is: an expected-guesses scan carries the estimator and the endgame search on top of the table read, and only the read gets faster, so the gain is a small share of a large number and it moves with which answers are drawn. It does not move with the fit, which is the thing worth knowing.

### The optimization that had to be taken back out

Numba hands each kernel to LLVM, and LLVM offers a `fastmath` flag: permission to reorder floating-point arithmetic for speed, on the reasoning that addition is associative. It is not quite associative in floating point, but the reordering is usually harmless.

Here it was not, and the damage was not to the arithmetic at all. The reduce reads a bucket and then writes a zero into it for the next guess. Restructuring the loop swapped those two memory operations, so **every bucket maximum read back as the zero its own loop had just written**. The license granted was over the order of the additions. What moved was a store.

It surfaced as one of the scorers returning exactly 3.0 for every guess and the search confidently preferring nonsense words — which is the good outcome, because a wrong answer that obvious gets found. The other two scorers happened to survive, and only because a logarithm or a call to the estimator sits between their read and their write and gives the compiler something it cannot reorder across. That is luck rather than protection, and not a thing to build on. None of the kernels carry the flag now. It costs nothing on the widest positions and up to about a tenth of the time in the middle of the range, which is a small price for scores that are what they claim to be.

### Scoring fewer guesses

One last parameter, because two things in this document turn out to be nothing but this parameter set differently. Every kernel takes a **pool** — the set of table rows it is being asked to score, standing in for all 12,947 — and returns one score per entry. That is what keeps deeper search affordable: a search exploring the replies to a bucket of ninety candidates does not need all thirteen thousand guesses. Hard mode is the same mechanism with the pool set to the candidate set itself.

That is the whole of what it costs to score a guess. What none of it prices is the thing every real game ends against.

## 9. The deadline, and what losing costs

Everything so far has quietly ignored the fact that real Wordle gives you six turns. `f` estimates guesses-to-finish with no deadline at all, which §7 explained is deliberate — but the next section's formulas need a cost for running out, so it has to be defined before they can use it.

The obvious approach is a second objective: minimize guesses, subject to some separate constraint about not losing. That gets awkward immediately, because now you need a weight trading the two against each other. Instead the deadline is folded into the *same units* as everything else. Define one number:

> **`loss_penalty`** — what it costs to run out of turns, charged on top of the six guesses already spent.

A lost game therefore scores `6 + loss_penalty`. Setting the penalty to 1 makes a loss score exactly 7, as if you had needed one more guess than you had. That is the **risk-neutral** setting: minimizing expected score then means exactly what it says, and losing carries no special horror beyond the turn it cost.

In the value recursion this shows up as the base case: with zero turns left, a position is worth `loss_penalty`. No separate deadline objective, no risk weighting, one recursion covering both "win fast" and "don't lose".

Raising the penalty above 1 buys solve rate at the expense of the mean, by making the search willing to spend a turn on safety. **The default is 5**, and §11 explains why that number rather than the principled 1. Everything below holds for any setting of it. Note that it is a *decision-time* price only: measured results always score a lost game as 7, whatever the penalty was, or rows run at different penalties could not be compared.

## 10. The endgame: stop estimating, start solving

`f` is a fitted approximation, and approximations are for when you cannot afford the truth. Once few candidates remain you can afford it — the true value is computable by brute force (`exact.py`):

$$V(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r V(S_r, t-1)\Big], \qquad V(S, 0) = \text{penalty}$$

with `t` the turns remaining. This is **expectimax**: you *minimize* over your guess, because you choose it, but take an *expectation* over the feedback, because you do not. It is not minimax. There is no adversary, only chance.

Naively the recursion is hopeless — thirteen thousand guesses times 243 patterns, all the way down. Four ideas make it affordable, and each of them is worth more than the speedup it buys.

**(a) Partition dedupe.** This is where §2 pays off. A guess *is* its partition, so two guesses that split `S` identically are perfectly interchangeable, and the branching factor becomes the number of *distinct partitions* rather than the number of words.

Grouping guesses that way means comparing, for each pair, the whole list of patterns they produce over the candidates — the guess's **feedback vector**, one number in `[0, 243)` per candidate, which is just the slice of that guess's table row the candidate columns pick out. Comparing thirteen thousand such vectors against each other directly is too slow. So each vector is squeezed down to a single 64-bit integer, and the integers are grouped instead.

With seven candidates or fewer the squeeze loses nothing at all: a pattern fits in 8 bits, seven of them fit in the 56 bits available, and the packed integer therefore *is* the vector. Equal integers mean identical partitions, and nothing needs checking.

Above seven the vector no longer fits and the integer becomes a hash — FNV-1a, a cheap mixing function that folds an input of any length down to 64 bits. Identical vectors always hash alike, and different ones almost never collide; but "almost never" is not a proof, so every guess is compared once against its group's representative and promoted to a group of its own if it really does split differently. A collision then costs one redundant representative and can never lose a distinct partition.

The hash is worth defending, since the direct approach is right there: `np.unique` will group the feedback *vectors* themselves and needs no collision check at all. It is a lexicographic sort, one pass per candidate, and it costs an order of magnitude more than hashing each vector to 64 bits and grouping those. One extra comparison per guess buys the exactness back.

There is a nice safety property here. A guess that could itself be the answer *always* splits distinctly, since only it produces `GGGGG` — so a real candidate can never be merged away into a probe. (Which is also why hard mode skips this step entirely. If every guess must be a candidate, no two of them can split `S` the same way, and there is nothing left to merge.)

The dedupe only bites while `S` is tiny, though. By a dozen candidates almost every guess in the list already splits differently, so beyond that the branching factor is effectively the whole word list. That, rather than the node count, is what makes the occasional deep position slow.

**(b) Closed-form leaves.** With two turns left, no search is needed at all. Play `g`, and every reply but `GGGGG` leaves you in some bucket with one turn to spend. There is nothing to think about on a last turn: you name the heaviest word in the bucket and you are right with probability `M_r / m_r`, writing `m_r` for bucket `r`'s mass and `M_r` for its largest single weight. So

$$V(S_r, 1) = 1 + \Big(1 - \frac{M_r}{m_r}\Big)\cdot\text{penalty}$$

Now average that over the buckets, each reached with probability `m_r / total`:

$$V(g,2) = 1 + \sum_{r \ne \texttt{GGGGG}} \frac{m_r}{\text{total}}\left[1 + \Big(1 - \frac{M_r}{m_r}\Big)\text{penalty}\right]$$

The `m_r` outside the bracket cancels the one inside it, which is the only step with any content — it turns a sum of ratios into a sum of the two quantities a scan can accumulate:

$$V(g,2) = 1 + \frac{A + (A - B)\cdot\text{penalty}}{\text{total}}$$

with `A` the mass and `B` the sum of bucket maxima over the non-winning buckets. Those two totals are all a guess's 243 buckets have to be reduced to, which is why `kernels._two_turn` keeps exactly the two accumulators it does. That is *one pass over the guesses* — exactly what heuristic scoring costs. So **two turns or fewer is always solved exactly, for free, at any candidate count.** The same two-turn position is reached along many different orders of guesses, about four calls in five being a repeat, so even this closed form is cached rather than recomputed (§12).

One detail in it is doing more work than it looks. The penalty there is a parameter rather than a constant. Called as a leaf, it is handed the real one and returns the true value of the position, which is the point of a leaf. The next idea calls the same code with a different penalty, for an entirely different purpose.

**(c) Branch and bound.** That closed form, evaluated at a *lower* penalty, doubles as a **lower bound** on `V(S,t)` for every larger `t`. Write `V₁` for the value computed with the penalty set to 1 and `V₅` for the shipped setting. What the search relies on is

$$V_1(S, 2) \le V_5(S, t) \qquad \text{for every } t \ge 2$$

which splits into two steps that chain left to right, and both of them need that 1.

*At a penalty of 1, turns in hand cannot make a position cheaper* — so `V₁(S,2) ≤ V₁(S,t)`. This one reads backwards, because more turns ought to be worth having. They are not here, and the reason is that running out is an *exit*. A position two turns from the deadline can take its last shot, miss, pay 1, and stop. A position with four turns in hand has to carry on until it actually wins, and no continuation ever finishes for less than the single guess that exit charges. Being allowed to leave is never worse than playing on.

*Charging less for a miss cannot raise the value* — so `V₁(S,t) ≤ V₅(S,t)`. Fix any policy at all and score its play twice, once charging 1 for a loss and once charging 5. The second charge is larger on the paths that lose and identical everywhere else, so no policy scores worse under the cheaper penalty — and neither does the best of them.

Join the two and the two-turn value at penalty 1 sits underneath every deeper value at the penalty actually in force, which is what makes it safe to prune with.

**The cap at 1 is not a safety margin — without it the bound is simply false.** At a penalty of 5 the first step reverses outright: the exit now costs five guesses rather than one, and a position near the deadline would much rather have the turns. The ten words ending in `-atch` are a real position and a familiar trap, and they show it:

| turns in hand `t` | 2 | 3 | 4 |
| --- | --- | --- | --- |
| value at penalty 5 | **1.95** | 1.53 | 1.46 |
| value at penalty 1 | 1.36 | 1.44 | 1.44 |

The bound needs those values to rise with `t`, and only the second row does. The family is cheap to begin with because the prior puts most of its mass on `hatch` alone; but with two turns and a miss charged at 5, the last turn is a guess at whatever the first failed to resolve, and the position prices at 1.95. A search that pruned the three-turn position because 1.95 already beat its incumbent would be discarding a subtree whose true value is 1.53. Hence `ExactEndgame.bound_penalty`, which is `min(loss_penalty, 1)`, and hence its use in the bound and nowhere else.

The search then does the standard thing with that bound. Guesses are ordered by it and walked in that order, holding on to the best complete value found so far — the **incumbent**. Within a guess, the running total starts as the sum of its buckets' bounds, and each bucket is replaced by its exact value one at a time, so the total only ever rises. The moment it passes the incumbent, the guess is abandoned unevaluated. Biggest-mass buckets go first, since they move the total fastest.

All of it is sound because the bound is **admissible** — it never overestimates, so nothing it prunes could have turned out to be the best.

**(d) Branch cap.** Only the fifty best splits by that bound are tried at all, and this is the one idea of the four that gives something up. The three above it are exact: dedupe merges guesses that are genuinely interchangeable, the closed form is a value rather than an estimate, and an admissible bound only ever prunes what could not have won. The cap prunes on no such guarantee. A position holding 1,177 distinct splits has 1,127 of them discarded unexamined, and if the best guess is among those, the search will never know. What comes back is an upper bound on the true value — optimal-within-the-cap rather than optimal, exactly as the code says.

What makes it pay is that the bound is a good *ranking* near the top, even where it is a loose estimate of the value. The optimum reliably sorts into the first few dozen, so the cap keeps it and discards the tail, where a loose bound costs nothing because those guesses were never going to win. Measured over more than a thousand real positions, caps of 25 and 50 pick the uncapped-optimal guess every single time, while the wall clock for the sweep falls from around five minutes to eight seconds. That is an empirical record and not a proof, and the distinction is worth keeping: nothing here rules out a position whose best guess sorts fiftieth.

Two details of it are easy to miss, and both bite.

**The cap is decided by a sort, and the bounds tie in bulk.** One real position holds 1,176 of its 1,177 splits at exactly 2.0, so which fifty are kept is settled almost entirely by the tie-break rather than by the bound. Order those ties arbitrarily and two runs of the same search examine two different sets of guesses. Both implementations therefore sort stably, which puts a tie in word order, and what counts as a tie is the bound *rounded* first — coarsely enough that two summation orders of the same bound cannot fall on opposite sides of it.

The general lesson is worth more than the fix: a threshold, a cap or a top-k standing on a key that ties in bulk is brittle no matter who evaluates it. Make the comparison as coarse as the quantity is actually meaningful, and let a deliberate rule take it from there.

**A *ranking* wants a wider cap than a *pick*.** The page shows five guesses. The winner sits comfortably clear of the cap, but the runners-up sit right against it, and measured against the uncapped search a cap of 50 gets every winner right while occasionally naming a worse fifth word. So `ExactEndgame.root_branch` is wider, and it applies only at the node being ranked — every node beneath it needs its winner alone.

### When there is no table to read

§8 priced what it costs to *read* the table. The search is where that price is worth asking about again, because one build has no table at all: the browser, where 167 MB can be neither downloaded nor rebuilt.

For choosing a guess, applying the rule afresh costs a few times a lookup and that is comfortable. This search is where it stops being comfortable. It visits hundreds of nodes for a single position and rescans the whole guess list at every one of them, so a set of thirty-odd candidates has the coloring rule run over the same half-million pairs several hundred times over. In the browser that turned one turn into 5.6 seconds, which is long enough that a tab looks dead.

The fix is §1's instinct applied at a much smaller scope. Every candidate the search will ever see belongs to the set it started from, since each bucket is a subset of the one above it. So the pairs it could possibly ask about are known before it starts. Score every guess against that starting set once, up front, and every node below reads a byte instead of coloring a word. That is 440 KB instead of 167 MB, one scan instead of hundreds, and the same turn drops from 5.6 seconds to 1.1. It gives up no exactness doing it — nothing outside those columns is ever asked for.

### Lookahead: the same recursion, one ply up

The search above waits until the game is nearly over. The same recursion runs at a position of any size, though, so long as you are willing to stop early and estimate the rest — which is what `--lookahead` does. Rather than scoring a guess with `E(g, S)`, it values each of that guess's replies by scoring *those* positions, and picks the guess by what comes back.

$$V_d(S, t) = \min_g \Big[1 + \sum_{r \ne \texttt{GGGGG}} p_r V_{d-1}(S_r, t-1)\Big], \qquad V_0(S, t) = \min_g E(g, S)$$

It is the expectimax above with two changes. The leaves are the heuristic rather than the truth, which is what lets it run at a width the exact search could never afford — and the exact search still takes any sub-position it can reach. And only the *guess* is capped, never the feedback: dropping a bucket would leave the sum something other than an expectation.

What it costs is a fan-out rather than a constant factor. Every guess kept expands into every one of its buckets, so each further ply multiplies the work by the cap again, and one ply at a cap of ten already runs about ten times slower than the estimator alone. It is off by default. §15 is where it earns that, or fails to.

## 11. Why the penalty is 5 and not 1

Risk neutrality is the principled default, so moving off it takes evidence. Start by looking at what the losses actually are.

They are not endgame blunders. At the sizes these positions reach, `exact.ExactEndgame` solves them outright, so the guess played is optimal. The solver is backing its prior. Holding `batty`, `catty`, `natty`, `patty`, `ratty` and `tatty` with four turns left, it plays `tatty`, which carries three quarters of that family's mass — and it turns down `bronc`, a word that splits the six of them five ways, because separating them spends a turn to insure against a case the prior puts under half a percent on. A loss is the prior being confidently wrong, and the penalty is what makes that insurance worth buying anyway.

| penalty | loss probability | mean (loss scored as 7) |
| --- | --- | --- |
| 1 | 0.1185% | 3.6607 |
| **5** | **0.0513%** | **3.6611** |
| 30 | 0.0417% | 3.6620 |

Well over half the loss probability vanishes between 1 and 5 for **four ten-thousandths of a guess**. Widening the exact search's reach adds the rest, and the two together cut the loss probability by about three fifths while the mean *improves* on the way — because reaching deeper replaces an estimate with a true value instead of trading one good thing for another.

That last column cannot pick the penalty, though, and the reason is instructive enough to be worth the detour. Scoring a loss at 7 is precisely the objective that a penalty of 1 optimizes, so ranking the settings by that column can only ever favor risk neutrality. The comparison is circular. Name the price of a loss instead — call it `L` guesses — and every setting becomes a line in `L`, so the choice falls out of where the lines cross. Penalty 1 wins below `L` ≈ 7.1, **5 wins between about 9 and 13**, and 30 only past 23.

Which leaves the question of where `L` really sits, and one end of the range can be measured rather than argued. Scoring a loss at 7 says a lost game would have needed exactly one more guess. It would not: lift the deadline and carry each losing game on from the position it died in, and the solver takes **1.2 more guesses** to finish, occasionally four (`analysis/loss_cost.py`). So the risk-neutral price of a loss is about **7.2**, not 7 — which already puts it past the point where penalty 1 gives way. Risk neutrality, done honestly, does not endorse the risk-neutral-looking setting.

What it endorses is the next setting up, which takes over at that very crossover, and nothing near 5. Continuing a lost game prices it in guesses, and that is the risk-neutral question by construction, so this fixes the leftmost line and cannot reach the rest. Choosing 5 is a claim of a different kind: that a failed puzzle costs more than the guesses it cost, which is a judgment about Wordle scoring you out of six rather than counting forever, and no continuation can measure it. It is also a cheap claim to get wrong, because very little rides on the choice: anywhere in the range where 5 wins, it beats the next setting by a fraction of a thousandth of a guess.

Where do the losses collect? Almost entirely on the never-curated words. Wordle is provably solvable in five guesses, but that theorem is about a solver that knows the answer is one of the 2,309 curated words, and this one spreads a tenth of its mass over the other 10,638. Sweep the curated list uniformly instead and it loses eleven words out of 2,309 — every one of them a word that has already been an answer, which is §5.1's repeat prior calling them unlikely, by construction.

## 12. Remembering positions

The four ideas of §10 cut the cost of solving a position once. This one is about not solving the same position twice, and it is the largest single speedup in the project.

The reason it works is a fact about the game rather than about the code. Fix the opening word. The answer decides the feedback — but feedback comes in only 243 varieties, so **all 12,947 possible answers lead to at most 243 distinct positions** at turn two. Playing every word as the answer to measure an opening therefore looks like a thirteen-thousand-wide computation and is really a two-hundred-wide one, repeated. The same collapse happens at every depth: many different paths arrive at the same candidate set. Storing each answer against the position it belongs to, instead of recomputing it every time that position comes round again, is worth about **a hundredfold** over a sweep of every answer (`analysis/ablations.py`).

A *position*, here, is the pair the answer really depends on: the surviving candidate set, and the number of turns left to spend on it. So `memo.Memo` keys on both. The candidate set is the raw bytes of that array of column numbers from §1 — bytes because an array cannot be a dictionary key and its bytes can, and because they are exact where a hash of them would not be. The turn count is in the key because the deadline is what the recursion bottoms out on, so the same set of words genuinely has different answers at different depths: a value found with four turns in hand is wrong for two, and §10's `-atch` family is the worked example.

There are three such caches, holding three different things, and they do not all expire together. `WordleSolver` keeps two — the guess it chose at a position, and the lookahead value of one — and both are dropped whenever `f` is refitted, because both are answers the old `f` gave. `ExactEndgame.memo` holds true values instead, which depend on the prior weights and never on the estimator, so a refit leaves every one of them correct and it is kept.

### Why it evicts rather than refuses

A cache needs a cap or it grows without limit. The obvious way to write one is to stop accepting entries once it is full:

```python
if len(memo) < limit:       # the tempting version
    memo[key] = value
```

That is the wrong way round, and the wrongness is not gradual. What such a cache declines to store are the *most recent* positions, which are exactly the ones the run in front of it is about to reuse. It ends up holding a frozen souvenir of whatever it happened to see first. Cross the limit and the hundredfold collapse above disappears all at once.

`memo.Memo` instead drops the **least recently used** entry — LRU, the standard policy of evicting whatever has gone longest untouched, on the reasoning that it is the least likely to be wanted next. It is a dictionary that remembers the order its entries were last touched: a read moves that entry to the newest end, and an insertion past the cap drops whatever sits at the oldest. That a *read* counts as a use is the entire difference between LRU and simply dropping the oldest insertion, and it is what keeps a position that is looked up constantly resident however long ago it was first computed.

The bookkeeping is not free — every read moves an entry, a couple of pointer writes — but the comparison is lopsided. Moving an entry costs on the order of a hundred nanoseconds. The miss it prevents costs a scoring pass over thirteen thousand guesses, which is milliseconds. The overhead is roughly ten thousand times smaller than the thing it buys, and that is not a close call.

It is worth being careful about what LRU does *not* leave you exposed to, because the obvious worry is not real. A sweep that plays two hundred openings in one process leaves each finished opening's positions behind, and almost none of them recur: two openings partition the word list differently, so the next run is asking about a different set of positions entirely. Dead weight, then — but not weight the live entries have to fight. Those stale entries are precisely the ones that have gone longest untouched, so every insertion evicts one of them before it can reach anything the current run is using. LRU sheds them in the right order without being told.

What `WordleSolver.reset_memo` buys is therefore memory rather than hit rate, and it clears only the two memos that depend on the estimator. The search's entries stay valid across everything and stay worth sharing.

Every piece of the guess rule is now on the table. §13 puts them in the order the solver applies them.

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

`argmin` is "the argument that minimizes" — not the smallest score but the *word* achieving it, since a guess is what the solver needs back. It is one call to `kernels.expected_guesses_all` and one `.argmin()` over the array it returns.

Every line that says 12,947 is the free choice of guess, and hard mode is the one variation on this dispatch: it replaces that pool with the candidate set itself, in the exact search as much as in the `argmin`, and changes nothing else about which branch runs. `f` stays the curve fitted for free play, so its scores read slightly optimistic under the restriction — it is pricing a probe the solver is no longer allowed to make.

Underneath that dispatch sits most of this document. The lookup that reads the feedback is §1's table. The `argmin` is §8's rearranged entropy inside its blocked, parallel, twice-scanned loop. `exact.solve` is §10's four ideas. And every branch consults §12's memo before doing any of it.

The heuristic `E(g,S)` is genuinely a **fallback**, running only where the game is too wide to solve outright. With both priors on and the shapes that ship, over every word the answer could be, the whole thing averages **3.0717 guesses** and opens with `salet`. Against the answers the NYT has actually set in the current regime it reads about 3.16 (§14.3).

That 3.0717 is what every part above adds up to, and it says nothing about which part earned it. Two questions are left. Whether the model of the editor underneath it was worth building at all is §14. Which piece of the machinery earned the number is §15. Neither answer falls where the work went.

## 14. Measuring a prior without fooling yourself

The solver is assembled, which is what makes it possible to ask the question §4 and §5 could only promise: what was the model of the editor actually worth? One trap first, then two questions — one the data settles emphatically, and one it cannot settle at all.

### 14.1 The evaluation needs three arms

A shape can only be judged on answers that actually happened. Score a prior against answers sampled from its own belief and it is grading its own homework, and the swing from doing that is larger than the effect being measured.

So the evaluation has three arms: the **22 real repeats**, the **48 off-list answers**, and the regime's **164 fresh curated answers**. Every one is replayed with the prior rebuilt from strictly earlier puzzles, and the arms are blended at the rates the regime itself shows. Every configuration measured has both priors switched on, against a baseline that also has both switched on with flat shapes inside them, so the only thing moving is how the mass spreads.

Every figure below is a difference from that baseline, in guesses, so negative is better:

| shapes | repeat (22) | off-list (48) | fresh (164) | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.318 ± 0.179 | −0.833 ± 0.113 | +0.012 ± 0.035 | −0.025 ± 0.040 |
| repeat `binned` alone | −0.273 ± 0.164 | +0.000 ± 0.111 | +0.110 ± 0.039 | +0.059 ± 0.038 |
| `binned` + `frequency` | −0.318 ± 0.166 | −0.354 ± 0.135 | +0.085 ± 0.039 | +0.005 ± 0.040 |
| **`binned` + `wordlebot`** | **−0.045 ± 0.213** | **−0.833 ± 0.120** | **+0.030 ± 0.036** | **−0.050 ± 0.042** |

The result is clean and slightly deflating. Each shape moves the arm it was built for, and moves it a long way — a quarter of a guess when the answer really is a repeat, five sixths of one when it is off-list. Each then pays a little of it back across the four fifths of answers that are neither. The two roughly cancel, and **every blend in the last column is inside its own standard error of zero**.

Unlike §4's table, these columns *are* comparable between rows, because every row is scored on the same real answers rather than on a world of its own devising. The table is comparable to nothing outside itself, though. Each row carries the `f` fitted for its own shapes, as §7 says it must, so a refit moves the whole table and the rows have to be refitted together or not at all.

The loosest term in every blend is the repeat arm, at ±0.16–0.21 on 22 events, which leaves the bottom two rows well inside a standard error of each other. Nothing in *this* table chooses between them. That is not the same as nothing choosing between them: §5.2 already scored the two curation shapes at the job they exist to do, and separated them cleanly — −6.46 nats per event against −8.15, over 48 events rather than 22. What the guesses cannot resolve, the modeling evidence can, and the vendored scrape earns its place there rather than here.

One more thing about that evaluation is worth telling, because it was a mistake and mistakes of this shape are easy to repeat. The fresh arm used to be a *census*: all 487 curated words still unused, played as if today. That is far more data than 164 real answers, and it looked like the better arm. It was training on the test set. `train.policy_iteration` draws its practice answers from the prior it is fitting, and that prior's fresh-curated group is, at fitting time, exactly those 487 words — so four fifths of the evaluation's weight sat on material every arm had already trained against.

What makes that dangerous is not that it favors one arm's words over another's. Both solvers reach their own tuned best on ground they know, so the arm carrying four fifths of the grade reports a tie — and a tie is not neutral. It hands the verdict to the two small arms, which are exactly where a sharpened shape looks its best, so the sharper shapes collect the credit of the repeat and off-list columns while the column that should have charged them for it says nothing.

`analysis/fresh_arm.py` prices what was being concealed: sharpening the repeat shape reads as costing essentially nothing on the census and about a tenth of a guess on the answers that really happened. Playing the census on those same days does not close the gap, and forcing both arms onto a single `f` mostly does — which is how you can tell it was the fit that differed and not the words.

The arm is 164 real answers now. That costs precision exactly where the variance lives, and what it keeps is the only advantage this evaluation has over a plain replay: 48 off-list events instead of the 17 that fall inside the regime. The two agree now, as they should, and both say the shapes buy nothing in guesses.

### 14.2 Do the two shapes need each other?

The guesses cannot say. Read the blended column of §14.1's table as though it added up and the pair does better than the two apart would predict — −0.050 against a sum of +0.034 — but that margin is 0.084 against an error of about half of it, and all three of those figures are inside their own error of zero to begin with. Adding up numbers that are individually indistinguishable from nothing settles nothing.

What each shape does have is a margin the prequential scoring *can* resolve at its own job, and both ship on that modeling evidence: the histogram's +1.18 ± 0.20 nats per event in §5.1, WordleBot's list against a flat spread in §5.2. Neither can point to guesses saved, together or apart.

There is a mechanism that predicts they should interfere, and it is worth knowing about even though these events cannot see it. Sharpening one side makes the other side's flatness the larger error in the weights. Concentrate the off-list mass onto its thousand-odd effective words and each of them ends up heavier than a used word does under a flat repeat spread — so the curation shape, on its own, promotes every off-list word above every past answer. When the answer turns out to be a `cigar`, that costs.

Whether the repeat shape buys it back is not a question 22 events can answer; resolving an effect that size on that arm would take something like 340 repeats, which at the current rate is eight years of Wordle. So the mechanism is unmeasured here rather than absent.

Interference of this general kind is real, and that much does get settled one level up. §14.3 finds the two *priors* these shapes live inside strongly entangled — by a different route, each miscalibrating the rate the other governs rather than one group outranking another.

### 14.3 Do the two *priors* need each other?

A shape only redistributes mass inside a group that its prior has already carved out. The prior decides whether the group exists at all: switch curation off and there is no curated-versus-not split to spread anything over, just 12,947 words sharing the mass equally, non-answers included. That is the larger question, and it has a much larger answer.

`analysis/replay.py --all-priors` replays the 203 puzzles of the current regime under all four on/off combinations, pairing every arm on the same answer:

| mean guesses | curation off | curation on | on − off |
| --- | --- | --- | --- |
| **repeat off** | 4.0394 | 3.5320 | −0.507 ± 0.056 |
| **repeat on** | 3.9409 | 3.1626 | −0.778 ± 0.057 |
| **on − off** | −0.099 ± 0.041 | −0.370 ± 0.058 | **−0.271 ± 0.069** |

Those four means do not add up, and the way they fail to is the finding. Each margin is a plain subtraction along a row or a column, so the corner can be reached two ways and comes out the same both times. That corner is the **interaction**, and it is larger than either prior buys on its own. The priors are strongly superadditive, even though §14.1 found their shapes are not. Neither is the senior partner: the repeat prior alone is worth almost nothing, and what it is really buying is the calibration of the other one.

One caveat rides along with those four cells. Each carries the `f` fitted for its own prior, as §7 says it must, so what is being compared is four *solvers* rather than four beliefs held by one solver. That is the honest comparison — a prior the estimator was never fitted to is not a configuration anyone would ship — but it does mean a refit moves every cell, and the four have to be refitted together.

The reason the two are entangled is calibration, and it shows plainest if you ask what rates each prior *implies*. Those follow from the weights alone, with no estimator anywhere in them, so the caveat above does not reach them. Set against the two rates the regime really shows:

| prior | implied repeat rate | implied off-list rate, given not a repeat |
| --- | --- | --- |
| none | 14.4% | 95.6% |
| curation only | 71.7% | 32.8% |
| repeat only | 10.0% | 95.6% |
| both | 10.0% | 10.0% |
| *observed* | *10.8%* | *9.4%* |

Every prior but the last is wildly wrong about the rate the *other* one governs, and that is the whole of the interaction.

Take them in turn. A uniform prior lands near enough on the repeat rate by pure accident — 1,870 used words out of 12,947 happens to be 14.4% — so adding the repeat prior to it corrects a column that was barely wrong while leaving that 95.6% exactly where it was. Hence a prior that buys almost nothing. Curation alone pulls the second column down to 32.8% and *creates* the repeat error in the act of doing it: confining the field to a 2,309-word list of which 1,822 are spent is, whether you meant it or not, a claim that the editor repeats herself seven times in ten. Only with both switched on is either rate right.

Per word, that correction reads as a used word falling sevenfold while a fresh curated word climbs fourfold — those being the 487 words the editor can still reach without either repeating herself or leaving the list.

It also prices what calibration costs on the repeats themselves, where curation alone beats the calibrated pair on the 22 answers that really were one. That is what a calibrated prior costs, not a fault in it. Move a class from an implied 71.7% down to 10.0% and you will pay on the occasions it turns up, in exchange for doing better across the four fifths of the regime that is neither a repeat nor off-list. Any honest prior makes that trade.

Half a guess and more, then, for getting the model of the editor roughly right. Hold on to that figure, because §15 is what every other kind of improvement in this project was worth.

## 15. A better map, and next to no better score

That is the thread running through the whole project, and it is the most interesting thing in it. Five separate attempts to value positions better have all come out the same way:

| the improvement | how much better the valuation got | what it bought on the mean |
| --- | --- | --- |
| one-ply lookahead | RMSE **0.0125** against the estimator's **0.1609** | −0.0080 ± 0.0036 over 20,000 games; −0.0018 ± 0.0036 under the shipped prior |
| the endgame search | exact — the true value, not a better estimate | under a hundredth of a guess, over the positions it reaches |
| the opening word | `f` ranks the top twelve at **+0.35** against truth, the full two hundred at **+0.74** | nothing left to buy: `salet` is already optimal, at zero regret |
| turn-conditioned `f` | — | nothing: `f` is never consulted where the deadline binds |
| the second guess | true values for all 51 positions `f` still decides | **0.0022** guesses, and the sign flips under a slightly different prior |

Read the middle column against the right one and the whole section is there. Every row buys a real improvement in valuation and none of them buys a score. The rest of this section is why.

### 15.1 The top of the ranking is a plateau

**One-ply lookahead** actually values the replies to a guess instead of estimating them. RMSE there is the root-mean-square of the errors against ground truth — the true expected guesses from a position, computable rather than estimable because the policy is deterministic, so a position can be played out once per candidate answer. That accuracy reaches the decisions: lookahead disagrees about which guess to play in two games in five, and it is right to disagree, giving up about a tenth as much value per decision. Then the games get played out and almost none of it arrives. The outcome moves in a fifth of them, a shade more often for the better than for the worse, and the net needs twenty thousand games to separate from zero at all — at roughly ten times the wall clock, and on the one configuration that ships it does not separate.

The tempting explanation is that you are taking an `argmin`, so the estimator's errors are **common-mode** across the guesses at one position and cancel in the comparison, leaving the ranking intact. They do not. The estimator ranks the ten guesses lookahead re-ranks at a rank correlation of about **+0.56** with their true values — orderings against orderings rather than the `r` of a least-squares fit, on a scale where +1 is identical and 0 is no agreement at all — and picks the best of them a little over half the time. A valuation whose errors cancelled would not be overruled two games in five by an accurate one.

What is true instead is that the top of the ranking is a **plateau**. Those ten guesses spread about a tenth of a guess between best and worst, and the estimator's pick gives up about a ninth of that spread while ordering the middle of the list almost at random. It finds the plateau reliably and cannot tell one spot on it from another — which costs very little, because there is very little there to tell apart.

**The endgame search** makes the same point more sharply, because where it reaches, its value is not a better estimate but the truth: a disagreement there is the estimator being wrong, full stop. Over the positions the search claims, the estimator picks a different guess about a third of the time and gives up under a hundredth of a guess doing it. At three and four candidates *every single disagreement is a tie* — a different word of exactly equal value, nothing to improve on. The ties thin out as the set grows, and by the largest sets the search reaches, most disagreements are real. The same plateau, found from the other side.

**The opening** is starker still, because `f` decides it alone: no search reaches a position of 12,947 candidates. Played out, `salet` is the best of the two hundred openings `f` likes best — zero regret, where regret is what a choice costs against the best available. Yet the two correlations in the table say `f` sorts the whole field far better than it sorts the top of it. That is the shape of the whole result in one line: `f` sorts the field coarsely and then cannot sort the top of it, and every decision it gets right is a decision where coarse was enough.

**Turn-conditioned `f`** never stood a chance. A curve conditioned on turns remaining differs from a flat one only where the deadline binds — and `f` is consulted in **none** of the positions with three or fewer turns left, the search having taken every one of them. The tiny gain it measured was a fact about an implementation since removed; the reason it could not have come out otherwise is structural, and still holds.

**The second guess** puts a number on the ceiling. Behind a fixed opening, the whole of turn 2 is one decision per feedback pattern — 193 reachable positions, of which the endgame search already solves 142 outright, leaving 51 that `f` decides alone. Play out the true value of the fifty best guesses at each, weight each position by the probability play reaches it, and the estimator gives up **0.0022 guesses**. That is not an estimate: substituting the ten decisions it gets wrong and sweeping every answer moves the mean by exactly that. Since turn 1 is already optimal and the search owns everything from turn 4, that 0.0022 is not one more example of the plateau. It is the *whole* of what any valuation improvement could ever be worth, anywhere in the game.

Then the number is asked to hold up somewhere else, and it does not. Score those same ten substitutions against a differently weighted field:

| weighted by | what the ten substitutions do |
| --- | --- |
| the default prior | save **0.0022** guesses |
| a flat weighting of the curated list | **cost** about ten times that |
| a flat weighting of all 12,947 | cost a few thousandths |
| the still-unused curated words | save again |

Blending the default prior a *tenth* of the way toward a flat one is enough to turn the sign.

The reason is that the gain was never an improvement so much as the residue of a reshuffle. Over a thousand answers change length; slightly more of the prior's mass improves than worsens; and 0.0022 is what is left standing after that near-cancellation, a couple of thousandths of the standard deviation of the changes it rides on. Which side wins is decided by weights that are a *model* of the editor rather than the editor.

That is sharper than the four cases above it, because it does not depend on any valuation being imperfect. **Playing perfectly the last position where the estimator decides anything is worth less than the error in the prior it would be playing perfectly for.** There is no depth of search, no further ply and no better-fitted `f` on the other side of that.

Improving how well the solver values its options, then, is worth almost nothing here. What is left is the options themselves.

### 15.2 The exceptions all change the choice set

The exceptions sharpen the rule, and every one of them changes *what the solver is choosing between* rather than how well it scores the choices.

**The shapes of §5** are the first. Together they are worth about five hundredths of a guess — small in itself, and still more than deeper search and better lookahead managed between them. A prior changes the quantity being optimized, which words are plausible answers at all, so it is a decision-side change wearing valuation-side clothes.

**The sharpened repeat prior** belongs here rather than among the five, and on real answers it reads as a small *cost* despite being enormously better as a model of the editor. The two figures are not in competition: changing the weights changes what a position is *worth*, so the two arms share no ground truth on which one guess is better than the other. §14.1 is where that is taken apart.

**Hard mode** is the same thing in its bluntest form. It improves no valuation whatsoever; it deletes words from the choice set. Under greedy entropy that is worth about a quarter of a guess, against the nothing at all that a thirteen-times-better valuation buys — take a choice away from a policy that was pricing it wrong and you gain more than by pricing everything else right. Under the expected-guesses objective, which already prices that choice correctly, the identical restriction *costs* a little: same intervention, opposite sign, entirely because of what the objective was already getting right.

That is the honest reading of the headline number too. The two thirds of a guess over greedy entropy is nearly all the possible-answer bonus and barely any curvature in `f` — and the bonus is the *prior* showing up inside the objective, `p_win` being nothing but the mass the prior puts on the word being guessed, which is why the split moves with how sharp the prior is. So is the curvature, which §6 showed falling to nothing wherever the curation prior is off. The search proper is barely in that two thirds at all: switching the whole of `exact.py` off costs a few ten-thousandths on the mean, since what it moves is the loss probability instead.

Modeling the editor beat improving the search, every time.

## 16. Why any of this finishes

One thread has run underneath everything above without being named. Choosing a single guess means scoring thirteen thousand words against a candidate set; the training loop does that a few million times; and the analysis sweeps run it over every word in the list. Five ideas make that affordable, and each one shows up more than once.

**Precompute whatever does not vary.** The feedback for a (guess, answer) pair is fixed forever, so it becomes a table (§1) — or, where the table cannot be held, the fraction of it one search will ever read (§8). The quantity `w log w` depends on the candidate but not on the guess, so it leaves the inner loop entirely (§8). `f` is always reached through an entropy, so it takes `x = log₂ n` rather than `n` and saves a round trip through exponentiation (§8). In every case the trick is noticing that a quantity sitting inside a doubly-nested loop only depends on the outer index.

**Cache whatever recurs.** Positions repeat constantly — at most 243 of them behind a fixed opening — so results are memoized on the candidate set, which is the largest single speedup here and the difference between an overnight sweep and a coffee break (§12). The compiled kernels are cached to disk (§1) and the fitted `f` to a JSON file (§7), for the same reason at longer timescales.

**Bound before you compute exactly.** The two-turn value is exact, costs one pass, and — evaluated at a capped penalty — is a valid lower bound on every deeper value, which makes it both a leaf and the ordering that prunes the search above it (§10). The branch cap sits on top of that as the one step here that gives up a guarantee, and the ablation below is what shows how little it is doing: the bound was already pruning to well inside the cap.

**Collapse things that are secretly the same.** §2's reframe — a guess is nothing but its partition — is what lets the search treat thousands of words as one move (§10). Recognizing an equivalence is usually cheaper than exploiting a numerical shortcut, and unlike a shortcut it never costs accuracy.

**Measure where the work actually is.** Three assumptions had to be given up here: that the wide positions dominate, when the small ones are scored far more often; that the cost is arithmetic, when it is memory traffic at one cache line per scattered lookup (§8); and that the branch cap was carrying the endgame, when the bound underneath it was. Each changed what the right optimization was, and the second is what stopped the search for further ones. The rule is the one this section is an instance of: an optimization you have not tried removing is a guess about its own value.

What each of those is worth is a question with an answer, since an idea can be taken back out and everything else left alone — which is what `analysis/ablations.py` does. Two warnings about reading it. The three groups are different operations, one scan and one search and one sweep, so a factor from one says nothing about another, and the naive whole is **not** the product of these parts. And every figure is a slowdown against the shipped code, so larger means the idea was worth more:

| taking one idea back out | at the opening | at a mid-game position |
| --- | --- | --- |
| `w log w` back inside the inner loop | 4.7× | 1.6× |
| no table — the coloring rule applied per pair | 45× | 11× |
| one core instead of sixteen | 9.3× | 8.2× |
| **all three at once, in plain Python** | **13,800×** | **3,500×** |

The other two levels are one figure each. Lifting the branch cap over 150 real endgame positions costs 2.6×, and playing all 12,947 answers with the three position caches held to a single entry costs **103×** — a figure that climbs with the length of the run, since the longer a memo has been filling the more of what a later game asks for it is already holding. The same ablation over 400 answers costs only 14.5×.

The bottom row of the table is what answers the question in the heading. A naive reading of `E(g, S)` — no table, no rearrangement, no compiled kernel — takes **about seven and a half minutes to choose a single opening guess**, and the endgame search is not even involved in that number. `train.policy_iteration` asks for a guess a quarter of a million times.

The branch cap's own figure is the one that misleads, though, and seeing that takes counting rather than timing. `analysis/branch_cap.py` solves 150 real positions twice, once keeping the fifty best splits and once keeping every one of them:

| over 150 real positions | cap 50 | no cap |
| --- | --- | --- |
| interior nodes searched | 184 | 184 |
| subtrees evaluated | 270 | 9,001 |
| splits evaluated at the busiest single node | 50 | 8,781 |

Read the first row first. The cap changes no interior node at all — the same 184 subtrees are searched either way — and it changes no answer either: all 150 positions come back with the same guess at the same value, which is the claim §10 makes for it, checked here rather than asserted. What it changes is the third row, and essentially only there. At 183 of those 184 nodes the bound cuts the loop after one or two splits and fifty is a limit nothing comes near; at exactly one, the bound fails to separate anything and the uncapped search grinds through 8,781 splits to arrive at the guess it already had.

So the cap is not what prunes the endgame. The admissible bound above it is, and it does that job so thoroughly that the cap is idle almost everywhere. What the cap insures against is the rare node where the bound tells you nothing, and it is worth being exact about the shape of that.

The 2.6× above averages the cap's effect over 150 positions, which reads as a modest saving taken evenly from all of them. It is the opposite: nothing whatsoever at 183 nodes out of 184, and an enormous saving at the one — 8,781 splits down to fifty. Time forty positions rather than a hundred and fifty and the figure falls to 1.0×, the smaller sample having missed that node altogether. Averaging a rare, huge effect over the occasions it does not fire is how it comes to look like a small, steady one.

The through-line is not that any of this was mechanical. §8's rearrangement is real algebra, and a two-turn formula serving as a leaf and as the bound above it at once is a trick with no meaning outside this game.

What the five share is not how much thought they took but where it was aimed — never at playing better, and none of them does: the solver picks the same words without them, the branch cap being the only one that could differ and the counts above saying it does not. What they bought instead was scale, and §15 is what it was spent on: twenty thousand games, two hundred openings played out in full, every answer swept again under each reweighting. The five are invisible in how the solver plays and indispensable to knowing that.

## 17. A year to fit `f` once

§16 priced each idea by taking it out, and warned that the factors do not compose — three different operations against three different baselines, so a product of them would be a number no run has ever produced. One composition is legitimate, though, and it is not a product of factors. Take the single operation that was measured end to end with nothing left in it, one opening scan at seven and a half minutes, and count how many of those the project performs.

Nothing in the right-hand column below was run to completion. Each figure is that one measured scan times a counted number of scans, which makes these projections where the rest of this document's numbers are measurements (`analysis/ablations.py --projection`):

| what the project does once | shipped | with none of it |
| --- | --- | --- |
| choose the opening guess | 33 ms | 7.5 minutes |
| play every answer once, all 12,947 games | 8.9 s | 67 days |
| `train` — fit `f` for all five shipped priors | 38.9 s | 365 days |

The count is the same kind of thing in both of the lower rows, and it is the memo that makes it so. Every game in a sweep and every rollout in a fit opens at the identical position, the full word list, so the shipped solver scans it once per process and hands the answer back to the other 12,946 games. Take the caches away and each game pays the seven and a half minutes again. What happens after the opening barely registers beside that: the position left after `salet` holds 268 candidates rather than 12,947, so every later turn of a game together adds a few percent to a cost the first turn has already set.

Divide the top row by the work inside it and the factor stops being an abstraction. One opening scan is 12,947 guesses against 12,947 candidates — **167,624,809 (guess, answer) pairs**, each one colored, each dropping its candidate's weight into one of 243 buckets. Pure Python spends about **2.7 microseconds** on a pair: a function call, the two-pass rule of §1 over five characters, a dictionary update. The shipped kernels spend about **0.2 nanoseconds** across sixteen cores, so roughly three nanoseconds on each, which is about what one cache line costs to arrive from memory. The whole 13,800× is the distance between calling a function and fetching one byte out of a table — and it is why §8 found the loop memory-bound, there being nothing else left inside it to be slow.

The ratio grows down the table because the levels stack. The first row is the scoring optimizations alone. The second puts the memo on top of them, which is not a faster scan but the opening scan happening once instead of thirteen thousand times. The third asks for more of the same and nothing new: policy iteration plays 2,000 games a round, four to seven rounds per prior, five priors, and every guess any of them asks for is priced by the first row.

None of which is this project with a longer wait attached. §15 rests on five measurements and every one of them is a sweep or a fit — at the right-hand column's prices the cheapest is a season's work, so none of them gets made, and the finding they add up to stays a thing that sounds plausible.

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

Named rather than given as line numbers, which drift with every edit. Every entry is greppable: `grep -rn "def score_guess" .`
