# Wordle solver

It runs in a browser at **[wordleoracle.vercel.app](https://wordleoracle.vercel.app/)**, as WASM with no server behind it. `wordle-solver-lac.vercel.app`, the address Vercel generated, is an alias of the same build.

```
python wordle_solver.py train        # fit + cache the estimator, every prior combination
python wordle_solver.py benchmark    # self-play: greedy entropy vs. expected guesses
python wordle_solver.py compare      # policy iteration once per form of f, then tabulate
python wordle_solver.py info --forms # what is cached, and which forms exist
```

The page is for solving a puzzle. The terminal is where the solver is fitted, measured and inspected.

The prior toggles, `--mode` and `--form` go after the subcommand:

```
python wordle_solver.py train --this-config-only --no-repeat-prior --form saturating
python wordle_solver.py benchmark --no-curation-prior --mode entropy
```

## Findings

* **The shipped solver averages <!--n results_tables.both.expected .2f-->3.18 guesses and loses <!--n loss_penalty.search.loss_full .3f-->0.028% of games**, opening `<!--s results_tables.both.opening-->salet<!--/s-->`, weighted by its prior ([Results](#results)).
* **It beats greedy entropy by <!--n results_tables.both.gain .2f-->0.60 guesses, and the priors are why.** On the real answers since the regime began the margin is <!--n replay.policies.entropy.delta .2f-->0.68 ± <!--n replay.policies.entropy.se .2f-->0.04. With no prior the fitted estimator is worth <!--n results_tables.none.gain .2f-->0.05 ([Results](#results)).
* **The two priors need each other.** On real answers their interaction is <!--n replay.priors.interaction-->−0.410 ± 0.059 guesses a game, and the repeat prior alone is worth nothing measurable ([What each prior is worth on real answers](#what-each-prior-is-worth-on-real-answers)).
* **The exact endgame search cuts losses and leaves the mean alone.** It takes the loss rate from <!--n loss_penalty.search.loss_none .3f-->0.076% to <!--n loss_penalty.search.loss_full .3f-->0.028% and moves the mean by about a thousandth ([What it is worth](#what-it-is-worth)).
* **A better valuation buys almost nothing.** One ply of lookahead takes ten times the wall clock, and under the default prior it costs <!--n lookahead.repeat1_curation1_r-binned_c-wordlebot.delta-->+0.0056 ± 0.0033 guesses. Averaged over four priors it saves <!--n lookahead.delta-->−0.0072 ± 0.0037. A perfect second guess is worth <!--n turn2_regret.census .3f-->0.004 ([Lookahead](#lookahead---lookahead-off-by-default), [Perfecting the second guess](#perfecting-the-second-guess)).
* **On the curated answer list it plays within a hundredth of a guess of the published optimum**, <!--n external_benchmark.fitted.gap .3f-->0.008 above it ([Against the known optimum](#against-the-known-optimum)).

## Layout

| file | contents |
| --- | --- |
| `wordle_data.py` | word lists, the feedback table, past solutions, tile/pattern conversion |
| `priors.py` | `PriorConfig` and the prior over answers it produces |
| `kernels.py` | numba scorers over all ~13k guesses: information gain, E(g, S), two-turn value |
| `fused_kernels.py` | the same scores off an answer-major copy of the table; opt-in, and a speed knob only |
| `estimator.py` | the forms of f(n), their least-squares fit, and the parameter cache |
| `exact.py` | exact endgame search, used wherever it is affordable |
| `memo.py` | the bounded LRU the solver and the search cache positions in |
| `solver.py` | `WordleSolver`: pick a guess, apply feedback, play a game out |
| `train.py` | policy iteration, self-play evaluation, form comparison |
| `wordle_solver.py` | the terminal front end: fitting, benchmarking, inspection |
| `wordle_game.py` | play Wordle yourself in the terminal, against a chosen or a sampled answer |
| `web/` | the page: solve, play and the word finder, over the WASM solver |
| `wording.py` | finds every string the site says, so `editor.py` can rewrite any of it |
| `editor.py` | the editor itself, previewing the static build it is editing, and `/dev`, charts of the analysis results |
| `unwrap_md.py` | checks that every `.md` is one line per paragraph; `--write` converts |
| `spelling.py` | checks documents, comments and docstrings for British spellings from `data/british_spellings.json`; `--write` fixes them |
| `data/`, `data/derived/` | the word lists and answer record; everything `feedback_table.py` regenerates |
| `feedback_table.py` | one-off: builds `feedback_table.npy` and the index maps from the word lists |
| `answer_like.py` | one-off: builds `answer_like.json`, the score behind `--curation-shape frequency` |
| `solutions.py` | brings `solutions.json` up to date from the NYT, one request per day since the last |
| `word_lists.py` | one-off: refetches `wordles.json` and `nonwordles.json` upstream |
| `wordlebot.py` | one-off: rescrapes WordleBot's weights, live and from the 2023 snapshot |
| `analysis/` | the measurement scripts behind every number in the prior sections below |
| `refresh.py` | the routine refresh in one command: new answers, the refit, the live scripts, the documents and the tests |

`answer_like.py`, `feedback_table.py`, `solutions.py`, `word_lists.py` and `wordlebot.py` report how they would change the file they build, and write nothing without `--write`.

## Results

Both priors on with the shipped shapes, six-guess limit, failures counted as 7. Every row is the fit archived for that form, replayed by `analysis/results_tables.py` without refitting. There is no sample and so no error bar. The solver is deterministic, so every one of the 12,947 words is played once and weighted by the prior, which computes each mean exactly.

<!--table results_tables.forms_table-->
| form | expression | terms | opening | mean | vs. best | loss probability |
| --- | --- | --- | --- | --- | --- | --- |
| spline | piecewise linear in x, knots at 2/4/6/8 | 5/6 | salet | 3.1771 | — | 0.0276% |
| cubic | a + b·x + c·x² + d·x³ | 4/4 | salet | 3.1772 | +0.0000 | 0.0285% |
| saturating | a + b·x + c·log2(1+x) | 3/3 | salet | 3.1775 | +0.0004 | 0.0274% |
| anchored | 1 + a·x + b·(1−1/n) + c·x² | 2/3 | raise | 3.1779 | +0.0007 | 0.0267% |
| anchored2 | 1 + a·x + b·(1−1/n) | 2/2 | salet | 3.1779 | +0.0008 | 0.0267% |
| sqrt | a + b·x + c·√x | 3/3 | salet | 3.1780 | +0.0008 | 0.0283% |
| invonly | a + b·(1−1/n) | 2/2 | salet | 3.1832 | +0.0061 | 0.0318% |
| loglinear | a + b·log2(n) | 2/2 | raise | 3.2155 | +0.0383 | 0.0420% |
| quadratic | a + b·x + c·x² | 2/3 | raise | 3.2155 | +0.0383 | 0.0420% |
| IG + 2·p_win | — | — | raise | 3.2182 | +0.0411 | 0.0653% |
| greedy entropy | — | — | raise | 3.7722 | +0.5951 | 0.0103% |
<!--/table-->

The opening word does not decide the order. <!--n results_tables.same_opening-->6 forms open `salet` and sit within <!--n results_tables.same_opening_spread .4f-->0.0061 guesses of each other. Of the <!--n results_tables.other_opening.count-->3 that open `raise`, `anchored` sits inside that spread, and `loglinear` and `quadratic` trail it. The figures are exact, so the ordering is a fact about this prior, but <!--n results_tables.same_opening_spread .4f-->0.0061 is far too little to choose a default on, and the next section does that another way. `loglinear` and `quadratic` are one function twice over: the monotonicity guard cuts `quadratic` back to `a + b·x`, and with `x = log2(n)` that is `loglinear`.

The loss column is the one place greedy entropy wins, and by a lot. *Losing less* explains why the estimator's six tenths of a guess cost <!--n head_to_head.baseline.loss_ratio .0f-->3 times the loss rate.

**`IG + 2·p_win` is the second reference**, and the only row that is neither a fitted curve nor a plain greedy rule (`WordleSolver(win_bonus=...)`). It scores information gain and adds two bits for the chance that the guess is itself the answer. The two bits come from two exact facts and no data. One candidate takes one guess and two take <!--c-->1.5, so a bit of information is worth half a guess. Winning outright saves a whole guess, which is two bits' worth. `SOLVER_EXPLAINED.md` *A straight-line `f` is entropy plus a bonus* has the identity showing that a linear `f` collapses to a policy of this shape. With nothing fitted it gains **<!--n results_tables.ig_over_entropy .3f-->0.554 guesses** over greedy entropy, and its loss rate is <!--n results_tables.ig_loss_ratio .0f-->6 times entropy's, which is the trade *Losing less* is about. That bet on the answer is why it is a reference, and greedy entropy, which makes no bet, stays the baseline.

**Concavity in x** is what puts the two forms below the cliff, and the identity that says so needs no measurement. With `f` exactly linear in x,

```
E(g, S) = 1 + a·(1 − p_win) + b·(H(S) − IG(g))
```

so minimizing E is the same as maximizing `b·IG + a·p_win`, which is greedy entropy plus a bonus for possible answers. That is why `loglinear` picks entropy's opening word and gains the least. Of the <!--n results_tables.entropy_gap .2f-->0.60 guesses between greedy entropy and the best form, about <!--n results_tables.both.bonus .2f-->0.56 is that bonus and about <!--n results_tables.both.curvature .2f-->0.04 is curvature. Everything above the cliff is concave and collects both terms. Below it, `loglinear` and `quadratic` are straight lines and collect only the first, and the guard cut `anchored` back to a curve with little bend left.

On this fit, fit quality sorts them at the cliff. The forms that open `salet` fit at an RMSE of <!--n results_tables.rmse_lo .3f-->0.507 to <!--n results_tables.rmse_hi .3f-->0.571, and the three that open `raise` all fit worse. The worst fit of the nine is <!--s results_tables.worst_fit.form-->`loglinear`<!--/s-->, at <!--n results_tables.worst_fit.rmse .3f-->0.607, and it opens <!--s results_tables.worst_fit.opening-->raise<!--/s--> <!--n results_tables.worst_fit.gap .3f-->0.038 behind the best. Under another prior the two can part. `SOLVER_EXPLAINED.md` *The form of `f` has to be refitted per shape* has the same thing seen across shapes.

Saturation is less of a liability under this prior than under a flat one. `invonly` flattens out completely (f(200) ≈ f(2000)) and still lands within <!--n results_tables.invonly_gap .4f-->0.0061 of the best, because with the mass concentrated the effective size rarely reaches the range where the curve has gone flat.

### Which form is the default

Every prior is fitted in **`spline`**. `--form` defaults to it, `compare` fits it alone unless `--forms` names others, and `form_default --promote` ranks only `spline` fits. The other eight forms stay in `estimator.FORMS`, and `--form` still plays any of them.

`analysis/form_default.py` without `--promote` is the comparison behind that choice. It replays every archived fit over every word the answer could be, weighted by each prior in turn. Nothing is held out, because the evaluation covers the whole space and the gaps are exact.

<!--table form_default.aggregate_table-->
| form | terms | mean gap to the prior's best | worst gap | worst rank of 9 |
| --- | --- | --- | --- | --- |
| **spline** | 5–6 | +0.0008 | +0.0024 | 7 |
| saturating | 3 | +0.0011 | +0.0029 | 9 |
| anchored2 | 2 | +0.0023 | +0.0072 | 7 |
| invonly | 2 | +0.0061 | +0.0204 | 7 |
| anchored | 2–3 | +0.0098 | +0.0410 | 8 |
| sqrt | 2–3 | +0.0110 | +0.0482 | 10 |
| cubic | 2–4 | +0.0203 | +0.0488 | 10 |
| loglinear | 2 | +0.0285 | +0.0485 | 10 |
| quadratic | 2–3 | +0.0285 | +0.0485 | 10 |
<!--/table-->

The top <!--n form_default.top_group-->3 are inside <!--n form_default.top_group_spread_mean .4f-->0.0014 guesses of each other on the mean gap and <!--n form_default.top_group_spread_worst .4f-->0.0048 on the worst, and the next two sit within a hundredth. That spread is about the size of the fit's own noise: refitting the same form from a different training seed moves its score by up to <!--n rust_fit.seed_spread.repeat1_curation1_r-binned_c-wordlebot .4f-->0.0000 under the default prior, and by <!--n rust_fit.seed_spread_max .4f-->0.0360 under `<!--s rust_fit.seed_spread_widest-->repeat0_curation1_c-wordlebot<!--/s-->`, where the seeds disagree on the opening word (`analysis/rust_fit.py` prints the per-seed scores). So which of the top group ranks first is mostly which fit came out luckiest. Across the eight priors the winners were <!--s form_default.winners-->`spline` ×5, `invonly`, `anchored2`, `saturating`<!--/s-->, and a refit reshuffles them.

So fitting several forms and keeping the winner chooses among fits that all do about equally well, and the choice moves on every refit. One form removes that churn, along with most of a refit's running time, which went to ranking every archive. `spline` is the one, because it has the smallest mean gap and the smallest worst gap in the table, <!--n form_default.spline.mean_gap .4f-->0.0008 and <!--n form_default.spline.worst_gap .4f-->0.0024 guesses. That worst gap is the most any prior gave up for the fixed choice.

`form_default --promote` ranks the fit already in play alongside the new `spline` archive and keeps it unless the archive beats it, stamping a kept fit with the date it was ranked so `tests/fit_provenance.py` knows it answers to the current record. A refit can improve a prior and cannot make one worse. A fit in play in another form does not compete, so the archive replaces it.

The comparison is a finding. `compare --forms all` for each prior and `form_default.py --shapes` remake it, and they are rerun when something that shapes `f` changes, such as a prior's shapes, the monotonicity guard or the endgame. `analysis/README.md` *Rerunning them* quotes when that last was.

The bottom of the table is the firm part of the result. `loglinear` and `quadratic` give up <!--n form_default.straight_line_gap .2f-->0.05 guesses wherever the curation prior is on, because the guard has cut them to a straight line and a straight line is greedy entropy plus the bonus. `cubic` and `sqrt` sit near the top under some priors and at the bottom under another, where the guard cut them back toward a line, so their worst gaps match the straight lines'. Those are the shapes to avoid. A straight line gives up <!--n form_default.straight_line_gap .2f-->0.05 where the top group spans <!--n form_default.top_group_spread_mean .4f-->0.0014.

Read this in preference to ranking the archives by their own stored holdout, which `compare` prints. The two orderings disagree, and the holdout is a 200-answer sanity check where this is the whole space.

All four combinations of the two switches, with the sharpened shapes, each carrying its own fitted `spline`. **Every row here is a configuration the page can be put into**, since the two priors are independent checkboxes on it. "The default prior" below means the last row. Every word is weighted by the solver's own prior, so each row asks how well the solver does if this model of the editor is right. The rows are **not** comparable to each other, since a concentrated prior makes the game easier. The opening word and the solve rate are the expected-guesses mode's:

<!--table results_tables.priors_table-->
| priors | form | opening | entropy mean | expected mean | solve rate |
| --- | --- | --- | --- | --- | --- |
| none | spline | tares | 4.1223 | 4.0718 | 99.95% |
| curation only | spline | salet | 3.8697 | 3.5282 | 99.98% |
| repeat only | spline | tares | 4.1016 | 4.0062 | 99.93% |
| **both (the default)** | spline | salet | 3.7722 | 3.1769 | 99.97% |
<!--/table-->

Nearly all of the gain comes from the curation prior. When the prior is close to flat, expected guesses and information gain rank guesses almost identically, so the estimator has little to add: with no priors it is worth **<!--n results_tables.none.gain .4f-->0.0505** guesses against entropy, and with both priors on it is worth **<!--n results_tables.both.gain .4f-->0.5953** from the same machinery. The identity above predicts this. Both figures are exact, so the small one is a real gain and not a sampling accident.

### Where the estimator's guesses come from

Against the fit-free reference the shipped solver wins <!--n head_to_head.gap .4f-->0.0413 guesses, and `analysis/head_to_head.py` says where. Both policies play all 12,947 words, so every word is a win for one side, a win for the other, or a tie:

<!--table head_to_head.table-->
| guesses | shipped | `IG + 2·p_win` |
| --- | --- | --- |
| 2 | **11.41%** | 8.62% |
| 3 | 62.92% | **64.76%** |
| 4 | **22.65%** | 23.33% |
| 5 | **2.61%** | 2.83% |
| 6 | **0.37%** | 0.39% |
| lost | **0.0285%** | 0.0653% |
<!--/table-->

The largest part of the margin is in the two-guess row. The shipped solver finishes in two on <!--n head_to_head.two.shipped .2f-->11.41% of the prior's mass and the reference on <!--n head_to_head.two.other .2f-->8.62%, of a whole margin of <!--n head_to_head.gap .4f-->0.0413. The reference wins back the three-guess row. This is what the fitted `f` and the priors are doing: they lean the second guess toward the words an editor is likely to pick, and when that guess is right the game ends there.

A gap of four hundredths of a guess sounds like two policies that agree on nearly everything, but they play about two games in five differently: the shipped solver finishes first on <!--n head_to_head.sooner.shipped .2f-->22.60% of the mass, the reference on **<!--n head_to_head.sooner.other .2f-->18.81%**, and they tie on <!--n head_to_head.same .0f-->59%. The reference loses <!--n head_to_head.lost.other-->431 words to the shipped solver's <!--n head_to_head.lost.shipped-->185, and its heaviest loss, `<!--s head_to_head.heaviest.other-->parer<!--/s-->` at <!--n head_to_head.heaviest.other.mass .4f-->0.0008% of the mass, weighs about the same as the shipped solver's heaviest, `<!--s head_to_head.heaviest.shipped-->foggy<!--/s-->` at <!--n head_to_head.heaviest.shipped.mass .4f-->0.0008%. Without an `f` the reference cannot see the deadline coming. The shipped policy can, and `--loss-penalty 5` sets how hard it steers away from one.

Because `f` estimates expected guesses, the policy is risk-neutral away from the endgame, where greedy entropy is implicitly cautious. `--loss-penalty 5` and the exact search together close about two thirds of the gap in loss probability. The estimator still loses <!--n head_to_head.baseline.loss_ratio .0f-->3 times as often as entropy does (see [Losing less](#losing-less)). `--loss-penalty 1` scores a loss as a seventh guess and gives that ground back.

Both margins are scored under the prior the shipped `f` was fitted to, so they assume the prior is right about the editor. Replayed against the <!--n replay.policies.puzzles-->266 real answers since the regime began, with the prior rebuilt each day from earlier puzzles (`analysis/replay.py --policies`), greedy entropy takes <!--n replay.policies.entropy.vs-->+0.6767 ± 0.0392 guesses a game more than the shipped solver, worse on <!--n replay.policies.entropy.worse-->174 games and better on <!--n replay.policies.entropy.better-->8. The reference reads <!--n replay.policies.ig.vs-->+0.0226 ± 0.0328, the same sign as the <!--n head_to_head.gap .3f-->0.041 above and well inside its noise. Resolving a gap of <!--c-->0.02 at two standard errors would take about <!--n replay.policies.ig.games_needed ,d-->2,863 puzzles.

## Against the known optimum

Optimal play is known for one setting. With the original <!--n external_benchmark.optimum.answers-->2,315 answers weighted uniformly and <!--n external_benchmark.optimum.allowed-->12,972 allowed guesses, the best possible strategy takes <!--n external_benchmark.optimum.total-->7,920 guesses in all, <!--n external_benchmark.optimum.mean .4f-->3.4212 a game, opens `salet` and never needs more than <!--n external_benchmark.optimum.worst-->5 ([Selby](https://sonorouschocolate.com/notes/index.php/The_best_strategies_for_Wordle)). `analysis/external_benchmark.py` fits `f` in that setting, with the candidates restricted to the answer list and every word allowed as a guess, and plays every answer once:

<!--table external_benchmark.table-->
| policy | opening | mean | vs. optimum | lost | games in 1 / 2 / 3 / 4 / 5 / 6 |
| --- | --- | --- | --- | --- | --- |
| fitted f | `reast` | 3.4296 | +0.0085 | 0 | 0 / 70 / 1237 / 942 / 60 / 0 |
| fitted f, opening salet | `salet` | 3.4249 | +0.0037 | 0 | 0 / 79 / 1218 / 966 / 44 / 2 |
| greedy entropy | `soare` | 3.5240 | +0.1029 | 0 | 0 / 31 / 1106 / 1104 / 67 / 1 |
| IG + 2·p_win | `soare` | 3.4621 | +0.0409 | 0 | 0 / 50 / 1214 / 974 / 70 / 1 |
<!--/table-->

The fitted policy finishes within a hundredth of a guess of the optimum, and within <!--n external_benchmark.salet.gap .3f-->0.004 behind `salet`. Greedy entropy is a tenth of a guess away. The comparison is close and not exact. This list is the <!--n external_benchmark.answers-->2,309 answers left after the NYT's removals, and the vocabulary has <!--n external_benchmark.allowed-->12,947 words. No optimum has been published for those lists, and six fewer answers could move it either way, so a gap this small says the policy is near optimal and does not say by how much.

Two things the shipped prior hides show up here. The estimator's own opening, `<!--s external_benchmark.fitted.opening-->reast<!--/s-->`, costs <!--n external_benchmark.opening_regret .4f-->0.0048 against `salet` behind the same policy, where under the shipped prior it picks `salet` with no regret ([The opening is not where the estimator is weak](#the-opening-is-not-where-the-estimator-is-weak)). And behind `salet` it needs a sixth guess for <!--n external_benchmark.salet.sixes-->2 answers that optimal play finishes in five.

## The two priors

The answer is not drawn uniformly, and the two departures from uniform can be switched on and off independently, from the command line and from a checkbox each on the page. **All four combinations ship**, each with its own cached fit and its own opening word. "The default prior" throughout this document means both switches on with the sharpened shapes, which is where the switches start.

* **repeat prior** (`--no-repeat-prior`): the editor reuses an already-played answer, so used words share `--repeat-prob` of the mass instead of their proportional share.
* **curation prior** (`--no-curation-prior`): some answers come from outside the curated 2022 list, so those words share `--noncurated-prob` of what is left.

The model is sequential, a repeat first and then curated or not on the remainder, so `--noncurated-prob` is conditional on not being a repeat and must be measured with that denominator.

Each group also has a **shape**, which is how its mass is spread over its members. The defaults are the sharpened shapes, worth <!--n replay.shapes.together.blend-->−0.160 ± 0.038 guesses over the flat spread the two priors used on their own.

* `--repeat-shape binned` (default) weights a past answer by where it first appeared, binned 200 wide and Laplace-smoothed, since reuse concentrates near the start of the run.
* `--curation-shape wordlebot` (default) weights an off-list word by WordleBot's own weight, keeping 5% of the mass outside that list as a hedge. `--curation-shape frequency` weights it by how answer-like the word is, which is slightly weaker and needs no vendored list.
* `--repeat-shape uniform`, `--curation-shape uniform` give the flat spread back.

Each shape is worth having on its own terms, and the two are **additive within noise**. *What the two shapes are worth together* below has the measurement. The shape is part of the cache key because the best form of `f` depends on it, and the dependence reaches the opening word. `anchored2` opens `salet` under the sharpened shapes, a hundredth behind the best form. Paired with a flat repeat shape and the sharpened curation shape, it opens `raise` and drops to seventh, four hundredths behind, with every form that kept its bend ahead of it.

### The regime the defaults describe

Measured over the <!--n regime.answers-->1,955 answers in `solutions.json`, through Wordle #<!--n regime.last_no d-->1955 (<!--s regime.last_date-->2026-10-25<!--/s-->). The puzzle number is `days_since_launch + 1`. The `id` field is an NYT internal key and stops matching after #506. It does not mark a word as off-list, which is decided by membership in the curated list and nothing else. `trace` came back on 2026-08-21 carrying id <!--c-->4413 and is a curated word.

<!--table regime.table-->
| window | puzzles | repeat rate | off-list \| not a repeat |
| --- | --- | --- | --- |
| 2021–2025 | 1,657 | 0.00% | 1.75% |
| since Wordle #1690 (2026-02-02) | 266 | 10.5% | 16.8% |
| its first half | 133 | 11.3% | 7.6% |
| its second half | 133 | 9.8% | 25.8% |
| **last 100** | **100** | **10.0%** | **31.1%** |
| last 50 | 50 | 10.0% | 37.8% |
| last 25 | 25 | 12.0% | 45.5% |
| all time | 1,955 | 1.4% | 3.7% |
<!--/table-->

The repeat default of 0.10 is the regime's rate, which has held since the regime began. The off-list rate is climbing, from <!--n regime.first_half.offlist .1f-->7.6% over the regime's first half to <!--n regime.second_half.offlist .1f-->25.8% over its second, so the regime's average is not the current rate. The off-list default of 0.30 is set on the last <!--n regime.recent.puzzles-->100 puzzles, which is the shortest window with enough off-list answers to read to a few points: <!--n regime.recent.offlist_events-->28 of them, and the rate is known to ±<!--n regime.recent.offlist_se .0f-->5 points. The shorter windows read higher. The last 50 holds <!--n regime.last50.offlist_events-->17 events and is known to ±<!--n regime.last50.offlist_se .0f-->7 points, and the last 25 holds <!--n regime.last25.offlist_events-->10 and is known to ±<!--n regime.last25.offlist_se .0f-->11, so it moves by several points whenever one puzzle enters or leaves it. So the last hundred says the rate is rising, both shorter windows put it above 0.30, and the sweep below says how far above the last hundred the board allows a setting to sit. Against the full record the two defaults are roughly <!--n regime.default_over_all.repeat .0f-->7× and <!--n regime.default_over_all.offlist .0f-->8× too high. Nothing was off-list before 2023, and `cigar` on 2026-02-02 was the first repeat in Wordle's history. Every repeat since falls in the last <!--n regime.since.puzzles-->266 puzzles, at a rate of <!--n regime.since.repeat .1f-->10.5%.

`analysis/prior_basin.py` sweeps the off-list probability with `f` held fixed and scores every setting on the whole regime, on the last <!--n prior_basin.recent_puzzles-->100 puzzles, and on the off-list answers inside that window, each paired against the shipped 0.30:

<!--table prior_basin.noncurated.table-->
| `noncurated_prob` | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.55 | 0.71 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all 266 | +0.038 | +0.019 | +0.015 | +0.030 | +0.000 | +0.000 | +0.143 | +0.233 | +0.511 |
| the last 100 | +0.090 | +0.090 | +0.080 | +0.090 | +0.000 | −0.010 | +0.040 | +0.150 | +0.420 |
| the 28 recent off-list | +0.393 | +0.357 | +0.286 | +0.321 | +0.000 | +0.000 | −0.071 | −0.250 | −0.214 |
<!--/table-->

On the whole regime the old default of 0.10 costs <!--s prior_basin.noncurated.p10.regime-->+0.019 ± 0.031<!--/s--> against the shipped 0.30, inside the error. On the last hundred it costs <!--s prior_basin.noncurated.p10.recent-->+0.090 ± 0.040<!--/s-->, and the difference sits on the off-list answers, <!--s prior_basin.noncurated.p10.recent_offlist-->+0.357 ± 0.128<!--/s--> a game over the recent ones against <!--s prior_basin.noncurated.p10.recent_fresh-->−0.016 ± 0.016<!--/s--> on the fresh curated answers beside them. Only the off-list answers separate the two settings by two standard errors, and the shapes shipped on margins of that size. The rate is the firmer evidence.

What the sweep does settle is the upper end. The board cannot tell 0.30 from <!--c-->0.40 on either window, and from 0.55 on it takes more guesses over the whole regime, so it allows the rates of both shorter windows. Between two settings the board ties on, the one nearer the better-measured rate wins, and the last hundred sits nearer 0.30. The sweep ties <!--c-->0.40 with 0.30 on the off-list answers alone as well, so <!--c-->0.40 is the setting to check again when the last hundred passes 0.35.

The two moves started together, and the likely reason is that the curated pool is running out. <!--n regime.curated_used-->1,856 of the 2,309 curated words are used, <!--n regime.curated_unused-->453 remain, and fresh ones are consumed at about <!--n regime.fresh_per_year .0f-->272 a year since the regime began, which leaves about <!--n regime.years_left .1f-->1.7 years. Going off-list and recycling old answers read as the same editorial response to the same shortage. That reading is a guess and not a measurement, but it is what both rates are set on: the year ahead, and not the record as a whole.

### What the id field indexes

`solutions.json` carries the NYT's own `id` per puzzle. It is neither the puzzle number nor stable per word. It is a position in one list that has only ever been appended to, and reading it that way accounts for every id in the record (`analysis/id_eras.py`).

| ids | what they index |
| --- | --- |
| 1 to <!--n id_eras.last_curated_id d-->2287 | the curated words in the order they were first scheduled, which is the order `wordles.json` is stored in, with <!--n id_eras.skipped-->22 slots skipped |
| <!--n id_eras.back_lo d-->2288 to <!--n id_eras.curated d-->2309 | where the words displaced by that skip came back |
| <!--n id_eras.block_lo d-->2310 to <!--n id_eras.block_hi d-->2397 | off-list words, appended in alphabetically sorted batches |
| <!--n id_eras.era2_lo d-->2398 on | the list re-sorted alphabetically, from the regime boundary |
| <!--n id_eras.tail_lo d-->4616 to <!--n id_eras.tail_hi d-->5261 | a later batch appended past the re-sorted list, off-list and curated words together |

The skip is one edit. <!--n id_eras.purge-->20 slots between positions <!--n id_eras.skip_first d-->285 and <!--n id_eras.skip_last d-->526 are never reached, and the words in them, `fanny`, `vomit`, `bobby`, `harry` and the rest, are a content purge, presumably at the NYT's acquisition. Of those, <!--n id_eras.dropped-->18 were dropped outright and <!--s id_eras.requeued-->`hasty` and `unlit`<!--/s--> were requeued past the end. <!--s id_eras.forward-->`gecko` and `shine`<!--/s--> were pulled forward from positions <!--s id_eras.forward_positions-->2266 and 2265<!--/s--> to fill two of the holes, so <!--n id_eras.skipped-->22 slots are skipped in all. <!--n id_eras.skipped-->22 out and <!--n id_eras.skipped-->22 back in leave the curated region <!--n id_eras.curated ,.0f-->2,309 wide, with the off-list appends starting at <!--n id_eras.block_lo d-->2310.

<!--n id_eras.era2_lo d-->2398 was not chosen either. It is where the end of the list had got to when `cigar` needed an entry its spent slot could not give it, and every repeat carries an era-2 id for that reason. Off-list answers since the boundary split: <!--n id_eras.era1_offlist-->15 sit in the old appended block and <!--n id_eras.era2_offlist-->25 carry ids from the later batch. The <!--n id_eras.era2_first_uses-->5 curated first-uses that took an era-2 id had a free era-1 slot and took a new number anyway, and nothing separates them from the <!--n id_eras.era1_first_uses-->193 that did not.

### What each prior is worth on real answers

Everything above argues the two priors from the answer record. This measures them against it. `analysis/replay.py --all-priors` replays the <!--n replay.priors.puzzles-->266 real puzzles of the current regime under all four combinations, rebuilding each prior from strictly earlier puzzles each day, and pairs every arm on the same answer. The pairing matters. The tables in *Results* weight each row by the prior being judged, so a prior that concentrates on words it happens to solve quickly scores well by construction. Here the yardstick is the sequence the NYT published, and no arm gets a say in it.

<!--table replay.priors.table-->
| priors | mean | vs. both | on the 28 repeats | on the 40 off-list | on the other 198 |
| --- | --- | --- | --- | --- | --- |
| none | 3.9662 | +0.805 ± 0.047 | +0.143 ± 0.152 | +0.500 ± 0.101 | +0.960 ± 0.051 |
| curation only | 3.5789 | +0.417 ± 0.047 | −0.179 ± 0.171 | −0.025 ± 0.084 | +0.591 ± 0.051 |
| repeat only | 3.9586 | +0.797 ± 0.050 | −0.143 ± 0.176 | +0.525 ± 0.124 | +0.985 ± 0.049 |
| **both (the default)** | **3.1617** | — | — | — | — |
<!--/table-->

Read the first column with `priors.build_weights` in mind. Switching the curation prior off leaves a uniform prior over all 12,947 words, non-answers included, and not a uniform prior over the curated list. So the <!--n replay.priors.contrast.curation_repeat_off-->+0.387 ± 0.041 that column shows for the curation prior is two things bundled together: telling the solver the answer is probably a curated word at all, and the `wordlebot` shape concentrating the off-list remainder onto about <!--n curation_shapes.wordlebot.support ,.0f-->911 words. Nothing measured here separates them.

**The two priors are strongly superadditive**, and the shapes below are not. Adding the repeat prior to a solver with no curation prior is worth <!--n replay.priors.contrast.repeat_curation_off-->+0.008 ± 0.036, next to nothing. Adding the same prior on top of the curation prior is worth <!--n replay.priors.contrast.repeat_curation_on-->+0.417 ± 0.047. The interaction is **<!--n replay.priors.interaction-->−0.410 ± 0.059**, and the three-arm blend agrees at <!--n replay.priors.interaction3-->−0.406 ± 0.053. The blend is a wider view of the same data and not an independent check, since its repeat arm is those same <!--n replay.priors.n_repeats-->28 events.

The mechanism is calibration. Each prior implies a repeat rate and an off-list rate, and the regime itself shows <!--n regime.since.repeat .1f-->10.5% and <!--n regime.since.offlist .1f-->16.8% (`analysis/regime.py`):

<!--table regime.implied_table-->
| prior | implied repeat rate | implied off-list rate, given not a repeat |
| --- | --- | --- |
| none | 14.9% | 95.9% |
| curation only | 59.5% | 66.1% |
| repeat only | 10.0% | 95.9% |
| both | 10.0% | 30.0% |
| *observed, the regime* | *10.5%* | *16.8%* |
| *observed, the last 100* | *10.0%* | *31.1%* |
<!--/table-->

Every prior but the last is far off on the rate the other one governs. Uniform lands near the repeat rate by accident: <!--n regime.used_words-->1,927 used words in 12,947 is <!--n regime.used_share .1f-->14.9%, so the repeat prior corrects a column that was barely wrong and leaves the <!--n regime.implied.none.offlist .1f-->95.9% untouched, and that is why it is worth almost nothing on its own. Curation alone is wrong in both columns. Its 0.30 is conditional on not repeating, and with the repeat prior off nothing marks a curated word as spent, so it puts <!--n regime.implied.curation_only.offlist .1f-->66.1% of the non-repeat mass off-list, and a 2,309-word list with <!--n regime.curated_used-->1,856 spent amounts to a prediction that the editor repeats herself nearly six times in ten. Only with both on is the repeat rate right, and the off-list rate then sits where *The regime the defaults describe* set it, above both windows by design. That is the whole of the interaction.

The same calibration explains the repeat arm, where `curation only` scores <!--n replay.priors.curation_only.repeats-->−0.179 ± 0.171 against `both` on the <!--n replay.priors.n_repeats-->28 answers that were repeats. A repeat averages <!--n regime.weight.used .1e-->5.2e-5 of the mass under `--repeat-prob 0.10` against the <!--n regime.weight.curation_only.used .1e-->3.1e-4 it would hold as an ordinary curated word, and any prior that moves a class from <!--n regime.implied.curation_only.repeat .1f-->59.5% to 10.0% does worse on the occasions the class turns up. In exchange the fresh curated word rises from <!--n regime.weight.curation_only.fresh .1e-->3.0e-4 to <!--n regime.weight.fresh .1e-->1.4e-3, worth <!--n replay.priors.curation_only.other-->+0.591 ± 0.051 across the <!--n regime.mixture.fresh .1f-->74.4% of the regime that is neither a repeat nor off-list.

### The repeat prior's shape

With `--repeat-shape uniform`, `build_weights` spreads the repeat mass evenly over all <!--n regime.used_words-->1,927 used words. The reuse is not uniform. <!--n drift.from_first_300-->26 of the <!--n drift.events-->28 repeats came from Wordle #1–#300 (median #<!--n drift.median_source .0f-->120), with only <!--s drift.outliers-->`sandy` (#1227), `smile` (#831)<!--/s--> outside, an <!--n drift.outlier_rate .0f-->7% outlier rate, too high to zero out.

Scored prequentially, where each model predicts repeat *i* from repeats 1…*i−1* only, so no parameter sees the point it is scored on:

<!--table repeat_shapes.table-->
| model | logL over 28 events | vs. uniform |
| --- | --- | --- |
| uniform over used (current) | −209.9 | — |
| binned, width 200, α = 0.5 | **−175.1** | **+34.8** |
| binned, width 200, α = 1.0 | −178.6 | +31.4 |
| best of 16 window variants (last K ∈ 3/5/8/all, ±25…±200) | −185.4 | +24.5 |
<!--/table-->

The winner is a Laplace-smoothed histogram over fixed bins, at <!--n repeat_shapes.gain +.2f-->+1.24 ± <!--n repeat_shapes.gain_se .2f-->0.18 nats per event (<!--n repeat_shapes.sigma .1f-->6.8σ, winning <!--n repeat_shapes.wins-->23 of <!--n repeat_shapes.events-->28). Scored in-sample the same comparison claims <!--n repeat_shapes.insample +.2f-->+1.55, so overfitting inflates it by about <!--n repeat_shapes.inflation .0f-->24% without manufacturing it.

The window row is the point of the exercise. A window has a boundary, a boundary is a parameter, and it has to be tuned on the same events it is then scored against. The best of sixteen tuned windows lands <!--n repeat_shapes.window_behind .0f-->10 nats behind the histogram. The histogram has one knob too, its bin width, and 200 is the best of <!--n repeat_shapes.widths-->8 widths from <!--n repeat_shapes.width.lo-->50 to <!--n repeat_shapes.width.hi-->600, so that is a tuned histogram against a tuned window. Across those widths its lead over the best window runs from <!--n repeat_shapes.width.lead_min +.1f-->−6.0 to <!--n repeat_shapes.width.lead_max +.1f-->+10.3 nats. What separates the two is how much a bad setting costs: the worst width scores <!--n repeat_shapes.width.worst .1f-->−191.4 and the worst window <!--n repeat_shapes.window_worst .1f-->−325.4. A window also needs a third knob besides `K` and the half-width, the share of mass to leave outside. With a hard cutoff a single repeat landing outside costs about 7 nats, as `sandy` and `smile` would.

The obvious rival is an age-based shape, with weight rising by time since last use instead of binning by where in the answer order the word first appeared. It does not beat the histogram. Refitting its rate prequentially scores <!--n repeat_shapes.age_refit .1f-->−191.2, well short of binned, because the rate is unstable on twenty-odd events. A rate fixed at 2/year reaches <!--n repeat_shapes.age_2 .1f-->−172.6, but only when chosen with hindsight on the same events, and it wins by piling mass on the oldest words and then losing heavily on `sandy` and `smile`. The Laplace floor is what lets binned survive its outliers.

On its own the shape is worth about seven hundredths of a guess on real answers. `analysis/replay.py --shapes` replays all <!--n replay.shapes.puzzles-->266 real puzzles of the current regime, rebuilding the prior from strictly earlier puzzles each day. Sharpening the repeat shape alone scores <!--n replay.shapes.repeat_alone.mean .4f-->3.2481 against the flat shapes' <!--n replay.shapes.uniform.mean .4f-->3.3158, a paired **<!--n replay.shapes.repeat_alone.vs-->−0.0677 ± 0.0246**. <!--n replay.shapes.repeat_alone.changed-->35 games change length, <!--n replay.shapes.repeat_alone.better-->26 for the better and <!--n replay.shapes.repeat_alone.worse-->9 for the worse. The repeat answers carry most of it, at <!--n replay.shapes.repeat_alone.repeats-->−0.536 ± 0.150 on the <!--n replay.shapes.n_repeats-->28 of them, and the <!--n replay.shapes.n_other-->198 that are neither repeats nor off-list move by <!--n replay.shapes.repeat_alone.other-->−0.015 ± 0.017.

The paired sd is <!--n replay.shapes.repeat_alone.sd .2f-->0.40 guesses, which puts the effect more than two standard errors from zero. The shape ships on the modeling evidence above, and the replay agrees.

The source window is not drifting forward at the rate a fixed replay lag would need. The drift rate is how far the source moves per puzzle, so **0 is a static window and 1.0 is a fixed replay lag**, since one puzzle a day means a constant lag drags the source forward one position a day. It is measured as a Theil–Sen slope, the median of the slopes between every pair of repeats. Over all <!--n drift.events-->28 events the rate is <!--n drift.theil_sen +.2f-->+0.10, with a 95% bootstrap interval of <!--n drift.theil_sen.lo95 +.2f-->−0.64 to <!--n drift.theil_sen.hi95 +.2f-->+0.73: it excludes a fixed replay lag and includes a static window. Over the first 18 it was <!--n drift.first18.theil_sen +.2f-->+0.78, with an interval of <!--n drift.first18.theil_sen.lo95 +.2f-->−0.25 to <!--n drift.first18.theil_sen.hi95 +.2f-->+1.56 that excluded neither. A least-squares slope over all <!--n drift.events-->28 is <!--n drift.drift-->−0.40 ± 0.64, and <!--n drift.drift_inside-->+0.12 ± 0.21 without <!--s drift.outliers-->`sandy` (#1227), `smile` (#831)<!--/s-->; the far-off sources set its sign. The running median of the sources on the editor's `/dev` page climbs over the early repeats and falls back over the later ones. The rank correlation between repeat order and source index fell the same way, from <!--n drift.first18.rho +.2f-->+0.46 (p = <!--n drift.first18.p .2f-->0.06) to <!--n drift.rho +.2f-->+0.09 (p = <!--n drift.p .2f-->0.65), because the newest repeats keep arriving from early in the answer order: <!--s drift.latest_two-->`major` and `stink`<!--/s-->, the latest two, came from <!--s drift.latest_two_sources-->#21 and #15<!--/s-->. The replay lag averages <!--n drift.lag_mean .2f-->4.51 years with an sd of <!--n drift.lag_sd .2f-->0.77, and <!--n drift.in_band-->22 of <!--n drift.events-->28 events sit between <!--c-->4.4 and <!--c-->5.1 years.

### The curation prior's shape

Weighting off-list words instead of spreading them flat is a large modeling gain on its own. Scored prequentially over the off-list answers, uniform is <!--n curation_shapes.uniform.nats .2f-->−9.27 nats per event. An answer-like score fitted against the curated list as positives is <!--n curation_shapes.frequency.nats .2f-->−8.03 at the shipped `t` = 0.35. WordleBot's weights, reconstructed by a third party, are <!--n curation_shapes.wordlebot.nats .2f-->−6.44, with the effective support falling from <!--n curation_shapes.uniform.support ,.0f-->10,638 words to <!--n curation_shapes.wordlebot.support ,.0f-->911. That list was scraped after every event it is scored on. A 2023 snapshot predates <!--n curation_shapes.post-->67 of the <!--n curation_shapes.events-->71, and on those <!--n curation_shapes.post-->67 it scores <!--n curation_shapes.snapshot.post .2f-->−6.60 against the live list's <!--n curation_shapes.wordlebot.post .2f-->−6.43.

Between the two sharpened shapes the modeling gain shows in guesses. `analysis/replay.py` runs `frequency` against the default `wordlebot` head to head. `wordlebot` wins by **<!--n replay.frequency.blend-->+0.054 ± 0.029** blended, and by <!--n replay.frequency.vs-->+0.0752 ± 0.0342 over the <!--n replay.puzzles-->266 regime puzzles. Nearly all of it is the off-list arm, <!--n replay.frequency.arm.offlist-->+0.437 ± 0.091 over its <!--n replay.arms.offlist-->71 events. The repeat arm reads <!--n replay.frequency.arm.repeat-->−0.250 ± 0.132, under two standard errors from zero. `wordlebot` ships on the modeling evidence first, and `frequency` is there for anyone who would rather not vendor a scrape.

### What the two shapes are worth together

The shapes convert only a little of their modeling gain into guesses. Measured on three arms, the <!--n replay.shapes.arms.repeats-->28 real repeats and <!--n replay.shapes.arms.offlist-->71 off-list answers with the prior rebuilt as of each puzzle, plus the regime's <!--n replay.shapes.arms.fresh-->198 fresh curated answers, blended at the regime's own mixture of <!--s replay.shapes.mixture-->10.5% / 15.0% / 74.4%<!--/s-->, against a baseline with both priors on and uniform shapes inside them:

<!--table replay.shapes.arm_table-->
| shapes | repeat | off-list | fresh | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.179 ± 0.179 | −0.761 ± 0.095 | −0.025 ± 0.038 | −0.114 ± 0.037 |
| repeat `binned` alone | −0.536 ± 0.150 | +0.014 ± 0.043 | −0.015 ± 0.017 | −0.066 ± 0.021 |
| `binned` + `frequency` | −0.429 ± 0.174 | −0.380 ± 0.107 | −0.005 ± 0.038 | −0.106 ± 0.037 |
| `binned` + `wordlebot` | −0.179 ± 0.193 | −0.817 ± 0.097 | −0.025 ± 0.039 | −0.160 ± 0.038 |
<!--/table-->

Concentrating the off-list mass on about <!--n curation_shapes.wordlebot.support ,.0f-->911 words gives each about <!--n regime.weight.offlist_listed .1e-->2.9e-4 against a used word's <!--n regime.weight.used .1e-->5.2e-5, so past answers stop competing for the solver's attention. That is the source of the penalty `wordlebot` alone carries on the repeat arm.

Every row reaches its own error, and the shipped pair is the largest, at <!--n replay.shapes.together.blend-->−0.160 ± 0.038, <!--n replay.shapes.together.blend_sigma .1f-->4.2σ from zero, and the same on the plain regime replay at <!--n replay.shapes.together.vs-->−0.1541 ± 0.0434. Whether the two shapes need each other the table cannot say. Apart they score <!--n replay.shapes.curation_alone.blend-->−0.114 ± 0.037 and <!--n replay.shapes.repeat_alone.blend-->−0.066 ± 0.021, summing to <!--n replay.shapes.apart +.3f-->−0.180, against <!--n replay.shapes.together.blend_delta +.3f-->−0.160 together, a gap of about two hundredths with errors of two to four hundredths on either side. `frequency` (<!--n replay.shapes.binned_frequency.blend-->−0.106 ± 0.037) sits a step behind `wordlebot` here, and the paired head-to-head above measures that step at under two standard errors.

**A census of the curated words still unused is the wrong fresh arm, because it is training on the test set.** Policy iteration draws its practice answers from the prior it is fitting, whose fresh-curated group is exactly those <!--n regime.curated_unused-->453 words, so <!--n regime.mixture.fresh .0f-->74% of the table's weight would sit on material every arm had trained against. Both solvers reach their own tuned best there and tie, and a tie on <!--n regime.mixture.fresh .0f-->74% of the grade leaves the small arms to decide the whole table. `analysis/fresh_arm.py` scores the two arms against each other for the repeat shape. Under the fits in play they agree, at <!--n fresh_arm.census_today-->+0.004 ± 0.009 on the census against <!--n fresh_arm.own.real-->−0.016 ± 0.018 on the answers that happened, and the gap between the two word sets is <!--n fresh_arm.own.gap .3f-->−0.017 with each arm on its own `f` and <!--n fresh_arm.sharp.gap .3f-->0.015 with both on one. What the census reads is whatever the two fits learned about it, and under other fits it has read <!--u-->a tenth of a guess kinder to the sharpened shape than the real answers, so the agreement is a fact about the fits and not a reason to use the arm.

The arm is the <!--n replay.shapes.arms.fresh-->198 real answers, which costs precision where the variance lives and keeps the one edge this table has over the regime replay above, the off-list arm's <!--n replay.shapes.arms.offlist-->71 events against the window's <!--n replay.shapes.n_offlist-->40. That is worth about a fifth, ±<!--n replay.shapes.together.blend_se .3f-->0.038 here against ±<!--n replay.shapes.together.se .3f-->0.043 there on the same contrast, and the two agree, both putting the shipped pair more than three standard errors clear of zero. The modeling evidence is the primary case for the shapes, and this table agrees with it.

The measurement needs `f` refitted per shape. Every prior is fitted in `spline` (*Which form is the default*), but a `spline` fitted under one pair of shapes is a different curve from one fitted under another. A fit carried across is a curve fitted to some other prior's weights, and the opening word can go with it.

### The group with no shape

Two of the three groups `priors.build_weights` builds get a shape, and the third and largest does not. The curated words not yet used carry <!--n regime.share.fresh .0f-->63% of the mass at the shipped probabilities, spread flat over the <!--n fresh_shapes.fresh_curated.pool_to-->466 that are left. `analysis/fresh_shapes.py` says flat is where it should stay.

The events are the <!--n fresh_shapes.fresh_curated.events-->1,338 first uses of a curated word from Wordle #507 (2022-11-07), the first puzzle `solutions.json` credits to an editor. Before it the answers walk the curated list in its stored order, so the pick is not a choice. Each model is normalized over the words still unused that day and never sees the event it is scored on, the same rule the other two shapes are held to:

<!--table fresh_shapes.fresh_curated.table-->
| shape | nats/event | vs. uniform | effective support |
| --- | --- | --- | --- |
| uniform over the pool (current) | −6.9687 | — | 466 |
| WordleBot weight | −6.9384 | **+0.0302 ± 0.0007** | 446 |
| WordleBot, 5% hedge (as shipped off-list) | −7.0557 | −0.0871 | 307 |
| list position, binned 100/200/400 | −6.9686 | +0.0001 | 466 |
<!--/table-->

WordleBot wins at <!--n fresh_shapes.fresh_curated.wordlebot.sigma .0f-->44σ on <!--n fresh_shapes.fresh_curated.wordlebot.wins-->1,297 of <!--n fresh_shapes.fresh_curated.events-->1,338 events, and the win is worth almost nothing: <!--n fresh_shapes.fresh_curated.wordlebot.gain .2f-->0.03 nats against a baseline of <!--n fresh_shapes.fresh_curated.uniform.nats .2f-->−6.97, moving the effective support from <!--n fresh_shapes.fresh_curated.pool_to-->466 words to <!--n fresh_shapes.fresh_curated.wordlebot.support .0f-->446. For comparison, the same weights on the off-list group are worth **<!--n fresh_shapes.off-list.wordlebot.gain +.4f-->+2.8159** and take its support from <!--n fresh_shapes.off-list.pool_to-->10,575 to <!--n fresh_shapes.off-list.wordlebot.support ,.0f-->900. So WordleBot's weights say a great deal about which off-list word the editor will pick and almost nothing about which curated word, even though the curated group holds <!--n regime.share.fresh .0f-->63% of the mass and the off-list group <!--n regime.share.offlist .0f-->27%.

One might blame the features tried, but a conditional logit over <!--n fresh_shapes.features-->6 of them (word frequency, WordleBot weight, list position, vowels, rare letters, distinct letters), fitted on every event at once and scored on what it was fitted on, gains only **<!--n fresh_shapes.ceiling +.4f-->+0.1067 nats/event**. That is a ceiling with hindsight built in, and anything prequential comes in under it. The best single feature is <!--s fresh_shapes.best_feature-->word frequency<!--/s--> at <!--n fresh_shapes.best_feature_gain +.4f-->+0.0653, so the editor leans slightly towards commoner words and otherwise draws close to uniformly from whatever the list has left.

The shipped hedge goes backwards here. Five percent is sized for a group where <!--n fresh_shapes.pool.offlist_unlisted-->9,746 of <!--n fresh_shapes.pool.offlist_words-->10,574 words are unlisted. Only <!--n fresh_shapes.pool.fresh_unlisted-->7 curated words are, so the same 5% hands each of those few several times what a listed word gets. A hedge has to be sized for its own group, and this one was sized for the other.

That is also why the two groups are not pooled and weighted by WordleBot throughout, the obvious simplification since the scrape covers curated and off-list words alike. WordleBot lists <!--n fresh_shapes.pool.offlist_listed-->828 off-list words carrying more total weight than all <!--n fresh_shapes.pool.fresh_listed-->458 listed fresh-curated words, so one pool implies the editor leaves the curated list on <!--n fresh_shapes.pool.pooled_rate .1f-->57.9% of non-repeat days against the <!--n fresh_shapes.pool.regime_rate .1f-->15.1% the regime ran at. It is the same failure the `curation only` arm makes on repeats, since the weight ranks words within a group and does not set the rate between two.

### The prior sits in a flat basin

None of the prior's settings is worth tuning. `analysis/prior_basin.py` sweeps both axes over the <!--n prior_basin.puzzles-->266 real regime puzzles with the shipped `f` held throughout. The last paragraph here says what that does to the edges.

Inside the groups there is nothing to tune. Tempering a shape raises its relative weights to a power, with 0 leaving the group flat and 1 being what ships. Both shapes cost a little more at each step toward flat, and no single step clears its own noise. Flattening the repeat shape costs **<!--n prior_basin.repeat_flattened-->+0.030 ± 0.024**, about one standard error, and the points between land within a hundredth or two of either end. Flattening the curation shape outright costs <!--n prior_basin.curation_flattened-->+0.068 ± 0.044, and every point from a quarter sharpened to fully sharpened lands within <!--n prior_basin.curation_span .3f-->0.045 guesses of the shipped setting, even though this is the shape with much the larger modeling gain.

Between the groups, the mass moves by a factor of twenty before the total notices:

<!--table prior_basin.table-->
| `repeat_prob` | 0.01–0.20 | 0.30 | 0.40 | 0.55 | 0.71 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- |
| all 266 | within ±0.02 | +0.098 | +0.124 | +0.150 | +0.244 | +0.586 |
| the 198 fresh | within ±0.03 | +0.167 | +0.202 | +0.237 | +0.384 | +0.778 |
<!--/table-->

`noncurated_prob` has the same shape, flat to <!--n prior_basin.noncurated.flat_to .2f-->0.40 and then <!--s prior_basin.noncurated.beyond-->+0.143, +0.233, +0.511<!--/s-->. The subgroups underneath are not flat. The repeat arm runs from <!--n prior_basin.repeat.repeat_lo-->+0.179 at `repeat_prob` <!--c-->0.01 to <!--n prior_basin.repeat.repeat_hi-->−0.250 at <!--n prior_basin.repeat.flat_to .2f-->0.20, and the fresh arm runs the other way, so the trade is real and cancels across the basin.

That reconciles this with *What each prior is worth on real answers*, where switching the repeat prior off costs <!--s replay.priors.contrast.repeat_curation_on-->+0.417 ± 0.047<!--/s--> guesses and the curation prior <!--s replay.priors.contrast.curation_repeat_on-->+0.797 ± 0.050<!--/s-->, each with the other on. Off is not a small probability but a wrong one. With curation still on and the repeat prior off, used and unused curated words share the mass equally, which implies a repeat rate of <!--n regime.implied.curation_only.repeat .1f-->59.5%. The settings that cost anything sit out at <!--n regime.implied.curation_only.repeat .0f-->59% and <!--n regime.implied.none.offlist .0f-->96%, and the basin covers everything from <!--c-->1% to <!--n prior_basin.repeat.flat_to .0%-->20%. The default 0.10 and the regime's measured <!--n prior_basin.regime_repeat .3f-->0.105 are both well inside it. The prior has to be roughly right, and beyond that the details do not reach the board.

The far ends read worse than they should, since a setting `f` was never fitted for carries a borrowed estimator on top of its own miscalibration. That inflates the edges and not the middle, so the basin is at least this wide.

### How fresh the answer record has to be

Every measurement above rebuilds the prior from strictly earlier puzzles on every day of the replay, and the prequential shape scores hold each model to the same rule. The deployed solver is not that current. `data/solutions.json` is refreshed by hand with `solutions.py --write`, and stops at yesterday when it is, so a day of lag is there the moment it is written. `analysis/refit_lag.py` measures what the gap costs, replaying the <!--n refit_lag.puzzles-->266 regime puzzles with the record truncated to `t − lag` instead of `t`:

<!--table refit_lag.table-->
| lag | mass moved | vs. a record rebuilt daily | games changed of 266 |
| --- | --- | --- | --- |
| 1 day | 0.14% | +0.0075 ± 0.0075 | 4 (3 worse, 1 better) |
| 3 days | 0.41% | −0.0075 ± 0.0119 | 7 (3 worse, 4 better) |
| 7 days | 0.93% | +0.0150 ± 0.0141 | 11 (7 worse, 4 better) |
| 14 days | 1.81% | +0.0113 ± 0.0203 | 20 (11 worse, 9 better) |
| 30 days | 3.69% | +0.0150 ± 0.0226 | 33 (18 worse, 15 better) |
<!--/table-->

Averaged over a refresh cycle, where refitting every k days leaves the lag uniform over 0…k−1, that is **<!--n refit_lag.every2-->+0.0038 ± 0.0038** every two days, <!--n refit_lag.every4-->+0.0000 ± 0.0062 every four and <!--n refit_lag.every7-->−0.0027 ± 0.0079 every week. None of it is distinguishable from nothing, and at 30 days the changed games split evenly enough to be coin flips.

Two things could account for a null like that. Either the weights barely move, or they move and the board does not care, as the basin above finds for every other way of perturbing this prior. Here it is the first. A lag of k days leaves k words carrying fresh-curated mass instead of used mass and changes nothing else, so a week relocates <!--n refit_lag.lag7.mass .2f-->0.93% of the prior's mass, against the <!--n refit_lag.yardstick .2f-->6.39% that flattening the repeat shape relocates, itself worth only <!--n prior_basin.repeat_flattened-->+0.030 ± 0.024.

Refitting `f` does not enter this. `estimator.cache_path` is keyed on the prior's slug alone, so one fit serves every day of a replay and every day of a deployment, and a fresher record changes the weights and nothing else. It changes them only where `priors.build_weights` reads it, which is under `repeat` alone. A repeat-off prior costs nothing to leave stale, which is why `--repeat-only` can skip its arms after a write.

## Choosing a guess

`--mode entropy` is the greedy scorer. It maximizes expected information gain and nothing else, so a guess earns nothing for being a possible answer. That makes it a clean reference for what rewarding that chance is worth.

`--mode expected` (the default) instead minimizes an estimate of how many guesses the game will take in total:

```
E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)
```

where the candidates in state `S` are bucketed by the feedback pattern `r` that guess `g` would produce, `p_r` is the bucket's prior mass, and `n_r` is its effective size `2 ** H` (the plain count when the prior is flat over it). Buckets of one or two candidates are worth exactly 1 and `2 - p_max`, so those values are used directly. Dropping the `GGGGG` term is why a guess that could be the answer beats an equally informative guess that could not.

Both are fallbacks, used where the game is too wide to search.

### Hard mode (`--hard`, off by default)

Every guess must be a word that could still be the answer. It is a flag on `benchmark` and a switch on the page.

**This is not the NYT's hard mode**, which is a rule about letters. It keeps every green where it is and uses every revealed letter somewhere, in as many copies as have been revealed. It says nothing about the letters ruled out, so it permits a guess that spends a gray letter, carries a copy more than the answer holds, or puts a yellow back in the square where it came up yellow. Every word that could still be the answer satisfies that rule, so what is played here is the narrower of the two, and the solver never sees the extra moves.

Widening it to the NYT's rule is worth nothing, and every word is played under both rules to say so (`analysis/hard_mode_rules.py`). Handing the solver the legal pool at each turn is one substitution again, but the pool is now a function of the rows played and not of the candidates, so the script plays the policy itself. `--check` shows that loop reproducing `common.every_answer` game for game.

| policy | guesses taken off the candidates | words changing length | what the wider rule is worth |
| --- | --- | --- | --- |
| expected, as shipped | <!--n hard_mode_rules.shipped.off_pool-->0 in 12,947 games | <!--n hard_mode_rules.shipped.changed-->0 | <!--n hard_mode_rules.shipped.worth +.4f-->+0.0000 |
| expected, endgame off | <!--n hard_mode_rules.no_endgame.off_pool-->56 | <!--n hard_mode_rules.no_endgame.changed-->25 (<!--n hard_mode_rules.no_endgame.better-->13 better, <!--n hard_mode_rules.no_endgame.worse-->12 worse) | <!--n hard_mode_rules.no_endgame.worth +.4f-->−0.0000 |
| **greedy entropy** | **<!--n hard_mode_rules.entropy.off_pool-->2,524** | **<!--n hard_mode_rules.entropy.changed-->859 (<!--n hard_mode_rules.entropy.better-->217 better, <!--n hard_mode_rules.entropy.worse-->642 worse)** | **<!--n hard_mode_rules.entropy.worth +.4f-->+0.0461** |

The objective never takes the freedom. The pool is the whole word list at turn one under either rule, and after that a word that cannot win has to be worth more than the win chance it gives up. The objective already weighs that trade, and it comes out against the probe every time. With the endgame off, so that the pool and not the search decides every turn, it does reach for a non-candidate <!--n hard_mode_rules.no_endgame.off_pool-->56 times, splitting <!--n hard_mode_rules.no_endgame.better-->13 words better and <!--n hard_mode_rules.no_endgame.worse-->12 worse for no movement in the mean. Only greedy entropy uses the wider rule freely, and it does worse for it. Entropy gives no credit for winning outright, so it takes the sharper split and forfeits games it could have ended, the same asymmetry SOLVER_EXPLAINED *A straight-line* turns on.

Both arms keep the endgame search restricted to the candidates, since its nodes would need a constraint history they are never given, so the figures above are a floor on what the NYT's rule is worth. Widening the search too is untested here. The reason to expect little from it is that the objective declined the wider pool at every turn where it was free to choose, which is an argument and not a measurement.

Mechanically it is one substitution. The pool of guesses to score becomes the candidate set instead of all 12,947 words (`WordleSolver.guess_pool`), which the kernels already take as a parameter, and the exact search narrows the same way. In the search that also removes the partition dedupe, since only a candidate itself can come back `GGGGG`, so no two candidates split S identically.

It is **not** a `train` flag. The fit is cached per prior and the restriction is not part of the slug, so training under it would overwrite the ordinary fit. `f` is therefore always the one fitted for unrestricted play, and its scores read a little low under the restriction, since the position it estimates is one where the solver would be allowed a probe that it is not allowed here. The restriction costs guesses for the same reason. A word that cannot win is often the sharper split.

What it costs, every word weighted by the row's own prior, the same words in both arms, failures counted as 7 (`analysis/hard_mode.py`). Both arms are exact, so the cost is a difference and not an estimate:

<!--table hard_mode.table-->
| priors | free | hard | cost | solve rate, free → hard |
| --- | --- | --- | --- | --- |
| none | 4.0718 | 4.4504 | +0.3785 | 99.95% → 92.28% |
| curation only | 3.5282 | 3.6348 | +0.1066 | 99.98% → 99.47% |
| repeat only | 4.0062 | 4.3671 | +0.3609 | 99.93% → 92.64% |
| **both (the default)** | **3.1769** | **3.2200** | **+0.0431** | **99.97% → 99.82%** |
| both, `frequency` | 3.2928 | 3.3344 | +0.0416 | 99.93% → 99.37% |
<!--/table-->

Under the default prior it costs **<!--n hard_mode.both.cost +.4f-->+0.0431** guesses, a small fraction of what the estimator is worth over greedy entropy. The flatness of the prior sets the size. With no priors the cost is <!--n hard_mode.none_over_both .0f-->9 times larger and the solve rate falls to **<!--n hard_mode.none.hard_solved .2f-->92.28%**. Two causes would fit, and nothing here separates them. A wide candidate set is where a probe helps most and the restriction forbids one, while concentrated mass collapses the set fast enough that the likely answers are decent probes anyway. Or the free-play `f` that hard mode borrows fits a flat prior's hard-mode positions worse than a concentrated one's.

The losses show it most. Even in the default row, where the mean barely moves, the loss probability grows <!--n hard_mode.both.loss_ratio .1f-->6.4 times. Hard mode arrives in the one-letter families of *The penalty* below and cannot spend a turn separating them, since the separating word is the one it is not allowed to play.

The opening word never changes in any row. Every one of the 12,947 words carries prior mass, so at the first guess every word is still a possible answer and the restriction binds nothing.

Under `--mode entropy` the restriction **helps**, by <!--n hard_mode.entropy.both.cost +.4f-->−0.1775 guesses on the default prior (<!--n hard_mode.entropy.both.free .4f-->3.7722 → <!--n hard_mode.entropy.both.hard .4f-->3.5947). That is the identity under *Results* read backwards. Greedy entropy is missing the bonus for guessing a word that could be the answer, and hard mode supplies it by force. It costs losses, with its solve rate falling from <!--n hard_mode.entropy.both.free_solved .2f-->99.99% to <!--n hard_mode.entropy.both.hard_solved .2f-->99.76%, and it still does not reach the expected-guesses policy's <!--n hard_mode.both.free .4f-->3.1769, which gets the same bonus for free and keeps the probes.

## Forms of f

Any basis linear in its parameters works, so every fit is a single least-squares solve. `estimator.FORMS` holds them, all written as `f(n) = base + sum_i params[i] * term_i(n)` with `x = log2(n)` and `base` either 0 or 1 (pinning `f(1) = 1`). The result is floored at 1.

f must not decrease anywhere in `[1, 12947]`, since a value function that rewards a larger candidate set would let the search prefer uninformative guesses. Terms are dropped from the end (most flexible first) until it is monotone.

## Policy iteration

`train` alternates:

1. **evaluate**: play 2,000 sampled games with the current policy, recording for every position its effective size and how many guesses were actually needed from there;
2. **improve**: refit `f` to those pairs. Acting greedily on the new `f` is the improved policy.

Round 0 is seeded with greedy entropy. Training games are played with no deadline, so targets are unbiased estimates of expected guesses and `f` keeps that meaning. The six-guess limit is a constraint layered on top.

The loop stops when the parameters settle, or when the measured mean stops improving (a greedy policy can cycle), and caches the parameters whose policy measured best: the form, opening word, fit history and holdout score, one file per prior combination in `estimator_cache/params_<slug>.json` (with `compare` archiving to `estimator_cache/forms/<slug>__<form>.json`). The slug names the shapes as well as the toggles, so a fit made under one shape is never read back under another. A cached file is refused if it was fitted with different prior probabilities, or predates the form registry.

**A prior with no fit does not play.** `load_active` raises `NotFitted` and names the `train` command that makes one. The alternative is to bend a fixed reference curve onto the chosen form and play that. The invented curve is not a measurement of any prior, and <!--u-->under the default one it read f(2000) ≈ 3.69 where the fitted `f` says 2.82. Played, it flipped the opening word from `salet` to `raise` and cost 0.034 ± 0.004 guesses paired over 20,000 games<!--/u-->, and since it was never saved, every run recomputed it. Entropy mode, which never consults `f`, runs on any prior.

The refusal carries the `train` command that ends it, so both front ends hand it over as it stands. The terminal prints it and exits 2 instead of raising a traceback, and the page shows it with the same command in it. Startup shows it on the veil instead of a stack trace, since there is nothing to debug.

### What the loop converges to

The loop does not converge to the best `f` it sees, so `policy_iteration` keeps the parameters whose policy measured best rather than the ones it stopped on, and that selection is itself a sample. The widest miss on the last refit was `<!--s fit_rounds.worst.slug-->repeat0_curation0<!--/s-->` fitted at `--exact-max <!--n fit_rounds.worst.reach-->60`, where the loop kept an `f` worth **<!--n fit_rounds.worst.kept .4f-->4.0716** guesses over every answer while one of its own rounds had produced **<!--n fit_rounds.worst.best .4f-->4.0707**, <!--n fit_rounds.worst.gap .3f-->0.001 better. Nothing guarantees otherwise. `f` carries at most six parameters and a whole candidate set is compressed to one scalar before it sees it, so the fixed point of the fit is nobody's optimum in particular.

The selection statistic looks like the weak link. `best_train_mean_guesses` is a mean over the same 2,000 answers the round was fitted on, and the loop takes an argmin over half a dozen correlated draws of it, which is the shape of a selection that overfits itself. It turns out to matter little. `analysis/fit_rounds.py` hands every distinct `f` any round produced to `every_answer --form --params` instead, over all 12,947 words weighted by the prior. The kept fit is the exact best of its own run in **<!--n fit_rounds.kept_is_best-->4 of <!--n fit_rounds.runs-->10** runs across the five priors and both settings, giving up <!--n fit_rounds.worst.gap .3f-->0.001 guesses in the worst of the others. `form_default --promote` collects what there is to win there. It ranks the new `spline` archive and the active fit by that same exact sweep, so the fit that ships is the sweep's choice and never the loop's sample estimate.

Round 0 is not a throwaway either. On `repeat1_curation0_r-binned` the `f` fitted from the entropy seed's rollouts outscores every later round at both settings, and it is what both runs keep.

### Warm starting from the previous fit

After `solutions.py --write` the previous fit is still in `estimator_cache/`, so seeding round 0 with it instead of with greedy entropy looks like a free head start. It is not, because the entropy seed is the one round that costs nothing. Entropy mode consults neither `f` nor the endgame search, so round 0 runs in about a second where an expected-mode round on a flat prior runs in thirty. Seeded at `--exact-max 60`, `repeat0_curation0` took the same four rounds and its loop went from <!--u-->31.8 s to 59.0 s<!--/u-->.

With the search off the seeding does save rounds, <!--u-->29 to 24 across the five priors and 27.5 s to 23.2 s, and it lands somewhere slightly worse. Scored over every answer the warm-started fits average **3.5571** against **3.5511** cold, and nearly all of the gap is one prior. `repeat0_curation1_c-wordlebot` reaches 3.5241 and opens `raise`, where the cold run reaches 3.4953 and opens `reast`<!--/u-->. Skipping the early rounds means skipping the transients, and the transients are where the good fits were.

The third arm shows that the round count is not set by how far the seed sits from the answer. Seeding with the current fit, the `f` the run is trying to reproduce, saves nothing over seeding with one from a fortnight earlier. It is 24 rounds either way.

None of it is kept. The seed argument those three arms needed is not in the code, so the figures above, like the turn-conditioned ones below, describe an implementation that no longer exists.

### Fitting in Rust

The whole loop has a native side. `core/src/train.rs` is `train.policy_iteration` term for term, `core/src/fit.rs` does the least-squares solve, and `core/src/rng.rs` reproduces `np.random.default_rng` (SeedSequence, PCG64 and `choice`'s own cumulative sum and right-hand search), so both implementations draw the same training answers. `wordle_solver.py train --rust` hands a fit over. `rust_train.py` declines where a setting has no flag instead of guessing, and `estimator.save_params` still writes the cache, so there is one writer and one format.

Two of the three pieces agree exactly. The sampler is bit for bit, over three seeds and all five priors, index by index. The fit is not, and there is a reason it cannot be. `np.linalg.lstsq` is LAPACK's SVD. When a spline hands it a rank-deficient problem, which happens routinely because a hinge beyond the widest position seen is a column of zeros, the SVD returns the minimum-norm solution and lands that column on exactly zero. `fit.rs` drops such a column before the solve, which is where the SVD would have left it anyway, and takes the full-rank remainder by Householder QR. A plain QR needs the column gone, since its back-substitution divides by a diagonal a dead column would leave at zero. The two agree to about 6e-13 across 27 dataset-and-form pairs and keep the same terms in every one, so the fit is the second thing `tests/rust_parity.py` compares on a tolerance, after the letter odds.

**Parity stops at the seed round.** Given identical parameters the two play identically, with the same guesses, the same rollouts and the same mean. But the first fit differs in its last bits, greedy policy iteration acts on that, and somewhere a tie breaks the other way. One rollout changes, the next fit changes, and from there the two walk different paths to different fixed points. Even the seed round's own holdout is downstream of it. So the Rust returns a different valid fit, and measurably not a worse one. `analysis/rust_fit.py` fits every prior both ways over <!--n rust_fit.seeds-->3 seeds and scores each `f` over all 12,947 answers weighted by the prior. The native fits lose on <!--n rust_fit.worse-->2 of the <!--n rust_fit.pairs-->15 pairs. The Rust is therefore the default and `--no-rust` fits in Python instead. Every cached fit is the Rust's, along with every measurement that moves with a fit.

A round of policy iteration is **<!--n timing.train.per_round.gain .2f-->1.11×** faster natively, <!--n timing.train.per_round.rust .3f-->0.833 s against <!--n timing.train.per_round.python .3f-->0.926 s, which is well short of the <!--n timing.sweep.expected.python_over_rust .1f-->1.5× the same port gets on a sweep. A sweep spends two thirds of its Python time outside the kernels, where Rust is roughly twice as quick, while a training round is mostly kernel scans that are already parallel and already at memory bandwidth.

A whole `train` for all five priors takes <!--n timing.train.rust .1f-->33.6 s against <!--n timing.train.python .1f-->36.2 s, <!--n timing.train.gain .2f-->1.08×, averaged over <!--n timing.train.seeds-->4 training seeds on an idle machine. One seed alone is a poor measure. Each fit stops after three rounds without improvement, the two trainers walk different paths after the seed round, and so they stop after different numbers of rounds. Which seed is timed moves the ratio more than either trainer's speed does.

The sweep's other trick does not carry over. Playing several games at once is worth <!--n timing.sweep.expected.chunking .2f-->1.08× on a sweep and slows a training round: <!--n timing.train.round.chunks1 .3f-->0.827 s serially against <!--n timing.train.round.chunks2 .3f-->0.987 s at two chunks and <!--n timing.train.round.chunks3 .3f-->1.157 s at three, over the five priors at 2,000 games. Training turns the endgame search off, and the search is what gives a core somewhere to go while a game is between scans. With nothing but scans left, splitting the memo three ways only makes each copy refill. Under `--exact-max 60` it reverses, with three chunks taking the default prior's loop from <!--n timing.train.exact60.chunks1 .1f-->15.4 s to <!--n timing.train.exact60.chunks3 .1f-->12.9 s, so `train --chunks` follows `--exact-max` unless it is set outright.

### Keeping the port in step

The Python is the oracle and `tests/rust_parity.py` diffs every ported piece against it. Four rules keep the two in agreement.

**Reduce float vectors through `sum::pairwise`.** `np.sum` adds pairwise, so its error grows as log n where a left-to-right loop grows as n. `core/src/sum.rs` transcribes numpy's order, and every place the Python spells `.sum()` on floats uses it. Two places accumulate left to right instead. `kernels.rs` does, because a bucket is filled by scattering candidates into 243 slots and the weight total has to match those buckets exactly, or a guess leaving every candidate in one bucket stops scoring exactly zero bits. `api::find`, which stands in for `kernels.ig_all`, does the same. A numpy `@` cannot be matched, since BLAS reduces in an order that is neither documented nor stable across builds. The letter odds and the fit are the two fields compared on a tolerance for that reason.

**Break a tie by word index, over a key rounded to `BRANCH_TIE_TOL`.** The endgame's branch cap keeps the best fifty splits and the bounds tie in bulk. Sorted unstably, each side searched a different fifty and returned runners-up <!--u-->0.02 guesses apart. Both sort stably over the rounded bound, so two bounds a last bit apart land in one bucket and the index decides. *The exact endgame* has the tolerance and why it is centered. A cap, a threshold or a top-k over a key that ties in bulk is brittle whoever evaluates it, since a compiler, a numpy version or a machine can move the answer as easily as a second implementation can. The fix is to make the comparison as coarse as the quantity is meaningful and let a stated rule break the ties.

**Bit-exactness is a means and not the goal.** Where two outputs are compared on a tolerance, the Rust is free to choose its own algorithm. `core/src/fit.rs` solves by Householder QR where `np.linalg.lstsq` uses an SVD, and agrees to 6e-13. What is not free is an order some exactness property rests on, like the kernel totals above, or a discrete cliff that turns a last bit into a different answer, like the branch cap before it was rounded. Transcribing numpy where neither applies costs code for nothing. `core/src/rng.rs` reproduces `SeedSequence` and PCG64 so one parity check can hand both sides the same rollouts, and passing the drawn answers in would have given the same check with none of that code.

**Port the optimizations with the algorithms.** `kernels.py` declares every kernel `parallel=True` and caches the all-rows pool. Leaving either out of the Rust made a whole sweep slower than the Python. `cargo build --release --features wordle_core/profile` times every kernel entry point, and `api_probe` prints those totals for one request. Measuring settled where the time went on each occasion that guessing had it wrong, including one loop outside the kernels that fetched a guess's letters once per candidate instead of once per guess. The browser also needs `source::Columns`, which the Python has no use for. *Without a server* has it.

No fast math is enabled on either side. `rustc` exposes no such flag and release-mode LLVM does not reassociate a float sum. `SOLVER_EXPLAINED.md` *The optimization that had to be taken back out* has what it did to the Python kernels.

What the native side is worth over a sweep, measured on an idle machine (`analysis/timing.py`):

| | expected | entropy |
| --- | --- | --- |
| Python, `common.every_answer` | <!--n timing.sweep.expected.python .1f-->9.4 s | |
| Rust, `every_answer` | <!--n timing.sweep.expected.rust .1f-->6.1 s | |
| keeping `game::SWEEP_CHUNKS` games in flight | <!--n timing.sweep.expected.chunking .2f-->1.08× | <!--n timing.sweep.entropy.chunking .2f-->1.16× |
| the answer-major dispatch, `--fused` | <!--n timing.sweep.expected.fused .2f-->1.04× | <!--n timing.sweep.entropy.fused .2f-->1.49× |

Two thirds of the Python's sweep is outside the kernels, and that part is about twice as fast in Rust. The kernels are memory-bound in both. Three games in flight is where chunking measured best, and one chunk per core runs half as fast again as serial. Per scan the fused dispatch is worth <!--n timing.scan.at100 .1f-->2.5× at <!--c-->100 candidates and <!--n timing.scan.at198 .1f-->1.9× at 200, against the Python's <!--n table_layout.expected.at130 .1f-->2.1× at <!--c-->130 and <!--n table_layout.expected.at400 .1f-->1.1× at <!--c-->400. Over whole games the Python gets <!--n fused_gain.expected.whole .2f-->1.03× in expected mode and <!--n fused_gain.entropy.whole .2f-->1.65× in entropy. `source::Columns` already hands the endgame a compact table over most of the band the fused layout rescues, so what is left for `fused` is the heuristic scans. `bench_position` prints the per-scan figures.

## The exact endgame

Once the candidate set is small, or few turns remain, `exact.py` computes the value outright:

```
V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
V(S, 0) = loss_penalty
```

with `t` the guesses left including this one. A loss penalty of 1 means running out costs one unit more than the six turns already spent, so a lost game scores 7. The deadline then needs no separate objective and no risk weighting.

V is non-decreasing in t only when a loss is priced at 1 or below. Running out is then an exit costing exactly one guess, and no continuation finishes for less, so having turns left over is never an advantage. The bound the search prunes with is therefore the two-turn value computed at `min(loss_penalty, 1)` (`ExactEndgame.bound_penalty`), which sits under every deeper value at the real penalty. Uncapped it would not. At a penalty of 5 the ordering reverses, and the ten `-atch` words are worth <!--n loss_penalty.bound.two_turn_at5 .3f-->1.828 with two turns left against <!--n loss_penalty.bound.three_turn_at5 .3f-->1.537 with three.

Five things make it affordable:

* **Partition dedupe.** Guesses that split S identically are interchangeable, so the branching factor is the number of distinct splits and not 12,947. A guess that is itself a candidate always splits distinctly (only it produces `GGGGG`), so a possible answer is never merged with a probe. It only helps while S is tiny, since by 12 candidates about 12.7k of the guesses already split differently.
* **Closed-form leaves.** One or two turns left, and one or two candidates left, are closed forms: `V(g, 2) = 1 + (A + (A − B)·penalty) / total` over the non-winning buckets, with A their mass and B the sum of their maxima. That is one pass over the guesses with no estimator, the cost of scoring them heuristically, so **two turns or fewer is always solved exactly** at any candidate count.
* **Memoization.** A candidate set is solved once and reused, since the same position is reached along many orders of guesses. Two-turn positions are where this matters most. Four such calls in five are repeats of one already answered, and <!--u-->one position in a 300-game sample came up 373 times<!--/u-->. Caching them instead of re-running the closed form is worth about <!--u-->30% of the search's total time, so the memo is consulted before the two-turn branch.
* **Branch and bound.** Guesses are tried in order of their two-turn value, computed only for the distinct splits, which are about a quarter of the pool. Within a guess each bucket's bound is replaced by its exact value one at a time, and the guess is abandoned as soon as the running total passes the incumbent.
* **Branch cap** (`--exact-branch`, default 50, `0` to disable). Only the best 50 splits by that bound are tried. Branch and bound finds the best guess early and can then spend an unbounded amount of time proving nothing beats it. The cap takes the answer and skips the proof. <!--u-->Over 1,275 real positions, caps of 25 and 50 pick the uncapped-optimal guess **every time** (a cap of 10 misses 5), while wall clock falls from 322 s to 8 s and the worst single position from 30 s to 1.9 s.<!--/u--> That saving is rare and not steady, which counting shows where timing cannot (`analysis/branch_cap.py`). Over 150 real positions the cap changes no interior node, the same <!--n branch_cap.nodes-->274 being searched either way, because the admissible bound is what prunes the search, and at <!--n branch_cap.unbound_nodes-->272 of those nodes it cuts the loop after one or two splits. At the remaining <!--n branch_cap.binding_nodes-->2 the bound separates nothing, and at the busier of them the uncapped search goes through <!--n branch_cap.busiest.none ,d-->8,781 splits to reach the guess it already had. So the cap exists for those nodes and changes nothing at the rest, and a figure timed over too few positions misses it.

Which fifty splits the cap keeps rests on how ties are broken. Two guesses that split the candidates identically score identically, with the same buckets filled in the same order, so `SPLIT_TIE_TOL` is 0 and the dedupe that merges one guess under two spellings gates on equality. Over half a million duplicate pairs the widest score spread inside a group is exactly zero. Bounds tie in bulk too, one position holding <!--n branch_cap.tied.tied ,d-->962 of its <!--n branch_cap.tied.splits ,d-->1,578 splits at the bound the cap's edge falls on, <!--n branch_cap.tied.at .1f-->2.0, so both implementations break the tie by word index. A tie is the bound rounded to `BRANCH_TIE_TOL`, which is 1e-11 against the roughly 1e-13 two reduction orders can open between the same bound computed twice. Bounds a last bit apart land in one bucket and the index decides between them, so the two agree without the weights beneath them agreeing bitwise. It is rounded and not floored, because those ties sit on whole values like the <!--n branch_cap.tied.at .1f-->2.0 above and a bucket edge through one of them splits the cluster the rounding is there to gather. `analysis/branch_tie.py` sweeps <!--n branch_tie.decades-->8 decades of tolerance and finds the same guess total throughout on <!--n branch_tie.one_total-->5 of the <!--n branch_tie.priors-->5 priors.

Deeper than two turns the search is gated on `--exact-max` (default 60), with a node budget as a backstop that falls back to the heuristic. A typical search visits 1–20 nodes and costs 1–30 ms. The value `solve` returns is therefore an upper bound on the true one, and the search is optimal within the cap and not provably optimal. Correctness is checked against an independent brute-force implementation of the recursion over random candidate sets.

The runners-up come from the same search. `ExactEndgame.top` puts the k-th best value where the incumbent goes, so a guess is still abandoned the moment it cannot reach the list, and the words behind the pick carry true values, never estimates. That is not free, since a looser cutoff kills fewer guesses on their bound. Over <!--n split_ranking.cost.positions-->200 real deep positions a top five costs <!--n split_ranking.cost.top5.mean_ms .0f-->25 ms a position against the pick's <!--n split_ranking.cost.pick.mean_ms .0f-->4 ms, and <!--n split_ranking.cost.top5.worst_ms ,.0f-->1,932 ms against <!--n split_ranking.cost.pick.worst_ms .0f-->122 ms on the worst one (`analysis/split_ranking.py`).

It also wants a wider cap than the pick does, which is `--exact-root-branch` (default 150). The winner sits far from the cap and the runners-up sit against it, so a cap that never costs the pick anything still costs the list something. <!--u-->Measured against the uncapped search over two independent samples of 80 positions each, 50 leaves three lists of the first sample and one of the second naming a worse guess, by 0.0055 and 0.012 guesses<!--/u-->, while **every winner is right at every cap tried**. Eighty is the smallest that comes back exactly uncapped on both samples, and 150 leaves the same margin over it that 50 leaves over 25. It applies only at the node being ranked. Everything the search recurses into needs its winner alone and keeps the cap of 50. <!--u-->The page's widest endgame position takes 197 ms instead of 94 ms for it, then 14 ms on every repeat once the memo holds it.<!--/u-->

Two of the figures above are the wider cap's. Over the same 200 positions it takes a top five from about <!--u-->12 ms to <!--n split_ranking.cost.top5.mean_ms .0f-->25 ms a position and from about <!--u-->520 ms to <!--n split_ranking.cost.top5.worst_ms ,.0f-->1,932 ms on the worst, while the pick does not move. That extra time goes on a tail nobody measuring the solver is measuring, so the wider cap comes back off with `WORDLE_ROOT_BRANCH=50` in the environment, or `--exact-root-branch 50` on a single run. Both implementations read the variable, so they stay comparable under it. The site is the one build with no way to set it, and the one that wants the wider list anyway.

### What it is worth

From the first state the search takes over, the whole continuation is optimal, so it cannot be worse than the estimator in expectation. It is also barely better. Every word played as the answer under the default prior (`analysis/loss_penalty.py --search`):

<!--table loss_penalty.search.table-->
| policy | mean | loss probability | solve rate | wall clock |
| --- | --- | --- | --- | --- |
| greedy entropy | 3.7722 | 0.0103% | 99.9897% | 4 s |
| no exact search | 3.1775 | 0.0756% | 99.9244% | 4 s |
| exact for t ≤ 2 only | 3.1779 | 0.0398% | 99.9602% | 4 s |
| exact for m ≤ 10 | 3.1778 | 0.0376% | 99.9624% | 5 s |
| exact for m ≤ 24 | 3.1777 | 0.0346% | 99.9654% | 7 s |
| **exact for m ≤ 60 (default)** | **3.1769** | **0.0285%** | **99.9715%** | **10 s** |
<!--/table-->

The mean moves by about a thousandth of a guess across that column. What the search moves is the loss probability, and most of that movement is in the part that costs nothing. `t ≤ 2` alone takes <!--n loss_penalty.search.loss_none .4f-->0.0756% to <!--n loss_penalty.search.loss_two .4f-->0.0398%, <!--n loss_penalty.search.two_share .0%-->76% of the whole reduction, at no wall clock. Everything deeper takes the rest at two to three times the wall clock. That share is coarse, and it moves with the fit and not with the record. A loss probability this small is a handful of words, so a single one can be a large share of the reduction. Which words they are is decided far from the endgame, by where the heuristic's earlier guesses route a word, and the search wins them back for the reason it exists.

The heaviest word it wins back under the current fit is `<!--s loss_penalty.search.rescued.word-->hater<!--/s-->`, at <!--n loss_penalty.search.rescued.weight .4f-->0.0008% alone <!--n loss_penalty.search.rescued.share .0%-->2% of the reduction. The two policies part on it at turn <!--n loss_penalty.search.rescued.turn-->2 with <!--n loss_penalty.search.rescued.at-->37 candidates, where the heuristic plays `<!--s loss_penalty.search.rescued.heuristic.guess-->tamer<!--/s-->` and the search `<!--s loss_penalty.search.rescued.search.guess-->reked<!--/s-->`. Those leave the answer among <!--n loss_penalty.search.rescued.heuristic.left-->9 and <!--n loss_penalty.search.rescued.search.left-->14 words, and an estimator that sees a count scores the first as the easier position. But the first is <!--s loss_penalty.search.rescued.heuristic.words-->`cater`, `hater`, `eater`, `water`, `dater`, `gater`, `oater`, `pater`, `rater`<!--/s-->, worth <!--n loss_penalty.search.rescued.heuristic.value .2f-->1.67 with <!--n loss_penalty.search.rescued.turns_left-->4 turns left, and the second is <!--s loss_penalty.search.rescued.search.words-->`cater`, `tamer`, `hater`, `taper`, `water`, `gater`, `mater`, `oater`, `pater`, `taber`, `tater`, `taver`, `tawer`, `taxer`<!--/s-->, worth <!--n loss_penalty.search.rescued.search.value .2f-->1.31. A size-only estimator cannot tell a bucket no guess splits from a larger one that a guess does. The exact search can, and <!--n loss_penalty.search.rescued.at-->37 candidates is inside its reach. Another fit routes other words and reads the share a third higher or lower, so it is quoted as most and not as a percentage.

Per decision it is sharper, and this is the one place a valuation can be checked against the truth. Over <!--n exact_decisions.blended.positions-->600 positions the search takes over, weighted by how often play reaches them (`analysis/exact_decisions.py`), the estimator picks a different guess **<!--n exact_decisions.blended.differ .1f-->32.7%** of the time and gives up **<!--n exact_decisions.blended.regret .4f-->0.0044 guesses** by it. What that costs depends on how much is left:

<!--table exact_decisions.blended.table-->
| candidates | positions | estimator differs | estimator regret | of which ties |
| --- | --- | --- | --- | --- |
| 3–4 | 131 | 36.8% | 0.0000 | all of them |
| 5–10 | 200 | 36.6% | 0.0017 | 33.9% |
| 11–60 | 269 | 29.1% | 0.0074 | 7.7% |
<!--/table-->

At three and four candidates every disagreement is a tie, a different word of equal value, so there is nothing to improve on. The ties thin out as the set grows, and by 11–60 most disagreements are real, which is where the <!--n exact_decisions.blended.regret .4f-->0.0044 comes from. The blend over four priors makes it look worse than it is under the one that ships: under the default prior alone the estimator differs on <!--n exact_decisions.shipped.differ .1f-->6.2% of positions and gives up <!--n exact_decisions.shipped.regret .4f-->0.0011.

The search's own regret reads <!--n exact_decisions.blended.search_regret .4f-->0.0002 instead of zero, because of the loss penalty and not any limit of the search. It optimizes a penalized score while this column measures mean guesses. At `--loss-penalty 1` it is <!--u-->0.000002, from a single position, which also shows the branch cap costs next to nothing at these sizes.

Refitting `f` under the search <!--u-->moves the curve by under 0.03 guesses everywhere (fit RMSE 0.63)<!--/u--> and buys nothing in play. Scored over every answer, the search-fitted `f` moves the mean by **<!--n fit_rounds.search_fit.mean +.4f-->+0.0000 guesses** on average across the <!--n fit_rounds.search_fit.priors-->5 priors and <!--n fit_rounds.search_fit.worst +.4f-->+0.0009 at worst, with <!--n fit_rounds.search_fit.identical-->0 of the five coming out identical, while the fitting loop takes about **<!--n fit_rounds.search_fit.loop_ratio .0f-->4×** as long. Which prior comes out worst is fit noise. `train` and `compare` therefore both keep only the free `t ≤ 2` part. `--exact-max 24` or `60` fits under the full policy on either.

## Losing less

### Greedy entropy loses less, not more

Every word played as the answer, under both priors:

<!--table loss_penalty.policies.table-->
| policy | mean | loss probability | solve rate |
| --- | --- | --- | --- |
| greedy entropy | 3.7722 | 0.0103% | 99.9897% |
| expected, no exact search, penalty 1 | 3.1775 | 0.0756% | 99.9244% |
| expected, exact-max 24, penalty 5 | 3.1777 | 0.0346% | 99.9654% |
| **expected, exact-max 60, penalty 5 (default)** | **3.1769** | **0.0285%** | **99.9715%** |
<!--/table-->

Entropy is not lossless, since <!--n loss_penalty.policies.entropy_lost-->74 words beat it, but it loses <!--n loss_penalty.policies.loss_ratio .0f-->3 times less often than the estimator, and the two do not fail on the same words, sharing only <!--n loss_penalty.policies.shared_lost-->20. What protects entropy is that it cannot see the prior. Separating candidates maximizes information, and separating never gambles. The estimator backs the prior instead, and loses when the prior is confidently wrong. *The penalty* below takes that apart.

The six tenths of a guess the estimator wins therefore come with about <!--n loss_penalty.policies.loss_ratio .0f-->3 times the loss rate, and `--loss-penalty` is the dial that trades some of the gain back for fewer losses. Neither rate is large enough for the default 200-game `benchmark` to notice. It finds zero losses <!--n loss_penalty.policies.benchmark_clean.entropy .1f-->98.0% of the time under entropy and <!--n loss_penalty.policies.benchmark_clean.default .1f-->94.5% under the default, so both look lossless there.

Wordle is provably solvable in five guesses, but that theorem is about a solver that knows the answer is one of the curated 2,309. This one hedges over all 12,947, and <!--n regime.share.offlist .0f-->27% of the prior mass sits on the 10,638 never-curated ones, which is where the losses collect. Of the <!--n loss_penalty.policies.default_lost-->185 words the default prior can lose on, <!--n loss_penalty.policies.default_lost_uncurated-->181 are never-curated, words like `aahed`, `anele`, `avyze`, `bizzy`. Playing all 2,309 curated words as the answer instead:

<!--table loss_penalty.curated.table-->
| policy | mean | worst | losses |
| --- | --- | --- | --- |
| greedy entropy | 3.9311 | 6 | 0 / 2309 |
| expected (defaults) | 3.8233 | 7 | 4 / 2309 |
<!--/table-->

The estimator wins the mean here too, and takes <!--n loss_penalty.curated.losses-->4 losses doing it. Every one of them, <!--s loss_penalty.curated.lost_words-->`brook`, `catty`, `foggy`, `musty`<!--/s-->, is a word that has already been an answer (<!--n loss_penalty.curated.lost_used-->4 of the <!--n loss_penalty.curated.losses-->4), which is the repeat prior working. The <!--n regime.curated_unused-->453 still-unused curated words hold <!--n regime.share.fresh .0f-->63% of the mass while the <!--n regime.curated_used-->1,856 already used share <!--n regime.share.used .0f-->10% of it, so sweeping the curated list uniformly asks the solver to do well on a distribution its own prior says is unlikely. Greedy entropy avoids that by ignoring the prior, the same blindness that costs it six tenths of a guess everywhere else.

The six-guess games in that table are wins. They are what hedging over 12,947 words costs, and closing them would take a full-depth optimizer over the restricted set.

### The penalty

`--loss-penalty` defaults to **5**, well above the risk-neutral value measured below. To see why, start with what a loss is. One might expect an endgame blunder, but at the sizes these positions reach `exact.ExactEndgame` solves them outright, so the guess played is the optimal one. What is happening is that the solver is backing its prior. Holding `batty`/`catty`/`natty`/`patty`/`ratty`/`tatty` with four turns left it plays `tatty`, which carries <!--u-->75% of that family's mass, and it declines `bronc`, a word splitting the six of them five ways, because separating spends a turn to insure against a <!--u-->0.4% case. A loss is the prior being confidently wrong, and the penalty makes the insurance worth buying.

Every word played as the answer, under each of the four prior combinations, blended equally:

<!--table loss_penalty.penalty.table-->
| `--loss-penalty` | loss probability | mean (solved) | mean (loss = 7) |
| --- | --- | --- | --- |
| 1 | 0.1049% | 3.6921 | 3.6953 |
| 2 | 0.0660% | 3.6933 | 3.6954 |
| **5 (default)** | **0.0431%** | **3.6944** | **3.6958** |
| 10 | 0.0335% | 3.6952 | 3.6963 |
| 30 | 0.0314% | 3.6956 | 3.6966 |
<!--/table-->

<!--n loss_penalty.penalty.first_to_default.loss_drop .0f-->59% of the loss probability disappears between 1 and 5, for **<!--n loss_penalty.penalty.first_to_default.mean +.4f-->+0.0004 guesses** on the penalized mean. Opening words are untouched, since the root never reaches the exact search.

That last column cannot settle which penalty is right. Scoring a loss at 7 is the objective `--loss-penalty 1` optimizes, so ranking settings by it always favors risk neutrality. The penalty is itself a price. When the turns run out the search scores the position at the penalty, on top of the six guesses spent, so a penalty of `p` plays as if a lost game cost `6 + p` guesses.

What a loss really costs can be measured. Scoring a loss at 7 assumes the lost game needed exactly one more guess. Lift the deadline, carry each losing game on from the position it died in, and the solver needs **<!--n loss_cost.penalty5.extra .1f-->1.3 more**, occasionally <!--n loss_cost.penalty5.worst-->3 (`analysis/loss_cost.py`). A loss costs about **<!--n loss_cost.penalty5.price .1f-->7.3** guesses, so the risk-neutral penalty is about <!--n loss_cost.penalty5.extra .1f-->1.3. Choosing 5 prices a loss at <!--c-->11, nearly four guesses more than it costs. That is a different kind of claim: that a failed puzzle costs more than the guesses it would have taken, because the game is scored out of six and not counted forever. It is a judgment and not a measurement, and no continuation can settle it.

### Widening the search's reach

The second lever is `--exact-max`, how large a candidate set the exact search will take on. At the default penalty, again over every word and every prior:

<!--table loss_penalty.reach.table-->
| `--exact-max` | loss probability | mean (loss = 7) |
| --- | --- | --- |
| 24 | 0.0579% | 3.6984 |
| 40 | 0.0477% | 3.6967 |
| **60 (default)** | **0.0431%** | **3.6958** |
| 80 | 0.0390% | 3.6950 |
| 100 | 0.0390% | 3.6942 |
<!--/table-->

This lever costs nothing in guesses. Reaching further replaces an estimate with a true value, so the mean improves all the way down the column. What it gains falls off sharply. Going 24 → 60 takes <!--n loss_penalty.reach.first_to_default.loss_drop .0f-->26% of the loss mass. 60 → <!--c-->100 takes a further <!--n loss_penalty.reach.default_to_last.loss_share .0%-->7% of it and <!--n loss_penalty.reach.default_to_last.mean .4f-->0.0016 guesses.

**Under the default prior it gains nothing.** Held at 60, 80 and <!--c-->100 the loss probability is <!--n loss_penalty.reach.shipped.60.loss .4f-->0.0285%, <!--n loss_penalty.reach.shipped.80.loss .4f-->0.0283% and <!--n loss_penalty.reach.shipped.100.loss .4f-->0.0282%, and the mean goes from <!--n loss_penalty.reach.shipped.60.mean .4f-->3.1769 to <!--n loss_penalty.reach.shipped.100.mean .4f-->3.1767. Reaching to <!--c-->100 wins back <!--n loss_penalty.reach.shipped.rescued-->2 of the 12,947 words and leaves <!--n loss_penalty.reach.shipped.100.curated-->4 of the <!--n loss_penalty.curated.losses-->4 curated losses standing. Every gain in the table above comes from the flat and curation-only rows. The `catty`/`ratty` family that beats the solver is not a position a deeper search can crack, because nothing is wrong with how they are played. See *The penalty*.

Time stops it at 60, and only the flat prior feels it. Reaching to <!--c-->100 runs <!--n loss_penalty.cost.flat.100.ratio .1f-->3.4× slower there, and the node budget starts firing, with <!--n loss_penalty.cost.flat.100.bailouts-->5 bailouts in four hundred games, the first of them at <!--n loss_penalty.cost.flat.first_bailout-->80. Each bailout is a position that falls back to the heuristic instead of being solved, so past 60 the exact search is less exact. Under the default prior the curve stays gentle, with <!--c-->100 costing <!--n loss_penalty.cost.shipped.100.ratio .2f-->1.29× what 60 does against the flat prior's <!--n loss_penalty.cost.flat.100.ratio .2f-->3.40×, because concentrated mass means the candidate set rarely gets that big. The penalty drives none of this, since 1, 5 and 30 run within <!--n loss_penalty.cost.flat.penalty_spread .0f-->4% of each other at either reach. The bound is capped at `min(penalty, 1)` while true values scale with the penalty, and that asymmetry does not show up as cost.

Between the penalty and the wider reach, the loss probability falls from <!--n loss_penalty.corner.loss .4f-->0.1096% to <!--n loss_penalty.penalty.5.loss .4f-->0.0431%, **a <!--n loss_penalty.corner.loss_drop .0f-->61% reduction**, and the mean improves by <!--n loss_penalty.corner.mean_gain .4f-->0.0022, since the reach gains more in the mean than the penalty gives up.

## Lookahead (`--lookahead`, off by default)

Above the exact search sits an optional expectimax with a branch cap of its own. Only the guess is pruned, never the feedback, since dropping a bucket would leave the sum something other than an expectation:

```
V_d(S, t) = min over the best `--lookahead-branch` guesses of
            [ 1 + sum_{r != GGGGG} p_r · V_{d-1}(S_r, t-1) ]
V_0(S, t) = min_g E(g, S)
```

with the exact search taking any sub-position it can reach. This is the exact search's `--exact-branch` device one level up. It is a fan-out at each node and not a frontier held to a constant size, so every kept guess expands into every one of its buckets and each further ply multiplies the work by the cap again. That growth is where the <!--u-->205 s below comes from.

Over <!--n lookahead.games-->20,000 paired games (<!--n lookahead.per_prior-->5,000 per prior, identical answers, `analysis/lookahead.py`), one ply at a cap of 10 is worth **<!--n lookahead.delta-->−0.0072 ± 0.0037** guesses for roughly <!--u-->10× the wall clock. At <!--n lookahead.sigma .1f-->2.0σ it is barely resolved, and it is seven thousandths of a guess. The arms are <!--n lookahead.repeat0_curation0.delta-->−0.0096 ± 0.0075 with no priors, <!--n lookahead.repeat0_curation1_c-wordlebot.delta-->−0.0122 ± 0.0091 and <!--n lookahead.repeat1_curation0_r-binned.delta-->−0.0124 ± 0.0080 with one each, and <!--n lookahead.repeat1_curation1_r-binned_c-wordlebot.delta-->+0.0056 ± 0.0033 under the default prior, where it costs guesses, at under two standard errors, on the one configuration that ships. Pooling the four arms brings the figure to the edge of resolution, and the quantity it resolves is too small to want. <!--u-->A cap of 25 measured no better, the opening word stays `salet` at every cap from 5 to 40, and two plies costs 205 s for the opening position alone.<!--/u-->

Its value is not a fixed property of the solver. Lookahead measures the room the base policy leaves above itself, so a better-fitted `f` should shrink it. That mechanism is not visible at this resolution. The last refit left lookahead worth less, the direction that mechanism predicts, by about one standard error of the difference, which is too little to show it. Read the figure against a run of the same vintage, never against a published one.

It barely matters despite valuing positions far better. The policy is deterministic, so the true expected guesses from a position can be computed by playing it out once per candidate answer, weighted by the prior. Each valuation is scored against what its own play achieves, over <!--n lookahead_accuracy.decisions.positions-->207 positions from real games, weighted by how often play reaches them (`analysis/lookahead_accuracy.py`):

<!--table lookahead_accuracy.blended.table-->
| valuation | true value | bias | mean abs error | RMSE |
| --- | --- | --- | --- | --- |
| `min_g E(g, S)` (what the solver scores positions at) | 2.7987 | +0.0205 | 0.0662 | 0.0866 |
| one-ply lookahead | 2.7881 | −0.0007 | 0.0025 | **0.0104** |
<!--/table-->

<!--n lookahead_accuracy.blended.ratio .0f-->8 times more accurate, and near enough unbiased. Those are the positions where the heuristic decides, which means turns two and three. By the fourth the candidate set is under `--exact-max` and both valuations return the same true value.

Accuracy in the level is not what drives play. A valuation wrong by a constant at a position still orders the guesses perfectly and costs nothing, so what matters is the ranking within a position. Playing out the true value of the ten guesses lookahead re-ranks, at those same positions:

<!--table lookahead_accuracy.decisions.table-->
|  | blended | none | curation | repeat | default |
| --- | --- | --- | --- | --- | --- |
| spread of true value, best to worst | 0.0917 | 0.0921 | 0.0853 | 0.0883 | 0.1010 |
| `min_g E(g, S)` regret | 0.0112 | 0.0205 | 0.0075 | 0.0146 | 0.0021 |
| one-ply lookahead regret | 0.0006 | 0.0018 | 0.0000 | 0.0005 | 0.0000 |
| the estimator picks the best | 54.4% | 37.8% | 54.8% | 42.5% | 82.5% |
| lookahead picks a different guess | 44.9% | 61.6% | 45.2% | 55.2% | 17.5% |
| rank correlation with the truth | +0.526 | +0.421 | +0.534 | +0.443 | +0.705 |
<!--/table-->

The estimator orders those ten barely better than a coin flip when the prior is flat, and much better than that under the default one, where its pick is the best available five games in six. Lookahead disagrees at nearly half of them, changes the length of <!--n lookahead.changed_share .0f-->22% of the games, nearly evenly in each direction (<!--n lookahead.better-->2,229 better against <!--n lookahead.worse-->2,164 worse), and by the decision regret it is right to disagree, giving up <!--n lookahead_accuracy.decisions.look_regret .4f-->0.0006 against <!--n lookahead_accuracy.decisions.base_regret .4f-->0.0112. Better valuation does produce better decisions at these positions, under every prior, though under the default prior the whole-game figure goes the other way.

What it barely changes is the guess count, and the first row is why. Those ten spread <!--n lookahead_accuracy.decisions.spread .4f-->0.0917 between best and worst, and the estimator's pick gives up a tenth of that. The top of the ranking is a plateau. The estimator finds it reliably and cannot tell one spot on it from another, which costs a hundredth of a guess because there is little to tell apart. Recovering most of it per decision still reads as <!--n lookahead.delta-->−0.0072 ± 0.0037 over a whole game, since a game only meets such a position where the exact search has not already taken over.

**This is the recurring result in this project.** A valuation improvement that is real, large, and correctly converted into decisions still buys almost nothing. It holds for the deep exact search, for one-ply lookahead, for turn-conditioned `f`, and for the sharpened repeat prior above. Only lookahead's has been taken apart at the level of the decision, and the other three were measured at the outcome.

### Conditioning f on turns remaining

`f(n)` estimates guesses-to-finish with no turn limit, so a position with four turns left is scored as one with two. Fitting `f` once per turn count produces curves with a large turn effect, <!--u-->a 2,000-candidate bucket being worth 4.97 with three turns left against 3.04 with five<!--/u-->, and changes nothing.

There was no room for it to. Where those curves differ is where the deadline binds, and the exact search has taken that decision long before. Counting every decision in <!--n exact_decisions.who.games ,d-->1,600 games (`analysis/exact_decisions.py --who-decides`):

<!--table exact_decisions.who.table-->
| turns left | `f` | exact search | closed form | one candidate |
| --- | --- | --- | --- | --- |
| 6 | 1,600 | 0 | 0 | 0 |
| 5 | 1,329 | 260 | 1 | 10 |
| 4 | 56 | 1,160 | 132 | 195 |
| 3 | 0 | 292 | 146 | 469 |
| 2 | 0 | 30 | 22 | 159 |
| 1 | 0 | 0 | 0 | 31 |
<!--/table-->

`f` takes **<!--n exact_decisions.who.f_late-->0** decisions in the <!--n exact_decisions.who.late-->1,149 positions with three or fewer turns left. The <!--n exact_decisions.who.f_at_four-->56 at four turns left are the last it sees. Every row where a turn-conditioned curve would say something different is a row nobody reads. <!--u-->Over 80,000 paired games the turn-conditioned fit measured losses 81 → 79 and a mean of +0.0005 ± 0.0007.<!--/u--> That implementation is not kept, and the table above is a fact about the solver as it stands.

### Conditioning f on the pattern

`f` reads one scalar, so ten candidates in the `-atch` family and ten well-separated ones are the same position to it. The feedback pattern is the one structural feature already in hand where `f` is evaluated, since `kernels._expected_guesses` iterates over the `r` of `E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)`, so `f(n) + offset[class(r)]` would cost one read from a 243-entry table and no rescan.

The bias it would correct is real and large. Fitting the offsets alongside `f` on 2,000 games of rollouts, with the green count entered as dummies and depth controlled for (`analysis/pattern_offsets.py`), a bucket reached by three greens needs **<!--n pattern_offsets.repeat1_curation0_r-binned.greens3-->+0.47 ± 0.06** guesses more than its effective size predicts under the repeat-only prior and <!--n pattern_offsets.repeat0_curation0.greens3-->+0.34 ± 0.06 under no prior at all, rising through three greens. The two priors carrying both shapes show <!--n pattern_offsets.repeat1_curation1_r-binned_c-frequency.greens3-->+0.16 ± 0.06 and <!--n pattern_offsets.repeat1_curation1_r-binned_c-wordlebot.greens3-->+0.08 ± 0.05, and curation alone <!--n pattern_offsets.repeat0_curation1_c-wordlebot.greens3-->+0.15 ± 0.05.

Play gets worse anyway. Hold `f` at the shipped fit and scale the correction from nothing up to the fitted value, and the mean over every answer climbs at every step on four of the five priors, <!--u-->while the fifth moves 0.0006 the other way<!--/u-->. Refitting `f` jointly with the offsets rescues none of it.

Gating the offset by bucket size shows that the problem is extrapolation and not a wrong correction. Applied only to buckets of 30 candidates or fewer, which is where the residuals were measured (a four-green bucket averages four candidates), <!--u-->it costs 0.0020 guesses averaged over the five priors and helps one of them. Applied to buckets of hundreds, which the residuals never sampled and which the widest scans are made of, it costs up to 0.18. A constant of +0.45 guesses<!--/u--> is small against a four-candidate bucket and enormous against a five-hundred-candidate bucket, and the ranking at the opening is decided by the latter.

What the correction does give, slightly and on all five priors, is a lower loss rate at the tightest gate, and `--loss-penalty` already makes that trade without 243 floats.

## The opening is not where the estimator is weak

Every game shares one position, and `f` decides it alone, since the candidate set is the whole list and no search reaches it. It is also the widest position there is, which is where `f` ranks worst. Playing out the two hundred openings it likes best, once per answer and weighted by the prior (`analysis/opening.py`):

<!--table opening.table-->
| ranks by `f` | mean true value | worst | best in band |
| --- | --- | --- | --- |
| 1–10 | 3.2082 | 3.2247 | **salet 3.1800** |
| 11–25 | 3.2233 | 3.2457 | parse 3.2042 |
| 26–50 | 3.2316 | 3.2494 | tales 3.2162 |
| 51–100 | 3.2369 | 3.2727 | trine 3.2094 |
| 101–200 | 3.2473 | 3.2745 | stile 3.2262 |
<!--/table-->

`<!--s opening.pick-->salet<!--/s-->` is the best of the <!--n opening.openings-->200 and **nothing beats it**. The opening costs <!--n opening.regret .4f-->0.0000 guesses of regret. The mean and the worst climb with rank, and the best in a band climbs with one dip, at 51–100, so a rival from further down would have to beat every word in ranks 11–200 while `f` ranked it below all of them.

Against that, `f` orders the same words badly. Across the top twelve its rank correlation with the truth is **<!--n opening.rho_top12 +.3f-->+0.280**. Across the top twenty-five it is <!--n opening.rho_top25 +.3f-->+0.481, and across the top two hundred **<!--n opening.rho +.3f-->+0.661**. It cannot separate near-ties and it separates a range well. Coarse discrimination and fine ordering are different jobs, and picking an opening needs only the first.

## Perfecting the second guess

With the opening optimal and the exact search owning every turn from the fourth, whatever the estimator still costs has to be in the second guess. That decision is small enough to evaluate outright. The opening never varies, so the entire turn-2 policy is one guess per feedback pattern: <!--n turn2_regret.positions-->193 reachable positions, of which the endgame search already solves <!--n turn2_regret.solved-->142 outright, leaving <!--n turn2_regret.decided-->51 where `f` decides alone. `analysis/turn2_regret.py` plays out the true value of the fifty best guesses at each of those (`common.guess_value`, one playout per candidate answer) and weights each position by its prior mass, which is the probability play reaches it. The estimator gives up **<!--n turn2_regret.census .5f-->0.00411 guesses**, at <!--n turn2_regret.moved-->17 positions of the <!--n turn2_regret.decided-->51.

Substituting those decisions and sweeping every answer recovers about that, <!--n turn2_regret.shipped_mean .5f-->3.17687 → <!--n turn2_regret.override_mean .5f-->3.17265 against a census that predicted <!--n turn2_regret.census .5f-->0.00411. Losses move the other way, <!--n turn2_regret.shipped_loss .4f-->0.0285% → <!--n turn2_regret.override_loss .4f-->0.0327%, since a playout carries no deadline, so a sixth guess costs it no more than a second.

Under a distribution other than the one that chose them, the sign does not always hold:

<!--table turn2_regret.table-->
| distribution | shipped | turn-2 optimum | delta |
| --- | --- | --- | --- |
| the default prior (what chose them) | 3.1769 | 3.1727 | **−0.0042** |
| uniform over all 12,947 | 4.4313 | 4.4467 | **+0.0154** |
| uniform over the curated list | 3.8233 | 3.8112 | **−0.0121** |
| uniform over the still unused | 2.9713 | 2.9581 | **−0.0132** |
| the 266 real regime answers | 4.1429 | 4.1203 | −0.0226 ± 0.0266 |
| all 1,955 real answers ever | 4.0307 | 4.0189 | −0.0118 ± 0.0111 |
<!--/table-->

The two real-answer rows are only comparable as differences. Both arms replay those answers under today's prior, which knows that every one of them has already been used and gives it repeat weight. That is why the levels sit well above the <!--n replay.priors.both.mean .4f-->3.1617 that *What each prior is worth on real answers* carries, where `analysis/replay.py` rebuilds the prior from strictly earlier puzzles each day. The levels are that handicap and not a regression, and the paired difference is unaffected, since both arms see the identical prior.

Those two rows do not resolve it, at <!--n turn2_regret.regime.delta-->−0.0226 ± 0.0266 over the regime and <!--n turn2_regret.alltime.delta-->−0.0118 ± 0.0111 over every answer ever. The answers the NYT published lean toward the substitution in both, each by about one standard error or less. The enumerated rows are exact and they disagree among themselves. Optimizing the second guess against one weighting of the word list makes it worse under another.

The gain is the residue of a reshuffle. **<!--n turn2_regret.churn-->2,198 of 12,947** answers change length, carrying <!--n turn2_regret.churn_mass .1f-->7.9% of the prior's mass, and <!--n turn2_regret.net +.4f-->−0.0042 is what survives of <!--n turn2_regret.improves .2f-->4.12% of the mass improving against <!--n turn2_regret.worsens .2f-->3.74% worsening. Set beside the <!--n turn2_regret.sd .2f-->1.30-guess standard deviation of the per-answer changes it rides on, the net is <!--n turn2_regret.net_fraction .4f-->0.0033 of it. The gain reverses under the flat weighting of the whole list, and holds under the flat weighting of the curated list and under the still-unused weighting.

The per-override breakdown separates the two kinds. Some overrides help or cost nothing under every enumerated weighting. `<!--s turn2_regret.flip.word-->sieve<!--/s-->` reads <!--n turn2_regret.flip.default +.4f-->−0.0001 under the default prior and **<!--n turn2_regret.flip.curated +.4f-->+0.0100** under the curated one, a sign flip and many times the magnitude. <!--n turn2_regret.everywhere-->6 of the <!--n turn2_regret.moved-->17 help under all four, and keeping only those is worth <!--n turn2_regret.everywhere_gain +.5f-->−0.00161. That rule is itself a fit to the distributions it was checked against, so what it shows is how little of the <!--n turn2_regret.paid .4f-->0.0042 was attributable to anything but the prior's particular weighting.

The recurring result stated under *Lookahead* above arrives here in its strongest form. The cases collected there each show a better valuation failing to buy guesses. This one gives the ceiling directly. **Playing the last position the estimator decides perfectly is worth less than the error in the prior it would be perfect for**, which bounds what a better valuation could gain there. The bound covers the fifty guesses `f` ranks highest at each turn-2 position, changed one position at a time, and inside it no search depth, no extra ply and no better-fitted `f` reaches past <!--n turn2_regret.paid .4f-->0.0042. What remains is a better model of the editor, and *The prior sits in a flat basin* measures that lever as pulled too.

## Playing it yourself

The page's *Play* view is the easier way to play. This is the same game in a terminal.

```
python wordle_game.py                  menu: type an answer, or draw one from a prior
python wordle_game.py --answer crane   play against a known answer
python wordle_game.py --random         draw the answer from the prior
```

During a game, `?` shows the solver's recommendation (Enter accepts it) and `quit` gives up. `--assist` shows it every turn, `--auto` hands the game over, and `--seed` makes a `--random` answer reproducible. The game only reads the word lists. The 167 MB feedback table is loaded lazily, and only for a hint.

The board is recapped at the end as colored blocks. `--share` adds the emoji grid underneath it for pasting elsewhere.

## In a browser

```
./build_site.sh                          build the page into site/
python -m http.server -d site 8000       serve it on http://127.0.0.1:8000
```

The page solves, plays and finds. The group beside *Settings* says which of the three the page is doing. It marks where you are, not where the button will take you.

Where a view has a terminal counterpart, the description below names it. Those are not directions to the terminal, which is where the solver is fitted, not used.

**Solve** takes a board and recommends the next guess. You enter the feedback by coloring tiles, and the terminal's prior toggles, `--mode`, `--form` and `--hard` all become switches. *Shape of f* leaves the choice to the cache by default and marks which form that is, which is `spline` for every prior. Alongside the board is a panel of alternatives and still-possible candidates, with the prior, the objective and the fitted f under *Advanced*. Each of those comes with an explanation of what it is doing, and the rates quoted in it are measured from `solutions.json` at the moment the board is built. Nothing is written into the page, so the explanation cannot fall behind the answer record.

*Set answer*, under the Solve board, is the solver tested against a word: `wordle_game.py --auto` with the whole panel showing. With an answer in the box the solver colors every row from it as you submit and the tiles stop being an input, so there is no pattern to enter wrong. Enter on an empty row plays the solver's pick, *Play it out* presses it until the game ends, and *Draw random answer* deals one from the prior the way *Play* does. What sets it apart from watching *Play* with *Assist* on is that the board stays editable. Delete a row, type another word there, and the rest of the game is re-solved and recolored from that point. The answer rides along with every board request like a switch does, so the switches keep re-running the solver over the same rows while the colors stay put.

**Play** is `wordle_game.py`. The *Answer* box takes a word of your own, which is the quickest way to set the solver a position, and it is the only setting that deals a new game. An answer already dealt cannot be given a different one. *Hint* is the terminal game's `?` and presses again to close, the *Assist* switch is `--assist` (Enter then plays its pick, which is `--auto`), *Reveal answer* is `quit`, and the emoji grid at the end is `--share`. The page does not receive the answer until the game is over.

The panel beside the board holds three cards, and all three fold away. They start closed.

*Solver's pick* holds the word the solver would play here and its score. *Alternatives* holds the next four behind it. Clicking one of those words puts it in the row you are typing. Both cards open when you turn *Assist* on, and both open when you press *Hint*. Neither is ever closed for you, so a card you opened with *Assist* off is still open on your next turn.

Those five are five different guesses, which late in a game is not the same as five different words. Once few candidates remain, many words give identical feedback against every one of them, and a ranking of words is then one guess spelled several ways. At four candidates or fewer the five best words are <!--n split_ranking.band2_4.distinct .1f-->4.8 different guesses, each standing for about <!--n split_ranking.band2_4.words .0f-->100 others (`analysis/split_ranking.py`). So the ranking names one word per distinct split, and a `+3` beside it opens the rest of that split as chips. Play any of them instead and the game goes exactly as it would have. A word that could itself be the answer never has any, since only it comes back all green. The pick carries the same list, under the score.

*Letter frequency* sits above those two and *Assist* does not touch it. You open and close it yourself, and it stays as you left it for the rest of the game. Each letter carries the chance that the answer contains it somewhere: the prior's weight on the remaining candidates that hold the letter, over the prior's weight on all of them. On an empty board E is at the top with about <!--n regime.letter_e .0f-->41%. Vowels are listed above the other letters.

The prior doing the weighting is the one the answer was drawn from, not the one the switches currently name, so turning a prior off mid-game rethinks the hint but leaves these numbers alone. A letter that no remaining candidate holds drops off the list, since 0% says nothing the gray keys have not said already. Letters at 100% stay and sort to the top, so late in a game the list opens with the ones the answer is known to contain.

**Find** is the page's own, and it runs backwards from everything else here. The answer is known and the guess is the unknown. Type an answer, color the pattern that a guess came back as, and you get every word in the list that would have produced that pattern against that answer. It is the question to ask of somebody else's share grid. Neither row describes the other, so there is no ghost word and nothing is grayed out, and a gray tile in the pattern is a color and not an empty tile waiting for a letter.

The match itself is exact and immediate, since the answer is one column of the feedback table and finding the words is a comparison over it instead of a scoring pass. The three numbers each word carries take more work:

* **Words left** is how many candidates would survive that guess given this pattern, so it ranks the words with hindsight. Smallest first is what would have told you the most about this particular answer.
* **Bits** is expected information gain, averaged over every answer the guess could have faced. It does not depend on the pattern at all, so it ranks the words as openings instead.
* **Chance** is the prior's own probability that the guess is itself the answer.

The last two move with the prior switches. Nothing else on the page reaches the finder, so the mode, the shape of f and hard mode are hidden while it is showing. The finder never asks for a guess, needs no fitted `f`, and works under any prior.

| key | solve | play | find |
| --- | --- | --- | --- |
| A–Z | write a letter into the selected tile | write a letter | write a letter of the answer |
| ← → | move along the row | move along the row | move along the row |
| ↑ ↓ | move between rows, never past the row being played | — | move between the answer and the pattern |
| Space | change the selected tile: gray → yellow → green | — | change the selected pattern tile |
| Enter | on an empty row take the suggestion, otherwise submit | guess, or play the solver's pick when it is showing | find every word that fits |

Going back up to an earlier row and correcting it is the point of ↑. Every submit sends the whole board, so a tile fixed four rows up re-runs the solver from there. You never have to delete the rows under it first.

### Editing the board

The page keeps its own copy of every row, and typing and coloring are local until Enter sends the whole board back. The solver only ever sees complete boards, so correcting an earlier row is the same operation as playing a new one. Rows the solver has not been told about yet are drawn with a dashed border.

Taking a word off is the exception, and goes back on its own. Writing one is a proposal, and Enter submits it. Removing one is not, since a board cannot half-forget a word it is no longer showing. However the word goes, by *Delete row* or backspaced away letter by letter, what remains has to be the board those words alone would have made, down to the shadow and every number in the panel.

Five rules fall out of that:

**A color belongs to the letter under it.** Clearing a letter clears its color, and typing a different letter into a tile resets that tile to gray. Otherwise a row typed over an old one comes back wearing the old one's feedback, a letter turning yellow because the word before it had a yellow there. Space on an empty tile does nothing for the same reason, since the color would not show and would then belong to whatever was typed there next.

**Typing at the end of a full word replaces the last letter**, as it does in any other box that is already full. A stray keypress costs one letter, not the word. Retyping a rejected guess from scratch therefore means clearing the row first.

**Backspace only deletes leftwards**, the tile under the cursor and then the one before it. Letters to the right of the cursor are left alone, so the first few letters of a row can go without disturbing the last ones. The backspace that empties the row takes the word off the board, and it counts as a deletion. A row cleared by hand is a row deleted, and the two cannot look different afterwards.

**Delete row empties the row in place and leaves the gap.** The rows under it stay where they were played, so the word that should have been there goes straight into the hole. Submit with the gap still empty and you get *Row 2 is empty* back. Nothing closes up and nothing slides.

A word only reaches the board because it was played, which makes a row that needs deleting nearly always a row that needs replacing. Closing the gap would claim you played those later words earlier than you did. The position is not free to move either. Both the endgame search and the estimator are fed `guesses_left = MAX_GUESSES - len(rows)`, so shifting rows up would change what the solver recommends.

An empty row with nothing under it is the row waiting to be played, so it is not sent. That makes *Delete row* on the last guess an undo, and it clears the board back to the opening recommendation. Only that gap-free case goes back on its own. A hole higher up waits for the word that fills it, because the board it makes is one the solver refuses anyway.

**The suggestion is a shadow, not a row.** The solver's pick is drawn faint in the row after the last real word, and it follows that word instead of standing at a fixed height. Take the last guess off and the board goes back, so what comes up into the vacated row is the pick for the shorter board. *Delete row* cannot touch the shadow itself, since there is nothing there to delete. Click it, or press Enter on an empty row, and it becomes a word in a row that deletes like any other.

There is exactly one free row, and the shadow is in it. Otherwise a row emptied by hand leaves a second free row under it, still selectable and sitting past the end of the board the shadow describes, and typing in that one makes the shadow vanish from a row you are not even in. The board is trimmed back to the last word and its free row.

The rules add up to one promise. Three words with the third taken off leaves exactly what two words played from the start would leave, and nothing on the page can tell which of the two happened.

One request is in flight at a time, since the solver answers one at a time anyway. A second one made meanwhile is held, not dropped. Dropped, it would leave the page showing something nobody asked for, a game view with no game dealt into it and every panel reading "—". The loading veil stays up until there is a board behind it for the same reason, retrying if the first request fails.

**A setting throws nothing away.** The three views share one set of switches, and changing any of them (a prior, the mode, the shape of f, hard mode) asks the solver again about the position you are already in. The solve board is re-run over the rows the solver has already accepted, so a word half typed into the open row is neither submitted nor lost. The game keeps its answer and re-asks the hint, and the finder searches again. Only the *Answer* box deals a new game.

The terminal menu works the other way round, with four combinations to choose between before the deal, because there the prior decides what the answer is drawn from. Here the answer was drawn under whatever was on at the time and it stays what it is. Turning a prior off mid-game rethinks the advice instead of dealing a new word, which is the only reading under which you can ask what the other model of the editor would have played from here.

Nothing is rebuilt behind your back either. A setting changed while a view is hidden marks that view for a re-run, and the re-run happens when you next look at it. The alternative is a view claiming a setting it was never built under.

### Rewriting what it says

`python editor.py` lists every string the site can say, each beside a box to rewrite it, with the page itself alongside. Save, and the page says it, with no restart.

What it shows alongside is `site/`, the build that gets deployed, running its solver in WASM. So the editor loads no word lists, no feedback table and no numba, and starts in the time it takes to read one directory.

Nothing is registered by hand. `wording.py` reads `index.html`, `app.js` and the two Rust files the solver answers from, and picks the text out by how it is used (between tags, assigned to `textContent`, passed to `say()` or `Err()`, and so on), so a string written the ordinary way is editable the moment it exists and the list cannot fall behind the site. Each is shown with the file, the line and where on the page it sits.

Rust needs a lookback the others do not. A message there is often several braces inside the call that says what it is for, as in `.ok_or_else(|| { if … })`, so cutting the context at a brace, as the JS scanner does, would throw away the only thing marking it as text. `wording._rs_statement` cuts at the semicolon instead and lets the enclosing function bound it. That difference takes the Rust from 20 of its messages to all of them, and it keeps `format!("g{}", …)` out, since a session id is built in a statement that says nothing about text.

A choice between two strings counts as one of those uses, and what marks it as text can be lines above the arm it is looking at, such as `why:` over a ternary laid out one arm per line. Reading only the arm's own line would miss the longest prose on the page, the two explanations of the priors.

Strings are found at a character offset, so the same sentence written twice is two entries and rewriting one leaves the other alone. Edits are applied to the served bytes of `index.html` and `app.js`, and saved in `wording.json` keyed by `<file>:<hash of the text>:<n>`, which survives the code being worked on where line numbers and offsets do not.

A value baked into a string is part of it. An edit may move `${count}` but not drop it, and saving is refused if one goes missing. Rust's bare `{}` counts as one of those too, since that is how `format!` takes an argument. The editor listens on this machine alone, and anyone who can reach it can rewrite the source it is built from.

**Save** leaves the source alone. The rewrites live in `wording.json` and are applied to the served bytes on the way out, and `build_site.sh` applies them too, so a rewrite that has only been saved still reaches the deployed page. **Write into the code** puts them in the files for good and clears `wording.json`. That is the end state to reach, because an override is matched to the code by wording and stays fragile as long as it sits outside it. Prose is re-wrapped to the indentation it lands in, and both phrases are appended to `wording_history.json`. That record is the only one kept. The bake edits the source in place with no undo, so read the source afterwards. A phrase written into the Rust needs `./build_site.sh` before the preview says it, since by then it is inside the compiled module and not in a file being served.

### Without a server

`./build_site.sh` writes `site/`, which is the three views with no Python behind them and nothing to deploy but files. It is <!--n site_size.total_mb .1f-->1.6 MB, <!--n site_size.wasm_kb .0f-->468 KB of that the WASM module and <!--n site_size.data_kb .0f-->782 KB data, and <!--n site_size.wire_kb .0f-->282 KB in all over the wire, since a host that serves it with brotli gets the module down to <!--n site_size.wasm_wire_kb .0f-->125 KB and `solutions.json` from <!--n site_size.solutions_kb .0f-->307 KB to <!--n site_size.solutions_wire_kb .0f-->16. Nothing else is asked of a host. `vercel.json` labels the module `application/wasm`, as streaming compilation requires, and where that label is wrong the worker compiles the bytes it already fetched instead, so the build deploys anywhere that serves files.

`site/` is generated and is not in the repository. `./deploy.sh` builds it and uploads it to Vercel in one step, which is the whole of the deployment. There is nothing to build on the far side, and the only state that has to survive a rebuild is `site/.vercel/`, the link to the Vercel project, which `build_site.sh` carries across. `./deploy.sh --preview` puts the same build on a throwaway URL to look at before promoting it.

The shim has to stand in for a server's failures too. `api()` throws on a reply that is not ok and renders a board on one that is. A refusal returned as <!--c-->200 would therefore be applied as a board, and the page would die reading rows off a payload that has none. The shim answers <!--c-->400 for any reply carrying an `error`, which is the whole of the distinction the page draws.

`web/app.js` ships unmodified. It reaches the solver through `fetch`, so `web/wasm-api.js` stands in for `fetch` on `/api/*` and routes into `wordle_core::api` instead of over a network. That is possible only because the page already sends the whole board on every submit, which makes that boundary a pure function of (path, body). Leaving `app.js` alone matters because `tests/board.test.js` loads it through `vm`, and a modified copy would mean the board tests exercise a different file from the one that ships.

The solver runs in a worker, `web/solver-worker.js`, and not on the page's thread. The widest position after the opening costs a little over <!--u-->200 ms in the browser, and spent on the main thread that is <!--u-->200 ms in which the page cannot repaint, scroll or take a keystroke. Moving it does not make it quicker, since the same scan takes the same time, but the page stays responsive throughout, which is most of what the delay cost. Measured against a 200 ms busy loop, the event loop turned over once during the block and some sixty thousand times during a solve of the same length. Making it faster would mean threads, which in a browser mean `SharedArrayBuffer`, cross-origin isolation headers and a nightly toolchain to rebuild `std` with atomics. Measured, they are worth about <!--u-->9x on this one position and <!--u-->2.4x over a whole sweep, and the gap between those two is how little of a game is scoring. That is not worth a browser build that no longer compiles on stable and that nothing in `tests/` can check.

**The table does not come with it.** 167 MB is not a download and not something a tab can rebuild, so the browser scores feedback live, with `source::Live` against `source::Table` chosen as a generic parameter so the inner loop dispatches statically. Live costs <!--n timing.live.median.ratio .1f-->2.5× a lookup on a median turn-two bucket and <!--n timing.live.worst.ratio .1f-->3.7× on the worst, measured natively on one position by `native/src/bin/bench_position.rs`: <!--n timing.live.median.live_ms .1f-->0.5 ms against the table's <!--n timing.live.median.table_ms .1f-->0.2 ms on a median turn-two bucket, and <!--n timing.live.worst.live_ms .1f-->15.2 ms against <!--n timing.live.worst.table_ms .1f-->4.1 ms on the worst. Cache lines predicted a wider gap than that, and a scattered read does cost a whole line, but the rows one position touches stay resident and most of its reads then hit. The absolute figure is what the page lives with: <!--u-->168 ms to rank every guess after one word, 743 ms for the widest thing the finder can be asked<!--/u-->.

Live scoring is felt in the endgame and almost nowhere else. The exact search visits hundreds of nodes for one position and would rescan all 12,947 guesses at every one of them, which for a set of 34 candidates means running the coloring rule over the same half-million pairs several hundred times. The Python solver reads the pairs out of the table it holds. `source::Columns` is the part of that table a tab can afford: every guess against the candidates the search began with, <!--u-->440 KB, filled in one scan and read from thereafter. It is exact, because every bucket the search recurses into is a subset of the set above it and nothing outside those columns is ever asked for. <!--u-->Without it, the two rescanning kernels are 84% of the position's cost and that 34-candidate turn takes 5.6 s. With it the same turn is 1.1 s.<!--/u-->

What survives between requests matters as much as what one of them costs. The page re-sends the whole board on every submit, and a tile recolored or a switch flipped is the same request as a word played. A `Solver` borrows its source and its weights and cannot outlive the request that built it. What it caches is a pure function of a position and can outlive one. So `solver::Memos` holds the three caches apart from the solver and `state` lends them back per request, keyed on the settings. Four settings' worth are held, so flipping a switch and flipping it back re-solves nothing. A repeated board costs 11 ms.

The exception is the opening, at 12,947 guesses against 12,947 candidates. That one position is a constant per setting, so `native/src/bin/build_site.rs` computes all <!--n site_size.openings-->180 of them ahead of time into `openings.json`: every prior against both modes, both hard-mode settings, and each of the nine forms of `f`. Entropy never consults `f`, so its share of those is one ranking written out under nine names.

Turn two is precomputed as well, for each of the four priors the two switches reach. It is affordable live but not comfortably so. An opening cuts the list into around <!--n site_size.turn2.per_prior .0f-->202 buckets, most of them small enough for the exact endgame to solve outright (<!--n turn2_regret.solved-->142 of `salet`'s <!--n turn2_regret.positions-->193), which is where the second above went. Each prior opens on its own word and was fitted in its own form, so each gets a set of its own: `<!--s results_tables.both.opening-->salet<!--/s-->` with both priors on, `<!--s results_tables.curation_only.opening-->salet<!--/s-->` with only curation, `<!--s results_tables.repeat_only.opening-->tares<!--/s-->` with only the repeat prior and `<!--s results_tables.none.opening-->tares<!--/s-->` with neither. That is <!--n site_size.turn2.rankings-->808 rankings, <!--n site_size.turn2.wire_kb .0f-->29 KB over the wire. The other two menus stay where the page leaves them, because covering every mode and every shape of `f` as well is <!--n site_size.openings-->180 settings, an hour of `build_site` and megabytes of `openings.json`. Off that path, with a first word the setting would not have played or the mode or the form changed, the search runs live. Nothing deeper is worth precomputing, since turn three costs a sixth of what turn two does even with an empty memo.

The editor rewrites the source it is scanning, which needs a checkout, so it is a local Python tool that previews this build. Play mode's answer sits in the module. The page never receives it until the game is over, but that is a statement about the page and not a guarantee.

`native/src/bin/api_probe.rs` runs the same endpoints from a shell. Debugging them through a browser is guesswork on a slow loop, which is why the API lives in `wordle_core` and not in the `wasm` crate, which does marshaling and nothing else. It reports what each request cost, and built with `--features wordle_core/profile` it reports where that time went per kernel. That is how the endgame's rescanning was found: two entry points holding <!--u-->84% of one position, which reading the recursion had not suggested.
