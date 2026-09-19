# Wordle solver

It runs in a browser at **[wordleoracle.vercel.app](https://wordleoracle.vercel.app/)** — the whole solver, in WASM, with no server behind it. That is a chosen name rather than Vercel's generated one, though still a subdomain of theirs, so a custom domain would move it again. The generated address the site used to answer on, `wordle-solver-lac.vercel.app`, is still aliased to the same build.

```
python wordle_solver.py train        # fit + cache the estimator, every prior combination
python wordle_solver.py benchmark    # self-play: greedy entropy vs. expected guesses
python wordle_solver.py compare      # policy iteration once per form of f, then tabulate
python wordle_solver.py info --forms # what is cached, and which forms exist
python wordle_solver.py play         # solve a puzzle here rather than on the page
```

Solving a puzzle is what the page is for; the terminal is where the solver is fitted, measured and inspected. `play` is kept as the page's oracle — `tests/front_ends_agree.py` diffs its parser against `web.py`'s, and with no second parser to diff, a switch meaning one thing in the browser and another on the command line would be invisible from both sides.

The prior toggles, `--mode` and `--form` go after the subcommand:

```
python wordle_solver.py train --this-config-only --no-repeat-prior --form saturating
python wordle_solver.py play --no-curation-prior --mode entropy
```

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
| `wordle_solver.py` | the terminal front end: fitting, benchmarking, inspection, and `play` as the page's oracle |
| `wordle_game.py` | play Wordle yourself, against a chosen or a sampled answer; `web.py` imports its emoji grid and its random deal |
| `web.py`, `web/` | both of the above in a browser, plus the word finder: stdlib HTTP server and a static page |
| `wording.py` | finds every string the site says, so `editor.py` can rewrite any of it |
| `editor.py` | the editor itself, previewing the static build it is editing |
| `unwrap_md.py` | checks that every `.md` is one line per paragraph; `--write` converts |
| `wordle_solver.ipynb` | the analysis behind the prior probabilities |
| `data/`, `data/derived/` | the word lists and answer record; everything `feedback_table.py` regenerates |
| `feedback_table.py` | one-off: builds `feedback_table.npy` from the word lists |
| `answer_like.py` | one-off: builds `answer_like.json`, the score behind `--curation-shape frequency` |
| `solutions.py` | brings `solutions.json` up to date from the NYT, one request per day since the last |
| `word_lists.py` | one-off: refetches `wordles.json` and `nonwordles.json` upstream |
| `wordlebot.py` | one-off: rescrapes WordleBot's weights, live and from the 2023 snapshot |
| `analysis/` | the measurement scripts behind every number in the prior sections below |

`answer_like.py`, `solutions.py`, `word_lists.py` and `wordlebot.py` verify against the file they would write and do nothing unless passed `--write`, since each one rewrites something the solver reads. `feedback_table.py` predates that rule and rebuilds in place as soon as it is run — or imported.

## Results

Both priors on with the shipped shapes, six-guess limit, failures counted as 7. Every row is the fit archived for that form, replayed rather than refitted, by `analysis/results_tables.py`. There is no sample here and so no error bar: the solver is deterministic, so every one of the 12,947 words is played once and weighted by the prior, which *computes* each mean rather than estimating it.

<!--table results_tables.forms_table-->
| form | expression | terms | opening | mean | vs. best | loss probability |
| --- | --- | --- | --- | --- | --- | --- |
| anchored2 | 1 + a·x + b·(1−1/n) | 2/2 | salet | 3.0641 | — | 0.0243% |
| saturating | a + b·x + c·log2(1+x) | 3/3 | salet | 3.0645 | +0.0004 | 0.0231% |
| cubic | a + b·x + c·x² + d·x³ | 4/4 | salet | 3.0645 | +0.0004 | 0.0234% |
| anchored | 1 + a·x + b·(1−1/n) + c·x² | 2/3 | salet | 3.0646 | +0.0006 | 0.0226% |
| sqrt | a + b·x + c·√x | 3/3 | salet | 3.0650 | +0.0010 | 0.0216% |
| invonly | a + b·(1−1/n) | 2/2 | salet | 3.0661 | +0.0020 | 0.0220% |
| spline | piecewise linear in x, knots at 2/4/6/8 | 4/6 | salet | 3.0672 | +0.0031 | 0.0232% |
| loglinear | a + b·log2(n) | 2/2 | raise | 3.1040 | +0.0399 | 0.0307% |
| quadratic | a + b·x + c·x² | 2/3 | raise | 3.1040 | +0.0399 | 0.0307% |
| IG + 2·p_win | — | — | raise | 3.1067 | +0.0426 | 0.0387% |
| greedy entropy | — | — | raise | 3.7569 | +0.6928 | 0.0034% |
<!--/table-->

The table divides where the opening word does, and the division is lopsided: <!--n results_tables.same_opening-->7 forms open `salet` inside <!--n results_tables.same_opening_spread .4f-->0.0031 guesses of each other, then a cliff of <!--n results_tables.cliff .3f-->0.037 to the two that open `raise`. Since these are exact figures, that ordering is a fact about this prior rather than a coin landing — though <!--n results_tables.same_opening_spread .4f-->0.0031 is still far too little to choose a *default* on, which is what the next section is for. `loglinear` and `quadratic` are one function twice over: the monotonicity guard cuts `quadratic` back to `a + b·x`, and with `x = log2(n)` that is what `loglinear` already was.

The loss column is the one place greedy entropy wins, and it wins by a lot: see *Losing less* for why the estimator's two thirds of a guess costs <!--n head_to_head.baseline.loss_ratio .0f-->7 times the loss rate.

**`IG + 2·p_win` is the second reference**, and the only row that is neither a fitted curve nor a plain greedy rule (`WordleSolver(win_bonus=...)`). It scores information gain and then pays a guess that could itself be the answer two bits for the chance that it is. Two exact facts fix that price with no data at all: one candidate takes one guess and two take 1.5, so a bit of information is worth half a guess, and winning outright saves a whole one — two bits' worth. `SOLVER_EXPLAINED.md` *A straight-line `f` is entropy plus a bonus* has the identity that makes a policy of that shape what a linear `f` collapses to. It is a reference and deliberately not the baseline: it buys **<!--n results_tables.ig_over_entropy .3f-->0.650 guesses** over greedy entropy with nothing fitted, and pays for them with <!--n results_tables.ig_loss_ratio .0f-->11 times the loss rate, which is the trade *Losing less* is about. Greedy entropy stays the baseline because it makes no such bet.

What puts the two stragglers below the cliff is **concavity in x**, and the identity that says so needs no measurement. With `f` exactly linear in x,

```
E(g, S) = 1 + a·(1 − p_win) + b·(H(S) − IG(g))
```

so minimizing E is the same as maximizing `b·IG + a·p_win`: greedy entropy plus a bonus for possible answers. That is why `loglinear` picks entropy's opening word and gains the least — of the <!--n results_tables.entropy_gap .2f-->0.69 guesses between greedy entropy and the best form, ~<!--n results_tables.both.bonus .2f-->0.65 is that bonus and ~<!--n results_tables.both.curvature .2f-->0.04 is curvature. Everything above the cliff is concave and collects both terms; the two below collect only the first.

Fit quality is not what sorts them, even where the two happen to agree. Under this prior the pair below the cliff are also the worst-fitting, at an RMSE of <!--n results_tables.rmse_line .3f-->0.571 against everything else's <!--n results_tables.rmse_lo .2f-->0.51–<!--n results_tables.rmse_hi .2f-->0.53 — but that is one fact rather than two, a straight line being both the worst fit and the shape the identity above collapses. `SOLVER_EXPLAINED.md` *The form of `f` has to be refitted per shape* has a case where the two come apart.

Saturation is not the liability under this prior that it is under a flat one. `invonly` flattens out completely (f(200) ≈ f(2000)) and still lands within <!--n results_tables.invonly_gap .4f-->0.0020 of the best, because with the mass concentrated the effective size rarely reaches the range where the curve has gone flat.

### Which form is the default

`--form` defaults to **`saturating`**, and the first thing to say about that default is how little it governs. **It never reaches play.** `estimator.load_active` reads whatever form each prior's own cached fit was made in, and every prior the page or the flags can select has one; the Rust falls back on the default only for a slug with no fit at all. So this section decides one thing: which family `train` fits when you do not name one, which matters for a prior nobody has tried yet and for nothing else.

That is still worth getting right, so `analysis/form_default.py` replays every archived fit over every word the answer could be, weighted by each prior in turn. Nothing is held out because nothing needs to be: the evaluation covers the whole space, and the gaps are exact.

<!--table form_default.aggregate_table-->
| form | terms | mean gap to the prior's best | worst gap | worst rank of 9 |
| --- | --- | --- | --- | --- |
| invonly | 2 | +0.0011 | +0.0031 | 7 |
| anchored2 | 2 | +0.0012 | +0.0043 | 6 |
| anchored | 2–3 | +0.0018 | +0.0049 | 8 |
| **saturating** | 3 | +0.0018 | +0.0045 | 9 |
| sqrt | 3 | +0.0020 | +0.0042 | 10 |
| spline | 4–6 | +0.0026 | +0.0043 | 9 |
| cubic | 2–4 | +0.0096 | +0.0407 | 8 |
| quadratic | 2–3 | +0.0244 | +0.0407 | 10 |
| loglinear | 2 | +0.0247 | +0.0407 | 10 |
<!--/table-->

The top <!--n form_default.top_group-->6 are inside <!--n form_default.top_group_spread_mean .4f-->0.0015 guesses of each other on the mean gap and <!--n form_default.top_group_spread_worst .4f-->0.0019 on the worst, and that spread is the size of the fit's own noise: refit the same form from a different training seed and its score moves by up to <!--n rust_fit.seed_spread_max .4f-->0.0008 under these priors (`analysis/rust_fit.py` prints the per-seed scores). So which of them ranks first is which fit came out luckiest, not a property of the form. Across the eight priors the winners were <!--s form_default.winners-->`anchored2` ×4, `invonly` ×2, `spline`, `saturating`<!--/s-->, and a refit reshuffles them.

That changes what a refit is for. It chooses between *fits* rather than forms, so `form_default --promote` ranks the fit already in play alongside the new archives and keeps it when nothing beats it — a refit can improve a prior and cannot demote one. And it is why `compare` fits four of the nine by default, <!--s form_default.main_forms-->`anchored2`, `invonly`, `saturating`, `spline`<!--/s-->: one per family — the anchored pair, the inverse alone, the line with a concave bend, the free-form — each the member that has won a prior, with `--forms all` for the rest. On this refit the four held every winner, so the set gave up <!--n form_default.main_forms_gap .4f-->0.0000 against fitting all nine; what it could give up is bounded by the spread above.

`saturating` stays the default because nothing argues for a change: it sits within <!--n form_default.saturating.worst_gap .4f-->0.0045 of the best form under every prior tried, and any of the top group would serve as well.

The bottom of the table is the firm part of the result. `loglinear` and `quadratic` give up <!--n form_default.straight_line_gap .2f-->0.04 guesses wherever the curation prior is on, because the guard has cut them to a straight line and a straight line is greedy entropy plus the bonus; `cubic` joins them under `frequency`. Those are the forms to avoid, and they lose by twenty times what separates the leaders.

Read this in preference to ranking the archives by their own stored holdout, which `compare` prints: the two orderings disagree, and the holdout is a 200-answer sanity check where this is the whole space.

All four combinations of the two switches, with the sharpened shapes, each carrying its own fitted form. **Every row here is a configuration the page can be put into**, the two priors being independent checkboxes on it; "the default prior" below always means the last row, never the only one that runs. Every word is weighted by *the solver's own prior*, so each row asks "if this model of the editor is right, how well does the solver do?" The rows are **not** comparable to each other, since a concentrated prior makes the game genuinely easier. The opening word and the solve rate are the expected-guesses mode's:

<!--table results_tables.priors_table-->
| priors | form | opening | entropy mean | expected mean | solve rate |
| --- | --- | --- | --- | --- | --- |
| none | invonly | tares | 4.1223 | 4.0694 | 99.89% |
| curation only | invonly | trace | 3.8455 | 3.4903 | 99.99% |
| repeat only | anchored2 | tares | 4.1036 | 4.0105 | 99.93% |
| **both (the default)** | anchored2 | salet | 3.7569 | 3.0641 | 99.98% |
<!--/table-->

The gain comes from the curation prior. When the prior is nearly flat, expected guesses and information gain rank guesses almost identically — with no priors the estimator is worth **<!--n results_tables.none.gain .4f-->0.0529** guesses against entropy, next to **<!--n results_tables.both.gain .4f-->0.6928** with both priors on. Thirteen times less for the same machinery, as the identity above predicts. It is small, but being an exact figure rather than a sample it is not a question of resolving it.

### Where the estimator's guesses come from

Against the fit-free reference the shipped solver wins <!--n head_to_head.gap .4f-->0.0426 guesses, and `analysis/head_to_head.py` says where every one of them is. Both policies play all 12,947 words, so a word either changes hands or it does not:

<!--table head_to_head.table-->
| guesses | shipped | `IG + 2·p_win` |
| --- | --- | --- |
| 2 | **14.32%** | 11.64% |
| 3 | 67.25% | **68.55%** |
| 4 | **16.47%** | 17.66% |
| 5 | **1.63%** | 1.87% |
| 6 | 0.30% | **0.25%** |
| lost | **0.0243%** | 0.0387% |
<!--/table-->

The two-guess row carries the margin. Finishing in two on <!--n head_to_head.two.shipped .2f-->14.32% of the prior's mass against <!--n head_to_head.two.other .2f-->11.64% is worth more than the <!--n head_to_head.gap .4f-->0.0426 by itself, and the reference wins only one row back, landing on three more often than the shipped solver does. That is the fitted `f` and the priors doing the one thing they are for: leaning the second guess toward the words an editor is likely to pick, which pays out as outright wins rather than as tidier splits. The one place the reference is genuinely safer is the six-guess row, where its refusal to gamble keeps a few more games off the deadline.

They are also nothing like the same policy. The shipped solver finishes sooner on <!--n head_to_head.sooner.shipped .2f-->21.11% of the prior's mass, the reference on **<!--n head_to_head.sooner.other .2f-->17.63%**, and they tie on <!--n head_to_head.same .0f-->61%. Two policies four hundredths of a guess apart disagree about two games in five. The reference's losses are worse than its count of them suggests, too: <!--n head_to_head.lost.other-->425 words against <!--n head_to_head.lost.shipped-->313, and its heaviest is `<!--s head_to_head.heaviest.other-->candy<!--/s-->` at <!--n head_to_head.heaviest.other.mass .4f-->0.0026% of the mass, three times heavier than anything the shipped solver drops — it has no `f` with which to price a deadline, so it walks into losses that `--loss-penalty 5` buys the shipped policy out of.

Because `f` estimates expected guesses, the policy is risk-neutral away from the endgame, where greedy entropy is implicitly cautious. `--loss-penalty 5` compensates, closing about two thirds of the gap in loss probability, but not for nothing and not the whole way: the estimator still loses <!--n head_to_head.baseline.loss_ratio .0f-->7 times as often as entropy does (see [Losing less](#losing-less)). `--loss-penalty 1` restores risk neutrality and gives that ground back.

## The two priors

The answer is not drawn uniformly, and the two departures from uniform can be switched on and off independently — from the command line, and from a checkbox each on the page. **All four combinations ship**, each with its own cached fit and its own opening word; what "the default prior" names throughout this document is both of them on with the sharpened shapes, which is where the switches start rather than the only place they go.

* **repeat prior** (`--no-repeat-prior`) — the editor reuses an already-played answer, so used words share `--repeat-prob` of the mass instead of their proportional share.
* **curation prior** (`--no-curation-prior`) — some answers come from outside the curated 2022 list, so those words share `--noncurated-prob` of what is left.

The model is sequential — a repeat first, then curated vs. non-curated on the remainder — so `--noncurated-prob` is *conditional on not being a repeat*, and must be measured with that denominator.

Each group also has a **shape**, which is how its mass is spread over its members. The defaults are the sharpened shapes, worth <!--n replay.shapes.together.blend-->−0.076 ± 0.038 guesses over the flat spread the two priors used on their own.

* `--repeat-shape binned` (default) weights a past answer by where it first appeared, binned 200 wide and Laplace-smoothed — reuse concentrates near the start of the run, not uniformly over everything played.
* `--curation-shape wordlebot` (default) weights an off-list word by WordleBot's own weight, keeping 5% of the mass outside that list as a hedge; `--curation-shape frequency` weights it by how answer-like the word is instead, which is slightly weaker but needs no vendored list.
* `--repeat-shape uniform`, `--curation-shape uniform` give the flat spread back.

Each shape is worth having on its own terms, and the two are **additive within noise**; *What the two shapes are worth together* below has the measurement. The shape is part of the cache key because the best form of `f` depends on it, and the dependence reaches the opening word: `spline` wins under the sharpened shapes and under the flat ones alike, yet pair a sharpened repeat shape with a flat curation shape and it opens `raise` instead of `salet` and drops to seventh, behind every form that stayed concave.

### The regime the defaults describe

Measured over the <!--n regime.answers-->1,917 answers in `solutions.json`, through Wordle #<!--n regime.last_no d-->1917 (<!--s regime.last_date-->2026-09-17<!--/s-->). The puzzle number is `days_since_launch + 1`. The `id` field is an NYT internal key and stops matching after #506 — it is *not* what marks a word as off-list, which is membership in the curated list and nothing else. `trace` came back on 2026-08-21 carrying id 4413 and is a curated word.

<!--table regime.table-->
| window | puzzles | repeat rate | off-list \| not a repeat |
| --- | --- | --- | --- |
| 2021–2025 | 1,657 | 0.00% | 1.75% |
| **since Wordle #1690 (2026-02-02)** | **228** | **10.5%** | **13.2%** |
| last 50 | 50 | 12.0% | 29.5% |
| all time | 1,917 | 1.3% | 3.1% |
<!--/table-->

Both defaults of 0.10 are right for the current regime and roughly 9× and 4× too high against the full record. Nothing was ever off-list before 2023, and `cigar` on 2026-02-02 was the first repeat in Wordle's history; every repeat since falls in the last <!--n regime.since.puzzles-->228 puzzles, about one in nine and a half.

The two moves started together because the curated pool is running out: <!--n regime.curated_used-->1,835 of the 2,309 curated words are used, <!--n regime.curated_unused-->474 remain, and fresh ones are consumed at ~<!--n regime.fresh_per_year .0f-->328/year — about <!--n regime.years_left .1f-->1.4 years left. Going off-list and recycling old answers are the same editorial response to the same constraint. Extrapolating both upward is defensible on that basis, and 0.10 is a forward-looking number rather than a historical one.

### What each prior is worth on real answers

Everything above argues the two priors *from* the answer record; this measures them against it. `analysis/replay.py --all-priors` replays the <!--n replay.priors.puzzles-->228 real puzzles of the current regime under all four combinations, rebuilding each prior from strictly earlier puzzles each day, and pairs every arm on the same answer. The pairing is the whole point: the tables in *Results* weight each row by the prior being judged, so a prior that concentrates on words it happens to solve quickly scores well by construction, whereas here the yardstick is the sequence the NYT actually published and no arm gets a say in it.

<!--table replay.priors.table-->
| priors | mean | vs. both | on the 24 repeats | on the 27 off-list | on the other 177 |
| --- | --- | --- | --- | --- | --- |
| none | 3.9912 | +0.833 ± 0.053 | +0.292 ± 0.165 | +0.259 ± 0.147 | +0.994 ± 0.055 |
| curation only | 3.5351 | +0.377 ± 0.052 | −0.333 ± 0.167 | +0.333 ± 0.177 | +0.480 ± 0.054 |
| repeat only | 3.9737 | +0.816 ± 0.052 | +0.208 ± 0.199 | +0.296 ± 0.167 | +0.977 ± 0.051 |
| **both (the default)** | **3.1579** | — | — | — | — |
<!--/table-->

Read that first column against `priors.build_weights` rather than against intuition. Switching the curation prior off does not leave a uniform prior over the curated list; it leaves a uniform prior over all 12,947 words, non-answers included. The <!--n replay.priors.contrast.curation_repeat_off-->+0.456 ± 0.053 the column charges for the curation prior, read as a cost, is therefore two things bundled — telling the solver the answer is probably a curated word at all, and the `wordlebot` shape concentrating the off-list remainder onto ~911 words — and nothing measured here separates them.

**The two priors are strongly superadditive**, which the shapes below turn out not to be. Adding the repeat prior to a solver with no curation prior buys <!--n replay.priors.contrast.repeat_curation_off-->+0.018 ± 0.037 — next to nothing. Adding the same prior on top of the curation prior buys <!--n replay.priors.contrast.repeat_curation_on-->+0.377 ± 0.052. The interaction is **<!--n replay.priors.interaction-->−0.360 ± 0.065**, and the three-arm blend agrees at <!--n replay.priors.interaction3-->−0.361 ± 0.060 — a wider view of the same data rather than an independent check, since its repeat arm is those same <!--n replay.priors.n_repeats-->24 events.

The mechanism is calibration. Ask what rates each prior implies, against the <!--n regime.since.repeat .1f-->10.5% repeat rate and the <!--n regime.since.offlist .1f-->13.2% off-list rate the regime really shows: uniform implies <!--n regime.implied.none.repeat .1f-->14.6% and <!--n regime.implied.none.offlist .1f-->95.7%, curation alone <!--n regime.implied.curation_only.repeat .1f-->72.4% and <!--n regime.implied.curation_only.offlist .1f-->33.1%, repeat alone <!--n regime.implied.repeat_only.repeat .1f-->10.0% and <!--n regime.implied.repeat_only.offlist .1f-->95.7%, and both 10.0% and 10.0% (`analysis/regime.py`). Every prior but the last is wildly wrong about the rate the *other* one governs. Uniform lands near enough on repeats by accident — <!--n regime.used_words-->1,893 used words in 12,947 is <!--n regime.used_share .1f-->14.6% — so the repeat prior corrects a column that was barely wrong and leaves the <!--n regime.implied.none.offlist .1f-->95.7% untouched, which is why it buys almost nothing. Curation alone pulls that second column to <!--n regime.implied.curation_only.offlist .1f-->33.1% and creates the repeat error doing it, a 2,309-word list with <!--n regime.curated_used-->1,835 spent amounting to a prediction that the editor repeats herself seven times in ten. Only with both on is either rate right, and that is the whole of the interaction.

That also prices the repeat arm, where `curation only` scores <!--n replay.priors.curation_only.repeats-->−0.333 ± 0.167 against `both` on the <!--n replay.priors.n_repeats-->24 answers that really were repeats. This is the ordinary cost of a calibrated prior, not a defect: a repeat averages 5.3e-5 under `--repeat-prob 0.10` against the 3.8e-4 it would hold as an ordinary curated word, and any prior that moves a class from <!--n regime.implied.curation_only.repeat .1f-->72.4% to 10.0% pays on the occasions the class turns up. What it buys is the fresh curated word rising from 3.9e-4 to 1.7e-3, worth <!--n replay.priors.curation_only.other-->+0.480 ± 0.054 across the <!--n regime.mixture.fresh .1f-->77.6% of the regime that is neither a repeat nor off-list.

### The repeat prior's shape

By default `build_weights` spreads the repeat mass uniformly over all <!--n regime.used_words-->1,893 used words, but the reuse is not uniform: <!--n drift.from_first_300-->22 of the <!--n drift.events-->24 repeats came from Wordle #1–#300 (median #<!--n drift.median_source .0f-->129), with only <!--s drift.outliers-->`sandy` (#1227), `smile` (#831)<!--/s--> outside — an <!--n drift.outlier_rate .0f-->8% outlier rate, too high to zero out.

Scored prequentially — each model predicts repeat *i* from repeats 1…*i−1* only, so no parameter ever sees the point it is scored on:

<!--table repeat_shapes.table-->
| model | logL over 24 events | vs. uniform |
| --- | --- | --- |
| uniform over used (current) | −179.7 | — |
| binned, width 200, α = 0.5 | **−152.6** | **+27.1** |
| binned, width 200, α = 1.0 | −155.5 | +24.2 |
| best of 16 window variants (last K ∈ 3/5/8/all, ±25…±200) | −161.5 | +18.2 |
<!--/table-->

The winner is a Laplace-smoothed histogram over fixed bins, at <!--n repeat_shapes.gain +.2f-->+1.13 ± <!--n repeat_shapes.gain_se .2f-->0.20 nats per event (<!--n repeat_shapes.sigma .1f-->5.5σ, winning <!--n repeat_shapes.wins-->19 of <!--n repeat_shapes.events-->24). Scored in-sample the same comparison claims <!--n repeat_shapes.insample +.2f-->+1.46, so overfitting inflates it ~<!--n repeat_shapes.inflation .0f-->29% without manufacturing it.

The window row is the point of the exercise. A window has a boundary, a boundary is a parameter, and it has to be tuned on the same events it is then scored against — so even the best of sixteen tuned windows lands <!--n repeat_shapes.window_behind .0f-->9 nats behind a histogram with nothing to tune. (A window needs a third knob besides `K` and the half-width: how much mass to leave outside. With a hard cutoff a single repeat landing outside costs ~7 nats, which is `sandy` and `smile` all over again.)

The obvious rival is an age-based shape, with weight rising by time since last use instead of binning by where in the answer order the word first appeared. It does not beat this. Refitting its rate prequentially scores <!--n repeat_shapes.age_refit .1f-->−168.9, well short of binned; the rate is unstable on twenty-odd events. A rate fixed at 2/year reaches <!--n repeat_shapes.age_2 .1f-->−150.4, but only when chosen with hindsight on the same events, and it wins by piling mass on the oldest words and then paying for it on `sandy` and `smile`. The Laplace floor is what makes binned survive its outliers.

On its own it changes nothing that matters, and may cost a little. Replaying all <!--n replay.shapes.puzzles-->228 real puzzles of the current regime, rebuilding the prior from strictly earlier puzzles each day, sharpening the repeat shape alone scores <!--n replay.shapes.repeat_alone.mean .4f-->3.2807 against the flat shapes' <!--n replay.shapes.uniform.mean .4f-->3.2368 — a paired **<!--n replay.shapes.repeat_alone.vs-->+0.0439 ± 0.0421**, with <!--n replay.shapes.repeat_alone.changed-->74 games changing length, <!--n replay.shapes.repeat_alone.better-->30 better and <!--n replay.shapes.repeat_alone.worse-->44 worse. The subgroups pull against each other rather than cancelling: <!--n replay.shapes.repeat_alone.repeats-->−0.333 ± 0.155 on the <!--n replay.shapes.n_repeats-->24 repeat answers against <!--n replay.shapes.repeat_alone.other-->+0.130 ± 0.041 on the <!--n replay.shapes.n_other-->177 that are neither repeats nor off-list, and repeats are only a ninth of games. `analysis/replay.py --shapes` produces this pair.

The paired sd is <!--n replay.shapes.repeat_alone.sd .2f-->0.64 guesses, so resolving a 0.02-guess effect at 2σ needs ~<!--n replay.shapes.repeat_alone.games_needed-->4,034 regime games at one per day, which is why this sat unshipped.

Whether the source window is drifting forward is likewise unresolved, and the evidence has weakened. The drift rate is how far the source moves per puzzle, so **0 is a static window and 1.0 is a fixed replay lag** — one puzzle a day means a constant lag drags the source forward one position a day. Over the first 18 events it was <!--n drift.first18.drift-->+0.55 ± 1.56, an interval covering both; over all <!--n drift.events-->24 it is <!--n drift.drift-->−0.06 ± 0.88, which has collapsed onto the static reading while still excluding nothing. The rank correlation between repeat order and source index fell the same way, <!--n drift.first18.rho +.2f-->+0.46 (p = <!--n drift.first18.p .2f-->0.04) to <!--n drift.rho +.2f-->+0.35 (p = <!--n drift.p .2f-->0.08), and it keeps falling because the newest repeats keep arriving from early in the answer order — <!--s drift.latest_two-->`click` and `focus`<!--/s-->, the latest two, came from <!--s drift.latest_two_sources-->#135 and #268<!--/s-->. The replay lag averages <!--n drift.lag_mean .2f-->4.40 years with an sd of <!--n drift.lag_sd .2f-->0.78, and <!--n drift.in_band-->21 of <!--n drift.events-->24 events sit between 4.4 and 5.1 years.

### The curation prior's shape

Weighting off-list words instead of flatly is a large modeling gain on its own. Scored prequentially over the off-list answers, uniform is <!--n curation_shapes.uniform.nats .2f-->−9.27 nats per event; an answer-like score fitted against the curated list as positives is <!--n curation_shapes.frequency.nats .2f-->−8.06 at the shipped `t` = 0.35; and WordleBot's weights, reconstructed by a third party and validated against a 2023 snapshot that predates <!--n curation_shapes.post-->54 of the <!--n curation_shapes.events-->58 events, are <!--n curation_shapes.wordlebot.nats .2f-->−6.45, with the effective support falling from <!--n curation_shapes.uniform.support ,.0f-->10,638 words to <!--n curation_shapes.wordlebot.support ,.0f-->911.

Between the two sharpened shapes the modeling gain does now show in guesses. `analysis/replay.py` runs `frequency` against the default `wordlebot` head to head: `wordlebot` wins by **<!--n replay.frequency.blend-->+0.081 ± 0.025** blended, and by <!--n replay.frequency.vs-->+0.0614 ± 0.0268 over the <!--n replay.puzzles-->228 regime puzzles. Nearly all of it is the off-list arm, <!--n replay.frequency.arm.offlist-->+0.466 ± 0.089 over its <!--n replay.arms.offlist-->58 events, where the repeat arm ties at <!--n replay.frequency.arm.repeat-->+0.000 ± 0.060: under the current fits the two shapes play the repeat answers identically, so nothing offsets the off-list gap the way an earlier refit's repeat arm did. `wordlebot` ships on the modeling evidence first, and `frequency` is there for anyone who would rather not vendor a scrape.

### What the two shapes are worth together

The shapes convert only a little of their modeling gain into guesses. Measured on three arms — the <!--n replay.shapes.arms.repeats-->24 real repeats and <!--n replay.shapes.arms.offlist-->58 off-list answers with the prior rebuilt as of each puzzle, plus the regime's <!--n replay.shapes.arms.fresh-->177 fresh curated answers — blended at its own mixture of <!--s replay.shapes.mixture-->10.5% / 11.8% / 77.6%<!--/s-->, against a baseline with both priors on and uniform shapes inside them:

<!--table replay.shapes.arm_table-->
| shapes | repeat | off-list | fresh | blended |
| --- | --- | --- | --- | --- |
| curation `wordlebot` alone | +0.042 ± 0.127 | −0.879 ± 0.118 | +0.107 ± 0.038 | −0.016 ± 0.035 |
| repeat `binned` alone | −0.333 ± 0.155 | −0.052 ± 0.124 | +0.130 ± 0.041 | +0.060 ± 0.039 |
| `binned` + `frequency` | −0.417 ± 0.146 | −0.397 ± 0.130 | +0.124 ± 0.044 | +0.006 ± 0.040 |
| `binned` + `wordlebot` | −0.417 ± 0.158 | −0.862 ± 0.117 | +0.090 ± 0.040 | −0.076 ± 0.038 |
<!--/table-->

Concentrating the off-list mass on ~<!--n curation_shapes.wordlebot.support ,.0f-->911 words gives each about <!--n regime.weight.offlist_listed .1e-->9.7e-5 against a used word's <!--n regime.weight.used .1e-->5.3e-5, so past answers stop competing for the solver's attention. That is where the residual penalty `wordlebot` alone carries on the repeat arm comes from.

The shipped pair is the one blend that reaches its own error: <!--n replay.shapes.together.blend-->−0.076 ± 0.038, <!--n replay.shapes.together.blend_sigma .1f-->2.0σ from zero, and the same on the plain regime replay at <!--n replay.shapes.together.vs-->−0.0789 ± 0.0462. Whether the two shapes need each other the table still cannot say: <!--n replay.shapes.curation_alone.blend-->−0.016 ± 0.035 and <!--n replay.shapes.repeat_alone.blend-->+0.060 ± 0.039 apart, summing to <!--n replay.shapes.apart +.3f-->+0.043, against <!--n replay.shapes.together.blend_delta +.3f-->−0.076 together — a gap of a tenth with errors of a twentieth on either side. `frequency` (<!--n replay.shapes.binned_frequency.blend-->+0.006 ± 0.040) sits a step behind `wordlebot` here, and the paired head-to-head above is what resolves that step.

**The fresh arm used to be a census of the curated words still unused, and that was training on the test set.** Policy iteration draws its practice answers from the prior it is fitting, whose fresh-curated group is exactly those <!--n regime.curated_unused-->474 words, so four-fifths of the table's weight sat on material every arm had trained against. It flattered nobody in particular — both solvers reach their own tuned best there and tie — which is precisely the problem, because a tie on 81% of the grade hands the verdict to the small arms. `analysis/fresh_arm.py` prices it: <!--n fresh_arm.census_today-->−0.004 ± 0.014 on the census against <!--n fresh_arm.own.real-->+0.130 ± 0.041 on the answers that really happened, still only <!--n fresh_arm.own.census-->+0.012 ± 0.014 when the census is played on those same days, and the gap between the two word sets falling from <!--n fresh_arm.own.gap .3f-->0.118 to <!--n fresh_arm.sharp.gap .3f-->0.019 once both arms share one `f`.

The arm is <!--n replay.shapes.arms.fresh-->177 real answers now, which costs precision where the variance lives and keeps the only edge this table has over the regime replay above: the off-list arm's <!--n replay.shapes.arms.offlist-->58 events against the window's <!--n replay.shapes.n_offlist-->27. That is worth about a fifth — ±<!--n replay.shapes.together.blend_se .3f-->0.038 here against ±<!--n replay.shapes.together.se .3f-->0.046 there, on the same contrast — and the two agree, both putting the shipped pair at the edge of what a season of puzzles can resolve. What the shapes have going for them first is the modeling evidence, and this table has begun to agree with it.

One thing this measurement still needs: the form of `f` must be refitted per shape. The three shape combinations each picked a different form on the last refit — <!--s form_default.shape_winners-->`spline`, `saturating`, `anchored2`<!--/s--> — so a form carried across is a form some other prior chose, and the opening word can go with it.

### The group with no shape

Two of the three groups `priors.build_weights` builds get a shape; the third and largest does not. The curated words not yet used carry 81% of the mass at the shipped probabilities, spread flat over the <!--n fresh_shapes.fresh_curated.pool_to-->475 of them that are left, and `analysis/fresh_shapes.py` is what says that flat is where it should stay.

The events are the <!--n fresh_shapes.fresh_curated.events-->1,329 first uses of a curated word from Wordle #507 (2022-11-07), which is the first puzzle `solutions.json` credits to an editor — before it the answers walk the curated list in its stored order, so the pick is not a choice. Each model is normalized over the words still unused that day and never sees the event it is scored on, the same rule the other two shapes are held to:

<!--table fresh_shapes.fresh_curated.table-->
| shape | nats/event | vs. uniform | effective support |
| --- | --- | --- | --- |
| uniform over the pool (current) | −6.9742 | — | 475 |
| WordleBot weight | −6.9442 | **+0.0300 ± 0.0007** | 455 |
| WordleBot, 5% hedge (as shipped off-list) | −7.0607 | −0.0865 | 315 |
| list position, binned 100/200/400 | −6.9741 | +0.0001 | 475 |
<!--/table-->

WordleBot wins at <!--n fresh_shapes.fresh_curated.wordlebot.sigma .0f-->44σ on <!--n fresh_shapes.fresh_curated.wordlebot.wins-->1,288 of <!--n fresh_shapes.fresh_curated.events-->1,329 events and the win is worth almost nothing: <!--n fresh_shapes.fresh_curated.wordlebot.gain .2f-->0.03 nats against a baseline of <!--n fresh_shapes.fresh_curated.uniform.nats .2f-->−6.97, moving the effective support from <!--n fresh_shapes.fresh_curated.pool_to-->475 words to <!--n fresh_shapes.fresh_curated.wordlebot.support .0f-->455. The same weights on the off-list group are worth **<!--n fresh_shapes.off-list.wordlebot.gain +.4f-->+2.8108**, taking its support from <!--n fresh_shapes.off-list.pool_to-->10,581 to <!--n fresh_shapes.off-list.wordlebot.support ,.0f-->905. The group holding 9% of the mass has structure worth recovering and the group holding 81% has next to none.

Nor is that a failure of the features tried. A conditional logit over <!--n fresh_shapes.features-->6 of them — word frequency, WordleBot weight, list position, vowels, rare letters, distinct letters — fitted on every event at once and scored on what it was fitted on, gains **<!--n fresh_shapes.ceiling +.4f-->+0.1064 nats/event**. That is a ceiling with hindsight built in, and anything prequential comes in under it. The best single feature is <!--s fresh_shapes.best_feature-->word frequency<!--/s--> at <!--n fresh_shapes.best_feature_gain +.4f-->+0.0655, so the editor leans very slightly towards commoner words and otherwise draws close to uniformly from whatever the list has left.

The shipped hedge going *backwards* is worth knowing before reusing it. Five percent is sized for a group where <!--n fresh_shapes.pool.offlist_unlisted-->9,746 of <!--n fresh_shapes.pool.offlist_words-->10,580 words are unlisted; only <!--n fresh_shapes.pool.fresh_unlisted-->7 curated words are, so the same 5% hands each of those few several times what a listed word gets. A shape is calibrated to its group, not to the weights underneath it.

That is also why the two groups are not simply pooled and weighted by WordleBot throughout, which is the obvious simplification since the scrape covers curated and off-list words alike. WordleBot lists <!--n fresh_shapes.pool.offlist_listed-->834 off-list words carrying more total weight than all <!--n fresh_shapes.pool.fresh_listed-->467 listed fresh-curated words, so one pool implies the editor leaves the curated list on <!--n fresh_shapes.pool.pooled_rate .1f-->57.7% of non-repeat days against the <!--n fresh_shapes.pool.regime_rate .1f-->13.2% the regime ran at. It is the same failure the `curation only` arm makes on repeats: the weight ranks words within a group and does not set the rate between two.

### The prior sits in a flat basin

None of the prior's settings is worth tuning, and `analysis/prior_basin.py` is what says so. It sweeps both axes over the <!--n prior_basin.puzzles-->228 real regime puzzles with `f` held fixed throughout, so a difference is the prior's doing rather than the fit's.

Inside the groups there is nothing at all. Tempering a shape — raising its relative weights to a power, 0 leaving the group flat and 1 being what ships — buys none of what it was tried for. The repeat shape wants all of its sharpening: flattening it costs **<!--n prior_basin.repeat_flattened-->+0.039 ± 0.026**, and the curve between runs monotonically under the fit made for the flat shapes. The curation shape wants only a little of its sharpening: flattening it outright costs <!--n prior_basin.curation_flattened-->+0.105 ± 0.028, and every point from a quarter sharpened to fully sharpened lands within <!--n prior_basin.curation_span .3f-->0.026 guesses of the shipped setting, despite this being the shape with much the larger modeling gain.

Between the groups, the mass moves by a factor of thirty before the total notices:

<!--table prior_basin.table-->
| `repeat_prob` | 0.01–0.30 | 0.40 | 0.55 | 0.71 | 0.90 |
| --- | --- | --- | --- | --- | --- |
| all 228 | within ±0.03 | +0.105 | +0.158 | +0.189 | +0.610 |
| the 177 fresh | within ±0.05 | +0.147 | +0.232 | +0.277 | +0.797 |
<!--/table-->

`noncurated_prob` has the same shape, flat to <!--n prior_basin.noncurated.flat_to .2f-->0.40 and then <!--s prior_basin.noncurated.beyond-->+0.158, +0.241, +0.443<!--/s-->. The subgroups underneath are not flat — the repeat arm runs from <!--n prior_basin.repeat.repeat_lo-->+0.333 at `repeat_prob` 0.01 to <!--n prior_basin.repeat.repeat_hi-->−0.333 at <!--n prior_basin.repeat.flat_to .2f-->0.30, and the fresh arm the other way — so the trade is real and it simply cancels across the basin.

That is what reconciles this with *What each prior is worth on real answers*, where switching a prior off costs 0.3 to 0.7 guesses. Off is not a small probability but a wrong one: with curation still on and the repeat prior off, used and unused curated words share the mass equally, which implies a repeat rate of <!--n regime.implied.curation_only.repeat .1f-->72.4%. The settings that cost anything live out at <!--n regime.implied.curation_only.repeat .0f-->72% and <!--n regime.implied.none.offlist .0f-->96%, and the basin covers everything from 1% to <!--n prior_basin.repeat.flat_to .0%-->30%. The default 0.10 and the regime's measured <!--n prior_basin.regime_repeat .3f-->0.105 are both well inside it, which is the useful thing to know — the prior has to be roughly right, and beyond that the details do not reach the board.

The far ends read worse than they should, since a setting `f` was never fitted for carries a borrowed estimator on top of its own miscalibration. That inflates the edges and not the middle, so the basin is at least this wide.

### How fresh the answer record has to be

Every measurement above rebuilds the prior from strictly earlier puzzles on every day of the replay, and the prequential shape scores hold each model to the same rule. The deployed solver is not that current: `data/solutions.json` is refreshed by hand with `solutions.py --write`, and stops at yesterday when it is, so a day of lag is there the moment it is written. `analysis/refit_lag.py` prices the gap, replaying the <!--n refit_lag.puzzles-->228 regime puzzles with the record truncated to `t − lag` instead of `t`:

<!--table refit_lag.table-->
| lag | mass moved | vs. a record rebuilt daily | games changed of 228 |
| --- | --- | --- | --- |
| 1 day | 0.16% | −0.0044 ± 0.0076 | 3 (1 worse, 2 better) |
| 3 days | 0.49% | −0.0044 ± 0.0116 | 7 (3 worse, 4 better) |
| 7 days | 1.11% | −0.0132 ± 0.0146 | 11 (4 worse, 7 better) |
| 14 days | 2.17% | −0.0044 ± 0.0170 | 15 (7 worse, 8 better) |
| 30 days | 4.48% | +0.0263 ± 0.0240 | 30 (18 worse, 12 better) |
<!--/table-->

Averaged over a refresh cycle, where refitting every k days leaves the lag uniform over 0…k−1, that is **<!--n refit_lag.every2-->−0.0022 ± 0.0038** every two days, <!--n refit_lag.every4-->+0.0011 ± 0.0059 every four and <!--n refit_lag.every7-->−0.0081 ± 0.0084 every week. None of it is distinguishable from nothing, and at 30 days the changed games split evenly enough to be coin flips.

Two things could account for a null like that: the weights barely move, or they move and the board does not care, which is what the basin above finds for every other way of perturbing this prior. Here it is the first. A lag of k days leaves k words carrying fresh-curated mass rather than used mass and changes nothing else, so a week relocates <!--n refit_lag.lag7.mass .2f-->1.11% of the prior's mass — against the <!--n refit_lag.yardstick .2f-->6.01% that flattening the repeat shape relocates, itself worth only <!--n prior_basin.repeat_flattened-->+0.039 ± 0.026.

Refitting `f` does not enter this. `estimator.cache_path` is keyed on the prior's slug alone, so one fit serves every day of a replay and every day of a deployment alike; a fresher record buys fresher weights and nothing more. And it buys those only where `priors.build_weights` reads it, which is under `repeat` alone — a repeat-off prior costs exactly nothing to leave stale, for the same reason `--repeat-only` can skip its arms after a write.

## Choosing a guess

`--mode entropy` is the greedy scorer: maximize expected information gain, and nothing else — being a possible answer earns a guess no credit at all, which is what keeps it a clean reference for what pricing that chance is worth.

`--mode expected` (the default) instead minimizes an estimate of how many guesses the game will take in total:

```
E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)
```

where the candidates in state `S` are bucketed by the feedback pattern `r` that guess `g` would produce, `p_r` is the bucket's prior mass, and `n_r` is its *effective size* `2 ** H` (the plain count when the prior is flat over it). Buckets of one or two candidates are worth exactly 1 and `2 - p_max`, so those are used directly rather than going through f. Dropping the `GGGGG` term is what makes a guess that could be the answer beat an equally informative guess that could not.

Both are only fallbacks, used where the game is too wide to search.

### Hard mode (`--hard`, off by default)

Every guess must be a word that could still be the answer. It is a flag on `play` and `benchmark` and a switch on the page.

**This is not the NYT's hard mode**, which is a rule about letters: keep every green where it is, and use every revealed letter somewhere, in as many copies as have been revealed. It says nothing about the letters ruled out, so it permits a guess that spends a gray letter, carries a copy more than the answer holds, or puts a yellow back in the square where it came up yellow. Every word that could still be the answer satisfies that rule, so what is played here is the narrower of the two and the solver never sees the extra moves.

Widening it to the NYT's rule buys nothing, and every word is played under both rules to say so (`analysis/hard_mode_rules.py`). Handing the solver the legal pool at each turn is one substitution again, but the pool is now a function of the rows played rather than of the candidates, so the script plays the policy itself; `--check` is what shows that loop reproducing `common.every_answer` game for game.

| policy | guesses taken off the candidates | words changing length | what the wider rule is worth |
| --- | --- | --- | --- |
| expected, as shipped | <!--n hard_mode_rules.shipped.off_pool-->0 in 12,947 games | <!--n hard_mode_rules.shipped.changed-->0 | <!--n hard_mode_rules.shipped.worth +.4f-->+0.0000 |
| expected, endgame off | <!--n hard_mode_rules.no_endgame.off_pool-->79 | <!--n hard_mode_rules.no_endgame.changed-->41 (<!--n hard_mode_rules.no_endgame.better-->21 better, <!--n hard_mode_rules.no_endgame.worse-->20 worse) | <!--n hard_mode_rules.no_endgame.worth +.4f-->−0.0000 |
| **greedy entropy** | **<!--n hard_mode_rules.entropy.off_pool-->3,099** | **<!--n hard_mode_rules.entropy.changed-->1,100 (<!--n hard_mode_rules.entropy.better-->302 better, <!--n hard_mode_rules.entropy.worse-->798 worse)** | **<!--n hard_mode_rules.entropy.worth +.4f-->+0.0631** |

The objective never takes the freedom. The pool is the whole word list at turn one under either rule, and after that a word that cannot win has to be worth more than the win chance it gives up — which is the trade the objective already prices, and it comes out against the probe every time. Switch the endgame off so that the pool rather than the search decides every turn and it does reach for a non-candidate <!--n hard_mode_rules.no_endgame.off_pool-->79 times, splitting <!--n hard_mode_rules.no_endgame.better-->21 words better and <!--n hard_mode_rules.no_endgame.worse-->20 worse for no movement in the mean. Only greedy entropy uses the wider rule freely, and it pays: entropy does not price winning outright, so it buys the sharper split and forfeits games it could have ended, which is the same asymmetry SOLVER_EXPLAINED *A straight-line* turns on.

Both arms keep the endgame search restricted to the candidates, since its nodes would need a constraint history they are never given, so the figures above are a floor on what the NYT's rule is worth rather than the whole of it. Widening the search too is untested here — the reason to expect little from it is that the objective declined the wider pool at every turn where it was free to choose, which is an argument rather than a measurement.

Mechanically it is one substitution. The pool of guesses to score becomes the candidate set instead of all 12,947 words (`WordleSolver.guess_pool`), which the kernels already take as a parameter, and the exact search narrows the same way. In the search that also removes the partition dedupe: only a candidate itself can come back `GGGGG`, so no two candidates split S identically and there is nothing left to merge.

It is deliberately **not** a `train` flag. The fit is cached per prior and the restriction is not part of the slug, so training under it would overwrite the ordinary fit. `f` is therefore always the one fitted for unrestricted play, and its scores read a little low under the restriction: the position it is estimating is one where the solver would be allowed a probe that it is not in fact allowed. The restriction costs guesses for that same reason. A word that cannot win is often the sharper split.

What it costs, every word weighted by the row's own prior, the same words in both arms, failures counted as 7 (`analysis/hard_mode.py`). Both arms are exact, so the cost is a difference rather than an estimate of one:

<!--table hard_mode.table-->
| priors | free | hard | cost | solve rate, free → hard |
| --- | --- | --- | --- | --- |
| none | 4.0694 | 4.4522 | +0.3827 | 99.89% → 92.33% |
| curation only | 3.4903 | 3.5903 | +0.1000 | 99.99% → 99.58% |
| repeat only | 4.0105 | 4.3726 | +0.3621 | 99.93% → 92.59% |
| **both (the default)** | **3.0641** | **3.0883** | **+0.0242** | **99.98% → 99.90%** |
| both, `frequency` | 3.1007 | 3.1284 | +0.0277 | 99.96% → 99.69% |
<!--/table-->

Under the default prior it costs **<!--n hard_mode.both.cost +.4f-->+0.0242** guesses, a small fraction of what the estimator is worth over greedy entropy. What sets the size is how flat the prior is: with no priors at all the cost is <!--n hard_mode.none_over_both .0f-->16 times larger and the solve rate falls to **<!--n hard_mode.none.hard_solved .2f-->92.33%**, because a wide candidate set with no probe available is exactly the position the restriction forbids you to fix. Concentrate the mass and the candidate set collapses fast enough that the likely answers are decent probes anyway.

The losses are where it really shows. Even in the default row, where the mean barely moves, the loss probability roughly quadruples. Hard mode arrives in the one-letter families of *The penalty* below and then cannot spend a turn separating them, since the separating word is the one it is not allowed to play.

The opening word never changes, in any row. Every one of the 12,947 words carries prior mass, so at the first guess every word is still a possible answer and the restriction binds nothing at all.

Under `--mode entropy` the restriction **helps**, by <!--n hard_mode.entropy.both.cost +.4f-->−0.2504 guesses on the default prior (<!--n hard_mode.entropy.both.free .4f-->3.7569 → <!--n hard_mode.entropy.both.hard .4f-->3.5065). That is the identity under *Results* read backwards: what greedy entropy is missing is the bonus for guessing a word that could be the answer, and hard mode supplies it by force. It pays in losses, its solve rate falling from <!--n hard_mode.entropy.both.free_solved .2f-->100.00% to <!--n hard_mode.entropy.both.hard_solved .2f-->99.83%, and it still does not reach the expected-guesses policy's <!--n hard_mode.both.free .4f-->3.0641 — which gets the same bonus for free, and keeps the probes.

## Forms of f

Any basis linear in its parameters works, so every fit is a single least-squares solve. `estimator.FORMS` holds them, all written as `f(n) = base + sum_i params[i] * term_i(n)` with `x = log2(n)` and `base` either 0 or 1 (pinning `f(1) = 1`); the result is floored at 1.

f must not decrease anywhere in `[1, 12947]` — a value function that rewards a larger candidate set would let the search prefer uninformative guesses — so terms are dropped from the end (most flexible first) until it is monotone.

## Policy iteration

`train` alternates:

1. **evaluate** — play 2,000 sampled games with the current policy, recording for every position its effective size and how many guesses were actually needed from there;
2. **improve** — refit `f` to those pairs; acting greedily on the new `f` is the improved policy.

Round 0 is seeded with greedy entropy. Training games are played with no deadline, so targets are unbiased estimates of expected guesses and `f` keeps that meaning; the six-guess limit is a constraint layered on top.

The loop stops when the parameters settle, or when the measured mean stops improving (a greedy policy can cycle), and caches the parameters whose policy measured best — the form, opening word, fit history and holdout score, one file per prior combination in `estimator_cache/params_<slug>.json` (with `compare` archiving to `estimator_cache/forms/<slug>__<form>.json`). The slug names the shapes as well as the toggles, so a fit made under one shape can never be read back under another. A cached file is refused if it was fitted with different prior probabilities, or predates the form registry.

**A prior with no fit does not play.** `load_active` raises `NotFitted` and names the `train` command that fixes it. The alternative was to bend a fixed reference curve onto the chosen form and play that, and the invented curve was never a measurement of any prior: under the default one it read f(2000) ≈ 3.69 where the fitted `f` says 2.82. It flipped the opening word from `salet` to `raise`, cost 0.034 ± 0.004 guesses paired over 20,000 games, and was never saved, so every run paid for it again. Entropy mode, which never consults `f`, still runs on any prior.

The refusal carries the `train` command that ends it, so both front ends can hand it over as it stands: the terminal prints it and exits 2 rather than a traceback, and the page reports it as a 503 with the same command in it. Startup shows it on the veil instead of a stack trace, since there is nothing to debug.

### What the loop converges to

It does not converge to the best `f` it sees, which is why `policy_iteration` keeps the parameters whose policy measured best rather than the ones it stopped on — and that selection is itself a sample. The widest miss on the last refit was `<!--s fit_rounds.worst.slug-->repeat0_curation1_c-wordlebot<!--/s-->` fitted at `--exact-max <!--n fit_rounds.worst.reach-->60`, where the loop kept an `f` worth **<!--n fit_rounds.worst.kept .4f-->3.5023** guesses over every answer while one of its own rounds had produced **<!--n fit_rounds.worst.best .4f-->3.4903**, <!--n fit_rounds.worst.gap .3f-->0.012 better. Nothing guarantees otherwise: `f` carries at most six parameters and a whole candidate set is compressed to one scalar before it ever sees it, so the fixed point of the fit is nobody's optimum in particular.

That makes the selection statistic look like the weak link, and it is not. `best_train_mean_guesses` is a mean over the same 2,000 answers the round was fitted on, and the loop takes an argmin over half a dozen correlated draws of it, which is the shape of a selection that overfits itself. Hand every distinct `f` any round produced to `every_answer --form --params` instead (`analysis/fit_rounds.py`) — all 12,947 words, weighted by the prior, computed rather than sampled — and the kept fit is the exact best of its own run in **<!--n fit_rounds.kept_is_best-->7 of <!--n fit_rounds.runs-->10** runs across the five priors and both settings, giving up <!--n fit_rounds.worst.gap .3f-->0.012 guesses in the worst of the others. What there is to win there, `form_default --promote` collects: it ranks every archived fit and the active one by that same exact sweep, so the fit that ships is never the loop's sample estimate but the sweep's.

Round 0 is not a throwaway either. On `repeat1_curation0_r-binned` the `f` fitted from the entropy seed's rollouts outscores every later round at both settings, and it is what both runs keep.

### Warm starting from the previous fit (tried, rejected)

After `solutions.py --write` the previous fit is still sitting in `estimator_cache/`, so seeding round 0 with it rather than with greedy entropy looks like a free head start. It is not, because the entropy seed is the one round that costs nothing — entropy mode consults neither `f` nor the endgame search, so round 0 runs in about a second where an expected-mode round on a flat prior runs in thirty. Seeded at `--exact-max 60`, `repeat0_curation0` took the same four rounds and its loop went from 31.8 s to 59.0 s.

With the search off the seeding does save rounds, 29 to 24 across the five priors and 27.5 s to 23.2 s, and it lands somewhere slightly worse for them. Scored over every answer the warm-started fits average **3.5571** against **3.5511** cold, and nearly all of the gap is one prior: `repeat0_curation1_c-wordlebot` reaches 3.5241 and opens `raise`, where the cold run reaches 3.4953 and opens `reast`. Skipping the early rounds means skipping the transients, and the transients are where the good fits were.

What says the round count is not set by how far the seed sits from the answer is the third arm. Seeding with the *current* fit, the very `f` the run is trying to reproduce, saves nothing over seeding with one from a fortnight earlier: 24 rounds either way.

None of it is kept. The seed argument those three arms needed was measured and taken back out, so the figures above are facts about an implementation that no longer exists, as the turn-conditioned ones below are.

### Fitting in Rust

The whole loop has a native side. `core/src/train.rs` is `train.policy_iteration` term for term, `core/src/fit.rs` does the least-squares solve, and `core/src/rng.rs` reproduces `np.random.default_rng` — SeedSequence, PCG64 and `choice`'s own cumulative sum and right-hand search — so that both implementations draw the same training answers rather than merely similar ones. `wordle_solver.py train --rust` hands a fit over; `rust_train.py` declines rather than guessing where a setting has no flag, and `estimator.save_params` still writes the cache, so there is one writer and one format.

Two of the three pieces agree exactly. The sampler is bit for bit, over three seeds and all five priors, index by index. The fit is not, and cannot be. `np.linalg.lstsq` is LAPACK's SVD, which meets the rank-deficient case a spline routinely hands it — a hinge beyond the widest position seen is a column of zeros — with a minimum-norm solution, landing that column on exactly zero. `fit.rs` drops such a column before the solve, which is where the SVD would have left it, and takes the full-rank remainder by Householder QR; that is what a plain QR needs, its back-substitution dividing by a diagonal a dead column would leave at zero. The two agree to about 6e-13 across 27 dataset-and-form pairs and keep the same terms in every one. That makes it the second thing `tests/rust_parity.py` compares on a tolerance, after the letter odds.

**Parity stops at the seed round, and that is as far as it can go.** Given identical parameters the two play identically — the same guesses, the same rollouts, the same mean. But the first fit differs in its last bits, greedy policy iteration acts on that, and somewhere a tie breaks the other way: one rollout changes, the next fit changes, and from there the two walk different paths to different fixed points. Even the seed round's own holdout is downstream of it. So the Rust returns a *different valid fit* rather than the Python's — different, and measurably not worse. `analysis/rust_fit.py` fits every prior both ways over <!--n rust_fit.seeds-->3 seeds and scores each `f` over all 12,947 answers weighted by the prior: the native fits lose on <!--n rust_fit.worse-->0 of the <!--n rust_fit.pairs-->15 pairs. It is therefore the default, `--no-rust` fits here instead, and every cached fit has been the Rust's since the 2026-09-17 refit, along with every measurement that moves with a fit.

What it buys is **<!--n timing.train.gain .2f-->1.21×** over a whole `train` — <!--n timing.train.python .1f-->40.4 s against <!--n timing.train.rust .1f-->33.4 s for all five priors, best of two on an idle machine — which is well short of the <!--n timing.sweep.expected.python_over_rust .1f-->1.5× the same port gets on a sweep. The reason is where the time goes: a sweep spends two thirds of its Python time outside the kernels, where Rust is roughly twice as quick, while a training round is mostly kernel scans that are already parallel and already at memory bandwidth.

The sweep's other trick does not carry over at all. Playing several games at once is worth <!--n timing.sweep.expected.chunking .2f-->1.01× there and *costs* here — <!--n timing.train.round.chunks1 .3f-->0.847 s a round serially against <!--n timing.train.round.chunks2 .3f-->0.984 at two chunks and <!--n timing.train.round.chunks3 .3f-->1.175 at three, over the five priors at 2,000 games. Training turns the endgame search off, and the search is what leaves a core somewhere to go while a game is between scans; with nothing but scans left, splitting the memo three ways only makes each copy refill. Under `--exact-max 60` it reverses, three chunks taking the default prior's loop from <!--n timing.train.exact60.chunks1 .1f-->17.4 s to <!--n timing.train.exact60.chunks3 .1f-->14.0 s, so `train --chunks` follows `--exact-max` unless it is set outright.

## The exact endgame

Once the candidate set is small, or few turns remain, `exact.py` computes the value outright:

```
V(S, t) = min over g of [ 1 + sum_{r != GGGGG} p_r * V(S_r, t-1) ]
V(S, 0) = loss_penalty
```

with `t` the guesses left including this one. A loss penalty of 1 means running out costs one unit more than the six turns already spent, so a lost game scores 7. The deadline then needs no separate objective and no risk weighting.

V is non-*decreasing* in t only when a loss is priced at 1 or below: running out is then an exit costing exactly one guess, and no continuation finishes for less, so having turns left over is never an advantage. The bound the search prunes with is accordingly the two-turn value computed at `min(loss_penalty, 1)` (`ExactEndgame.bound_penalty`), which sits under every deeper value at the real penalty. Uncapped it would not: at a penalty of 5 the ordering reverses, and the ten `-atch` words are worth <!--n loss_penalty.bound.two_turn_at5 .3f-->1.914 with two turns left against <!--n loss_penalty.bound.three_turn_at5 .3f-->1.570 with three.

Four things make it affordable:

* **Partition dedupe.** Guesses that split S identically are interchangeable, so the branching factor is the number of distinct splits rather than 12,947. A guess that is itself a candidate always splits distinctly (only it produces `GGGGG`), so a possible answer is never merged with a probe. It only helps while S is tiny: by 12 candidates ~12.7k of the guesses already split differently.
* **Closed-form leaves.** One or two turns left, and one or two candidates left, are closed forms — `V(g, 2) = 1 + (A + (A − B)·penalty) / total` over the non-winning buckets, A their mass and B the sum of their maxima. One pass over the guesses, no estimator, exactly the cost of scoring them heuristically, so **two turns or fewer is always solved exactly** at any candidate count.
* **Memoization.** A candidate set is solved once and reused, since the same position is reached along many different orders of guesses. Two-turn positions are where this pays: four such calls in five are repeats of one already answered, and one position in a 300-game sample came up 373 times. Caching them instead of re-running the closed form is worth about 30% of the search's total time, which is why the memo is consulted *before* the two-turn branch and not after it.
* **Branch and bound.** Guesses are tried in order of their two-turn value — computed only for the distinct splits, which are about a quarter of the pool — and within a guess each bucket's bound is replaced by its exact value one at a time, abandoning as soon as the running total passes the incumbent.
* **Branch cap** (`--exact-branch`, default 50, `0` to disable). Only the best 50 splits by that bound are tried. Branch and bound finds the best guess early and can then spend an unbounded amount of time *proving* nothing beats it; the cap buys the answer and skips the proof. Over 1,275 real positions, caps of 25 and 50 pick the uncapped-optimal guess **every time** (a cap of 10 misses 5), while wall clock falls from 322 s to 8 s and the worst single position from 30 s to 1.9 s. That saving is rare rather than steady, which counting shows where timing cannot (`analysis/branch_cap.py`): over 150 real positions the cap changes no interior node at all, the same 184 being searched either way, because what prunes the search is the admissible bound and at 183 of those nodes it cuts the loop after one or two splits. At the remaining one the bound separates nothing and the uncapped search grinds through 8,781 splits to reach the guess it already had. So the cap is insurance against that node rather than a discount on the rest, and a figure timed over too few positions misses it entirely. Two guesses that split the candidates identically score identically — the same buckets, filled in the same order — so `SPLIT_TIE_TOL` is 0 and the dedupe that merges one guess under two spellings gates on equality; it was 1e-9 for headroom, and over half a million duplicate pairs the widest score spread inside a group is exactly zero. Bounds tie in bulk — one position holds 1,176 of its 1,177 splits at exactly 2.0 — so which fifty the cap keeps rests on the tie-break, and both implementations break it by word index for that reason. What counts as a tie is the bound rounded to `BRANCH_TIE_TOL`, which is 1e-11 against the ~1e-13 two reduction orders can open between the same bound computed twice: bounds a last bit apart land in one bucket and the index decides between them, so the two agree without the weights beneath them agreeing bitwise. Rounded and not floored, because those ties sit on whole values like the 2.0 above and a bucket edge through one of them splits the very cluster the rounding is there to gather — `analysis/branch_tie.py` sweeps eight decades of tolerance and finds one outcome throughout when the bucket is centred, and five when its edge is not.

Deeper than two turns the search is gated on `--exact-max` (default 60), with a node budget as a backstop that falls back to the heuristic. A typical search visits 1–20 nodes and costs 1–30 ms. The value `solve` returns is therefore an upper bound on the true one, and the search is optimal-within-the-cap rather than provably optimal. Correctness is checked against an independent brute-force implementation of the recursion over random candidate sets.

The runners-up come from the same search. `ExactEndgame.top` puts the k-th best value where the incumbent goes, so a guess is still abandoned the moment it cannot reach the list, and the words behind the pick carry true values rather than estimates — a ranking never mixes the two. That is not free, since a looser cutoff kills fewer guesses on their bound: over <!--n split_ranking.cost.positions-->200 real deep positions a top five costs <!--n split_ranking.cost.top5.mean_ms .0f-->18 ms a position against the pick's <!--n split_ranking.cost.pick.mean_ms .0f-->4 ms, and <!--n split_ranking.cost.top5.worst_ms ,.0f-->1,108 ms against <!--n split_ranking.cost.pick.worst_ms .0f-->141 ms on the worst one (`analysis/split_ranking.py`).

It also wants a wider cap than the pick does, which is `--exact-root-branch` (default 150). Five answers are not one answer: the winner sits far from the cap and the runners-up sit right against it, so a cap that never costs the pick anything still costs the list something. Measured against the uncapped search over two independent samples of 80 positions each, 50 leaves three lists of the first sample and one of the second naming a worse guess — by 0.0055 and 0.012 guesses — while **every winner is right at every cap tried**. Eighty is the smallest that comes back exactly uncapped on both samples, and 150 leaves the same margin over it that 50 leaves over 25. It applies only at the node being ranked; everything the search recurses into needs its winner alone and keeps the cap of 50. The page's widest endgame position pays 197 ms for it instead of 94 ms, then 14 ms on every repeat once the memo holds it.

Two of the figures above are the wider cap's: over the same 200 positions it takes a top five from about 12 ms to <!--n split_ranking.cost.top5.mean_ms .0f-->18 ms a position and from about 520 ms to <!--n split_ranking.cost.top5.worst_ms ,.0f-->1,108 ms on the worst, while the pick does not move at all. That price buys a tail nobody measuring the solver is measuring, so it comes back off with `WORDLE_ROOT_BRANCH=50` in the environment, or `--exact-root-branch 50` on a single run. Both implementations read the variable, which is what keeps them comparable under it. The site is the one build with no way to set it, and the one that wants the wider list anyway.

### What it is worth

From the first state the search takes over, the whole continuation is optimal, so it cannot be worse than the estimator in expectation. It is also barely better. Every word played as the answer under the default prior (`analysis/loss_penalty.py --search`):

<!--table loss_penalty.search.table-->
| policy | mean | loss probability | solve rate | wall clock |
| --- | --- | --- | --- | --- |
| greedy entropy | 3.7569 | 0.0034% | 99.9966% | 4 s |
| no exact search | 3.0646 | 0.0660% | 99.9340% | 4 s |
| exact for t ≤ 2 only | 3.0648 | 0.0408% | 99.9592% | 4 s |
| exact for m ≤ 10 | 3.0647 | 0.0407% | 99.9593% | 5 s |
| exact for m ≤ 24 | 3.0645 | 0.0398% | 99.9602% | 7 s |
| **exact for m ≤ 60 (default)** | **3.0641** | **0.0243%** | **99.9757%** | **9 s** |
<!--/table-->

The mean spans three ten-thousandths of a guess across that entire column, from no search at all to the default. What the search moves is the loss probability, and most of that movement is in the part that costs nothing: `t ≤ 2` alone takes <!--n loss_penalty.search.loss_none .4f-->0.0660% to <!--n loss_penalty.search.loss_two .4f-->0.0408%, <!--n loss_penalty.search.two_share .0%-->60% of the whole reduction, at no wall clock. Everything deeper buys the rest for two to three times the wall clock. That share is coarse, though, and moves with the fit rather than with the record: a loss probability this small is a handful of curated words, so a single one can be a third of the reduction. Which words they are is decided far from the endgame, by where the heuristic's earlier guesses route a word, and the search wins them back for the reason it exists. The heaviest word it wins back under the current fit is `<!--s loss_penalty.search.rescued.word-->fogey<!--/s-->`, at <!--n loss_penalty.search.rescued.weight .4f-->0.0140% alone <!--n loss_penalty.search.rescued.share .0%-->34% of the reduction, and the two policies part on it at turn <!--n loss_penalty.search.rescued.turn-->3 with <!--n loss_penalty.search.rescued.at-->40 candidates, where the heuristic plays `<!--s loss_penalty.search.rescued.heuristic.guess-->boney<!--/s-->` and the search `<!--s loss_penalty.search.rescued.search.guess-->chomp<!--/s-->`. Those leave the answer among <!--n loss_penalty.search.rescued.heuristic.left-->13 and <!--n loss_penalty.search.rescued.search.left-->12 words — much the same position to an estimator that sees a count — but the first is <!--s loss_penalty.search.rescued.heuristic.words-->`covey`, `gooey`, `cooey`, `cozey`, `fogey`, `hokey`, `homey`, `hooey`, `jokey`, `mopey`, `pogey`, `pokey`, `zooey`<!--/s-->, worth <!--n loss_penalty.search.rescued.heuristic.value .2f-->2.03 with <!--n loss_penalty.search.rescued.turns_left-->3 turns left, and the second is <!--s loss_penalty.search.rescued.search.words-->`woven`, `woken`, `boney`, `bogey`, `boxen`, `fogey`, `gonef`, `jokey`, `wowee`, `woxen`, `yogee`, `zowee`<!--/s-->, worth <!--n loss_penalty.search.rescued.search.value .2f-->1.29. A size-only estimator cannot tell a bucket no guess splits from one of the same size that a guess does; the exact search can, and <!--n loss_penalty.search.rescued.at-->40 candidates is inside its reach. Another fit routes other words and reads the share a third higher or lower, which is why it is quoted as *most* and not as a percentage worth keeping.

Per decision it is sharper, and this is the one place a valuation can be checked against the truth rather than against a better guess at it. Over <!--n exact_decisions.blended.positions-->600 positions the search takes over, weighted by how often play reaches them (`analysis/exact_decisions.py`), the estimator picks a different guess **<!--n exact_decisions.blended.differ .1f-->37.2%** of the time and gives up **<!--n exact_decisions.blended.regret .4f-->0.0076 guesses** by it. What that costs depends entirely on how much is left:

<!--table exact_decisions.blended.table-->
| candidates | positions | estimator differs | estimator regret | of which ties |
| --- | --- | --- | --- | --- |
| 3–4 | 134 | 35.4% | 0.0000 | all of them |
| 5–10 | 219 | 35.5% | 0.0066 | 28.3% |
| 11–60 | 247 | 39.2% | 0.0113 | 9.9% |
<!--/table-->

At three and four candidates every disagreement is a tie — a different word of equal value, where `f` cannot be improved on because there is nothing to improve. The ties thin out as the set grows, and by 11–60 most disagreements are real, which is where the <!--n exact_decisions.blended.regret .4f-->0.0076 comes from. Blended over the four priors it reads worse than it plays: under the default prior alone the estimator differs on only <!--n exact_decisions.shipped.differ .1f-->7.6% of games and gives up <!--n exact_decisions.shipped.regret .4f-->0.0004.

The search's own regret reads <!--n exact_decisions.blended.search_regret .4f-->0.0010 rather than zero, and that is the loss penalty rather than any limit of the search: it optimizes a penalized score while this column measures mean guesses. At `--loss-penalty 1` it is 0.0000 exactly, which also says the branch cap costs nothing at these sizes.

Refitting `f` under the search moves the curve by under 0.03 guesses everywhere (fit RMSE 0.63) and buys next to nothing in play. Scored over every answer rather than on a sample, the search-fitted `f` is worth **<!--n fit_rounds.search_fit.mean +.4f-->−0.0024 guesses** on average across the <!--n fit_rounds.search_fit.priors-->5 priors and <!--n fit_rounds.search_fit.worst +.4f-->−0.0120 at worst, <!--n fit_rounds.search_fit.identical-->4 of the five coming out identical, while the fitting loop takes about **<!--n fit_rounds.search_fit.loop_ratio .0f-->4×** as long. `train` and `compare` therefore both keep only the free `t ≤ 2` part; `--exact-max 24` or `60` fits under the full policy on either.

## Losing less

### Greedy entropy loses less, not more

Every word played as the answer, under both priors:

<!--table loss_penalty.policies.table-->
| policy | mean | loss probability | solve rate |
| --- | --- | --- | --- |
| greedy entropy | 3.7569 | 0.0034% | 99.9966% |
| expected, no exact search, penalty 1 | 3.0646 | 0.0660% | 99.9340% |
| expected, exact-max 24, penalty 5 | 3.0645 | 0.0398% | 99.9602% |
| **expected, exact-max 60, penalty 5 (default)** | **3.0641** | **0.0243%** | **99.9757%** |
<!--/table-->

Entropy is not lossless — 72 words beat it — but it loses seven times less often than the estimator, and the two are not failing on the same words: they share 25. What protects entropy is that it cannot see the prior. Separating candidates is what maximizes information, and separating is also what never gambles; the estimator backs the prior instead, and pays for it when the prior is confidently wrong, which *The penalty* below takes apart.

The two thirds of a guess the estimator wins are therefore not free. They cost about <!--n loss_penalty.policies.loss_ratio .0f-->7 times the loss rate, and `--loss-penalty` is the dial that buys some of it back. Neither rate is large enough for the default 200-game `benchmark` to notice: it finds zero losses <!--n loss_penalty.policies.benchmark_clean.entropy .1f-->99.3% of the time under entropy and <!--n loss_penalty.policies.benchmark_clean.default .1f-->95.3% under the default, which is why both look lossless there.

Wordle is provably solvable in five guesses, but that theorem is about a solver that knows the answer is one of the curated 2,309. This one hedges over all 12,947, and 9% of the prior mass sits on the 10,638 never-curated ones, which is where the losses collect: of the <!--n loss_penalty.policies.default_lost-->313 words the default prior can lose on, <!--n loss_penalty.policies.default_lost_uncurated-->301 are never-curated — words like `aahed`, `anele`, `avyze`, `bizzy`. Playing all 2,309 curated words as the answer instead:

<!--table loss_penalty.curated.table-->
| policy | mean | worst | losses |
| --- | --- | --- | --- |
| greedy entropy | 3.9272 | 6 | 0 / 2309 |
| expected (defaults) | 3.7731 | 7 | 12 / 2309 |
<!--/table-->

The estimator wins the mean here too, and takes <!--n loss_penalty.curated.losses-->12 losses doing it. Every one of them — <!--s loss_penalty.curated.lost_words-->`brook`, `catty`, `foggy`, `handy`, `hater`, `holly`, `irate`, `joker`, `mushy`, `nicer`, `ratty`, `water`<!--/s--> — is a word that has already been an answer (<!--n loss_penalty.curated.lost_used-->12 of the <!--n loss_penalty.curated.losses-->12), which is the repeat prior working rather than failing. The <!--n regime.curated_unused-->474 still-unused curated words hold 81% of the mass while the <!--n regime.curated_used-->1,835 already used share 10% of it, so sweeping the curated list uniformly asks the solver to do well on a distribution its own prior says is unlikely. Greedy entropy dodges that by ignoring the prior altogether — the same blindness that costs it six tenths of a guess everywhere else.

The six-guess games in that table are wins rather than near-misses: they are what hedging over 12,947 words costs, and closing them would take a full-depth optimizer over the restricted set.

### The penalty

`--loss-penalty` defaults to **5** rather than the risk-neutral 1, and the reason starts with what a loss actually is. It is not an endgame blunder: at the sizes these positions reach, `exact.ExactEndgame` solves them outright, so the guess played is the optimal one. The solver is backing its prior. Holding `batty`/`catty`/`natty`/`patty`/`ratty`/`tatty` with four turns left it plays `tatty`, which carries 75% of that family's mass, and it declines `bronc` — a word splitting the six of them five ways — because separating spends a turn to insure against a 0.4% case. A loss is the prior being confidently wrong, and the penalty is what makes the insurance worth buying anyway.

Every word played as the answer, under each of the four prior combinations, blended equally:

<!--table loss_penalty.penalty.table-->
| `--loss-penalty` | loss probability | mean (solved) | mean (loss = 7) |
| --- | --- | --- | --- |
| 1 | 0.1159% | 3.6546 | 3.6582 |
| 2 | 0.0774% | 3.6558 | 3.6582 |
| **5 (default)** | **0.0533%** | **3.6569** | **3.6586** |
| 10 | 0.0441% | 3.6577 | 3.6591 |
| 30 | 0.0416% | 3.6583 | 3.6596 |
<!--/table-->

<!--n loss_penalty.penalty.first_to_default.loss_drop .0f-->54% of the loss probability disappears between 1 and 5, for **<!--n loss_penalty.penalty.first_to_default.mean +.4f-->+0.0004 guesses** on the penalized mean. Opening words are untouched, since the root never reaches the exact search.

That last column cannot settle which penalty is right, and it is worth seeing why: scoring a loss at 7 is exactly the objective `--loss-penalty 1` optimizes, so ranking settings by it can only favor risk neutrality. Name the price of a loss instead — L guesses — and each setting becomes a line in L, so the choice falls out of where the lines cross:

<!--table loss_penalty.crossover.table-->
| a loss is worth | best penalty |
| --- | --- |
| under 7.1 | 1 |
| 7.1 – 8.6 | 2 |
| **8.6 – 12.6** | **5** |
| 12.6 – 24.8 | 10 |
| over 24.8 | 30 |
<!--/table-->

5 is right if a lost game is worth between nine and thirteen guesses, two to seven worse than scraping a win at the wire.

One end of that is measurable rather than arguable. Scoring a loss at 7 assumes the lost game needed exactly one more guess; lift the deadline, carry each losing game on from the position it died in, and the solver needs **<!--n loss_cost.penalty5.extra .1f-->1.2 more**, occasionally <!--n loss_cost.penalty5.worst-->4 (`analysis/loss_cost.py`). The risk-neutral price of a loss is therefore about **7.2** — past the point where penalty 1 gives way, and nowhere near 5. Priced honestly, risk neutrality endorses 2. Choosing 5 is the separate claim that a failed puzzle costs more than the guesses it would have taken, which is a judgment about being scored out of six rather than counting forever, and no continuation can measure it.

Little rides on the choice: anywhere in that range the best setting beats the next by under 0.0002 guesses.

### Widening the search's reach

The second lever is `--exact-max`, how large a candidate set the exact search will take on. At the default penalty, again over every word and every prior:

<!--table loss_penalty.reach.table-->
| `--exact-max` | loss probability | mean (loss = 7) |
| --- | --- | --- |
| 24 | 0.0848% | 3.6617 |
| 40 | 0.0595% | 3.6597 |
| **60 (default)** | **0.0533%** | **3.6586** |
| 80 | 0.0473% | 3.6580 |
| 100 | 0.0453% | 3.6575 |
<!--/table-->

This lever costs nothing in guesses: reaching further replaces an estimate with a true value rather than trading one thing for another, so the mean improves all the way down the column. What it buys falls off sharply, though. Going 24 → 60 takes <!--n loss_penalty.reach.first_to_default.loss_drop .0f-->37% of the loss mass; 60 → 100 takes a further <!--n loss_penalty.reach.default_to_last.loss_share .0%-->9% of it and <!--n loss_penalty.reach.default_to_last.mean .4f-->0.0010 guesses.

**Under the default prior it buys nothing at all.** Held at 60, 80 and 100 the loss probability is <!--n loss_penalty.reach.shipped.60.loss .4f-->0.0243%, <!--n loss_penalty.reach.shipped.80.loss .4f-->0.0241% and <!--n loss_penalty.reach.shipped.100.loss .4f-->0.0240% and the mean <!--n loss_penalty.reach.shipped.60.mean .4f-->3.0641 to <!--n loss_penalty.reach.shipped.100.mean .4f-->3.0640; reaching to 100 rescues <!--n loss_penalty.reach.shipped.rescued-->6 words out of 12,947 and leaves <!--n loss_penalty.reach.shipped.100.curated-->12 of the <!--n loss_penalty.curated.losses-->12 curated losses standing. Every gain in the table above comes from the flat and curation-only rows. The families that beat the solver — `water`/`hater`, `catty`/`ratty` — are not positions a deeper search can crack, because nothing is wrong with how they are being played; see *The penalty*.

What stops it at 60 is time, and only the flat prior pays that. Reaching to 100 runs <!--n loss_penalty.cost.flat.100.ratio .1f-->3.4× slower there, and the node budget starts firing — <!--n loss_penalty.cost.flat.100.bailouts-->5 bailouts in four hundred games, the first of them already at <!--n loss_penalty.cost.flat.first_bailout-->80, each one a position that falls back to the heuristic instead of being solved, so past 60 the "exact" search is quietly less exact. Under the default prior the curve stays gentle, 100 costing <!--n loss_penalty.cost.shipped.100.ratio .2f-->1.29× what 60 does against the flat prior's <!--n loss_penalty.cost.flat.100.ratio .2f-->3.40×, because concentrated mass means the candidate set rarely gets that big. The penalty drives none of this: 1, 5 and 30 run within <!--n loss_penalty.cost.flat.penalty_spread .0f-->4% of each other at either reach, which is worth saying because the bound is capped at `min(penalty, 1)` while true values scale with the penalty, and that asymmetry might have been expected to show up as cost. It does not.

Between the penalty and the wider reach, the loss probability falls from <!--n loss_penalty.corner.loss .4f-->0.1356% to <!--n loss_penalty.penalty.5.loss .4f-->0.0533% — **a <!--n loss_penalty.corner.loss_drop .0f-->61% reduction** — and the mean *improves* by <!--n loss_penalty.corner.mean_gain .4f-->0.0028, the reach gaining more than the penalty spends.

## Lookahead (`--lookahead`, off by default)

Above the exact search sits an optional expectimax with a branch cap of its own. Only the guess is pruned, never the feedback — dropping a bucket would leave the sum something other than an expectation:

```
V_d(S, t) = min over the best `--lookahead-branch` guesses of
            [ 1 + sum_{r != GGGGG} p_r · V_{d-1}(S_r, t-1) ]
V_0(S, t) = min_g E(g, S)
```

with the exact search taking any sub-position it can reach. This is the exact search's `--exact-branch` device one level up: a fan-out at each node rather than a frontier held to a constant size, so every kept guess expands into every one of its buckets and each further ply multiplies the work by the cap again. That growth, not a constant factor, is where the 205 s below comes from.

Over <!--n lookahead.games-->20,000 paired games (<!--n lookahead.per_prior-->5,000 per prior, identical answers, `analysis/lookahead.py`), one ply at a cap of 10 is worth **<!--n lookahead.delta-->−0.0083 ± 0.0039** guesses for roughly 10× the wall clock. At <!--n lookahead.sigma .1f-->2.1σ that is barely resolvable — and it is eight thousandths of a guess. The arms are <!--n lookahead.repeat0_curation0.delta-->−0.0124 ± 0.0068 with no priors, <!--n lookahead.repeat0_curation1_c-wordlebot.delta-->+0.0006 ± 0.0108 and <!--n lookahead.repeat1_curation0_r-binned.delta-->−0.0198 ± 0.0082 with one each, and <!--n lookahead.repeat1_curation1_r-binned_c-wordlebot.delta-->−0.0014 ± 0.0030 under the default prior — inside a standard error of nothing, on the one configuration that ships. What the pooling buys is resolution, and what it resolves is a quantity too small to want. A cap of 25 measured no better, the opening word stays `salet` at every cap from 5 to 40, and two plies costs 205 s for the opening position alone.

Its value is not a fixed property of the solver: what lookahead measures is the room the base policy leaves above itself, so a better-fitted `f` should shrink it. That mechanism is not visible at this resolution — the last refit improved every arm's base policy and left lookahead worth marginally more, not less — which is what an effect of eight thousandths against errors of four looks like. Read the figure against a run of the same vintage, never against a published one.

The interesting part is that it barely matters *despite* valuing positions far better. The policy is deterministic, so the true expected guesses from a position can be computed rather than estimated — play it out once per candidate answer, weighted by the prior. Each valuation is scored against what its own play achieves, over 222 positions from real games, weighted by how often play reaches them (`analysis/lookahead_accuracy.py`):

<!--table lookahead_accuracy.blended.table-->
| valuation | true value | bias | mean abs error | RMSE |
| --- | --- | --- | --- | --- |
| `min_g E(g, S)` (what the solver scores positions at) | 2.7545 | +0.0981 | 0.1274 | 0.1580 |
| one-ply lookahead | 2.7454 | +0.0026 | 0.0044 | **0.0138** |
<!--/table-->

Thirteen times more accurate, and near enough unbiased. Those are the positions where the heuristic is the one deciding, which means turns two and three; by the fourth the candidate set is under `--exact-max` and both valuations return the same true value.

Accuracy in the level is not what drives play. A valuation wrong by a constant at a position still orders the guesses perfectly and costs nothing, so what matters is the ranking *within* a position. Playing out the true value of the ten guesses lookahead re-ranks, at those same positions:

<!--table lookahead_accuracy.decisions.table-->
|  | blended | none | curation | repeat | default |
| --- | --- | --- | --- | --- | --- |
| spread of true value, best to worst | 0.0989 | 0.0946 | 0.0779 | 0.0949 | 0.1280 |
| `min_g E(g, S)` regret | 0.0102 | 0.0146 | 0.0078 | 0.0151 | 0.0034 |
| one-ply lookahead regret | 0.0015 | 0.0018 | 0.0031 | 0.0008 | 0.0002 |
| the estimator picks the best | 56.4% | 38.1% | 64.5% | 42.5% | 80.5% |
| lookahead picks a different guess | 42.8% | 51.1% | 46.3% | 54.1% | 19.5% |
| rank correlation with the truth | +0.553 | +0.450 | +0.538 | +0.493 | +0.731 |
<!--/table-->

The estimator orders those ten barely better than a coin flip when the prior is flat, and much better than that under the default one, where its pick is the best available five games in six. Lookahead disagrees in two games in five, changes the length of a <!--n lookahead.changed_share .0f-->24% of them (<!--n lookahead.better-->2,446 better against <!--n lookahead.worse-->2,311 worse), and is right to disagree — <!--n lookahead_accuracy.decisions.look_regret .4f-->0.0015 given up against <!--n lookahead_accuracy.decisions.base_regret .4f-->0.0102. Better valuation does buy better decisions here, under every prior.

What it barely buys is guesses, and the first row is why. Those ten spread <!--n lookahead_accuracy.decisions.spread .4f-->0.0989 between best and worst, and the estimator's pick gives up a tenth of that. The top of the ranking is a plateau: the estimator finds it reliably and cannot tell one spot on it from another, which costs a hundredth of a guess because there is little there to tell apart. Recovering most of it per decision still reads as <!--n lookahead.delta-->−0.0083 ± 0.0039 over a whole game, a game only meeting such a position where the exact search has not already taken over.

**This is the recurring result in this project**: a valuation improvement that is real, large, and correctly converted into decisions still buys almost nothing. It holds for the deep exact search, for one-ply lookahead, for turn-conditioned `f`, and for the sharpened repeat prior above — though only lookahead's has been taken apart at the level of the decision, the other three measured at the outcome.

### Conditioning f on turns remaining (tried, reverted)

`f(n)` estimates guesses-to-finish with no turn limit, so a position with four turns left is scored as one with two. Fitting `f` once per turn count produces curves with a large turn effect — a 2,000-candidate bucket worth 4.97 with three turns left against 3.04 with five — and changes nothing.

It could not have done. Where those curves differ is where the deadline binds, and the exact search has taken that decision long before. Counting every decision in 1,600 games (`analysis/exact_decisions.py --who-decides`):

<!--table exact_decisions.who.table-->
| turns left | `f` | exact search | closed form | one candidate |
| --- | --- | --- | --- | --- |
| 6 | 1,600 | 0 | 0 | 0 |
| 5 | 1,298 | 292 | 4 | 6 |
| 4 | 97 | 1,116 | 128 | 191 |
| 3 | 0 | 309 | 133 | 449 |
| 2 | 0 | 37 | 28 | 165 |
| 1 | 0 | 0 | 0 | 32 |
<!--/table-->

`f` takes **<!--n exact_decisions.who.f_late-->0** decisions in the <!--n exact_decisions.who.late-->1,153 positions with three or fewer turns left; the <!--n exact_decisions.who.f_at_four-->97 at four turns left are the last it ever sees. Every row where a turn-conditioned curve would say something different is a row nobody reads. The outcome measured when it was tried, over 80,000 paired games, was losses 81 → 79 and a mean of +0.0005 ± 0.0007 — a fact about an implementation since removed, where the table above is a fact about the solver as it stands.

### Conditioning f on the pattern (tried, rejected)

`f` reads one scalar, so ten candidates in the `-atch` family and ten well-separated ones are the same position to it. The feedback pattern is the one structural feature already in hand where `f` is evaluated — `kernels._expected_guesses` is iterating over exactly the `r` of `E(g, S) = 1 + sum over r != GGGGG of p_r * f(n_r)` — so `f(n) + offset[class(r)]` would cost one read from a 243-entry table and no rescan of anything.

The bias it would correct is real and large. Fitting the offsets alongside `f` on 2,000 games of rollouts, with the green count entered as dummies and depth controlled for (`analysis/pattern_offsets.py`), a bucket reached by three greens needs **<!--n pattern_offsets.repeat1_curation0_r-binned.greens3-->+0.28 ± 0.06** guesses more than its effective size predicts under the repeat-only prior and <!--n pattern_offsets.repeat0_curation0.greens3-->+0.16 ± 0.07 under no prior at all, rising with the green count throughout. The three priors carrying a curation shape show about 0.05 and nothing worth chasing.

Play gets worse anyway. Hold `f` at the shipped fit and scale the correction from nothing up to the fitted value, and the mean over every answer climbs at every step on four of the five priors — `repeat0_curation0` goes 4.0699 → 4.2473 and its opening from `tares` to `estop` — while the fifth moves 0.0006 the other way. Refitting `f` jointly with the offsets rescues none of it.

Gating the offset by bucket size says why, and the answer is extrapolation rather than a wrong correction. Applied only to buckets of 30 candidates or fewer, which is where the residuals were measured — a four-green bucket averages four candidates — it costs 0.0020 guesses averaged over the five priors and helps one of them. Applied to buckets of hundreds, which the residuals never sampled and which the widest scans are made of, it costs up to 0.18. A constant of +0.45 guesses is a small thing against a four-candidate bucket and an enormous one against a five-hundred-candidate bucket, and the ranking at the opening is decided by exactly the latter.

What the correction does buy, slightly and on all five priors, is a lower loss rate at the tightest gate. That trade already has a dial in `--loss-penalty`, which does not need 243 floats to turn.

## The opening is not where the estimator is weak

Every game shares one position, and `f` decides it alone — the candidate set is the whole list, so nothing reaches down to it. It is also the widest position there is, which is where `f` ranks worst. Playing out the two hundred openings it likes best, once per answer and weighted by the prior (`analysis/opening.py`):

<!--table opening.table-->
| ranks by `f` | mean true value | worst | best in band |
| --- | --- | --- | --- |
| 1–10 | 3.0897 | 3.1057 | **salet 3.0641** |
| 11–25 | 3.1059 | 3.1317 | slate 3.0916 |
| 26–50 | 3.1197 | 3.1365 | carle 3.1037 |
| 51–100 | 3.1239 | 3.1530 | trine 3.0923 |
| 101–200 | 3.1354 | 3.1689 | sient 3.1093 |
<!--/table-->

`<!--s opening.pick-->salet<!--/s-->` is the best of the <!--n opening.openings-->200 and **nothing beats it**: the opening costs <!--n opening.regret .4f-->0.0000 guesses of regret. Both columns climb with rank, the best in a band as steadily as the mean, so a rival from further down would have to beat every word in ranks 11–200 while `f` ranked it below all of them.

What makes that worth stating is how badly `f` orders the very same words. Across the top twelve its rank correlation with the truth is **<!--n opening.rho_top12 +.3f-->+0.448**; across the top twenty-five it is <!--n opening.rho_top25 +.3f-->+0.431, and across the top two hundred **<!--n opening.rho +.3f-->+0.679**. It cannot separate near-ties and it separates a range perfectly well — coarse discrimination and fine ordering are different jobs, and picking an opening needs only the first.

## Perfecting the second guess (tried, not shipped)

With the opening optimal and the exact search owning every turn from the fourth, whatever the estimator still costs has to be in the second guess. That decision is small enough to price outright rather than sample. The opening never varies, so the entire turn-2 policy is one guess per feedback pattern — <!--n turn2_regret.positions-->193 reachable positions, of which the endgame search already solves <!--n turn2_regret.solved-->142 outright, leaving <!--n turn2_regret.decided-->51 where `f` decides alone. `analysis/turn2_regret.py` plays out the true value of the fifty best guesses at each of those (`common.guess_value`, one playout per candidate answer) and weights each position by its prior mass, which is exactly the probability play reaches it. The estimator gives up **<!--n turn2_regret.census .5f-->0.00336 guesses**, at <!--n turn2_regret.moved-->11 positions of the <!--n turn2_regret.decided-->51.

Substituting those decisions and sweeping every answer pays precisely that, <!--n turn2_regret.shipped_mean .5f-->3.06409 → <!--n turn2_regret.override_mean .5f-->3.06071 against a census that predicted <!--n turn2_regret.census .5f-->0.00336. Losses move the other way, <!--n turn2_regret.shipped_loss .4f-->0.0243% → <!--n turn2_regret.override_loss .4f-->0.0400%, since a playout carries no deadline and so prices a sixth guess as freely as a second.

Then it meets a distribution it was not chosen under, and the sign does not hold:

<!--table turn2_regret.table-->
| distribution | shipped | turn-2 optimum | delta |
| --- | --- | --- | --- |
| the default prior (what chose them) | 3.0641 | 3.0607 | **−0.0034** |
| uniform over all 12,947 | 4.4877 | 4.4835 | **−0.0042** |
| uniform over the curated list | 3.7731 | 3.7809 | **+0.0078** |
| uniform over the still unused | 2.9368 | 2.9305 | **−0.0063** |
| the 228 real regime answers | 4.0658 | 4.0614 | −0.0044 ± 0.0211 |
| all 1,917 real answers ever | 3.9854 | 3.9969 | +0.0115 ± 0.0074 |
<!--/table-->

Read the two real-answer rows as differences and not as levels. Both arms replay those answers under *today's* prior, in which every one of them is a word already used and priced at repeat weight, where `analysis/replay.py` rebuilds the prior from strictly earlier puzzles each day and reports the <!--n replay.priors.both.mean .4f-->3.1579 that *What each prior is worth on real answers* carries. The levels near 4.0 are that handicap and not a regression; the paired difference is unaffected, both arms seeing the identical prior.

Those two rows fail to resolve it — <!--n turn2_regret.regime.delta-->−0.0044 ± 0.0211 over the regime and <!--n turn2_regret.alltime.delta-->+0.0115 ± 0.0074 over every answer ever, the answers the NYT really published leaning against the substitution without settling it. The enumerated rows are exact and they disagree among themselves, which is the finding. Optimizing the second guess against one weighting of the word list makes it worse under another.

Two things say why. The gain is the residue of a reshuffle rather than an improvement: **<!--n turn2_regret.churn-->1,365 of 12,947** answers change length, carrying <!--n turn2_regret.churn_mass .1f-->4.2% of the prior's mass, and <!--n turn2_regret.net +.4f-->−0.0034 is what survives of <!--n turn2_regret.improves .2f-->2.26% of the mass improving against <!--n turn2_regret.worsens .2f-->1.91% worsening. Set beside the <!--n turn2_regret.sd .2f-->1.30-guess standard deviation of the per-answer changes it rides on, the net is <!--n turn2_regret.net_fraction .4f-->0.0026 of it. And the reversal does not need the whole distance: blend the default prior toward uniform-over-curated and the substitution has stopped paying <!--n turn2_regret.blend_crossing .2f-->0.30 of the way there.

The per-override breakdown separates the two kinds. Some overrides help or cost nothing under every enumerated weighting; `<!--s turn2_regret.flip.word-->indol<!--/s-->` reads <!--n turn2_regret.flip.default +.4f-->−0.0002 under the default prior and **<!--n turn2_regret.flip.curated +.4f-->+0.0065** under the curated one, a sign flip and many times the magnitude. <!--n turn2_regret.everywhere-->5 of the <!--n turn2_regret.moved-->11 help under all four, and keeping only those is worth <!--n turn2_regret.everywhere_gain +.5f-->−0.00015 — but that rule is itself a fit to the distributions it was checked against, so what it really shows is how little of the <!--n turn2_regret.paid .4f-->0.0034 was ever attributable to anything but the prior's particular weighting.

So the recurring result stated under *Lookahead* above arrives here in its strongest form. The cases collected there each show a better valuation failing to buy guesses; this one gives the ceiling directly. **Playing the last position the estimator decides perfectly is worth less than the error in the prior it would be perfect for**, which bounds every valuation improvement that could ever be made there — no search depth, no extra ply and no better-fitted `f` reaches past <!--n turn2_regret.paid .4f-->0.0034. What remains is a better model of the editor, and *The prior sits in a flat basin* is the measurement saying that lever has been pulled too.

## Playing it yourself

The page's *Play* view is the one to reach for. This is the same game in a terminal, and the source of the two pieces `web.py` imports to build that view: `wordle_game.TILE_BLOCK` and `wordle_game.pick_random_answer`.

```
python wordle_game.py                  menu: type an answer, or draw one from a prior
python wordle_game.py --answer crane   play against a known answer
python wordle_game.py --random         draw the answer from the prior
```

During a game, `?` shows the solver's recommendation (Enter accepts it) and `quit` gives up. `--assist` shows it every turn, `--auto` hands the game over, and `--seed` makes a `--random` answer reproducible. The game only reads the word lists — the 167 MB feedback table is loaded lazily, and only for a hint.

The board is recapped at the end as colored blocks. `--share` adds the emoji grid underneath it, which is the one worth pasting somewhere else.

## In a browser

```
python web.py                 serve on http://127.0.0.1:8000 and open a browser
python web.py --port 9000     somewhere else
python web.py --no-browser    do not open one
```

Both terminal front ends are there, plus one thing that has no terminal equivalent. The group beside *Settings* says which of the three the page is doing. It marks where you are, not where the button will take you.

The two mappings below explain the page by way of the simpler thing. They are not directions to the terminal, which is where the solver is fitted rather than used.

**Solve** is `wordle_solver.py play`. You enter the feedback by coloring tiles instead of typing `NYGNN`, and the prior toggles, `--mode`, `--form` and `--hard` all become switches. *Shape of f* leaves the choice to the cache by default and marks which form that is, since the winning form is fitted per prior and moves with the toggles. Alongside the board is a panel of alternatives and still-possible candidates, with the prior, the objective and the fitted f under *Advanced*. Each of those comes with an explanation of what it is doing, and the rates quoted in it are measured from `solutions.json` at the moment the board is built. Nothing is written into the page, so the explanation can't fall behind the answer record the way a sentence quoting "about 10%" would.

*Set answer*, under the Solve board, is the solver tested against a word: `wordle_game.py --auto` with the whole panel showing. With an answer in the box the server colors every row from it as you submit and the tiles stop being an input, so there is no pattern to enter wrong; Enter on an empty row plays the solver's pick, *Play it out* presses it until the game ends, and *Draw random answer* deals one from the prior the way *Play* does. What sets it apart from watching *Play* with *Assist* on is that the board stays editable: delete a row, type another word there, and the rest of the game is re-solved and recolored from that point. The answer rides along with every board request like a switch does, so the switches keep re-running the solver over the same rows while the colors, being the truth, stay put.

**Play** is `wordle_game.py`. The *Answer* box takes a word of your own, which is the quickest way to set the solver a position, and it is the only setting that deals a new game — an answer already dealt cannot be given a different one. *Hint* is the terminal game's `?` and presses again to close, the *Assist* switch is `--assist` (Enter then plays its pick, which is `--auto`), *Reveal answer* is `quit`, and the emoji grid at the end is `--share`. The answer stays on the server until the game is over, so the network traffic does not spoil it.

The panel beside the board holds three cards, and all three fold away. They start closed.

*Solver's pick* holds the word the solver would play here and its score. *Alternatives* holds the next four behind it; clicking one of those words puts it in the row you are typing. Both cards open when you turn *Assist* on, and both open when you press *Hint*. Neither is ever closed for you, so a card you opened with *Assist* off is still open on your next turn.

Those five are five different *guesses*, which late in a game is not the same as five different words. Once few candidates remain, many words give identical feedback against every one of them, and a ranking of words is then one guess spelled several ways: at four candidates or fewer the five best words are 4.7 different guesses, each standing for a hundred and eighty-odd others (`analysis/split_ranking.py`). So the ranking names one word per distinct split, and a `+3` beside it opens the rest of that split as chips — play any of them instead and the game goes exactly as it would have. A word that could itself be the answer never has any, since only it comes back all green. The pick carries the same list, under the score.

*Letter frequency* sits above those two and *Assist* doesn't touch it. You open and close it yourself, and it stays as you left it for the rest of the game. Each letter carries the chance that the answer contains it somewhere: the prior's weight on the remaining candidates that hold the letter, over the prior's weight on all of them. On an empty board E is at the top with about 40%. Vowels are listed above the other letters.

The prior doing the weighting is the one the answer was drawn from, not the one the switches currently name, so turning a prior off mid-game rethinks the hint but leaves these numbers alone. A letter that no remaining candidate holds drops off the list, since 0% says nothing the gray keys have not said already. Letters at 100% stay and sort to the top, so late in a game the list opens with the ones the answer is known to contain.

**Find** is the page's own, and it runs backwards from everything else here. The answer is known and the guess is the unknown. Type an answer, color the pattern that a guess came back as, and you get every word in the list that would have produced that pattern against that answer. It is the question you ask of somebody else's share grid. Neither row describes the other. That is why there is no ghost word and nothing is grayed out, and why a gray tile in the pattern is a color rather than an empty tile waiting for a letter.

The match itself is exact and immediate, since the answer is one column of the feedback table and finding the words is a comparison over it instead of a scoring pass. The three numbers each word carries take more work:

* **Words left** is how many candidates would survive that guess *given this pattern*, so it ranks the words with hindsight — smallest first is what would have told you the most about this particular answer.
* **Bits** is expected information gain, averaged over every answer the guess could have faced. It does not depend on the pattern at all, so it ranks the words as openings instead.
* **Chance** is the prior's own probability that the guess is itself the answer.

The last two move with the prior switches. Nothing else on the page reaches the finder at all, which is why the mode, the shape of f and hard mode are hidden while it is showing. The finder never asks for a guess, needs no fitted `f`, and works under any prior.

| key | solve | play | find |
| --- | --- | --- | --- |
| A–Z | write a letter into the selected tile | write a letter | write a letter of the answer |
| ← → | move along the row | move along the row | move along the row |
| ↑ ↓ | move between rows, never past the row being played | — | move between the answer and the pattern |
| Space | change the selected tile: gray → yellow → green | — | change the selected pattern tile |
| Enter | on an empty row take the suggestion, otherwise submit | guess, or play the solver's pick when it is showing | find every word that fits |

Going back up to an earlier row and correcting it is the point of ↑. Every submit sends the whole board, so a tile fixed four rows up re-runs the solver from there. You never have to delete the rows under it first.

### Editing the board

The page keeps its own copy of every row, and typing and coloring are local until Enter sends the whole board back. The solver only ever sees complete boards, which is what makes correcting an earlier row the same operation as playing a new one. Rows the solver has not been told about yet are drawn with a dashed border.

Taking a word off is the exception, and goes back on its own. Writing one is a proposal, which is what Enter is for; removing one is not, since a board cannot half-forget a word it is no longer showing. However the word goes — *Delete row*, or backspaced away letter by letter — what remains has to be the board those words alone would have made, down to the shadow and every number in the panel.

Five rules fall out of that, and they are the answers to the things that used to go wrong:

**A color belongs to the letter under it.** Clearing a letter clears its color, and typing a different letter into a tile resets that tile to gray. Otherwise a row typed over an old one comes back wearing the old one's feedback — a letter turning yellow for no reason because the word before it had a yellow there. Space on an *empty* tile does nothing for the same reason: the color would not show, and would then belong to whatever was typed there next.

**Typing at the end of a full word replaces the last letter**, as it does in any other box that is already full. A stray keypress costs one letter, not the word. Retyping a rejected guess from scratch therefore means clearing the row first.

**Backspace only ever deletes leftwards** — the tile under the cursor, then the one before it. Letters to the right of the cursor are left alone, so the first few letters of a row can go without disturbing the last ones. The backspace that empties the row is the one that takes the word off the board, and it counts as a deletion: a row cleared by hand is a row deleted, and the two cannot look different afterwards.

**Delete row empties the row in place and leaves the gap.** The rows under it stay where they were played, so the word that should have been there goes straight into the hole. Submit with the gap still empty and you get *Row 2 is empty* back. Nothing closes up and nothing slides.

The refusal is deliberate. A word only reaches the board because it was played, which makes a row that needs deleting nearly always a row that needs *replacing*; closing the gap would quietly claim you played those later words earlier than you did. The position is not free to move either. Both the endgame search and the estimator are fed `guesses_left = MAX_GUESSES - len(rows)`, so shifting rows up would change what the solver recommends.

An empty row with nothing under it is a different thing — it is the row waiting to be played, so it is simply not sent. That is what makes *Delete row* on the last guess an undo, and what clears the board back to the opening recommendation. Only that gap-free case goes back on its own; a hole higher up waits for the word that fills it, because the board it makes is one the solver refuses anyway.

**The suggestion is a shadow, not a row.** The solver's pick is drawn faint in the row after the last real word, and it follows that word rather than standing at a fixed height. Take the last guess off and the board goes back, so what comes up into the vacated row is the pick for the shorter board — the board is a row shorter and says so. *Delete row* cannot touch the shadow itself, there being nothing there to delete. Take it — click it, or press Enter on an empty row — and it stops being a shadow: it is then a word in a row, and deletes like any other.

There is exactly one free row, and the shadow is in it. Otherwise a row emptied by hand leaves a second free row under it, still selectable and sitting past the end of the board the shadow describes, and typing in that one makes the shadow vanish from a row you are not even in. The board is trimmed back to the last word and its free row.

The rules are one promise between them. Three words with the third taken off leaves exactly what two words played from the start would leave, and nothing on the page can tell you which of the two happened.

One request is in flight at a time, since the solver is behind a single lock anyway. A second one made meanwhile is *held*, not dropped. Dropping it used to leave the page showing something nobody had asked for: a game view with no game dealt into it, every panel reading "—", and it would stay that way. The loading veil stays up until there is a board behind it for the same reason, retrying if the first request fails.

**A setting throws nothing away.** The three views share one set of switches, and changing any of them — a prior, the mode, the shape of f, hard mode — asks the solver again about the position you are already in. The solve board is re-run over the rows the server has already accepted, so a word half typed into the open row is neither submitted nor lost; the game keeps its answer and re-asks the hint; the finder searches again. Only the *Answer* box deals a new game.

The terminal menu works the other way round, with four combinations to choose between before the deal, because there the prior decides what the answer is drawn from. Here the answer was drawn under whatever was on at the time and it stays what it is. Turning a prior off mid-game rethinks the advice instead of dealing a new word, and that is the only reading under which you can ask what the *other* model of the editor would have played from here.

Nothing is rebuilt behind your back either. A setting changed while a view is hidden marks that view for a re-run, and the re-run happens when you next look at it. The alternative would be a view quietly claiming a setting it was never built under.

### Rewriting what it says

`python editor.py` lists every string the site can say, each beside a box to rewrite it, with the page itself alongside. Save, and the page says it — no restart.

What it shows alongside is `site/`, the build that gets deployed, running its solver in WASM. So the editor loads no word lists, no feedback table and no numba, and starts in the time it takes to read one directory; `web.py` served it before and had to warm the whole solver first, to change a sentence.

Nothing is registered by hand. `wording.py` reads `index.html`, `app.js`, `web.py` and the two Rust files the static build answers from, and picks the text out by how it is used (between tags, assigned to `textContent`, passed to `say()`, `SolverError()` or `Err()`, and so on), so a string written the ordinary way is editable the moment it exists and the list cannot fall behind the site. Each is shown with the file, the line and where on the page it sits.

Rust needs a lookback the others do not. A message there is often several braces inside the call that says what it is for — `.ok_or_else(|| { if … })` — so cutting the context at a brace, which is what the JS scanner does, would throw away the only thing marking it as text. `wording._rs_statement` cuts at the semicolon instead and lets the enclosing function bound it. That one difference is what takes the Rust from 20 of its messages to all of them, and it keeps `format!("g{}", …)` out, since a session id is built in a statement that says nothing about text.

A choice between two strings counts as one of those uses, and what marks it as text can be lines above the arm it is looking at — `why:` over a ternary laid out one arm per line. Reading only the arm's own line missed the longest prose on the page, the two explanations of the priors, until 2026-08-10.

Strings are found at a character offset, so the same sentence written twice is two entries and rewriting one leaves the other alone. Edits are applied to the served bytes of `index.html` and `app.js`, and saved in `wording.json` keyed by `<file>:<hash of the text>:<n>` — which survives the code being worked on, since neither line numbers nor offsets do. What the *server* says is matched on the way out instead, values and all: `Row {index + 1} contradicts the rows above it` becomes a pattern, and whatever it captured goes back into the rewritten version. That is by wording rather than by position, so a message written twice in `web.py` is one entry.

A value baked into a string is part of it: an edit may move `${count}` but not drop it, and saving is refused if one goes missing. Rust's bare `{}` counts as one of those too, since that is how `format!` takes an argument. The editor listens to this machine alone; anyone who can reach it can rewrite the source it is built from.

**Save** leaves the source alone. The rewrites live in `wording.json` and are applied to the served bytes on the way out — and `build_site.sh` applies them too, so a rewrite that has only been saved still reaches the deployed page. **Write into the code** puts them in the files for good and clears `wording.json`. That is the end state worth reaching, because an override is matched to the code by wording and stays fragile for exactly as long as it sits outside it. Prose is re-wrapped to the indentation it lands in, and both phrases are appended to `wording_history.json`. That record is the only one kept. The bake edits the source in place with no undo, so it is worth reading the source afterwards. A phrase written into the Rust needs `./build_site.sh` before the preview says it, since by then it is inside the compiled module rather than in a file being served.

**Every message the page can show is written twice**, once in `web.py` and once in the Rust the static build answers from. Both are listed and both have to be edited. Carrying an edit from one to the other automatically was tried and does not work: the two write their placeholders differently, and `{index + 1}` is not a name Rust can have, so an exact match reaches about a third of them and says nothing about the rest. What catches the omission instead is `tests/rust_parity.py`'s message check, which feeds both implementations the same nine bad requests and prints the two wordings side by side when they have come apart.

Nothing to install: the server is `http.server` and the page is three static files in `web/`, with three more behind the editor. Startup warms the feedback table and the numba kernels in the background while the page waits. That makes the first load the slow one. A single lock covers the solver. Scoring ~13k guesses already uses every core, so two requests at once would only make both slower.

### Without a server

`./build_site.sh` writes `site/`, which is the same three views with no Python behind them and nothing to deploy but files. It is 1.6 MB, 445 KB of that the WASM module and 686 KB the data the server used to hold — 355 KB in all over the wire, since a host that serves it with brotli gets the module down to 142 KB and `solutions.json` from 287 KB to 24. Nothing else is asked of a host. `vercel.json` labels the module `application/wasm`, which is what streaming compilation insists on, and where that label is wrong the worker compiles the bytes it already fetched instead — so the build deploys anywhere that serves files.

`site/` is generated and is not in the repository. `./deploy.sh` builds it and uploads it to Vercel in one step, which is the whole of the deployment: there is nothing to build on the far side, and the only state that has to survive a rebuild is `site/.vercel/`, the link to the Vercel project, which `build_site.sh` carries across. `./deploy.sh --preview` puts the same build on a throwaway URL to look at before promoting it.

What stands in for the server has to stand in for its failures too. `web.py` refuses a contradictory board with a 400, and `app.js` is written to that: `api()` throws on a reply that is not ok and renders a board on one that is. A refusal returned as 200 is therefore applied *as a board*, and the page dies reading rows off a payload that has none. The shim answers 400 for any reply carrying an `error`, which is the whole of the distinction the page draws; which of 400, 404 and 503 the server would have chosen is not reconstructed, since nothing on the page tells them apart and the only clue here is message text the editor may rewrite.

`web/app.js` is byte for byte the file `web.py` serves. It reaches the solver through `fetch`, so `web/wasm-api.js` stands in for `fetch` on `/api/*` and routes into `wordle_core::api` instead of over a network — which is possible only because the page already sends the whole board on every submit, making that boundary a pure function of (path, body) rather than a conversation. Leaving `app.js` alone matters beyond tidiness: `tests/board.test.js` loads it through `vm`, and a modified copy would mean the board tests exercise a different file from the one that ships.

The solver runs in a worker, `web/solver-worker.js`, and not on the page's thread. The widest position after the opening costs a little over 200 ms in the browser, and spent on the main thread that is 200 ms in which the page cannot repaint, scroll or take a keystroke. Moving it does not make it quicker — the same scan takes the same time — but the page stays responsive throughout, which is most of what the delay actually cost. Measured against a deliberate 200 ms block for comparison, the event loop turned over once during the block and some sixty thousand times during a solve of the same length. Making it genuinely faster would mean threads, which in a browser mean `SharedArrayBuffer`, cross-origin isolation headers and a nightly toolchain to rebuild `std` with atomics. Measured, they are worth about 9x on this one position and 2.4x over a whole sweep — the gap between those two being how little of a game is scoring at all. That was not judged to be worth a browser build that no longer compiles on stable and that nothing in `tests/` can check.

**The table does not come with it.** 167 MB is not a download and not something a tab can rebuild, so the browser scores feedback live — `source::Live` against `source::Table`, chosen as a generic parameter so the inner loop dispatches statically. Live costs about 3.5× a lookup, measured natively on one position by `native/src/bin/bench_position.rs`: 0.7 ms against the table's 0.2 ms on a median turn-two bucket, and 18.5 ms against 5.3 ms on the worst. Cache lines predicted a wider gap than that, and a scattered read does cost a whole line, but the rows one position touches stay resident and most of its reads then hit. The ratio is real and beside the point either way, because the absolute figure is what the page lives with: 168 ms to rank every guess after one word, 743 ms for the widest thing the finder can be asked.

Live scoring is felt in the endgame and almost nowhere else. The exact search visits hundreds of nodes for one position and would rescan all 12,947 guesses at every one of them, which for a set of 34 candidates means running the coloring rule over the same half-million pairs several hundred times; `web.py` never pays that, reading the pairs out of the table it already holds. `source::Columns` is the part of that table a tab can afford — every guess against the candidates the search began with, 440 KB, filled in one scan and read from thereafter. It is exact rather than an approximation, because every bucket the search recurses into is a subset of the set above it and nothing outside those columns is ever asked for. Without it, the two rescanning kernels are 84% of the position's cost and that 34-candidate turn takes 5.6 s; with it the same turn is 1.1 s.

What survives between requests matters as much as what one of them costs. `web.py` keeps a solver per setting for the life of the process and its memo with it, so a board solved once answers in 8 ms every time after — the common case rather than a lucky one, since the page re-sends the whole board on every submit, and a tile recolored or a switch flipped is the same request as a word played. A `Solver` borrows its source and its weights and cannot outlive the request that built it; what it caches is a pure function of a position and outlives one happily. So `solver::Memos` holds the three caches apart from the solver and `state` lends them back per request, under the key `web.py` would have kept its solver under. Four settings' worth are held, which is why flipping a switch and flipping it back re-solves nothing. A repeated board costs 11 ms.

The exception is the opening, at 12,947 guesses against 12,947 candidates. That one position is a constant per setting, so `native/src/bin/build_site.rs` computes all 180 of them ahead of time into `openings.json` — every prior against both modes, both hard-mode settings, and each of the nine forms of `f`. Entropy never consults `f`, so its share of those is one ranking written out under nine names.

Turn two is precomputed as well, for each of the four priors the two switches reach. It is affordable live but not comfortably so: an opening cuts the list into around 190 buckets, most of them small enough to hand the position to the exact endgame — 143 of `salet`'s 193 — which is where the second above went. Each prior opens on its own word and was fitted in its own form, so each gets a set of its own: `salet` with both priors on, `trace` with only curation, `tares` for the remaining two. That is 790 rankings, 40 KB over the wire. The other two menus stay where the page leaves them, because covering every mode and every shape of `f` as well is 180 settings, an hour of `build_site` and megabytes of `openings.json`. Off that path — a first word the setting would not have played, or the mode or the form changed — the search runs live. Nothing deeper is worth precomputing either: turn three costs a sixth of what turn two does even with an empty memo.

Two things do not survive. The editor rewrites the source it is scanning, which needs a checkout, so it stays a local Python tool — one that now previews this build rather than the Python's. And play mode's answer no longer sits out of reach — the module holds it, and the page still never receives it until the game is over, but "reading the network traffic does not spoil it" is now a statement about a page rather than a guarantee.

`native/src/bin/api_probe.rs` runs the same endpoints from a shell. Debugging them through a browser is guesswork on a slow loop, and this is why the API lives in `wordle_core` rather than in the `wasm` crate, which does marshalling and nothing else. It reports what each request cost, and built with `--features wordle_core/profile` it reports where that time went per kernel. That is how the endgame's rescanning was found: two entry points holding 84% of one position, which no amount of reading the recursion had suggested.
