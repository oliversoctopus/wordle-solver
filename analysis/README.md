# analysis/

Every measured number in `README.md` and `SOLVER_EXPLAINED.md` comes from one of the scripts below. They read `data/` and the estimator cache and print, and what they print they also record: `common.record` writes each figure to `results/<script>.json`, and `doc_numbers.py` at the root carries it into the documents from there. `--write` rewrites every bound figure and lists what moved, and the bare command is the check `tests/run.py` runs.

So a figure in a document comes from a results file, or it carries a marker saying why not, and `doc_numbers.py --unbound` fails on one that does neither. `<!--c-->` marks a definition or a worked example. `<!--u-->` marks a figure measured once with no script here, and *What is not reproduced here* lists them.

A script that sweeps every answer hands that sweep to the native binary where `./build.sh` has built one, and `analysis/rust.py` decides. It is worth about <!--n timing.sweep.expected.python_over_rust .1f-->1.6× on a whole script and the lengths are identical, since `tests/rust_parity.py --every-answer` plays all 12,947 both ways and diffs them per answer. `WORDLE_NO_RUST=1` keeps everything in Python, and the two are compared under it. A setting the flags cannot express falls back to Python on its own.

```
python analysis/regime.py           the regime table and the pool arithmetic
python analysis/repeat_shapes.py    prequential comparison of repeat shapes
python analysis/drift.py            whether the source window is moving
python analysis/curation_shapes.py  prequential comparison of curation shapes
python analysis/fresh_shapes.py     the third group, and why it has no shape
python analysis/id_eras.py          the NYT's id field: what it indexes and when it changed
python analysis/hard_mode.py        the cost of the hard-mode restriction
python analysis/hard_mode.py --all-priors --mode entropy   the same restriction under greedy entropy
python analysis/hard_mode_rules.py  the value of the NYT's looser rule
python analysis/hard_mode_rules.py --no-endgame   the same with the endgame search off
python analysis/hard_mode_rules.py --mode entropy  the same under greedy entropy
python analysis/lookahead.py        the value of one ply of lookahead
python analysis/lookahead_accuracy.py   how well the two valuations price a position
python analysis/lookahead_accuracy.py --decisions   the stake in the choice
python analysis/exact_decisions.py  the same question for the endgame search
python analysis/exact_decisions.py --who-decides   which valuation takes each turn
python analysis/opening.py          whether the opening f picks is the best one
python analysis/turn2_regret.py     the regret left in the second guess, and whether it holds
python analysis/loss_penalty.py     the penalty and the endgame search's reach
python analysis/loss_penalty.py --exact-max 24 40 60 80 100   the reach table README quotes
python analysis/loss_penalty.py --policies    entropy against the estimator
python analysis/loss_penalty.py --search      the search's contribution, by its reach
python analysis/loss_penalty.py --cost        the time cost of reaching further
python analysis/loss_penalty.py --bound       why the pruning bound caps the penalty
python analysis/loss_cost.py        the true cost of a lost game
python analysis/branch_cap.py       the branch cap's pruning, counted not timed
python analysis/ablations.py        the same questions in wall clock -- needs an idle machine
python analysis/replay.py           the shapes in guesses (slow)
python analysis/replay.py --shapes       the four shape combinations against uniform shapes
python analysis/replay.py --all-priors   each prior on real answers
python analysis/prior_basin.py      how much of the prior's setting reaches the board, and the off-list default (slow)
python analysis/fresh_arm.py        why the fresh arm and the real fresh answers disagree
python analysis/refit_lag.py        the cost of a stale answer record
python analysis/form_default.py     which form to fit by default (slower)
python analysis/fit_rounds.py       what policy iteration converges to, and the search's cost to it
python analysis/pattern_offsets.py  the share of f's error due to the pattern behind the bucket
python analysis/branch_tie.py       the branch cap's tie tolerance, and the band it sits in
python analysis/rust_fit.py         the native trainer's fit against the Python's
python analysis/results_tables.py   the two tables in README *Results* (slowest)
python analysis/head_to_head.py     the shipped solver against the reference, game by game
python analysis/head_to_head.py --baseline   against greedy entropy instead
python analysis/split_ranking.py    how much of a ranking is one guess twice
python analysis/table_layout.py     both feedback-table layouts, kernel by kernel
python analysis/fused_gain.py       the answer-major dispatch in games
python analysis/timing.py           the Rust port's sweep and trainer wall clocks, against the Python's
python analysis/timing.py --train   also the trainer probes; overwrites and restores the cache
python analysis/site_size.py        what the static build weighs, and what openings.json holds
```

Wall clocks on one machine, with the native sweep where a script can use it. A script whose figures are wall-clock ratios needs an idle machine. The others do not care what else is running.

| script | default | with flags | notes |
| --- | --- | --- | --- |
| `regime.py`, `repeat_shapes.py`, `drift.py`, `curation_shapes.py` | seconds | | |
| `fresh_shapes.py` | 3 min | `--events 300` under 1 min | nearly all of it the ceiling fit |
| `id_eras.py` | seconds | | |
| `hard_mode.py` | | `--all-priors` 2 min | native sweep |
| `hard_mode_rules.py` | 1 min | `--free` 3 min | plays the policy itself; `--check` reproduces `every_answer` game for game |
| `lookahead.py` | minutes | | |
| `lookahead_accuracy.py` | 7 min | `--decisions` 13 min | |
| `exact_decisions.py` | under 1 min | | |
| `opening.py` | 16 min | `--every-word` most of a day | about five seconds an opening |
| `turn2_regret.py` | 5 min | | most of it the census of fifty guesses at fifty-one positions |
| `loss_penalty.py` | 9 min | `--shipped-only --skip-reach` 1 min; `--policies`, `--search` under 1 min; `--cost` 6 min; `--bound` seconds | `--cost` is a timing probe and wants an idle machine |
| `loss_cost.py` | minutes | | two sweeps per prior plus a playout per losing word |
| `branch_cap.py` | 2 min | | counts, never times |
| `ablations.py` | under 1 min | `--sweep-games 12947` 15 min; `--projection` free | wall clock end to end; idle machine only |
| `replay.py` | 40 s | `--all-priors`, `--shapes` 2 min | |
| `prior_basin.py` | 7 min | | |
| `fresh_arm.py` | 2 min | | |
| `refit_lag.py` | 2 min | | |
| `results_tables.py` | 2 min | | native sweep |
| `form_default.py` | 12 min | | native sweep; replays every archived fit |
| `fit_rounds.py` | 8 min | | native sweep |
| `pattern_offsets.py` | seconds | | |
| `branch_tie.py` | 12 min | | Python only: its arms are orderings only `exact.py` has |
| `rust_fit.py` | 8 min | | Python and Rust fits, thirty sweeps |
| `split_ranking.py` | seconds | | its second half is a timing probe |
| `table_layout.py` | 4 min | | timing probe; read the ratios |
| `fused_gain.py` | under 1 min | | timing probe; see below |
| `timing.py` | 2 min | `--train` 5 min more | idle machine only |
| `site_size.py` | seconds | | reads `site/`; needs `brotli` on the path |

The native sweep is worth about <!--n timing.sweep.expected.python_over_rust .1f-->1.6× on the scripts that use it. The memos are why any of this finishes. Behind a fixed opening the 12,947 answers reach at most 243 distinct positions, and a run comes mostly out of cache once they fill. `memo.Memo` evicts its least recently used entry instead of refusing new ones, and `WordleSolver.reset_memo` is for a sweep that wants each run independent of the last. A run taking far longer than its row above is one to interrupt and diagnose.

`fused_gain.py` draws its answers uniformly from the whole list, not from the prior. The dispatch speeds up the table read alone, which in expected mode is a small share of a scan carrying the estimator and the endgame search. The gain is <!--n fused_gain.expected.whole .2f-->1.04× on its own population and <!--u-->1.09× on answers drawn from the prior, under either fit, so a figure that looks like it moved has usually changed population instead.

`loss_penalty.py --cost` times how much work the wider reach finds to do, and that is a fact about the prior. <!--u-->The answer log alone takes it from 1.20× to 1.49× on the shipped row, with the code and the fit fixed, because thirteen more used words push the search's mean candidate set from 18.0 to 20.7 and the share of its calls above the default cap from 5.2% to 8.1%.<!--/u--> Rerun it after `solutions.py --write` like any other measurement here.

| script | backs |
| --- | --- |
| `regime.py` | README *The regime the defaults describe*, SOLVER_EXPLAINED §4; its last block the implied-rate table in §14.3, which is why the two priors need each other |
| `repeat_shapes.py` | README *The repeat prior's shape* table, SOLVER_EXPLAINED §5.1 |
| `drift.py` | the closing paragraph of README *The repeat prior's shape* |
| `fresh_shapes.py` | README *The group with no shape* |
| `curation_shapes.py` | SOLVER_EXPLAINED §5.2: the shape table, the snapshot's forward-prediction table, and why the hedge is flat; the opening of README *The curation prior's shape* |
| `id_eras.py` | the id evidence for the off-list block in SOLVER_EXPLAINED §5.2, and the reconstruction in README *What the id field indexes* |
| `hard_mode.py` | the table and the entropy paragraph under README *Hard mode* |
| `hard_mode_rules.py` | the paragraph under README *Hard mode* saying the shipped restriction is not the NYT's rule, and what widening it to theirs would be worth |
| `lookahead.py` | the paired figures in README *Lookahead*, not its accuracy table |
| `lookahead_accuracy.py` | the accuracy and decision tables in README *Lookahead*, SOLVER_EXPLAINED §15 |
| `exact_decisions.py` | the per-decision table under README *The exact endgame*, its `--who-decides` the one under *Conditioning f on turns remaining*, SOLVER_EXPLAINED §15 |
| `opening.py` | README *The opening is not where the estimator is weak*, SOLVER_EXPLAINED §15 |
| `turn2_regret.py` | README *Perfecting the second guess*, and the last case in SOLVER_EXPLAINED §15 |
| `loss_penalty.py` | the whole of README *Losing less*, its `--search` the table under *The exact endgame*, its `--bound` the `-atch` values in SOLVER_EXPLAINED §10(c), SOLVER_EXPLAINED §10 and §11 |
| `loss_cost.py` | the continuation figures in SOLVER_EXPLAINED §11 |
| `branch_cap.py` | the branch-cap table in SOLVER_EXPLAINED §16, the claim in §10(d) that the cap never costs the winner, and the node counts under README *The exact endgame* |
| `ablations.py` | the ablation table in SOLVER_EXPLAINED §16, and the memo and branch-cap figures beside it; its `--projection` the whole of §17 |
| `prior_basin.py` | README *The prior sits in a flat basin*, and the off-list sweep under *The regime the defaults describe* |
| `fresh_arm.py` | the fresh-arm paragraphs of SOLVER_EXPLAINED §14.1 and README *What the two shapes are worth together* |
| `refit_lag.py` | README *How fresh the answer record has to be*, and the two paragraphs closing SOLVER_EXPLAINED §14.1 |
| `replay.py` | the head-to-head paragraph under README *The curation prior's shape*; its `--shapes` the whole of README *What the two shapes are worth together* and the regime replay under *The repeat prior's shape*; its `--all-priors` the whole of *What each prior is worth on real answers* |
| `form_default.py` | README *Which form is the default* |
| `results_tables.py` | both tables in README *Results*, and its `--part priors` the bonus/curvature split in SOLVER_EXPLAINED §6 |
| `head_to_head.py` | README *Where the estimator's guesses come from* |
| `split_ranking.py` | the *Alternatives* card under README *In a browser*, and the runners-up paragraph closing *The exact endgame* |
| `table_layout.py` | SOLVER_EXPLAINED §8 *Storing the table the other way round*, the per-kernel figures |
| `fused_gain.py` | the same section's whole-game figures |
| `timing.py` | the sweep, chunking, fused-dispatch and `train` wall clocks in README *Fitting in Rust* and *Keeping the port in step*, and the per-scan figures of `bench_position` |

## Rerunning them

Not all of them after every `solutions.py --write`. The scripts are of two kinds.

A **live** figure describes the shipped solver as it stands and moves with the record or the fit: the regime rates, the shape scores, the prior replays, which form each prior plays, the results tables, hard mode's cost. Its scripts are the routine refresh, about fifteen minutes between them:

* `regime.py`, `repeat_shapes.py`, `drift.py`, `curation_shapes.py`, `id_eras.py`
* `form_default.py --promote`, `replay.py` in its three modes, `results_tables.py`, `hard_mode.py --all-priors` and `--mode entropy`, `hard_mode_rules.py` in its three modes
* `turn2_regret.py`, whose enumerated rows are a finding but whose real-answer rows replay the record
* `prior_basin.py`, whose recent-window rows follow the record and back the off-list default
* `site_size.py`, after `./build_site.sh`

A **finding** answered a question once and is a property of the design that a refit re-confirms: lookahead is worth nothing, the branch cap never costs the winner, the fit does not converge to its best round, the native fit is as good, a stale record costs nothing. Those are rerun when the code they measure changes, and each one's results file says when that last was. The table below quotes it.

Nothing warns about a stale fit either way. The estimator cache is keyed on the `PriorConfig` alone, so refit before rerunning `replay.py` or it measures an old policy against new answers.

| finding | last measured |
| --- | --- |
| `lookahead.py`, `lookahead_accuracy.py` | <!--s lookahead.measured-->2026-09-21<!--/s--> |
| `exact_decisions.py`, `opening.py` | <!--s opening.measured-->2026-09-21<!--/s--> |
| `loss_penalty.py`, `loss_cost.py`, `branch_cap.py` | <!--s loss_penalty.measured-->2026-09-21<!--/s--> |
| `fit_rounds.py`, `rust_fit.py`, `pattern_offsets.py` | <!--s fit_rounds.measured-->2026-09-21<!--/s--> |
| `fresh_arm.py`, `refit_lag.py`, `fresh_shapes.py` | <!--s fresh_arm.measured-->2026-09-21<!--/s--> |
| `ablations.py`, `table_layout.py`, `fused_gain.py`, `timing.py` | <!--s timing.measured-->2026-09-20<!--/s--> |

**Not every arm moves.** `priors.build_weights` consults the used-word mask only under `PriorConfig.repeat`, so a repeat-off prior is the same distribution after the write as before it, its fit is bit-for-bit the same fit, and every figure weighted by it over every word is the same figure. `priors.depends_on_answer_record` is that rule in one line. `wordle_solver.py train --repeat-only` skips the two of the five fits it exempts, and `--repeat-only` drops the same arms from every script here that sweeps priors, printing which it dropped so a table assembled from two runs says so.

The exemption is about where the weights come from, not what an arm is called. The scripts that replay the real answer record (`replay.py`, `prior_basin.py`, `fresh_arm.py`) score their `none` and `curation only` arms over a log that just got longer, so every arm of theirs moves and they take no such flag. Carrying an old row into a new table holds only while nothing else moved either. A `word_lists.py --write`, a fresh `wordlebot.py` scrape or a change to `noncurated_prob` invalidates a repeat-off row as surely as a new answer would, and nothing records which run a row came from.

`replay.py --dump PATH` writes every arm's per-game guess counts out as JSON, <!--n regime.since.puzzles-->230 regime games plus the three arms' <!--n replay.shapes.arms.repeats-->24, <!--n replay.shapes.arms.offlist-->59 and <!--n replay.shapes.arms.fresh-->178 events, one row per arm. With a dump in hand, a contrast nobody thought to print, such as a pair the tables skip or a subgroup split some other way, becomes arithmetic over that file instead of another replay. None is kept here. A dump is true only of the fit and the answer log it was made against, and nothing would catch a checked-in one drifting out of step with either, while `--all-priors` makes a fresh one in two minutes.

**Check the method before believing a changed number.** Several scripts score a frozen prefix of the record, and those figures cannot move whatever the answer log does. If they still come out, a number that moved moved because the data did. They are bound to the results files like everything else, so `doc_numbers.py` fails when they stop coming out. The fixed points:

* `repeat_shapes.py --events 18` scores only the first 18 repeats: uniform <!--n repeat_shapes.first18.uniform .1f-->−134.5, binned <!--n repeat_shapes.first18.binned .1f-->−116.1 at <!--n repeat_shapes.first18.binned.vs +.1f-->+18.4, α = 1.0 at <!--n repeat_shapes.first18.binned_a1 .1f-->−118.4, and <!--n repeat_shapes.first18.gain +.3f-->+1.024 ± <!--n repeat_shapes.first18.gain_se .3f-->0.235 nats/event at <!--n repeat_shapes.first18.sigma .1f-->4.4σ winning <!--n repeat_shapes.first18.wins-->14 of 18.
* `regime.py`'s 2021–2025 row (<!--n regime.early.puzzles-->1,657 / <!--n regime.early.repeat .2f-->0.00% / <!--n regime.early.offlist .2f-->1.75%) and `curation_shapes.py`'s uniform baseline (<!--n curation_shapes.uniform.nats .2f-->−9.27).
* `fresh_shapes.py --events 300` scores a frozen prefix of the fresh-curated picks: uniform <!--n fresh_shapes.first300.fresh_curated.uniform.nats .4f-->−7.4093, WordleBot weight <!--n fresh_shapes.first300.fresh_curated.wordlebot.nats .4f-->−7.3911 at <!--n fresh_shapes.first300.fresh_curated.wordlebot.gain +.4f-->+0.0182 ± <!--n fresh_shapes.first300.fresh_curated.wordlebot.se .4f-->0.0009 winning <!--n fresh_shapes.first300.fresh_curated.wordlebot.wins-->286 of <!--c-->300, the shipped 5% hedge <!--n fresh_shapes.first300.fresh_curated.hedge.nats .4f-->−7.4625, and a ceiling of <!--n fresh_shapes.first300.ceiling +.4f-->+0.0547 nats/event. Its off-list rows and its pooled-rate block are not frozen and move with the record like anything else here.
* The first half of `id_eras.py`. The id matching the file position for <!--n id_eras.exact_positions-->284 puzzles, the drift climbing to <!--n id_eras.skipped-->22, and the runs of <!--n id_eras.run1.words-->12 and <!--n id_eras.run2.words-->8 in the appended block are settled history, and only its era-2 fit moves as answers arrive.

**An older answer log is reproducible in place.** `load_data(mask_last=n)` drops the most recent n entries of `solutions.json` and rebuilds the priors from what is left, and every front end takes it as `--mask-last`. Wind the log back to the day a figure was recorded, rerun, and whether the data moved or the code did stops being a matter of opinion. The fit does not wind back with it. `estimator.cache_path` is keyed on the `PriorConfig` slug alone, so the current `f` loads against the truncated log without complaint. Refitting is the wrong fix, because `train --mask-last n` writes to that same path and replaces the prior's live fit with one describing a short log, under a filename recording nothing about the difference. Copy `params_<slug>.json` aside first, or take the stale `f` as the known cost of a cheap check and say which you did.

## Enumerating instead of sampling

`hard_mode.py`, `loss_penalty.py`, `results_tables.py` and `form_default.py` do not draw answers and report a standard error. The solver is deterministic, so a game's outcome is a pure function of the answer, and playing all 12,947 words once with the prior as weights computes the mean and the loss probability. There is no sampling distribution left to put an interval around, and `common.every_answer` is the one implementation they share.

It matters most where the effect is small. The loss penalty is worth about <!--n loss_penalty.penalty.first_to_default.mean +.4f-->+0.0005 guesses, while a 20,000-game sample carries an error near <!--u-->0.009, twenty times larger, and only pairing made those figures readable. Removing the noise also moved conclusions. <!--u-->Under the flat prior the nine forms of `f` span 0.0013 guesses, where a 3,000-answer holdout showed 0.0110<!--/u-->, and most of that spread was noise.

Three things change shape in the output. A loss rate becomes the exact prior mass on the words that lose, which is not the fraction of words that lose. "Which games changed" becomes a list of words. A block sd or a holdout seed stops existing, so a ranking is a fact about the prior and cannot move on a rerun, which makes a small gap harder to dismiss.

Enumerating is usually the cheaper option too. The memos in `WordleSolver` and `ExactEndgame` saturate after a few thousand positions, so most of a sweep runs out of cache. The marginal rate climbs past <!--u-->4,000 games a second, and the arm that looks slowest from a cold start is not the one that costs the most.

`replay.py` is the exception that stays sampled. It replays the real answer sequence, so its sample is a historical record and not a draw from the prior, and there is nothing to enumerate.

That also lets its arms be compared across different priors, which no enumerated script can offer. An enumerated mean weights every word by the prior on trial, so two such rows answer the same question about two different populations and the gap between them means little. `results_tables.py` prints that warning above its own prior table, and `head_to_head.py` states the weighting and leaves the reader to draw it. A replay scores every arm on the answers the NYT published, which none of them had a hand in choosing, so `--all-priors` is where the four combinations can be put side by side.

`lookahead_accuracy.py` and `exact_decisions.py` sit between the two. Every value they report is exact, a position played out once per candidate answer, but which positions they visit, and how heavily each counts, comes from a sample of games. Their means carry a sampling error that the enumerated ones do not. The machinery is shared. `common.collect_positions`, `common.guess_value` and their neighbors are the one implementation both read.

## The three fits `--shapes` needs

Comparing the sharpened shapes against the uniform ones needs fits for `repeat1_curation1`, `repeat1_curation1_r-binned` and `repeat1_curation1_c-wordlebot`, and a prior with no fit raises instead of playing an invented `f`. All three are in `estimator_cache/`, fitted the way the shipped five were: the forms archived by `wordle_solver.py compare`, ranked by `form_default.py --shapes`, and the winner written as the active fit by `--promote`. `rust_train` declines a prior outside `priors.ALL_CONFIGS`, so these three are fitted in Python. They came out <!--s form_default.shape_winners-->`spline`, `cubic`, `cubic`<!--/s-->.

The three are not part of a routine refit. The shape question is a finding (README *What the two shapes are worth together*), so their fits stay as they are until it is reopened, and then all eight are refitted together against one answer record. Every arm in a `--shapes` table has to carry a fit of the same history, or the table is comparing solvers and not shapes.

## What is not reproduced here

`replay.py`'s arms are comparable to each other, but a run is not comparable to an older run. Its numbers depend on the fitted estimator as much as on the data, and a refit can move a configuration's winning form, so its output cannot be diffed across one. The same holds for `form_default.py`, which scores the fits in `estimator_cache/forms/`, so a refit moves every row.

`hard_mode.py` is the one that survives a refit. Its two arms play the same `f`, so a refit moves both columns together and mostly cancels in the paired difference. Read the difference across refits, and read the two means only against each other. `lookahead.py` does not get that protection even though it also pairs two arms on one fit. It measures how much room the base policy leaves above itself, so a better-fitted `f` should shrink it. Expect that as a direction and not as a prediction. The effect is a few thousandths of a guess against a standard error of about half that, and the last refit improved every arm's base policy while leaving lookahead worth marginally more.

Every figure this section names carries the `<!--u-->` marker where it stands, so `doc_numbers.py --unbound` passes it and a reader can tell it from a bound one. Several claims about the endgame search have no script, and all of them rest on position samples that were not kept: that a branch cap of 25 measures no better and that two plies costs <!--u-->205 s for the opening position alone (README *Lookahead*), the 1,275-position comparison of capped against uncapped search and its wall clock, the two 80-position samples that sized the wider root cap and the narrower cap's <!--u-->12 ms and <!--u-->520 ms, the runners-up <!--u-->0.02 guesses apart under an unstable sort, and the memo's 373-repeat position and its share of about <!--u-->30% of the search's time (README *The exact endgame*, SOLVER_EXPLAINED §10(d) and §12). They are timing and tie-break facts, so they do not move with `solutions.json`, and nothing here would catch them going stale either.

The browser figures in README *Without a server* were read off a tab and a profile build, and no script drives either: the <!--u-->200 ms turn on the page's thread, the <!--u-->168 ms and <!--u-->743 ms rankings, the <!--u-->440 KB column table with its turn dropping from <!--u-->5.6 s to <!--u-->1.1 s, and the two kernels holding <!--u-->84% of one position. `bench_position` measures the live-against-table scans natively and `timing.py` records those.

Three claims about the fit rest on implementations that were not kept, and none has a script here. Warm starting policy iteration from the cached fit (README *Warm starting from the previous fit*) needed a seed argument on `train.policy_iteration`. Conditioning `f` on the feedback pattern (README *Conditioning f on the pattern*) needed a kernel carrying the offsets, and `pattern_offsets.py` reproduces that measurement's residual half but not its verdict. Conditioning `f` on the turns remaining (README *Conditioning f on turns remaining*) needed one curve per turn count. All three were measured on the objective everything else here uses, every answer weighted by the prior, and all three were rejected on it.

Smaller figures with the same status:

* the search refit moving the curve by under <!--u-->0.03 guesses at an RMSE of <!--u-->0.63, and the search regret of <!--u-->0.0000 at penalty 1
* the penalty settings within <!--u-->0.0002 guesses of each other across the price of a loss, and `tatty` carrying <!--u-->75% of its family against a <!--u-->0.4% case
* the r = <!--u-->0.98 of the four-term answer-likeness model, and the four feature weights `answer_like.py` fits outside this directory
* the tenth of a guess by which the census once read kinder to the sharpened repeat shape than the real answers did, under fits since replaced (README *What the two shapes are worth together*, SOLVER_EXPLAINED §14.1)
* the marginal rate past <!--u-->4,000 games a second, the 20,000-game error near <!--u-->0.009 and the holdout span of <!--u-->0.0110
* the reach's cost moving from <!--u-->1.20× to 1.49×<!--/u--> with the answer log, and `fused_gain.py`'s <!--u-->1.09× on answers drawn from the prior

Two figures from the original measurements are not reproduced exactly. The window variants' spread was <!--u-->−124…−129<!--/u--> there. The best variant here matches and the worst is far worse, so the original defined the band differently. Only the best of the sixteen is quoted, which is the number the argument needs. The drift rate's standard error was ±<!--u-->0.39 there against ±<!--n drift.first18.drift_se .2f-->1.56 here. The slope agrees to two decimals, so the parameterization matches and the error model does not.
