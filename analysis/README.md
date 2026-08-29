# analysis/

Every measured number in `README.md` and `SOLVER_EXPLAINED.md` about the two priors, about which form of `f` is the default, about what hard mode costs, about what the fit itself converges to, about what the branch cap's tie tolerance is worth, about whether the native trainer's fit is as good, and about the loss penalty comes from one of these. They read `data/` and the estimator cache and print; they write nothing.

A script that sweeps every answer hands that sweep to the native binary where one has been built, which is `analysis/rust.py` deciding and `./build.sh` providing. It is worth about 1.7× on a whole script and the lengths are identical, not close: `tests/rust_parity.py --every-answer` plays all 12,947 both ways and diffs them per answer. `WORDLE_NO_RUST=1` keeps everything in Python, which is what the two are compared under, and a setting the flags cannot express falls back on its own rather than quietly measuring something else.

```
python analysis/regime.py           the regime table and the pool arithmetic
python analysis/repeat_shapes.py    prequential comparison of repeat shapes
python analysis/drift.py            is the source window moving?
python analysis/curation_shapes.py  prequential comparison of curation shapes
python analysis/id_eras.py          what the NYT's id field indexes, and when it changed
python analysis/hard_mode.py        what the hard-mode restriction costs
python analysis/hard_mode_rules.py  and what the NYT's looser rule would be worth
python analysis/lookahead.py        what one ply of lookahead is worth
python analysis/lookahead_accuracy.py   how well the two valuations price a position
python analysis/lookahead_accuracy.py --decisions   what is at stake in the choice
python analysis/exact_decisions.py  the same question asked of the endgame search
python analysis/exact_decisions.py --who-decides   which valuation takes each turn
python analysis/opening.py          is the opening f picks the best one?
python analysis/turn2_regret.py     what is left in the second guess, and whether it holds
python analysis/loss_penalty.py     the penalty and the endgame search's reach
python analysis/loss_penalty.py --policies    entropy against the estimator
python analysis/loss_penalty.py --search      what the search adds, by its reach
python analysis/loss_penalty.py --cost        what reaching further costs in time
python analysis/loss_penalty.py --bound       why the pruning bound caps the penalty
python analysis/loss_cost.py        what a lost game would really have cost
python analysis/branch_cap.py       what the branch cap prunes, counted not timed
python analysis/ablations.py        the same questions in wall clock -- needs an idle machine
python analysis/replay.py           what the shapes are worth in guesses (slow)
python analysis/replay.py --shapes       the four shape combinations against uniform shapes
python analysis/replay.py --all-priors   what each prior is worth, on real answers
python analysis/prior_basin.py      how much of the prior's setting reaches the board (slow)
python analysis/fresh_arm.py        why the fresh arm and the real fresh answers disagree
python analysis/form_default.py     which form to fit by default (slower)
python analysis/fit_rounds.py       what policy iteration converges to, and what the search costs it
python analysis/pattern_offsets.py  how much of f's error is the pattern that produced the bucket
python analysis/branch_tie.py       what the branch cap's tie tolerance costs, and the band it sits in
python analysis/rust_fit.py         whether the native trainer's fit is as good as the Python's
python analysis/results_tables.py   the two tables in README *Results* (slowest)
python analysis/head_to_head.py     the shipped solver against the reference, game by game
python analysis/head_to_head.py --baseline   against greedy entropy instead
python analysis/split_ranking.py    how much of a ranking is one guess twice
python analysis/table_layout.py     both feedback-table layouts, kernel by kernel
python analysis/fused_gain.py       what the answer-major dispatch is worth in games
```

The first four are seconds. `hard_mode.py` is two minutes for `--all-priors`, both arms over every word — cheap because the restriction it measures is also what makes a turn cheap to score. `hard_mode_rules.py` is about a minute for its two arms over every word and three for `--free`; it cannot hand a sweep to the native binary or to `common.every_answer`, neither of which can be told about a guess pool, so it plays the policy itself and `--check` is what shows that loop reproducing `every_answer` game for game. `lookahead_accuracy.py` is seven minutes over the four priors and thirteen with `--decisions`, which plays out ten guesses at each position rather than one; `exact_decisions.py` is under a minute, since the positions it asks about are the small ones. `opening.py` is sixteen minutes for the two hundred openings it defaults to, about five seconds each, and `--every-word` holds that rate over all 12,947 — the best part of a day. `turn2_regret.py` is five minutes, most of it the census: fifty guesses played out at each of the fifty-one positions the estimator decides, where the sweeps that follow come almost entirely out of the memo those playouts filled. `loss_penalty.py` is nine minutes for both sweeps over the four priors, a minute with `--shipped-only --skip-reach`, and well under one for `--policies` and about half for `--search`; `--cost` is a timing probe rather than a measurement and takes about six, while `--bound` values a single ten-word family and is seconds. `loss_cost.py` is two whole-answer sweeps per prior plus a short deadline-free playout for each losing word. `branch_cap.py` is a couple of minutes and counts rather than times, so it does not care what else the machine is doing. `ablations.py` does care, and is the exception in this directory: it is wall clock end to end and no figure of it survives background load, so run it on an idle machine or not at all. It is under a minute in its default form and about a quarter of an hour with `--sweep-games 12947`, nearly all of that the crippled sweep it measures the memo against. `--projection` adds nothing to the clock, being arithmetic over the opening scan it already timed and the rounds the cached fits recorded — but it inherits that scan's need for an idle machine, since a naive reading taken under load is what every figure it prints is a multiple of. `replay.py` is 40 s in its default form and about two minutes for either `--all-priors` or `--shapes`, which play four arms and five where the default plays two; `prior_basin.py` is seven minutes, its two sweeps replaying the regime forty times between them, and `fresh_arm.py` under two; `results_tables.py` is about two minutes, and `form_default.py` the slow one — it replays 45 fitted policies where the others replay one or two. Those two and `hard_mode.py` are the ones the native sweep carries; the figures here are with it, and about 1.7× longer without. `fit_rounds.py` is about eight minutes and leans on that sweep as heavily, being ten policy iterations and forty-odd whole-answer plays; `pattern_offsets.py` is six seconds, since it fits one model per prior off rollouts played with the endgame search off and mostly out of the memo. `branch_tie.py` is fifty whole-answer sweeps and cannot hand one of them to the native binary — its arms are orderings only `exact.py` has, so `WORDLE_NO_RUST=1` is set for it rather than chosen — and it is twelve minutes for that reason. `rust_fit.py` is eight, being thirty fits and thirty whole-answer sweeps, and it cannot hand those to the native binary either — half of them are the Python's own fits, which is the thing being compared.

`fused_gain.py` draws its answers uniformly from the whole list rather than from the prior, which is worth knowing before quoting its expected-mode figure: the dispatch speeds up the table read alone, and in expected mode that read is a small share of a scan carrying the estimator and the endgame search too. The gain is 1.02× on its own population and 1.09× on answers drawn from the prior, under either fit — so a figure that looks like it moved has usually changed population instead.

`split_ranking.py` is under ten seconds, since the positions it looks at are the small ones and nothing is played out. `table_layout.py` is about four minutes and `fused_gain.py` under one. Those two are timing probes rather than measurements, as the second half of `split_ranking.py` is: they move with the machine and with what else is running on it, and what they compare is two ways of arranging the same arithmetic, so read the ratios and not the milliseconds.

`loss_penalty.py --cost` looks like one of those and is not, which is worth knowing before skipping it. What it times is how much work the wider reach finds to do, and that is a fact about the prior: the answer log alone takes it from 1.20× to 1.49× on the shipped row, holding the code and the fit fixed, because thirteen more used words push the search's mean candidate set from 18.0 to 20.7 and the share of its calls above the default cap from 5.2% to 8.1%. Rerun it after `solutions.py --write` like any other measurement here.

Those are wall clocks on one machine, and they are far shorter than the sizes suggest for one reason: the memos fill after a few thousand positions and the rest of a run comes out of cache. Behind a fixed opening the 12,947 answers reach at most 243 distinct positions, so the memo turns a 12,947-wide computation into a 240-wide one, and that ratio is most of why any of this finishes at all.

Losing it is not a graceful slowdown but a hundredfold one, which is why `memo.Memo` drops its least recently used entry rather than refusing new ones once full, and why `WordleSolver.reset_memo` is there for a sweep that wants each run independent of the last. A run taking far longer than its row above is worth interrupting and diagnosing rather than waiting out.

| script | backs |
| --- | --- |
| `regime.py` | README *The regime the defaults describe*, SOLVER_EXPLAINED §4; its last block the implied-rate table in §14.3, which is why the two priors need each other |
| `repeat_shapes.py` | README *The repeat prior's shape* table, SOLVER_EXPLAINED §5.1 |
| `drift.py` | the closing paragraph of README *The repeat prior's shape* |
| `curation_shapes.py` | SOLVER_EXPLAINED §5.2 — the shape table, the snapshot's forward-prediction table, and why the hedge is flat; the opening of README *The curation prior's shape* |
| `id_eras.py` | the id evidence for the off-list block in SOLVER_EXPLAINED §5.2. Its header carries the rest of the reconstruction, which that section deliberately does not: the 2022 purge, where the displaced words went, and what a repeat costs in numbering |
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
| `prior_basin.py` | README *The prior sits in a flat basin* |
| `fresh_arm.py` | the fresh-arm paragraphs of SOLVER_EXPLAINED §14.1 and README *What the two shapes are worth together* |
| `replay.py` | the head-to-head paragraph under README *The curation prior's shape*; its `--shapes` the whole of README *What the two shapes are worth together* and the regime replay under *The repeat prior's shape*; its `--all-priors` the whole of *What each prior is worth on real answers* |
| `form_default.py` | README *Which form is the default* |
| `results_tables.py` | both tables in README *Results*, and its `--part priors` the bonus/curvature split in SOLVER_EXPLAINED §6 |
| `head_to_head.py` | README *Where the estimator's guesses come from* |
| `split_ranking.py` | the *Alternatives* card under README *In a browser*, and the runners-up paragraph closing *The exact endgame* |
| `table_layout.py` | SOLVER_EXPLAINED §8 *Storing the table the other way round*, the per-kernel figures |
| `fused_gain.py` | the same section's whole-game figures, 1.09× and 1.63× |

## Rerunning them

Rerun after `solutions.py --write`. Every one of these numbers moves with the answer log, and nothing warns you: the estimator cache is keyed on the `PriorConfig` alone, so a stale fit is loaded silently. Refit before rerunning `replay.py` or it measures an old policy against new answers.

**Not every arm moves.** `priors.build_weights` consults the used-word mask only under `PriorConfig.repeat`, so a repeat-off prior is the same distribution after the write as before it, its fit is bit-for-bit the same fit, and every figure weighted by it over every word is the same figure. `priors.depends_on_answer_record` is that rule in one line; `wordle_solver.py train --repeat-only` skips the two of the five fits it exempts, and `--repeat-only` drops the same arms from every script here that sweeps priors, printing which it dropped so a table assembled from two runs says so.

The exemption is about where the weights come from, not what an arm is called. The scripts that replay the real answer record — `replay.py`, `prior_basin.py`, `fresh_arm.py` — score their `none` and `curation only` arms over a log that just got longer, so every arm of theirs moves and they take no such flag. And carrying an old row into a new table is honest only while nothing else moved either: a `word_lists.py --write`, a fresh `wordlebot.py` scrape or a change to `noncurated_prob` invalidates a repeat-off row as surely as a new answer would, and nothing records which run a row came from.

`replay.py --dump PATH` writes every arm's per-game guess counts out as JSON — 190 regime games plus the three arms' 21, 46 and 154 events, one row per arm. Take one whenever you run the thing, and a contrast nobody thought to print becomes arithmetic over that file rather than another replay: a pair the tables skip, a subgroup split some other way. None is kept here at present. A dump is true only of the fit and the answer log it was made against, and nothing would catch a checked-in one drifting out of step with either — worth weighing against the two minutes `--all-priors` takes to make a fresh one.

**Check the method before believing a changed number.** `repeat_shapes.py --events 18` scores only the first 18 repeats, so its answer is frozen whatever the answer log does — uniform −134.5, binned −116.1 at +18.4, α = 1.0 at −118.4, and +1.024 ± 0.235 nats/event at 4.4σ winning 14 of 18. If those still come out, a number that moved moved because the data did. `regime.py`'s 2021–2025 row (1,657 / 0.00% / 1.75%) and `curation_shapes.py`'s uniform baseline (−9.27) are fixed points for the same reason. So is the whole first half of `id_eras.py`: the id matching the file position for 284 puzzles, the drift climbing to 22, and the runs of 4, 12 and 7 in the appended block are settled history, and only its era-2 fit moves as answers arrive.

**An older answer log is reproducible in place**, which is the other half of that question. `load_data(mask_last=n)` drops the most recent n entries of `solutions.json` and rebuilds the priors from what is left — the same mechanism that keeps today's unplayed answer out of the used-word prior at the default of 1 — and every front end takes it as `--mask-last`. Wind the log back to the day a figure was recorded, rerun, and whether the data moved or the code did stops being a matter of opinion. The fit does not wind back with it: `estimator.cache_path` is keyed on the `PriorConfig` slug alone, so the current `f` loads against the truncated log without complaint. Refitting is the wrong fix rather than the slow one — `train --mask-last n` writes to that same path, so it replaces the prior's live fit with one describing a deliberately short log, under a filename recording nothing about the difference. Copy `params_<slug>.json` aside first, or take the stale `f` as the known cost of a cheap check and say which you did.

## Enumerating instead of sampling

`hard_mode.py`, `loss_penalty.py`, `results_tables.py` and `form_default.py` do not draw answers and report a standard error. They do not have to: the solver is deterministic, so a game's outcome is a pure function of the answer, and playing all 12,947 words once with the prior as weights *computes* the mean and the loss probability. There is no sampling distribution left to put an interval around, and `common.every_answer` is the one implementation they share.

It matters most where the effect is small. The loss penalty is worth about +0.0005 guesses, while a 20,000-game sample carries an error near 0.009 — twenty times larger, and only pairing ever made those figures readable. Removing the noise also moved conclusions, not just digits: under the flat prior the nine forms of `f` span 0.0013 guesses rather than the 0.0110 a 3,000-answer holdout showed, which is most of what used to separate them.

Three things change shape in the output. A loss *rate* becomes the exact prior mass on the words that lose, which is not the fraction of words that lose. "Which games changed" becomes a list of words. And a block sd or a holdout seed stops existing, so a ranking is a fact about the prior rather than something that might move on a rerun — which makes a small gap harder to dismiss, not easier.

Enumerating is usually the cheaper option too. The memos in `WordleSolver` and `ExactEndgame` saturate after a few thousand positions, so most of a sweep runs out of cache — the marginal rate climbs past 4,000 games a second, and the arm that looks slowest from a cold start is not the one that costs the most.

`replay.py` is the exception that stays sampled, and for a good reason: it replays the *real* answer sequence, so its sample is a historical record rather than a draw from the prior. There is nothing to enumerate.

That is also what lets its arms be compared across *different* priors, which no enumerated script can offer. An enumerated mean weights every word by the prior on trial, so two such rows answer the same question about two different populations and the gap between them means little — `results_tables.py` prints exactly that warning above its own prior table, and `head_to_head.py` states the weighting and leaves you to draw it. A replay scores every arm on the answers the NYT published, which none of them had a hand in choosing, so `--all-priors` is where the four combinations can be put side by side.

`lookahead_accuracy.py` and `exact_decisions.py` sit between the two. Every value they report is exact — a position played out once per candidate answer — but which positions they visit, and how heavily each counts, comes from a sample of games. Their means carry a sampling error that the enumerated ones do not. The machinery is shared: `common.collect_positions`, `common.guess_value` and their neighbors are the one implementation both read.

## The three fits `--shapes` needs

Comparing the sharpened shapes against the uniform ones needs fits for `repeat1_curation1`, `repeat1_curation1_r-binned` and `repeat1_curation1_c-wordlebot`, and a prior with no fit raises rather than playing an invented `f`. All three are in `estimator_cache/` now, fitted the way the shipped five were: nine forms archived by `wordle_solver.py compare`, ranked exactly the way `form_default.py` ranks them, and the winner written as the active fit. They came out `spline`, `anchored` and `spline`, and all three open on `salet`.

All eight are refitted together against one answer record, which is what makes the run legitimate — every arm in a `--shapes` table has to carry a fit of the same history, or the table is comparing solvers rather than shapes.

## What is not reproduced here

`replay.py`'s arms are comparable to each other, but a run is not comparable to an older run. Its numbers depend on the fitted estimator as much as on the data, and a refit can move a configuration's winning form, so its output cannot be diffed across one. Four of the eight moved in the 2026-08-23 refit and all five had moved in the one before it. The same holds for `form_default.py`: it scores the fits in `estimator_cache/forms/`, so a refit moves every row.

`hard_mode.py` is the one that survives a refit, and it is worth knowing why: its two arms play the *same* `f`, so a refit moves both columns together and mostly cancels in the paired difference. Read the difference across refits; read the two means only against each other. `lookahead.py` does *not* get that protection even though it also pairs two arms on one fit: what it measures is how much room the base policy leaves above itself, so a better-fitted `f` should shrink it. Expect that as a direction and not as a prediction — the effect is a few thousandths of a guess against a standard error of about half that, and the 2026-08-23 refit improved every arm's base policy while leaving lookahead worth marginally more.

Several claims about the endgame search have no script, and all of them rest on position samples that were not kept: that a branch cap of 25 measures no better and that two plies costs 205 s for the opening position alone (README *Lookahead*); the 1,275-position comparison of capped against uncapped search, and the memo's 373-repeat position and its ~30% share of the search's time (README *The exact endgame*, SOLVER_EXPLAINED §10(d) and §12). They are timing and tie-break facts rather than answer-log ones, so they do not move with `solutions.json` — but nothing here would catch them going stale either.

Two claims about the fit rest on implementations that were not kept, and neither has a script here. Warm starting policy iteration from the cached fit (README *Warm starting from the previous fit*) needed a seed argument on `train.policy_iteration`; conditioning `f` on the feedback pattern (README *Conditioning f on the pattern*) needed a kernel carrying the offsets, and `pattern_offsets.py` reproduces that measurement's residual half but not its verdict. Both were measured on the objective everything else here uses — every answer, weighted by the prior — and both were rejected on it.

Two figures the documents used to carry could not be reproduced exactly. The window variants' *spread* was −124…−129; the best variant here matches, but the worst is far worse, so the original must have defined the band more robustly. Only the best of the sixteen is quoted now, which is the number the argument actually needs. The drift rate's standard error was ±0.39 against ±1.56 here; the slope agrees to two decimals, so the parameterization is right and the error model is not.
