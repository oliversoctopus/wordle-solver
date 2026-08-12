# analysis/

Every measured number in `README.md` and `SOLVER_EXPLAINED.md` about the two
priors, about which form of `f` is the default, and about what hard mode costs,
comes from one of these. They read `data/` and the estimator cache and print;
they write nothing.

```
python analysis/regime.py           the regime table and the pool arithmetic
python analysis/repeat_shapes.py    prequential comparison of repeat shapes
python analysis/drift.py            is the source window moving?
python analysis/curation_shapes.py  prequential comparison of curation shapes
python analysis/hard_mode.py        what the hard-mode restriction costs
python analysis/replay.py           what the shapes are worth in guesses (slow)
python analysis/form_default.py     which form to fit by default (slower)
python analysis/results_tables.py   the two tables in README *Results* (slowest)
```

The first four are seconds. `hard_mode.py` is 15 s for the default prior and two
minutes for `--all-priors`, both arms at 20,000 games — cheap because the
restriction it measures is also what makes a turn cheap to score. `replay.py`
plays a few thousand real games and takes a few minutes; `form_default.py`
replays 45 fitted policies and takes longer again; `results_tables.py` is an
hour and a quarter at its default sizes, most of it the flat-prior rows, where
more candidates survive each guess and the endgame search runs on bigger sets.

| script | backs |
| --- | --- |
| `regime.py` | README *Both defaults describe a regime six months old*, SOLVER_EXPLAINED §4.1 |
| `repeat_shapes.py` | README *Sharpening the repeat prior* table, SOLVER_EXPLAINED §4.2 |
| `drift.py` | the closing paragraph of README *Sharpening the repeat prior* |
| `curation_shapes.py` | SOLVER_EXPLAINED §4.3 including why the hedge is flat, the opening of README *Shaping the two priors* |
| `hard_mode.py` | the table and the entropy paragraph under README *Hard mode* |
| `replay.py` | the head-to-head paragraph under README *Shaping the two priors*, not the table's uniform-baseline rows |
| `form_default.py` | README *Which form is the default* |
| `results_tables.py` | both tables in README *Results* |

## Rerunning them

Rerun after `solutions.py --write`. Every one of these numbers moves with the
answer log, and nothing warns you: the estimator cache is keyed on the
`PriorConfig` alone, so a stale fit is loaded silently. Refit before rerunning
`replay.py` or it measures an old policy against new answers.

**Check the method before believing a changed number.** `repeat_shapes.py
--events 18` scores only the first 18 repeats and should print the figures the
README carried before the 2026-08-10 update — uniform −134.5, binned −116.1 at
+18.4, α = 1.0 at −118.4, and +1.024 ± 0.235 nats/event at 4.4σ winning 14 of
18. If those still come out, a number that moved moved because the data did.
`regime.py`'s 2021–2025 row (1,657 / 0.00% / 1.75%) and `curation_shapes.py`'s
uniform baseline (−9.27) are fixed points for the same reason.

## `replay.py` lost its uniform-shape arms

It used to compare the sharpened shapes against the uniform ones, which needed
fits for `repeat1_curation1`, `repeat1_curation1_r-binned` and
`repeat1_curation1_c-wordlebot`. Those fits are no longer in the repo, and since
2026-08-10 a prior with no fit raises instead of playing an invented `f`. The
arms were removed rather than retrained. What is left is the comparison both
remaining fits support: curation `frequency` against the shipped `wordlebot`,
over the regime and by arm.

## What is not reproduced here

`replay.py` is only comparable against itself. Its numbers depend on the fitted
estimator as much as on the data, and every configuration's winning form moved
in the 2026-08-10 refit, so its output cannot be diffed against a run from
before that. The same holds for `form_default.py`: it scores the fits in
`estimator_cache/forms/`, so a refit moves every row.

`hard_mode.py` is the one that survives a refit, and it is worth knowing why: its
two arms play the *same* `f`, so a refit moves both columns together and mostly
cancels in the paired difference. Read the difference across refits; read the
two means only against each other.

Three rows of the README *Shaping the two priors* table are history now, not
output: `wordlebot` alone, `binned` alone, and the uniform baseline they are
measured against. Reproducing them means fitting the three uniform-shape
configurations again. That is a deliberate choice about what this repo ships,
and not something a rerun should do behind your back.

Two figures the documents used to carry could not be reproduced exactly. The
window variants' *spread* was −124…−129; the best variant here matches, but the
worst is far worse, so the original must have defined the band more robustly.
Only the best of the sixteen is quoted now, which is the number the argument
actually needs. The drift rate's standard error was ±0.39 against ±1.56 here;
the slope agrees to two decimals, so the parameterization is right and the error
model is not.
