# analysis/

Every measured number in `README.md` and `SOLVER_EXPLAINED.md` about the two
priors comes from one of these. They read `data/` and print; they write nothing.

```
python analysis/regime.py           the regime table and the pool arithmetic
python analysis/repeat_shapes.py    prequential comparison of repeat shapes
python analysis/drift.py            is the source window moving?
python analysis/curation_shapes.py  prequential comparison of curation shapes
python analysis/replay.py           what the shapes are worth in guesses (slow)
```

The first four are seconds. `replay.py` plays a few thousand real games and
takes a few minutes.

| script | backs |
| --- | --- |
| `regime.py` | README *Both defaults describe a regime six months old*, SOLVER_EXPLAINED §4.1 |
| `repeat_shapes.py` | README *Sharpening the repeat prior* table, SOLVER_EXPLAINED §4.2 |
| `drift.py` | the closing paragraph of README *Sharpening the repeat prior* |
| `curation_shapes.py` | SOLVER_EXPLAINED §4.3, the opening of README *Shaping the two priors* |
| `replay.py` | the replay paragraph and the three-arm table |

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

## What is not reproduced here

`replay.py` is only comparable against itself. Its numbers depend on the fitted
estimator as much as on the data, and every configuration's winning form moved
in the 2026-08-10 refit, so its output cannot be diffed against a run from
before that.

Two figures the documents used to carry could not be reproduced exactly. The
window variants' *spread* was −124…−129; the best variant here matches, but the
worst is far worse, so the original must have defined the band more robustly —
only the best of the sixteen is quoted now, which is the number the argument
actually needs. And the drift rate's standard error was ±0.39 against ±1.56
here; the slope agrees to two decimals, so the parameterization is right and the
error model is not.
