"""What the shapes are worth in guesses, replayed against real answers.

Two measurements, both of which need the solver rather than just the answer log,
so this is the slow one -- a few minutes.

1. The regime replay: binned against uniform over every real puzzle since the
   regime began, with the prior rebuilt from strictly earlier puzzles each day.
2. The three-arm table: repeats / off-list / still-unused, blended at the
   regime's own mixture.  A shape can only be judged on answers that actually
   happened; scoring it against answers sampled from its own belief is grading
   its own homework.

Backs the closing paragraphs of README *Sharpening the repeat prior* and the
table in *Shaping the two priors*.
"""

import copy

import numpy as np

import common
from priors import PriorConfig
from solver import WordleSolver
from wordle_data import load_data

data = load_data(mask_last=0)
sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
START = common.regime_start(sol)
END = len(words)

UNIFORM = PriorConfig(repeat_shape="uniform", curation_shape="uniform")
BINNED = PriorConfig(repeat_shape="binned", curation_shape="uniform")


def as_of(t):
    """`data` with its answer history truncated to strictly before puzzle t."""
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:t]
    return view


def guesses(config, t, answer):
    solver = WordleSolver(data=as_of(t), config=config, quiet=True)
    out = solver.play(data.word2ind[answer])
    return len(out[0] if isinstance(out, tuple) else out)


common.header("1. regime replay: binned vs uniform repeat shape")
lb, lu, repeat = [], [], []
seen = set(words[:START])
for t in range(START, END):
    w = words[t]
    lb.append(guesses(BINNED, t, w))
    lu.append(guesses(UNIFORM, t, w))
    repeat.append(w in seen)
    seen.add(w)
lb, lu, repeat = np.array(lb), np.array(lu), np.array(repeat)
mean, se, d = common.paired(lb, lu)
print(f"puzzles replayed        {len(lb)}")
print(f"binned                  {lb.mean():.4f} guesses")
print(f"uniform                 {lu.mean():.4f} guesses")
print(f"paired difference       {mean:+.4f} +- {se:.4f}   (negative = binned better)")
print(f"paired sd               {d.std(ddof=1):.2f}")
print(f"games changing length   {int((d != 0).sum())} "
      f"({int((d < 0).sum())} better, {int((d > 0).sum())} worse)")
mr, ser, _ = common.paired(lb[repeat], lu[repeat])
mn, sen, _ = common.paired(lb[~repeat], lu[~repeat])
print(f"  on the {int(repeat.sum()):3d} repeats     {mr:+.3f} +- {ser:.3f}")
print(f"  on the other {int((~repeat).sum()):3d}      {mn:+.3f} +- {sen:.3f}")
print(f"games needed for 0.02 at 2 sigma: {(2*d.std(ddof=1)/0.02)**2:,.0f}")

common.header("2. three arms, blended at the regime mixture")
reps = [(t, w) for t, w, _ in common.repeat_events(words)]
offs = common.offlist_events(words, curated)
fresh = sorted(curated - set(words))
rep_rate = sum(1 for t, _ in reps if t >= START) / (END - START)
off_rate = sum(1 for t, _ in offs if t >= START) / (END - START)
mix = (rep_rate, off_rate, 1 - rep_rate - off_rate)
print(f"arms: {len(reps)} repeats, {len(offs)} off-list, {len(fresh)} still unused")
print(f"mixture: {100*mix[0]:.1f}% / {100*mix[1]:.1f}% / {100*mix[2]:.1f}%")
print()

ROWS = (("curation `wordlebot` alone",
         PriorConfig(repeat_shape="uniform", curation_shape="wordlebot")),
        ("repeat `binned` alone",
         PriorConfig(repeat_shape="binned", curation_shape="uniform")),
        ("`binned` + `frequency`",
         PriorConfig(repeat_shape="binned", curation_shape="frequency")),
        ("`binned` + `wordlebot`",
         PriorConfig(repeat_shape="binned", curation_shape="wordlebot")))


def arm(config, events, at_end=False):
    return [guesses(config, END if at_end else t, w) for t, w in events]


fresh_events = [(END, w) for w in fresh]
base = (arm(UNIFORM, reps), arm(UNIFORM, offs), arm(UNIFORM, fresh_events, True))

print(f"  {'shapes':26s} {'repeat':>16s} {'off-list':>16s} {'fresh':>16s} "
      f"{'blended':>16s}")
print("  " + "-" * 94)
for label, config in ROWS:
    got = (arm(config, reps), arm(config, offs), arm(config, fresh_events, True))
    cells = [common.paired(g, b) for g, b in zip(got, base)]
    blend = sum(m * c[0] for m, c in zip(mix, cells))
    blend_se = np.sqrt(sum((m * c[1]) ** 2 for m, c in zip(mix, cells)))
    body = " ".join(f"{c[0]:+9.3f}+-{c[1]:.3f}" for c in cells)
    print(f"  {label:26s} {body} {blend:+9.3f}+-{blend_se:.3f}")

print()
print("  These columns are comparable across rows -- every row is scored against")
print("  the same real answers -- but not against numbers from a run whose")
print("  estimator was fitted separately.  Refit before comparing across time.")
