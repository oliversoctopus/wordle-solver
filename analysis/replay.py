"""What a curation shape is worth in guesses, replayed against real answers.

Two measurements, both of which need the solver rather than just the answer log,
so this is the slow one -- a few minutes.

1. The regime replay: every real puzzle since the regime began, with the prior
   rebuilt from strictly earlier puzzles each day.
2. The three-arm table: repeats / off-list / still-unused, blended at the
   regime's own mixture.  A shape can only be judged on answers that actually
   happened; scoring it against answers sampled from its own belief is grading
   its own homework.

Both compare the two shape combinations that have a cached fit, `frequency`
against the shipped `wordlebot`.  The uniform shapes were arms here until
2026-08-10 and are gone with their fits: an unfitted f is no longer allowed to
play, and this is a measurement of shapes, not of who has been trained.
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

SHIPPED = PriorConfig(repeat_shape="binned", curation_shape="wordlebot")
RIVAL = PriorConfig(repeat_shape="binned", curation_shape="frequency")


def as_of(t):
    """`data` with its answer history truncated to strictly before puzzle t."""
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:t]
    return view


def guesses(config, t, answer):
    solver = WordleSolver(data=as_of(t), config=config)
    out = solver.play(data.word2ind[answer])
    return len(out[0] if isinstance(out, tuple) else out)


common.header("1. regime replay: curation `frequency` vs `wordlebot`")
lr, ls, repeat, offlist = [], [], [], []
seen = set(words[:START])
for t in range(START, END):
    w = words[t]
    lr.append(guesses(RIVAL, t, w))
    ls.append(guesses(SHIPPED, t, w))
    repeat.append(w in seen)
    # off-list *given not a repeat*, the order the model draws in
    offlist.append(w not in seen and w not in curated)
    seen.add(w)
lr, ls = np.array(lr), np.array(ls)
repeat, offlist = np.array(repeat), np.array(offlist)
mean, se, d = common.paired(lr, ls)
print(f"puzzles replayed        {len(lr)}")
print(f"frequency               {lr.mean():.4f} guesses")
print(f"wordlebot (shipped)     {ls.mean():.4f} guesses")
print(f"paired difference       {mean:+.4f} +- {se:.4f}   (negative = frequency better)")
print(f"paired sd               {d.std(ddof=1):.2f}")
print(f"games changing length   {int((d != 0).sum())} "
      f"({int((d < 0).sum())} better, {int((d > 0).sum())} worse)")
# the two subgroups either shape is meant to move
for label, mask in (("repeats", repeat), ("off-list", offlist)):
    if mask.any():
        m, s, _ = common.paired(lr[mask], ls[mask])
        print(f"  on the {int(mask.sum()):3d} {label:9s} {m:+.3f} +- {s:.3f}")
rest = ~(repeat | offlist)
m, s, _ = common.paired(lr[rest], ls[rest])
print(f"  on the other {int(rest.sum()):3d}      {m:+.3f} +- {s:.3f}")
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


def arm(config, events, at_end=False):
    return [guesses(config, END if at_end else t, w) for t, w in events]


fresh_events = [(END, w) for w in fresh]
base = (arm(SHIPPED, reps), arm(SHIPPED, offs), arm(SHIPPED, fresh_events, True))

print(f"  {'shapes':26s} {'repeat':>16s} {'off-list':>16s} {'fresh':>16s} "
      f"{'blended':>16s}")
print("  " + "-" * 94)
got = (arm(RIVAL, reps), arm(RIVAL, offs), arm(RIVAL, fresh_events, True))
cells = [common.paired(g, b) for g, b in zip(got, base)]
blend = sum(m * c[0] for m, c in zip(mix, cells))
blend_se = np.sqrt(sum((m * c[1]) ** 2 for m, c in zip(mix, cells)))
body = " ".join(f"{c[0]:+9.3f}+-{c[1]:.3f}" for c in cells)
print(f"  {'`binned` + `frequency`':26s} {body} {blend:+9.3f}+-{blend_se:.3f}")

print()
print("  Against `binned` + `wordlebot`, on the same real answers.  Not")
print("  comparable to a run whose estimator was fitted separately: refit both")
print("  configurations before comparing across time.")
