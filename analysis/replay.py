"""What a prior is worth in guesses, replayed against real answers.

Two measurements, both of which need the solver rather than just the answer log,
so this is the slow one -- a few minutes.

1. The regime replay: every real puzzle since the regime began, with the prior
   rebuilt from strictly earlier puzzles each day.
2. The three-arm table: repeats / off-list / still-unused, blended at the
   regime's own mixture.  A prior can only be judged on answers that actually
   happened; scoring it against answers sampled from its own belief is grading
   its own homework, which is what makes this the one place arms under
   *different* priors can be put side by side.

By default the arms are the two curation shapes that have a cached fit,
`frequency` against the shipped `wordlebot`.  The uniform shapes were arms here
until 2026-08-10 and are gone with their fits: an unfitted f is no longer
allowed to play, and this is a measurement of shapes, not of who has been
trained.  `--all-priors` switches to the four on/off combinations instead.

Every arm plays the same real answers and every mean is an unweighted average
over them, so the arms are comparable to each other however much their priors
disagree.  A whole run is not comparable to an older run: each arm is a prior
*and* the f fitted for it, so a refit moves the numbers as surely as a longer
answer log does.

    python analysis/replay.py                the two curation shapes
    python analysis/replay.py --all-priors   the four prior combinations

Backs the head-to-head paragraph under README *The curation prior's shape*, and
with `--all-priors` the table under *What each prior is worth on real answers*.
"""

import argparse
import copy
import itertools
import time

import numpy as np

import common
from priors import PriorConfig
from solver import WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--all-priors", action="store_true",
                    help="the four prior combinations against the shipped one, "
                         "rather than the two curation shapes")
parser.add_argument("--dump", metavar="PATH",
                    help="write every arm's per-game guess counts here, so a "
                         "contrast nobody thought to print can be computed "
                         "later without replaying anything")
args = parser.parse_args()
STARTED = time.time()

data = load_data(mask_last=0)
sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
START = common.regime_start(sol)
END = len(words)

SHIPPED = PriorConfig(repeat_shape="binned", curation_shape="wordlebot")
RIVAL = PriorConfig(repeat_shape="binned", curation_shape="frequency")

if args.all_priors:
    # the reference is `both`, which is the shipped prior with the shipped shapes
    BASE = ("both", SHIPPED)
    ARMS = [("none", PriorConfig(repeat=False, curation=False)),
            ("curation only", PriorConfig(repeat=False)),
            ("repeat only", PriorConfig(curation=False))]
    SUBTITLE = "the four prior combinations"
else:
    BASE = ("wordlebot (shipped)", SHIPPED)
    ARMS = [("frequency", RIVAL)]
    SUBTITLE = "curation `frequency` vs `wordlebot`"

EVERY = ARMS + [BASE]
WIDTH = max(len(label) for label, _ in EVERY)

# each prior's marginal, with the other prior off and then on
CONTRASTS = {
    "curation, repeat off": ("none", "curation only"),
    "curation, repeat on": ("repeat only", "both"),
    "repeat, curation off": ("none", "repeat only"),
    "repeat, curation on": ("curation only", "both"),
}
CWIDTH = max(len(label) for label in CONTRASTS) + 1

# the part-2 table has to hold whichever labels it ends up printing
TWIDTH = max(WIDTH, CWIDTH) if args.all_priors else WIDTH


def as_of(t):
    """`data` with its answer history truncated to strictly before puzzle t."""
    view = copy.copy(data)
    view.past_solutions = data.past_solutions[:t]
    return view


def guesses(config, t, answer):
    solver = WordleSolver(data=as_of(t), config=config)
    out = solver.play(data.word2ind[answer])
    return len(out[0] if isinstance(out, tuple) else out)


common.header(f"1. regime replay: {SUBTITLE}")
runs = {label: [] for label, _ in EVERY}
repeat, offlist = [], []
seen = set(words[:START])
for t in range(START, END):
    w = words[t]
    for label, config in EVERY:
        runs[label].append(guesses(config, t, w))
    repeat.append(w in seen)
    # off-list *given not a repeat*, the order the model draws in
    offlist.append(w not in seen and w not in curated)
    seen.add(w)
runs = {label: np.array(run) for label, run in runs.items()}
repeat, offlist = np.array(repeat), np.array(offlist)
rest = ~(repeat | offlist)

print(f"puzzles replayed        {END - START}")
for label, _ in EVERY:
    print(f"  {label:{WIDTH}s} {runs[label].mean():.4f} guesses")
print(f"\nagainst `{BASE[0]}`, paired on the same answers "
      f"(negative = the arm is better)\n")
for label, _ in ARMS:
    mean, se, d = common.paired(runs[label], runs[BASE[0]])
    print(f"  {label:{WIDTH}s} {mean:+.4f} +- {se:.4f}   sd {d.std(ddof=1):.2f}, "
          f"{int((d != 0).sum())} games changed "
          f"({int((d < 0).sum())} better, {int((d > 0).sum())} worse)")
    # the two subgroups a prior is meant to move, and everything else
    for name, mask in (("repeats", repeat), ("off-list", offlist),
                       ("other", rest)):
        if mask.any():
            m, s, _ = common.paired(runs[label][mask], runs[BASE[0]][mask])
            print(f"  {'':{WIDTH}s}   on the {int(mask.sum()):3d} {name:9s} "
                  f"{m:+.3f} +- {s:.3f}")
    print(f"  {'':{WIDTH}s}   games needed for 0.02 at 2 sigma: "
          f"{(2 * d.std(ddof=1) / 0.02) ** 2:,.0f}")

if args.all_priors:
    # What turning one prior on is worth with the other off, against what it is
    # worth with the other on.  All four arms played the same answers, so each
    # is a paired contrast like any other, and the interaction is the same
    # number read either way round.
    print()
    for label, (a, b) in CONTRASTS.items():
        contrast = runs[a] - runs[b]
        print(f"  {label:{CWIDTH}s} {contrast.mean():+.4f} +- "
              f"{contrast.std(ddof=1) / np.sqrt(len(contrast)):.4f}")
    inter = ((runs["none"] - runs["repeat only"])
             - (runs["curation only"] - runs["both"]))
    print(f"  {'interaction':{CWIDTH}s} {inter.mean():+.4f} +- "
          f"{inter.std(ddof=1) / np.sqrt(len(inter)):.4f}")

    # every pair, so a contrast this script did not think to name can still be
    # read off without replaying anything
    print(f"\n  all six pairs, paired on the same answers\n")
    for a, b in itertools.combinations([label for label, _ in EVERY], 2):
        d = runs[a] - runs[b]
        print(f"  {a:{WIDTH}s} - {b:{WIDTH}s} {d.mean():+.4f} +- "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.4f}   sd {d.std(ddof=1):.3f}")

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


def three_arms(config):
    return (arm(config, reps), arm(config, offs), arm(config, fresh_events, True))


played = {label: three_arms(config) for label, config in EVERY}
base = played[BASE[0]]


def blended(cells):
    """A row's three paired cells, and their blend at the regime mixture."""
    blend = sum(m * c[0] for m, c in zip(mix, cells))
    blend_se = np.sqrt(sum((m * c[1]) ** 2 for m, c in zip(mix, cells)))
    body = " ".join(f"{c[0]:+9.3f}+-{c[1]:.3f}" for c in cells)
    return f"{body} {blend:+9.3f}+-{blend_se:.3f}"


print(f"  {'arm':{TWIDTH}s} {'repeat':>16s} {'off-list':>16s} {'fresh':>16s} "
      f"{'blended':>16s}")
print("  " + "-" * (TWIDTH + 68))
for label, _ in ARMS:
    cells = [common.paired(g, b) for g, b in zip(played[label], base)]
    print(f"  {label:{TWIDTH}s} {blended(cells)}")

if args.all_priors:
    print()
    for label, (a, b) in CONTRASTS.items():
        cells = [common.paired(g, h) for g, h in zip(played[a], played[b])]
        print(f"  {label:{TWIDTH}s} {blended(cells)}")
    cells = [common.paired(np.array(a) - np.array(b), np.array(c) - np.array(d))
             for a, b, c, d in zip(played["none"], played["repeat only"],
                                   played["curation only"], base)]
    print(f"  {'interaction':{TWIDTH}s} {blended(cells)}")

print()
print(f"  Against `{BASE[0]}`, on the same real answers.  The arms are")
print("  comparable to each other; the run is not comparable to one made")
print("  against a different fit, so refit every arm before comparing across")
print("  time.")

if args.dump:
    from wordle_data import write_json
    write_json(args.dump, {
        "regime": {"start": START, "end": END,
                   "answers": words[START:END],
                   "guesses": {label: [int(g) for g in runs[label]]
                               for label, _ in EVERY}},
        "arms": {
            "mixture": list(mix),
            "events": {"repeat": [w for _, w in reps],
                       "offlist": [w for _, w in offs],
                       "fresh": fresh},
            "guesses": {label: [[int(g) for g in a] for a in played[label]]
                        for label, _ in EVERY},
        },
    })
    print(f"\n  raw per-game guess counts written to {args.dump}")

print(f"\n  {time.time() - STARTED:.0f} s")
