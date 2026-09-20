"""What a prior is worth in guesses, replayed against real answers.

Two measurements: the regime replay, every real puzzle since the regime began
with the prior rebuilt from earlier puzzles each day; and the three-arm table,
repeats, off-list and the regime's fresh curated answers, blended at the
regime's mixture.  The default arms are the two curation shapes; `--shapes`
runs the four sharpened combinations against uniform, `--all-priors` the four
prior combinations.  Backs README *The curation prior's shape*, *What the two
shapes are worth together* and *What each prior is worth on real answers*.
"""

import argparse
import itertools
import time

import numpy as np

import common
from priors import PriorConfig
from solver import WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--all-priors", action="store_true",
                  help="the four prior combinations against the shipped one, "
                       "rather than the two curation shapes")
mode.add_argument("--shapes", action="store_true",
                  help="the four sharpened shape combinations against uniform "
                       "shapes, both priors on throughout")
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

# the three `--shapes` rows §6.3 puts side by side: each shape sharpened alone,
# and the pair, so their blends can be checked for additivity
CURATION_ALONE = "curation `wordlebot` alone"
REPEAT_ALONE = "repeat `binned` alone"
TOGETHER = "`binned` + `wordlebot`"

if args.all_priors:
    # the reference is `both`, which is the shipped prior with the shipped shapes
    BASE = ("both", SHIPPED)
    ARMS = [("none", PriorConfig(repeat=False, curation=False)),
            ("curation only", PriorConfig(repeat=False)),
            ("repeat only", PriorConfig(curation=False))]
    SUBTITLE = "the four prior combinations"
elif args.shapes:
    # both priors on in every arm; only the shapes inside them differ
    BASE = ("uniform shapes", PriorConfig(repeat_shape="uniform",
                                          curation_shape="uniform"))
    ARMS = [(CURATION_ALONE, PriorConfig(repeat_shape="uniform",
                                         curation_shape="wordlebot")),
            (REPEAT_ALONE, PriorConfig(repeat_shape="binned",
                                       curation_shape="uniform")),
            ("`binned` + `frequency`", RIVAL),
            (TOGETHER, SHIPPED)]
    SUBTITLE = "the four shape combinations"
else:
    BASE = ("wordlebot (shipped)", SHIPPED)
    ARMS = [("frequency", RIVAL)]
    SUBTITLE = "curation `frequency` vs `wordlebot`"

EVERY = ARMS + [BASE]
WIDTH = max(len(label) for label, _ in EVERY)

# what the documents quote is recorded under the mode's own prefix
TAG = "priors." if args.all_priors else "shapes." if args.shapes else ""
KEYS = {"frequency": "frequency", CURATION_ALONE: "curation_alone",
        REPEAT_ALONE: "repeat_alone", "`binned` + `frequency`": "binned_frequency",
        TOGETHER: "together", "none": "none", "curation only": "curation_only",
        "repeat only": "repeat_only", "wordlebot (shipped)": "shipped",
        "uniform shapes": "uniform", "both": "both"}
neg = common.neg


def rec(key, value):
    return common.record(TAG + key, value)


def pm(mean, se, digits=3):
    return neg(f"{mean:+.{digits}f} \u00b1 {se:.{digits}f}")

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


def guesses(config, t, answer):
    solver = WordleSolver(data=common.as_of(data, t), config=config)
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
rec("puzzles", END - START)
for label, _ in EVERY:
    print(f"  {label:{WIDTH}s} {runs[label].mean():.4f} guesses")
    rec(f"{KEYS[label]}.mean", float(runs[label].mean()))
print(f"\nagainst `{BASE[0]}`, paired on the same answers "
      f"(negative = the arm is better)\n")
for name, mask in (("repeats", repeat), ("offlist", offlist), ("other", rest)):
    rec(f"n_{name}", int(mask.sum()))
regime_rows = []
for label, _ in ARMS:
    mean, se, d = common.paired(runs[label], runs[BASE[0]])
    print(f"  {label:{WIDTH}s} {mean:+.4f} +- {se:.4f}   sd {d.std(ddof=1):.2f}, "
          f"{int((d != 0).sum())} games changed "
          f"({int((d < 0).sum())} better, {int((d > 0).sum())} worse)")
    key = KEYS[label]
    rec(f"{key}.vs", pm(mean, se, 4))
    rec(f"{key}.delta", float(mean))
    rec(f"{key}.se", float(se))
    rec(f"{key}.sigma", float(abs(mean) / se))
    rec(f"{key}.changed", int((d != 0).sum()))
    rec(f"{key}.better", int((d < 0).sum()))
    rec(f"{key}.worse", int((d > 0).sum()))
    cells = [f"{runs[label].mean():.4f}", pm(mean, se, 3)]
    # the two subgroups a prior is meant to move, and everything else
    for name, mask in (("repeats", repeat), ("off-list", offlist),
                       ("other", rest)):
        if mask.any():
            m, s, _ = common.paired(runs[label][mask], runs[BASE[0]][mask])
            print(f"  {'':{WIDTH}s}   on the {int(mask.sum()):3d} {name:9s} "
                  f"{m:+.3f} +- {s:.3f}")
            rec(f"{key}.{name.replace('-', '')}", pm(m, s))
            cells.append(pm(m, s))
    print(f"  {'':{WIDTH}s}   games needed for 0.02 at 2 sigma: "
          f"{(2 * d.std(ddof=1) / 0.02) ** 2:,.0f}")
    rec(f"{key}.games_needed", int(round((2 * d.std(ddof=1) / 0.02) ** 2)))
    rec(f"{key}.sd", float(d.std(ddof=1)))
    regime_rows.append((label, *cells))
if args.all_priors:
    regime_rows.append((f"**{BASE[0]} (the default)**",
                        f"**{runs[BASE[0]].mean():.4f}**", "\u2014", "\u2014",
                        "\u2014", "\u2014"))
    rec("table", common.md_table(
        ["priors", "mean", "vs. both", f"on the {int(repeat.sum())} repeats",
         f"on the {int(offlist.sum())} off-list",
         f"on the other {int(rest.sum())}"], regime_rows))

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
        rec("contrast." + label.replace(", ", "_").replace(" ", "_"),
            pm(contrast.mean(), contrast.std(ddof=1) / np.sqrt(len(contrast))))
    inter = ((runs["none"] - runs["repeat only"])
             - (runs["curation only"] - runs["both"]))
    print(f"  {'interaction':{CWIDTH}s} {inter.mean():+.4f} +- "
          f"{inter.std(ddof=1) / np.sqrt(len(inter)):.4f}")
    rec("interaction", pm(inter.mean(), inter.std(ddof=1) / np.sqrt(len(inter))))

    # the same four means as a 2 x 2 grid, each margin as on minus off
    def on_minus_off(a, b):
        d = runs[b] - runs[a]
        return pm(d.mean(), d.std(ddof=1) / np.sqrt(len(d)))

    rec("grid_table", common.md_table(
        ["mean guesses", "curation off", "curation on", "on \u2212 off"],
        [("**repeat off**", f"{runs['none'].mean():.4f}",
          f"{runs['curation only'].mean():.4f}",
          on_minus_off("none", "curation only")),
         ("**repeat on**", f"{runs['repeat only'].mean():.4f}",
          f"{runs['both'].mean():.4f}", on_minus_off("repeat only", "both")),
         ("**on \u2212 off**", on_minus_off("none", "repeat only"),
          on_minus_off("curation only", "both"),
          "**" + pm(inter.mean(), inter.std(ddof=1) / np.sqrt(len(inter))) + "**")]))

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
fresh = common.fresh_events(words, curated, START)
rep_rate = sum(1 for t, _ in reps if t >= START) / (END - START)
off_rate = sum(1 for t, _ in offs if t >= START) / (END - START)
mix = (rep_rate, off_rate, 1 - rep_rate - off_rate)
print(f"arms: {len(reps)} repeats, {len(offs)} off-list, {len(fresh)} fresh, "
      f"every one an answer that really happened")
print(f"mixture: {100*mix[0]:.1f}% / {100*mix[1]:.1f}% / {100*mix[2]:.1f}%")
rec("arms.repeats", len(reps))
rec("arms.offlist", len(offs))
rec("arms.fresh", len(fresh))
rec("mixture", f"{100*mix[0]:.1f}% / {100*mix[1]:.1f}% / {100*mix[2]:.1f}%")
print()


def arm(config, events):
    return [guesses(config, t, w) for t, w in events]


def three_arms(config):
    return (arm(config, reps), arm(config, offs), arm(config, fresh))


played = {label: three_arms(config) for label, config in EVERY}
base = played[BASE[0]]


def blend_of(cells):
    """A row's blend at the regime mixture, and its standard error."""
    blend = sum(m * c[0] for m, c in zip(mix, cells))
    return blend, np.sqrt(sum((m * c[1]) ** 2 for m, c in zip(mix, cells)))


def blended(cells):
    """A row's three paired cells, and their blend at the regime mixture."""
    blend, blend_se = blend_of(cells)
    body = " ".join(f"{c[0]:+9.3f}+-{c[1]:.3f}" for c in cells)
    return f"{body} {blend:+9.3f}+-{blend_se:.3f}"


def absolute(arms):
    """One arm's own mean on each of the three, and their blend."""
    cells = [(np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))) for a in arms]
    blend, blend_se = blend_of(cells)
    body = " ".join(f"{c[0]:9.3f}+-{c[1]:.3f}" for c in cells)
    return f"{body} {blend:9.3f}+-{blend_se:.3f}"


HEAD = (f"  {'arm':{TWIDTH}s} {'repeat':>16s} {'off-list':>16s} {'fresh':>16s} "
        f"{'blended':>16s}\n  " + "-" * (TWIDTH + 68))

# every arm's own mean guesses, before any of them is subtracted from another
print("  guesses each arm takes\n")
print(HEAD)
for label, _ in EVERY:
    print(f"  {label:{TWIDTH}s} {absolute(played[label])}")

print(f"\n  against `{BASE[0]}`, paired on the same events\n")
print(HEAD)
rows = {}
arm_table = []
for label, _ in ARMS:
    rows[label] = [common.paired(g, b) for g, b in zip(played[label], base)]
    print(f"  {label:{TWIDTH}s} {blended(rows[label])}")
    key = KEYS[label]
    blend, blend_se = blend_of(rows[label])
    rec(f"{key}.blend", pm(blend, blend_se))
    rec(f"{key}.blend_delta", float(blend))
    rec(f"{key}.blend_se", float(blend_se))
    rec(f"{key}.blend_sigma", float(abs(blend) / blend_se))
    for name, cell in zip(("repeat", "offlist", "fresh"), rows[label]):
        rec(f"{key}.arm.{name}", pm(cell[0], cell[1]))
    arm_table.append((label, *(pm(c[0], c[1]) for c in rows[label]),
                      pm(blend, blend_se)))
rec("arm_table", common.md_table(
    ["shapes" if args.shapes else "priors", "repeat", "off-list", "fresh",
     "blended"], arm_table))

# The mixture is itself counted off those same puzzles, and the blend above
# treats it as exact.  Its error only matters where the arms disagree, so widen
# each row by the multinomial term and see whether the row moves.
n = END - START
mix_var = [m * (1 - m) / n for m in mix]
print(f"\n  with the mixture's own error ({n} puzzles: "
      + ", ".join(f"{100 * m:.1f}+-{100 * v ** 0.5:.1f}%"
                  for m, v in zip(mix, mix_var)) + ")\n")
for label, _ in ARMS:
    blend, blend_se = blend_of(rows[label])
    # the blend is fresh + w_rep*(repeat - fresh) + w_off*(off-list - fresh), so
    # what the rates multiply is each arm's gap to `fresh`
    gaps = [rows[label][i][0] - rows[label][2][0] for i in (0, 1)]
    slack = (mix_var[0] * gaps[0] ** 2 + mix_var[1] * gaps[1] ** 2
             - 2 * mix[0] * mix[1] / n * gaps[0] * gaps[1])
    print(f"  {label:{TWIDTH}s} {blend:+9.3f}+-"
          f"{np.sqrt(blend_se ** 2 + slack):.3f}")
    rec(f"{KEYS[label]}.blend_wide", pm(blend, np.sqrt(blend_se ** 2 + slack)))

if args.all_priors:
    print()
    for label, (a, b) in CONTRASTS.items():
        cells = [common.paired(g, h) for g, h in zip(played[a], played[b])]
        print(f"  {label:{TWIDTH}s} {blended(cells)}")
    cells = [common.paired(np.array(a) - np.array(b), np.array(c) - np.array(d))
             for a, b, c, d in zip(played["none"], played["repeat only"],
                                   played["curation only"], base)]
    print(f"  {'interaction':{TWIDTH}s} {blended(cells)}")
    rec("interaction3", pm(*blend_of(cells)))

if args.shapes:
    apart = blend_of(rows[CURATION_ALONE])[0] + blend_of(rows[REPEAT_ALONE])[0]
    print(f"\n  sharpened apart they sum to {apart:+.3f}, together "
          f"{blend_of(rows[TOGETHER])[0]:+.3f}")
    rec("apart", float(apart))

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
                       "fresh": [w for _, w in fresh]},
            "guesses": {label: [[int(g) for g in a] for a in played[label]]
                        for label, _ in EVERY},
        },
    })
    print(f"\n  raw per-game guess counts written to {args.dump}")

print(f"\n  {time.time() - STARTED:.0f} s")
