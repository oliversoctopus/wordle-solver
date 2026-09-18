"""What is left in the second guess, and whether taking it survives the prior.

Behind a fixed opening the whole turn-2 policy is at most 243 decisions, one per
feedback pattern, so it can be priced exhaustively rather than sampled.  Each
position is scored by playing out the true value of the top `--top` guesses --
`common.guess_value`, the same device `lookahead_accuracy.py --decisions` uses --
and weighted by its exact prior mass, which is the probability play reaches it.
That matters because `f` decides turn 2 and almost nothing else: the opening is a
single position already known to be optimal, and by four turns left the exact
search has taken over.  Whatever the estimator still costs is here.

Something is here, and it is 0.003 guesses.  Substituting those decisions and
sweeping every answer reproduces the figure end to end, which is the check that
the census measured what it meant to.

Then the point of the script.  Those decisions were chosen to minimize a mean
under the shipped prior, and the prior is a model of the editor rather than the
editor, so the last stage scores the identical substitution under distributions
it was not chosen under -- three other enumerated ones, and the answers the NYT
really published.  The sign does not survive.  What the gain rides on is a
reshuffle of a tenth of the word list, nearly half of which moves the wrong way,
and a small nudge to the prior's weighting decides which half wins.

The two real-answer rows are differences and not levels.  Both arms replay those
answers under today's prior, where every one of them is a word already used and
priced at repeat weight, so both sit far above the 3.1684 `replay.py` reports
from a prior rebuilt as of each puzzle.  That handicap is common to the arms and
cancels in the pair, which is the only quantity read off those rows.

So this backs a negative result, and it is the sharpest of the cases in
SOLVER_EXPLAINED *A better map, and next to no better score*: not that a
better valuation fails to pay, but that at the last position where the
estimator decides anything, playing
*optimally* is worth less than the error in the prior it would be optimal for.
README *Perfecting the second guess* is the write-up.

Nothing is written and no override is cached.  They are derived here each run,
since they belong to one opening, one prior and one fit.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import EXPECTED, MAX_TURNS, WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--top", type=int, default=50,
                    help="guesses played out per position, best first by f")
parser.add_argument("--blend", type=float, nargs="*",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
                    help="how far to blend the prior toward uniform-over-curated")
args = parser.parse_args()

data = load_data()
solver = WordleSolver(data=data, config=PriorConfig(), mode=EXPECTED)
everything = solver.all_candidates()


def turn2_positions(solver, opening):
    """(pattern, mass, candidates) per feedback the opening can return.

    The winning pattern is dropped: that game is over and has no turn 2.
    """
    seen, out = set(), []
    win = int(solver.feedback(opening, opening))
    for answer in range(len(solver.weights)):
        pattern = int(solver.feedback(opening, answer))
        if pattern == win or pattern in seen:
            continue
        seen.add(pattern)
        live = solver.filter(opening, pattern, everything)
        out.append((pattern, float(solver.weights[live].sum()), live))
    return sorted(out, key=lambda row: -row[1])


def census(solver, opening, top):
    """The regret of every turn-2 decision, and the guess that would beat it.

    A position the exact search already solves is skipped rather than played
    out: its guess is the true optimum and a playout could only confirm it.
    """
    rows, skipped = [], 0
    for pattern, mass, live in turn2_positions(solver, opening):
        if solver.solved_exactly(live, MAX_TURNS - 1):
            skipped += 1
            continue
        pick, _ = solver.best_guess(live, MAX_TURNS - 1)
        words, _ = common.shortlist(solver, live, top)
        if pick not in words:
            words.append(pick)
        values = {g: common.guess_value(solver, live, g) for g in words}
        # ties go to the lower word index, as every other ranking here does
        best = min(values, key=lambda g: (values[g], g))
        rows.append((pattern, mass, len(live), pick, best,
                     values[pick] - values[best]))
    return rows, skipped


def sweep(solver, opening, overrides):
    """Guesses taken with each word as the answer, the opening forced and the
    second guess substituted wherever ``overrides`` names one."""
    lengths = np.empty(len(solver.weights))
    for answer in range(len(lengths)):
        guess, live = opening, everything
        lengths[answer] = MAX_TURNS + 1
        for turn in range(1, MAX_TURNS + 1):
            if turn == 2:
                guess = overrides.get(int(solver.feedback(opening, answer)),
                                      guess)
            if guess == answer:
                lengths[answer] = turn
                break
            live = solver.filter(guess, solver.feedback(guess, answer), live)
            if turn < MAX_TURNS:
                guess, _ = solver.best_guess(live, MAX_TURNS - turn)
    return lengths


def uniform_over(words):
    """A flat distribution over a word list, as weights over the whole list."""
    weights = np.zeros(len(solver.weights))
    for word in words:
        if word in data.word2ind:
            weights[data.word2ind[word]] += 1.0
    return weights / weights.sum()


# ------------------------------------------------------------------ the census
opening, _ = solver.best_guess(everything, MAX_TURNS)
common.header(f"every turn-2 decision behind {data.all_words[opening]}")
start = time.perf_counter()
rows, skipped = census(solver, opening, args.top)
moved = [row for row in rows if row[5] > 1e-12]
reach = sum(row[1] * row[5] for row in rows)

print(f"  {len(rows) + skipped} positions, {skipped} already solved exactly")
print(f"  top {args.top} guesses played out at each of the other {len(rows)}, "
      f"{time.perf_counter() - start:.0f}s\n")
print(f"  {'pattern':>7} {'m':>6} {'mass':>8} {'f picks':>9} {'better':>9} "
      f"{'regret':>9} {'x mass':>9}")
for pattern, mass, size, pick, best, regret in sorted(
        moved, key=lambda row: -row[1] * row[5]):
    print(f"  {pattern:>7} {size:>6} {mass:>8.4f} "
          f"{data.all_words[pick]:>9} {data.all_words[best]:>9} "
          f"{regret:>+9.4f} {mass * regret:>9.5f}")
print(f"\n  a better second guess exists at {len(moved)} of {len(rows)}")
print(f"  turn-2 regret, weighted by reach: {reach:.5f} guesses")
common.record("positions", len(rows) + skipped)
common.record("solved", skipped)
common.record("decided", len(rows))
common.record("moved", len(moved))
common.record("census", float(reach))

# ------------------------------------------------------- does it reach the mean
overrides = {row[0]: row[4] for row in moved}
base = sweep(solver, opening, {})
full = sweep(solver, opening, overrides)
shipped_mean, shipped_loss = common.exact_stats(base, solver.weights)

common.header("substituting those decisions, over every answer")
print(f"  {'policy':<24} {'mean':>9} {'loss prob':>11} {'vs shipped':>12}")
print(f"  {'shipped':<24} {shipped_mean:>9.5f} {shipped_loss * 100:>10.4f}% "
      f"{'--':>12}")

# each on its own, since a decision that trades guesses for losses is not one
# worth having and the total would hide it; the last stage reads these again
single = {pattern: sweep(solver, opening, {pattern: word})
          for pattern, word in overrides.items()}
kept = {pattern: word for pattern, word in overrides.items()
        if common.exact_stats(single[pattern], solver.weights)[1]
        <= shipped_loss + 1e-12}
safe = sweep(solver, opening, kept)
for label, lengths in ((f"all {len(overrides)} overrides", full),
                       (f"the {len(kept)} that lose less", safe)):
    mean, loss = common.exact_stats(lengths, solver.weights)
    print(f"  {label:<24} {mean:>9.5f} {loss * 100:>10.4f}% "
          f"{mean - shipped_mean:>+12.5f}")
print(f"\n  the census predicted {reach:.5f} and the sweep pays "
      f"{shipped_mean - float(solver.weights @ full):.5f}")
full_mean, full_loss = common.exact_stats(full, solver.weights)
common.record("shipped_mean", shipped_mean)
common.record("override_mean", full_mean)
common.record("paid", shipped_mean - full_mean)
common.record("shipped_loss", shipped_loss * 100)
common.record("override_loss", full_loss * 100)

# ------------------------------------------- but under a distribution it is not
curated = [w for w in data.curated_list if w in data.word2ind]
used = set(data.past_solutions)
solutions = common.solutions()
regime = [r["solution"] for r in solutions[common.regime_start(solutions):]]
alltime = [r["solution"] for r in solutions]
# the object under test is the turn-2 optimum itself, so every override is in
# it; the loss filter above is a refinement of a thing that has to hold first
delta = full - base

enumerated = [
    ("shipped prior (what chose them)", solver.weights),
    ("uniform over all 12,947", uniform_over(data.all_words)),
    ("uniform over the curated list", uniform_over(curated)),
    ("uniform over the still unused",
     uniform_over([w for w in curated if w not in used])),
]

common.header("the same substitution, scored where it was not chosen")
print(f"  all {len(overrides)} of them: the turn-2 optimum under the "
      f"shipped prior\n")
print(f"  {'distribution':<33} {'shipped':>9} {'override':>9} {'delta':>9}")
table = []
neg = common.neg
for (label, weights), key in zip(enumerated, ("default", "all", "curated", "unused")):
    print(f"  {label:<33} {float(weights @ base):>9.4f} "
          f"{float(weights @ full):>9.4f} {float(weights @ delta):>+9.4f}")
    cell = neg(f"{float(weights @ delta):+.4f}")
    table.append((label.replace("shipped prior", "the default prior"),
                  f"{float(weights @ base):.4f}", f"{float(weights @ full):.4f}",
                  f"**{cell}**" if abs(float(weights @ delta)) > 0.002 else cell))
    common.record(f"enumerated.{key}", float(weights @ delta))
for label, words in (("the real regime answers", regime),
                     ("every real answer ever", alltime)):
    seen = np.array([delta[data.word2ind[w]] for w in words
                     if w in data.word2ind])
    weights = uniform_over(words)
    se = seen.std(ddof=1) / np.sqrt(len(seen))
    print(f"  {label + f' ({len(seen)})':<33} {float(weights @ base):>9.4f} "
          f"{float(weights @ full):>9.4f} {seen.mean():>+9.4f} +/- "
          f"{se:.4f}  "
          f"({int((seen < 0).sum())} better, {int((seen > 0).sum())} worse)")
    key = "regime" if "regime" in label else "alltime"
    shown = (f"the {len(seen)} real regime answers" if key == "regime"
             else f"all {len(seen):,} real answers ever")
    table.append((shown, f"{float(weights @ base):.4f}",
                  f"{float(weights @ full):.4f}",
                  neg(f"{seen.mean():+.4f} \u00b1 {se:.4f}")))
    common.record(f"{key}.delta", neg(f"{seen.mean():+.4f} \u00b1 {se:.4f}"))
    common.record(f"{key}.sigma", float(abs(seen.mean()) / se))
    common.record(f"{key}.n", len(seen))
common.record("table", common.md_table(
    ["distribution", "shipped", "turn-2 optimum", "delta"], table))

print("\n  each on its own, over the four enumerated distributions:\n")
print(f"  {'pattern':>7} {'word':>9}" +
      "".join(f"{label.split()[-1]:>13}" for label, _ in enumerated))
everywhere = {}
for pattern, word in overrides.items():
    on_its_own = single[pattern] - base
    scored = [float(weights @ on_its_own) for _, weights in enumerated]
    if max(scored) <= 1e-12:
        everywhere[pattern] = word
    print(f"  {pattern:>7} {data.all_words[word]:>9}" +
          "".join(f"{value:>+13.5f}" for value in scored))

# keeping only the ones that help everywhere is itself a fit to the four
# distributions above, and this is what the fit would have left to show for it
gain = float(solver.weights @ (sweep(solver, opening, everywhere) - base))
print(f"\n  {len(everywhere)} of the {len(overrides)} help under all four, and "
      f"together they are worth {gain:+.5f}\n  under the shipped prior, against "
      f"the {float(solver.weights @ delta):+.5f} of the whole set")
common.record("everywhere", len(everywhere))
common.record("everywhere_gain", float(gain))
# the widest sign flip among the overrides, between the shipped prior and the
# curated list, is the example the documents give
flips = []
for pattern, word in overrides.items():
    on_its_own = single[pattern] - base
    own = float(solver.weights @ on_its_own)
    cur = float(enumerated[2][1] @ on_its_own)
    if own < 0 < cur:
        flips.append((cur - own, data.all_words[word], own, cur))
if flips:
    _, word, own, cur = max(flips)
    common.record("flip.word", word)
    common.record("flip.default", float(own))
    common.record("flip.curated", float(cur))

# --------------------------------------------------------- why it does not hold
common.header("what the gain is made of")
weights = solver.weights
churn = delta != 0
print(f"  answers whose length changes   {int(churn.sum())} of {len(delta)}")
print(f"  prior mass on them             {100 * weights[churn].sum():.1f}%")
print(f"  mass that improves             {100 * weights[delta < 0].sum():.2f}%")
print(f"  mass that worsens              {100 * weights[delta > 0].sum():.2f}%")
print(f"  net                            {float(weights @ delta):+.5f}")
print(f"  sd of the per-answer change    {delta[churn].std():.4f} guesses")
print(f"  net as a fraction of that      "
      f"{abs(float(weights @ delta)) / delta[churn].std():.4f}")
common.record("churn", int(churn.sum()))
common.record("churn_mass", 100 * float(weights[churn].sum()))
common.record("improves", 100 * float(weights[delta < 0].sum()))
common.record("worsens", 100 * float(weights[delta > 0].sum()))
common.record("net", float(weights @ delta))
common.record("sd", float(delta[churn].std()))
common.record("net_fraction", abs(float(weights @ delta)) / float(delta[churn].std()))

flat = uniform_over(curated)
print("\n  blending the prior toward uniform-over-curated:\n")
print(f"  {'blend':>7} {'delta':>10}")
crossed = None
for t in args.blend:
    mixed = (1 - t) * weights + t * flat
    print(f"  {t:>7.2f} {float(mixed @ delta):>+10.5f}")
    if crossed is None and float(mixed @ delta) >= 0:
        crossed = t
common.record("blend_crossing", crossed if crossed is not None else 1.0)
