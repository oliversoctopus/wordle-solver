"""What the loss penalty and the endgame search's reach are worth, exactly.

The solver is deterministic, so a game's outcome is a pure function of the
answer.  Playing all 12,947 words once and weighting by the prior therefore
computes the mean and the loss probability rather than estimating them: no
sampling error, no error bars, and "which games changed" is a word list instead
of a count.  It is also faster than sampling, because the memos in
`solver.WordleSolver` and `exact.ExactEndgame` saturate after a few thousand
games and the rest of the sweep runs out of cache.

Backs the whole of README *Losing less*, and SOLVER_EXPLAINED sections 9 and 10.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig
from solver import (DEFAULT_EXACT_MAX, DEFAULT_LOSS_PENALTY, ENTROPY, EXPECTED,
                    WordleSolver, sample_answers)
from wordle_data import load_data

MAX_GUESSES = 6

# the four toggle combinations the two tables blend over, in the README's order
TOGGLE_CONFIGS = [PriorConfig(repeat=r, curation=c)
                  for r in (False, True) for c in (False, True)]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--penalties", type=float, nargs="+",
                    default=[1, 2, 5, 10, 30])
parser.add_argument("--exact-max", type=int, nargs="+", default=[24, 40, 60],
                    help="reach sweep")
# the two levers interact, so each sweep needs to say where the other one sits
parser.add_argument("--at-reach", type=int, default=DEFAULT_EXACT_MAX,
                    help="the reach the penalty sweep runs at")
parser.add_argument("--at-penalty", type=float, default=DEFAULT_LOSS_PENALTY,
                    help="the penalty the reach sweep runs at")
parser.add_argument("--shipped-only", action="store_true",
                    help="just the shipped prior, not the four combinations")
parser.add_argument("--skip-reach", action="store_true",
                    help="the penalty sweep alone")
parser.add_argument("--policies", action="store_true",
                    help="the policy comparison and the curated sweep instead")
parser.add_argument("--search", action="store_true",
                    help="what the search adds, by how far it is allowed to reach")
parser.add_argument("--cost", action="store_true",
                    help="time the reach instead of scoring it")
parser.add_argument("--cost-games", type=int, default=400,
                    help="answers per timed configuration")
args = parser.parse_args()

data = load_data()
curated = set(common.load("wordles.json"))
configs = [PriorConfig()] if args.shipped_only else TOGGLE_CONFIGS


def play_all(solver):
    """Guesses taken for every word as the answer, a loss counted as one over."""
    lengths = np.empty(len(solver.weights))
    for answer in range(len(lengths)):
        guesses, _ = solver.play(answer, max_guesses=MAX_GUESSES,
                                 deadline=MAX_GUESSES)
        solved = guesses[-1] == answer
        lengths[answer] = len(guesses) if solved else MAX_GUESSES + 1
    return lengths, solver.weights


def outcomes(config, penalty, exact_max):
    return play_all(WordleSolver(data=data, config=config, mode=EXPECTED,
                                 loss_penalty=penalty, exact_max=exact_max))


def stats(lengths, weights):
    """Exact loss probability and the two means, under this prior's weights."""
    lost = lengths > MAX_GUESSES
    won = ~lost
    return {
        "loss_prob": float(weights[lost].sum()),
        "losing_words": int(lost.sum()),
        "mean_solved": float(weights[won] @ lengths[won] / weights[won].sum()),
        "mean_penalized": float(weights @ lengths),
        # score(L) = intercept + L * loss_prob, for a loss worth L guesses
        "intercept": float(weights[won] @ lengths[won]),
        "lost_mask": lost,
    }


BLENDED = ("loss_prob", "losing_words", "mean_solved", "mean_penalized", "intercept")


def blend(rows):
    """Equal weight per prior, matching the equal games per arm of the old runs."""
    return {key: float(np.mean([row[key] for row in rows])) for key in BLENDED}


def policies():
    """Greedy entropy against the estimator, and the curated-list sweep.

    Every row plays the same 12,947 answers, so a difference between two rows is
    a difference in policy and nothing else.  Which words each row loses is a
    set, not a count, so "are the losses the same losses" is answered by
    intersecting them rather than by a test.
    """
    config = PriorConfig()
    rows = {
        "greedy entropy": WordleSolver(data=data, config=config, mode=ENTROPY),
        "expected, no exact search, penalty 1": WordleSolver(
            data=data, config=config, mode=EXPECTED, use_exact=False,
            loss_penalty=1.0),
        "expected, exact-max 24, penalty 5": WordleSolver(
            data=data, config=config, mode=EXPECTED, exact_max=24),
        "expected, exact-max 60, penalty 5 (default)": WordleSolver(
            data=data, config=config, mode=EXPECTED),
    }

    common.header("Policies, over every word as the answer")
    print(f"  {'policy':<44} {'mean':>8} {'loss prob':>10} {'solve rate':>11}")
    played = {}
    for label, solver in rows.items():
        lengths, weights = play_all(solver)
        played[label] = lengths
        row = stats(lengths, weights)
        print(f"  {label:<44} {row['mean_penalized']:8.4f} "
              f"{row['loss_prob'] * 100:9.4f}% {(1 - row['loss_prob']) * 100:10.4f}%")

    labels = list(rows)
    entropy = played[labels[0]] > MAX_GUESSES
    default = played[labels[-1]] > MAX_GUESSES
    print(f"\n  entropy loses {int(entropy.sum())} words, the default "
          f"{int(default.sum())}, sharing {int((entropy & default).sum())}")

    curated_idx = np.array([i for i, word in enumerate(data.all_words)
                            if word in curated])
    print(f"\n  all {len(curated_idx)} curated words as the answer, unweighted:")
    for label, lengths in played.items():
        sub = lengths[curated_idx]
        lost = [data.all_words[curated_idx[i]] for i in np.where(sub > MAX_GUESSES)[0]]
        print(f"    {label:<44} mean {sub.mean():.4f}  worst {int(sub.max())}  "
              f"losses {len(lost)} / {len(curated_idx)}")
        if lost:
            print(f"      {' '.join(sorted(lost))}")


def search():
    """What the endgame search adds, by how far it is allowed to reach.

    Every row plays the same 12,947 answers under the shipped prior, so a
    difference between two rows is the reach and nothing else.  `exact_max=0`
    leaves only the part that is free -- two turns or fewer, which costs one pass
    over the guesses either way.  The penalty is inert in the first two rows,
    since it only ever enters through the search.
    """
    config = PriorConfig()
    rows = {
        "greedy entropy": WordleSolver(data=data, config=config, mode=ENTROPY),
        "expected, no exact search": WordleSolver(
            data=data, config=config, mode=EXPECTED, use_exact=False),
        "expected, exact for t <= 2 only": WordleSolver(
            data=data, config=config, mode=EXPECTED, exact_max=0),
        "expected, exact for m <= 10": WordleSolver(
            data=data, config=config, mode=EXPECTED, exact_max=10),
        "expected, exact for m <= 24": WordleSolver(
            data=data, config=config, mode=EXPECTED, exact_max=24),
        "expected, exact for m <= 60 (default)": WordleSolver(
            data=data, config=config, mode=EXPECTED),
    }

    common.header("What the endgame search is worth, over every word as the answer")
    print(f"  {'policy':<38} {'mean':>8} {'loss prob':>10} "
          f"{'solve rate':>11} {'wall':>7}")
    for label, solver in rows.items():
        start = time.perf_counter()
        lengths, weights = play_all(solver)
        elapsed = time.perf_counter() - start
        row = stats(lengths, weights)
        print(f"  {label:<38} {row['mean_penalized']:8.4f} "
              f"{row['loss_prob'] * 100:9.4f}% {(1 - row['loss_prob']) * 100:10.4f}% "
              f"{elapsed:6.0f}s")


def costs():
    """What reaching further costs in time, and whether the penalty drives it.

    Wall clock does not travel between machines, so what this reports is the
    ratio to the shipped reach.  A sampled set of answers rather than all of
    them: the point is the shape of the curve, and the memos would otherwise
    flatten it by saturating partway through.
    """
    for label, config in (("flat", PriorConfig(repeat=False, curation=False)),
                          ("shipped", PriorConfig())):
        answers = sample_answers(
            WordleSolver(data=data, config=config, mode=EXPECTED).weights,
            args.cost_games, seed=4242)

        def timed(cap, penalty):
            solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                                  exact_max=cap, loss_penalty=penalty)
            solver.play(int(answers[0]), max_guesses=MAX_GUESSES,
                        deadline=MAX_GUESSES)          # warm the kernels
            start = time.perf_counter()
            for answer in answers:
                solver.play(int(answer), max_guesses=MAX_GUESSES,
                            deadline=MAX_GUESSES)
            return time.perf_counter() - start, solver.exact_bailouts

        common.header(f"{label}: what the reach costs, {args.cost_games} answers")
        caps = (24, 40, DEFAULT_EXACT_MAX, 80, 100)
        timings = {cap: timed(cap, DEFAULT_LOSS_PENALTY) for cap in caps}
        base = timings[DEFAULT_EXACT_MAX][0]
        print(f"  {'exact-max':>10} {'seconds':>9} {'vs default':>11} {'bailouts':>9}")
        for cap in caps:
            seconds, bailouts = timings[cap]
            print(f"  {cap:>10} {seconds:>9.1f} {seconds / base:>10.2f}x {bailouts:>9}")

        print(f"\n  and whether the penalty moves it:")
        print(f"  {'exact-max':>10} {'penalty':>8} {'seconds':>9} {'bailouts':>9}")
        for cap in (DEFAULT_EXACT_MAX, 100):
            for penalty in (1.0, 5.0, 30.0):
                seconds, bailouts = timed(cap, penalty)
                print(f"  {cap:>10} {penalty:>8g} {seconds:>9.1f} {bailouts:>9}")
        print()


def crossovers(grid, settings):
    """Which penalty is best, as a function of what a loss is really worth.

    Scoring a loss at 7 is the objective `--loss-penalty 1` already optimizes,
    so ranking policies by it can only ever favor risk neutrality.  Naming the
    price of a loss instead makes each policy a line in that price, and the
    choice falls out of where the lines cross.
    """
    common.header("Which penalty, for what a loss is worth")
    lines = {s: (row["intercept"], row["loss_prob"])
             for s in settings
             for row in [blend([grid[s, c.slug] for c in configs])]}

    print("  crossovers, in guesses per loss:")
    for low, high in zip(settings, settings[1:]):
        (i_low, p_low), (i_high, p_high) = lines[low], lines[high]
        if p_low <= p_high:  # the higher penalty loses no less often: never worth it
            print(f"    penalty {low:>4} is never displaced by {high}")
            continue
        print(f"    penalty {low:>4} gives way to {high:<5} at L = "
              f"{(i_high - i_low) / (p_low - p_high):6.2f}")

    print("\n  best penalty by what a loss costs, and by how much:")
    for price in (7, 8, 9, 10, 12, 15, 20, 25, 40):
        scores = {s: i + price * p for s, (i, p) in lines.items()}
        ranked = sorted(scores, key=scores.get)
        margin = scores[ranked[1]] - scores[ranked[0]] if len(ranked) > 1 else 0.0
        print(f"    L = {price:>3}: penalty {ranked[0]:>4}   "
              f"(ahead of {ranked[1]} by {margin:.5f} guesses)")


def words_of(mask):
    return [data.all_words[i] for i in np.where(mask)[0]]


def flips(baseline, arm):
    """Words the arm rescues, and words it loses that the baseline did not."""
    return (words_of(baseline & ~arm), words_of(~baseline & arm))


def sweep(label, settings, run, baseline_key):
    """Run one sweep over every prior and print the blended table."""
    common.header(label)
    grid = {}
    for setting in settings:
        for config in configs:
            lengths, weights = run(config, setting)
            grid[setting, config.slug] = stats(lengths, weights)

    print(f"  {'setting':>8} {'loss prob':>10} {'words':>7} {'mean solved':>12} "
          f"{'mean (loss=7)':>14}")
    have_baseline = any(setting == baseline_key for setting in settings)
    for setting in settings:
        row = blend([grid[setting, c.slug] for c in configs])
        marker = " *" if setting == baseline_key else "  "
        print(f"  {setting:>8}{marker}{row['loss_prob'] * 100:9.4f}% "
              f"{row['losing_words']:7.1f} {row['mean_solved']:12.4f} "
              f"{row['mean_penalized']:14.4f}")
    print(f"\n  blended equally over {len(configs)} prior(s); "
          f"* is the shipped default")

    for config in configs:
        print(f"\n  {config.slug}")
        base = (grid[baseline_key, config.slug]["lost_mask"]
                if have_baseline else None)
        for setting in settings:
            row = grid[setting, config.slug]
            cur = [w for w in words_of(row["lost_mask"]) if w in curated]
            print(f"    {setting:>6}: {row['loss_prob'] * 100:8.4f}% over "
                  f"{row['losing_words']:>4} words, mean {row['mean_penalized']:.4f}, "
                  f"{len(cur):>3} curated")
            if base is not None and setting != baseline_key:
                fixed, broken = flips(base, row["lost_mask"])
                print(f"            vs default: fixes {len(fixed)}, breaks {len(broken)}")
            if cur:
                print(f"            curated losses: {' '.join(sorted(cur))}")
    return grid


if args.cost:
    costs()
elif args.search:
    search()
elif args.policies:
    policies()
else:
    penalty_grid = sweep(
        f"The penalty, at exact-max {args.at_reach}", args.penalties,
        lambda config, penalty: outcomes(config, penalty, args.at_reach),
        DEFAULT_LOSS_PENALTY)

    if len(args.penalties) > 1:
        print()
        crossovers(penalty_grid, sorted(args.penalties))

    if not args.skip_reach:
        print()
        sweep(f"The endgame search's reach, at penalty {args.at_penalty:g}",
              args.exact_max,
              lambda config, cap: outcomes(config, args.at_penalty, cap),
              DEFAULT_EXACT_MAX)
