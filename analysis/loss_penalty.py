"""What the loss penalty and the endgame search's reach are worth, exactly.

The solver is deterministic, so playing all 12,947 words once and weighting
by the prior computes the mean and the loss probability, and "which games
changed" is a word list.  Backs the whole of README *Losing less*, and
SOLVER_EXPLAINED *The deadline, and what losing costs* and *Why the penalty
is 5 and not 1*.
"""

import argparse
import time

import numpy as np

import common
from priors import PriorConfig, build_weights
from solver import (DEFAULT_EXACT_MAX, DEFAULT_LOSS_PENALTY, ENTROPY, EXPECTED,
                    WordleSolver, sample_answers)
from wordle_data import MAX_GUESSES, load_data

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
parser.add_argument("--bound", action="store_true",
                    help="why the pruning bound caps the penalty at 1")
parser.add_argument("--bound-family", default="atch",
                    help="the suffix whose family --bound values")
parser.add_argument("--cost-games", type=int, default=400,
                    help="answers per timed configuration")
common.add_prior_filter(parser)
args = parser.parse_args()

data = load_data()
curated = set(common.load("wordles.json"))
configs = common.filter_priors(
    [PriorConfig()] if args.shipped_only else TOGGLE_CONFIGS, args)


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
    played, table, losses = {}, [], {}
    for label, solver in rows.items():
        lengths, weights = play_all(solver)
        played[label] = lengths
        row = stats(lengths, weights)
        print(f"  {label:<44} {row['mean_penalized']:8.4f} "
              f"{row['loss_prob'] * 100:9.4f}% {(1 - row['loss_prob']) * 100:10.4f}%")
        losses[label] = row["loss_prob"] * 100
        cells = (label, f"{row['mean_penalized']:.4f}",
                 f"{row['loss_prob'] * 100:.4f}%", f"{(1 - row['loss_prob']) * 100:.4f}%")
        table.append(tuple(f"**{c}**" for c in cells) if "default" in label else cells)
    common.record("policies.table", common.md_table(
        ["policy", "mean", "loss probability", "solve rate"], table))
    common.record("policies.loss_ratio", losses[list(rows)[-1]] / losses[list(rows)[0]])
    # what a 200-game benchmark sees: the chance it meets no loss at all
    for key, label in (("entropy", list(rows)[0]), ("default", list(rows)[-1])):
        common.record(f"policies.benchmark_clean.{key}",
                      100 * (1 - losses[label] / 100) ** 200)

    labels = list(rows)
    entropy = played[labels[0]] > MAX_GUESSES
    default = played[labels[-1]] > MAX_GUESSES
    print(f"\n  entropy loses {int(entropy.sum())} words, the default "
          f"{int(default.sum())}, sharing {int((entropy & default).sum())}")
    common.record("policies.default_lost", int(default.sum()))
    common.record("policies.entropy_lost", int(entropy.sum()))
    common.record("policies.shared_lost", int((entropy & default).sum()))
    never = np.array([w not in curated for w in data.all_words])
    common.record("policies.default_lost_uncurated", int((default & never).sum()))
    common.record("policies.default_lost_uncurated_mass",
                  100 * weights[default & never].sum() / weights[default].sum())

    curated_idx = np.array([i for i, word in enumerate(data.all_words)
                            if word in curated])
    print(f"\n  all {len(curated_idx)} curated words as the answer, unweighted:")
    table = []
    for label, lengths in played.items():
        sub = lengths[curated_idx]
        lost = [data.all_words[curated_idx[i]] for i in np.where(sub > MAX_GUESSES)[0]]
        print(f"    {label:<44} mean {sub.mean():.4f}  worst {int(sub.max())}  "
              f"losses {len(lost)} / {len(curated_idx)}")
        if lost:
            print(f"      {' '.join(sorted(lost))}")
        if label in (labels[0], labels[-1]):
            shown = "greedy entropy" if label == labels[0] else "expected (defaults)"
            table.append((shown, f"{sub.mean():.4f}", str(int(sub.max())),
                          f"{len(lost)} / {len(curated_idx)}"))
        if label == labels[-1]:
            common.record("curated.losses", len(lost))
            common.record("curated.lost_words", ", ".join(f"`{w}`" for w in sorted(lost)))
            used = {r["solution"] for r in common.solutions()}
            common.record("curated.lost_used", sum(1 for w in lost if w in used))
    common.record("curated.table", common.md_table(
        ["policy", "mean", "worst", "losses"], table))


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
    table, losses, masks = [], {}, {}
    for label, solver in rows.items():
        start = time.perf_counter()
        lengths, weights = play_all(solver)
        elapsed = time.perf_counter() - start
        row = stats(lengths, weights)
        masks[label] = row["lost_mask"]
        print(f"  {label:<38} {row['mean_penalized']:8.4f} "
              f"{row['loss_prob'] * 100:9.4f}% {(1 - row['loss_prob']) * 100:10.4f}% "
              f"{elapsed:6.0f}s")
        losses[label] = row["loss_prob"] * 100
        shown = label.replace("expected, ", "").replace("<=", "\u2264")
        cells = (shown, f"{row['mean_penalized']:.4f}",
                 f"{row['loss_prob'] * 100:.4f}%",
                 f"{(1 - row['loss_prob']) * 100:.4f}%", f"{elapsed:.0f} s")
        if "default" in label:
            cells = tuple(f"**{c}**" for c in cells)
        table.append(cells)
    common.record("search.table", common.md_table(
        ["policy", "mean", "loss probability", "solve rate", "wall clock"], table))
    none, two, full = (losses["expected, no exact search"],
                       losses["expected, exact for t <= 2 only"],
                       losses["expected, exact for m <= 60 (default)"])
    common.record("search.loss_none", none)
    common.record("search.loss_two", two)
    common.record("search.loss_full", full)
    common.record("search.two_share", (none - two) / (none - full))
    # the loss probability is a handful of curated words, so name the heaviest
    # one the full search wins that `t <= 2` alone loses, and its share
    rescued = masks["expected, exact for t <= 2 only"] & ~masks[
        "expected, exact for m <= 60 (default)"]
    heaviest = int(np.argmax(np.where(rescued, weights, 0.0)))
    share = 100 * weights[heaviest] / ((none - full) / 100)
    print(f"  heaviest word the full search rescues from t <= 2: "
          f"{data.all_words[heaviest]} at {100 * weights[heaviest]:.4f}%, "
          f"{share:.0f}% of the reduction")
    common.record("search.rescued.word", data.all_words[heaviest])
    common.record("search.rescued.weight", 100 * weights[heaviest])
    common.record("search.rescued.share", share / 100)
    rescue(rows["expected, exact for t <= 2 only"],
           rows["expected, exact for m <= 60 (default)"], heaviest)


def rescue(heuristic, searched, answer):
    """Where the two policies part on one answer, and what each side leaves.

    Both play the same game until the search first overrules the heuristic;
    the answer's bucket after each guess is then valued exactly with the turns
    that remain, which is the comparison the estimator could not make.
    """
    words = data.all_words
    cands = heuristic.all_candidates()
    for turn in range(1, MAX_GUESSES + 1):
        left = MAX_GUESSES - turn + 1
        h, _ = heuristic.best_guess(cands, left)
        x, _ = searched.best_guess(cands, left)
        if h != x:
            break
        cands = heuristic.filter(h, heuristic.feedback(h, answer), cands)
    else:
        return
    common.record("search.rescued.turn", turn)
    common.record("search.rescued.at", len(cands))
    common.record("search.rescued.turns_left", left - 1)
    print(f"  the two part at turn {turn}, {len(cands)} candidates: "
          f"heuristic {words[h]}, search {words[x]}")
    for tag, guess in (("heuristic", h), ("search", x)):
        bucket = heuristic.filter(guess, heuristic.feedback(guess, answer), cands)
        _, value = searched.exact.solve(bucket, left - 1)
        print(f"    {tag} {words[guess]} leaves {len(bucket)}, worth {value:.2f} "
              f"with {left - 1} turns left: {' '.join(words[c] for c in bucket)}")
        common.record(f"search.rescued.{tag}.guess", words[guess])
        common.record(f"search.rescued.{tag}.left", len(bucket))
        common.record(f"search.rescued.{tag}.value", value)
        common.record(f"search.rescued.{tag}.words",
                      ", ".join(f"`{words[c]}`" for c in bucket))


def costs():
    """What reaching further costs in time, and whether the penalty drives it.

    Wall clock does not travel between machines, so what this reports is the
    ratio to the shipped reach.  A sampled set of answers rather than all of
    them: what is wanted is the shape of the curve, and the memos would otherwise
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
            common.record(f"cost.{label}.{cap}.ratio", seconds / base)
            common.record(f"cost.{label}.{cap}.bailouts", int(bailouts))
        first_bailout = next((cap for cap in caps if timings[cap][1]), None)
        common.record(f"cost.{label}.first_bailout", first_bailout or 0)

        print(f"\n  and whether the penalty moves it:")
        print(f"  {'exact-max':>10} {'penalty':>8} {'seconds':>9} {'bailouts':>9}")
        spread = 0.0
        for cap in (DEFAULT_EXACT_MAX, 100):
            runs = {}
            for penalty in (1.0, 5.0, 30.0):
                seconds, bailouts = timed(cap, penalty)
                runs[penalty] = seconds
                print(f"  {cap:>10} {penalty:>8g} {seconds:>9.1f} {bailouts:>9}")
            spread = max(spread, max(runs.values()) / min(runs.values()) - 1)
        common.record(f"cost.{label}.penalty_spread", 100 * spread)
        print()


def bound():
    """Why `ExactEndgame.bound_penalty` caps the penalty at 1.

    The two-turn value prunes the search above it, which needs V to be
    non-decreasing in the turns remaining.  That holds at a penalty of 1, where
    running out costs exactly the guess that continuing would have, and fails at
    the shipped 5, where a position near the deadline would rather have the
    turns.  One family shows both.
    """
    from exact import ExactEndgame

    family = sorted(data.word2ind[word] for word in data.all_words
                    if word.endswith(args.bound_family))
    candidates = np.array(family, dtype=np.int64)
    weights = build_weights(data, PriorConfig())
    share = weights[candidates] / weights[candidates].sum()

    common.header(f"-{args.bound_family}: V by turns left, by penalty")
    print("  " + "  ".join(f"{data.all_words[i]} {p:.3f}"
                           for i, p in zip(candidates, share)))
    turns = (2, 3, 4)
    print(f"\n  {'penalty':>8} " + " ".join(f"{'t=' + str(t):>8}" for t in turns))
    by_penalty = {}
    for penalty in (1.0, DEFAULT_LOSS_PENALTY):
        # uncapped, so the t=2 column is the bound the search would use if
        # `bound_penalty` did not exist
        search = ExactEndgame(data.feedback_table, weights,
                              loss_penalty=penalty, max_branch=None)
        values = [search.solve(candidates, t)[1] for t in turns]
        rising = all(a <= b for a, b in zip(values, values[1:]))
        print(f"  {penalty:>8g} " + " ".join(f"{v:>8.4f}" for v in values)
              + ("   rising, so admissible" if rising
                 else "   falls: the two-turn value is not a bound"))
        by_penalty[penalty] = values
    print()
    if args.bound_family == "atch":
        shown = {1.0: "value at penalty 1", DEFAULT_LOSS_PENALTY: "value at penalty 5"}
        rows_out = []
        for penalty in (DEFAULT_LOSS_PENALTY, 1.0):
            cells = [f"{v:.2f}" for v in by_penalty[penalty]]
            if penalty != 1.0:
                cells[0] = f"**{cells[0]}**"
            rows_out.append((shown[penalty], *cells))
        common.record("bound.table", common.md_table(
            ["turns in hand `t`"] + [str(t) for t in turns], rows_out))
        common.record("bound.two_turn_at5", by_penalty[DEFAULT_LOSS_PENALTY][0])
        common.record("bound.three_turn_at5", by_penalty[DEFAULT_LOSS_PENALTY][1])


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
    tag = "penalty" if label.startswith("The penalty") else "reach"
    table, blended = [], {}
    for setting in settings:
        row = blend([grid[setting, c.slug] for c in configs])
        blended[setting] = row
        marker = " *" if setting == baseline_key else "  "
        print(f"  {setting:>8}{marker}{row['loss_prob'] * 100:9.4f}% "
              f"{row['losing_words']:7.1f} {row['mean_solved']:12.4f} "
              f"{row['mean_penalized']:14.4f}")
        key = f"{tag}.{setting:g}."
        common.record(key + "loss", row["loss_prob"] * 100)
        common.record(key + "mean", row["mean_penalized"])
        common.record(key + "solved", row["mean_solved"])
        cells = [f"{setting:g}", f"{row['loss_prob'] * 100:.4f}%"]
        if tag == "penalty":
            cells.append(f"{row['mean_solved']:.4f}")
        cells.append(f"{row['mean_penalized']:.4f}")
        if setting == baseline_key:
            cells = [f"**{cells[0]} (default)**"] + [f"**{c}**" for c in cells[1:]]
        table.append(tuple(cells))
    head = ["`--loss-penalty`", "loss probability", "mean (solved)", "mean (loss = 7)"] \
        if tag == "penalty" else ["`--exact-max`", "loss probability", "mean (loss = 7)"]
    common.record(f"{tag}.table", common.md_table(head, table))
    if tag == "penalty":
        # the three-row form SOLVER_EXPLAINED *Why the penalty is 5* carries
        short = []
        for setting in (1, 5, 30):
            if setting not in blended:
                continue
            row = blended[setting]
            cells = (f"{setting:g}", f"{row['loss_prob'] * 100:.4f}%",
                     f"{row['mean_penalized']:.4f}")
            if setting == baseline_key:
                cells = tuple(f"**{c}**" for c in cells)
            short.append(cells)
        common.record("penalty.short_table", common.md_table(
            ["penalty", "loss probability", "mean (loss scored as 7)"], short))
    if have_baseline:
        first = blended[settings[0]]
        last = blended[settings[-1]]
        base_row = blended[baseline_key]
        common.record(f"{tag}.default_to_last.loss_share",
                      (base_row["loss_prob"] - last["loss_prob"]) / first["loss_prob"])
        common.record(f"{tag}.default_to_last.mean",
                      base_row["mean_penalized"] - last["mean_penalized"])
        common.record(f"{tag}.first_to_default.loss_drop",
                      100 * (1 - base_row["loss_prob"] / first["loss_prob"]))
        common.record(f"{tag}.first_to_default.mean",
                      base_row["mean_penalized"] - first["mean_penalized"])
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
            if config == PriorConfig():
                common.record(f"{tag}.shipped.{setting:g}.loss", row["loss_prob"] * 100)
                common.record(f"{tag}.shipped.{setting:g}.mean", row["mean_penalized"])
                common.record(f"{tag}.shipped.{setting:g}.words", int(row["losing_words"]))
                common.record(f"{tag}.shipped.{setting:g}.curated", len(cur))
                if setting == settings[-1] and have_baseline:
                    common.record(f"{tag}.shipped.rescued",
                                  int(grid[baseline_key, config.slug]["losing_words"]
                                      - row["losing_words"]))
            if base is not None and setting != baseline_key:
                fixed, broken = flips(base, row["lost_mask"])
                print(f"            vs default: fixes {len(fixed)}, breaks {len(broken)}")
            if cur:
                print(f"            curated losses: {' '.join(sorted(cur))}")
    return grid


if args.bound:
    bound()
elif args.cost:
    costs()
elif args.search:
    search()
elif args.policies:
    policies()
else:
    sweep(
        f"The penalty, at exact-max {args.at_reach}", args.penalties,
        lambda config, penalty: outcomes(config, penalty, args.at_reach),
        DEFAULT_LOSS_PENALTY)

    if not args.skip_reach:
        print()
        sweep(f"The endgame search's reach, at penalty {args.at_penalty:g}",
              args.exact_max,
              lambda config, cap: outcomes(config, args.at_penalty, cap),
              DEFAULT_EXACT_MAX)
        # the corner the two sweeps share nothing with: the lowest penalty at
        # the shortest reach, against the shipped pair
        lowest, shortest = min(args.penalties), min(args.exact_max)
        corner = blend([stats(*outcomes(c, lowest, shortest)) for c in configs])
        shipped = blend([stats(*outcomes(c, DEFAULT_LOSS_PENALTY, DEFAULT_EXACT_MAX))
                         for c in configs])
        common.record("corner.loss", corner["loss_prob"] * 100)
        common.record("corner.loss_drop",
                      100 * (1 - shipped["loss_prob"] / corner["loss_prob"]))
        common.record("corner.mean_gain",
                      corner["mean_penalized"] - shipped["mean_penalized"])
        print(f"\n  from penalty {lowest:g} at reach {shortest} to the shipped pair: "
              f"loss {corner['loss_prob'] * 100:.4f}% -> "
              f"{shipped['loss_prob'] * 100:.4f}%, mean "
              f"{corner['mean_penalized']:.4f} -> {shipped['mean_penalized']:.4f}")
