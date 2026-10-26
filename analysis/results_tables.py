"""The two tables in README *Results*, replayed over every possible answer.

Part one scores every archived form under the shipped prior; part two plays
each prior combination in both modes.  Both weight by the prior in question,
so the means are computed and the gaps carry no sampling error.  Neither
refits: run `wordle_solver.py train` first if the fits are stale.
"""

import argparse
import json
import os

import numpy as np

import common
import estimator
from priors import PriorConfig
from solver import DERIVED_WIN_BONUS, ENTROPY, EXPECTED, WordleSolver
from wordle_data import (MAX_GUESSES, N_PATTERNS, WIN_PATTERN, load_data,
                         num_to_feedback)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--part", choices=("forms", "priors", "both"), default="both")
common.add_prior_filter(parser)
args = parser.parse_args()


def archived(config, form):
    path = estimator.form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def signed(x, places):
    """A table cell with its sign, and a true minus sign."""
    return f"{x:+.{places}f}".replace("-", "−")


def opening_of(solver):
    guess, _ = solver.best_guess(solver.all_candidates(), MAX_GUESSES)
    return solver.data.all_words[guess]


if args.part in ("forms", "both"):
    config = PriorConfig()
    common.header(f"forms under {config.describe()}")
    print("  every word as the answer, weighted by this prior; exact\n")

    runs, meta, weights = {}, {}, None
    rmse, expression = {}, {}
    for form in sorted(estimator.FORMS):
        record = archived(config, form)
        if record is None:
            continue
        solver = WordleSolver(config=config, params=record["params"], form=form)
        weights = solver.weights
        runs[form] = common.every_answer(solver)
        meta[form] = (f"{record.get('terms_used')}/{len(record.get('terms', []))}",
                      record.get("opening_word", "?"))
        rmse[form] = record.get("fit_rmse", float("nan"))
        expression[form] = common.pretty(estimator.FORMS[form].description)

    greedy = WordleSolver(config=config, mode=ENTROPY)
    runs["greedy entropy"] = common.every_answer(greedy)
    meta["greedy entropy"] = ("--", opening_of(greedy))

    # the reference that needs no fit: greedy entropy paying for the win chance
    priced = WordleSolver(config=config, mode=ENTROPY, win_bonus=DERIVED_WIN_BONUS)
    label = f"IG + {DERIVED_WIN_BONUS:g}*p_win"
    runs[label] = common.every_answer(priced)
    meta[label] = ("--", opening_of(priced))

    scored = {f: common.exact_stats(runs[f], weights) for f in runs}
    best = min(scored, key=lambda f: scored[f][0])
    print(f"  {'form':15s} {'terms':>6s} {'opening':>8s} {'mean':>8s} "
          f"{'vs best':>9s} {'loss prob':>10s}")
    for form in sorted(scored, key=lambda f: scored[f][0]):
        mean, loss = scored[form]
        print(f"  {form:15s} {meta[form][0]:>6s} {meta[form][1]:>8s} "
              f"{mean:8.4f} {mean - scored[best][0]:+9.4f} {loss * 100:9.4f}%")
    print(f"\n  best: {best}")

    order = sorted(scored, key=lambda f: scored[f][0])
    table = []
    for form in order:
        mean, loss = scored[form]
        gap = mean - scored[best][0]
        table.append((common.pretty(form), expression.get(form, "\u2014"),
                      meta[form][0].replace("--", "\u2014"), meta[form][1],
                      f"{mean:.4f}", f"+{gap:.4f}" if form != best else "\u2014",
                      f"{loss * 100:.4f}%"))
    common.record("forms_table", common.md_table(
        ["form", "expression", "terms", "opening", "mean", "vs. best",
         "loss probability"], table))
    common.record("best_form", best)
    fitted = [f for f in order if f in rmse]
    same = [f for f in fitted if meta[f][1] == meta[best][1]]
    others = [f for f in fitted if f not in same]
    common.record("same_opening", len(same))
    common.record("same_opening_spread", scored[same[-1]][0] - scored[best][0])
    common.record("cliff", scored[others[0]][0] - scored[same[-1]][0])
    entropy = scored["greedy entropy"]
    common.record("ig_over_entropy", entropy[0] - scored[label][0])
    common.record("ig_loss_ratio", scored[label][1] / entropy[1])
    common.record("entropy_gap", entropy[0] - scored[best][0])
    common.record("rmse_line", rmse[others[0]])
    common.record("rmse_lo", min(rmse[f] for f in same))
    common.record("rmse_hi", max(rmse[f] for f in same))
    # the worst fit of the nine, its play, and the range the rest fit in
    worst = max(fitted, key=rmse.get)
    common.record("worst_fit.form", f"`{worst}`")
    common.record("worst_fit.rmse", rmse[worst])
    common.record("worst_fit.gap", scored[worst][0] - scored[best][0])
    common.record("worst_fit.opening", meta[worst][1])
    rest = [rmse[f] for f in fitted if f != worst]
    common.record("worst_fit.rest_lo", min(rest))
    common.record("worst_fit.rest_hi", max(rest))
    # the forms that open on another word than the best, and how they fit
    common.record("other_opening.count", len(others))
    common.record("other_opening.rmse_hi", max(rmse[f] for f in others))
    common.record("invonly_gap", scored["invonly"][0] - scored[best][0])

    # the fitted curve SOLVER_EXPLAINED section 6 reads: the spline under the
    # shipped shapes, and the same form under flat shapes beside it
    sizes = (2, 4, 16, 100, 1000, 12947)
    rows = []
    for label, cfg in (("shipped shapes", PriorConfig()),
                       ("flat shapes", PriorConfig(repeat_shape="uniform",
                                                   curation_shape="uniform"))):
        # the fit in play, which promote may have kept over the newer archive
        record = estimator.load_record(cfg)
        if record is None or record.get("form") != "spline":
            record = archived(cfg, "spline")
        if record is None:
            continue
        params = np.array(record["params"])
        values = [float(estimator.f_value(np.array([n]), params, "spline")[0])
                  for n in sizes]
        cells = [f"{v:.2f}" for v in values]
        if label == "shipped shapes":
            cells = [f"**{c}**" for c in cells]
            terms = [f"{params[0]:.3f}", f"{params[1]:.3f}\u00b7x"]
            slopes, slope = [params[1]], params[1]
            # a hinge that prints as 0.000 is no bend, so it is left out
            for knot, coef in zip((2, 4, 6, 8), params[2:]):
                if round(coef, 3) != 0.0:
                    sign = "\u2212" if coef < 0 else "+"
                    terms.append(f"{sign} {abs(coef):.3f}\u00b7(x\u2212{knot})\u208a")
                    slope += coef
                    slopes.append(slope)
            common.record("spline.equation", " ".join(
                [terms[0], "+", terms[1]] + terms[2:]))
            common.record("spline.bends", len(slopes) - 1)
            common.record("spline.slopes", ", ".join(f"{x:.3f}" for x in slopes))
            common.record("spline.final_slope", float(slopes[-1]))
            common.record("spline.last_knot", max(
                k for k, c in zip((2, 4, 6, 8), params[2:]) if round(c, 3) != 0.0))
        else:
            common.record("flat_spline.f1000", values[4])
            common.record("flat_spline.f16", values[2])
            common.record("flat_spline.top_hinge", float(params[-1]))
        rows.append((f"`f(n)`, {label}", *cells))
    common.record("fn_table", common.md_table(
        ["n"] + [f"{n:,}" for n in sizes], rows))

if args.part in ("priors", "both"):
    common.header("prior combinations, both modes")
    print("  every word as the answer, weighted by that row's own prior, so the "
          "rows are\n  not comparable to each other; both modes in a row are "
          "exact under it\n")
    print(f"  {'priors':14s} {'opening':>8s} {'entropy':>9s} {'expected':>9s} "
          f"{'solved':>8s} {'form':>11s}")
    rows, table = [], []
    for label, config in common.filter_priors([
        ("none", PriorConfig(repeat=False, curation=False)),
        ("curation only", PriorConfig(repeat=False)),
        ("repeat only", PriorConfig(curation=False)),
        ("both", PriorConfig()),
    ], args):
        means = {}
        for mode in (ENTROPY, EXPECTED):
            solver = WordleSolver(config=config, mode=mode)
            means[mode] = common.exact_stats(common.every_answer(solver),
                                             solver.weights)
            if mode == EXPECTED:
                opening, form = opening_of(solver), solver.form.name
        # the linear form sits between the two: entropy plus the bonus and
        # nothing else, so it splits the gap into the bonus and the curvature
        record = archived(config, "loglinear")
        linear = WordleSolver(config=config, params=record["params"],
                              form="loglinear")
        straight = common.exact_stats(common.every_answer(linear),
                                      linear.weights)[0]
        rows.append((label, means[ENTROPY][0], straight, means[EXPECTED][0]))
        shown = f"**{label} (the default)**" if label == "both" else label
        table.append((shown, form, opening, f"{means[ENTROPY][0]:.4f}",
                      f"{means[EXPECTED][0]:.4f}",
                      f"{(1 - means[EXPECTED][1]) * 100:.2f}%"))
        key = label.replace(" ", "_")
        common.record(f"{key}.gain", means[ENTROPY][0] - means[EXPECTED][0])
        common.record(f"{key}.opening", opening)
        common.record(f"{key}.form", form)
        common.record(f"{key}.expected", means[EXPECTED][0])
        common.record(f"{key}.entropy", means[ENTROPY][0])
        print(f"  {label:14s} {opening:>8s} {means[ENTROPY][0]:9.4f} "
              f"{means[EXPECTED][0]:9.4f} {(1 - means[EXPECTED][1]) * 100:7.2f}% "
              f"{form:>11s}")

    common.header("what the gap between the two modes is made of")
    print("  the linear f is greedy entropy plus the possible-answer bonus "
          "(section 13),\n  so what it leaves is curvature\n")
    print(f"  {'priors':14s} {'entropy':>9s} {'loglinear':>10s} {'best':>9s} "
          f"{'bonus':>9s} {'curvature':>10s}")
    split = []
    for label, entropy, straight, best in rows:
        print(f"  {label:14s} {entropy:9.4f} {straight:10.4f} {best:9.4f} "
              f"{entropy - straight:+9.4f} {straight - best:+10.4f}")
        split.append((label, f"{entropy:.4f}", f"{straight:.4f}", f"{best:.4f}",
                      signed(entropy - straight, 4), signed(straight - best, 4)))
        key = label.replace(" ", "_")
        common.record(f"{key}.bonus", entropy - straight)
        common.record(f"{key}.curvature", straight - best)
    common.record("priors_table", common.arm_table(
        ["priors", "form", "opening", "entropy mean", "expected mean",
         "solve rate"], table, "priors_table"))
    common.record("split_table", common.arm_table(
        ["priors", "entropy", "linear", "best", "bonus", "curvature"], split,
        "split_table"))
    # the same rows as SOLVER_EXPLAINED sections 4 and 6 lay them out
    order = ["none", "repeat only", "curation only", "both"]
    by_label = {label: (e, l, b) for label, e, l, b in rows}
    bold = lambda label, cells: (  # noqa: E731
        (f"**{label} (the default)**", *(f"**{c}**" for c in cells))
        if label == "both" else (label, *cells))
    common.record("entropy_table", common.arm_table(
        ["priors on", "greedy entropy"],
        [(label, f"{by_label[label][0]:.2f}") for label in order if label in by_label],
        "entropy_table"))
    common.record("bonus_table", common.arm_table(
        ["priors on", "possible-answer bonus", "curvature in `f`"],
        [bold(label, (signed(by_label[label][0] - by_label[label][1], 3),
                      signed(by_label[label][1] - by_label[label][2], 3)))
         for label in order if label in by_label], "bonus_table"))
    # `none` reads no answer record, so a run that dropped it keeps its bonus
    none_bonus = (by_label["none"][0] - by_label["none"][1]
                  if "none" in by_label else common.last("none.bonus"))
    common.record("bonus_ratio",
                  (by_label["both"][0] - by_label["both"][1]) / none_bonus)

# what the cached fits converged through: the default prior's rounds, and the
# round counts across the five priors `train` fits
common.header("the rounds the cached fits took")
history = [h["mean_guesses"] for h in estimator.load_record(PriorConfig())["history"]]
print(f"  default prior: " + " -> ".join(f"{m:.2f}" for m in history))
common.record("rounds.seed", history[0])
common.record("rounds.iter1", history[1])
common.record("rounds.iter2", history[2])
fitted = [PriorConfig(repeat=False, curation=False), PriorConfig(repeat=False),
          PriorConfig(curation=False), PriorConfig(),
          PriorConfig(curation_shape="frequency")]
iterations = [estimator.load_record(c)["iterations"] for c in fitted]
print(f"  rounds per prior: {min(iterations)} to {max(iterations)} "
      f"over {len(fitted)} priors")
common.record("rounds.min", min(iterations))
common.record("rounds.max", max(iterations))
common.record("rounds.priors", len(fitted))

# the partition SOLVER_EXPLAINED opens with: one guess against the whole list
common.header("the buckets `crane` makes of the whole list")
data = load_data()
counts = np.bincount(data.feedback_table[data.word2ind["crane"]], minlength=N_PATTERNS)
order = np.argsort(-counts, kind="stable")
rows = [(f"`{num_to_feedback(int(p))}`", f"{int(counts[p]):,}") for p in order[:3]]
rows += [("…", f"({int((counts > 0).sum())} non-empty buckets)"),
         ("`GGGGG`", f"{int(counts[WIN_PATTERN])} (`crane` itself)")]
for pattern, size in rows:
    print(f"  {pattern:10s} {size}")
common.record("crane.buckets", int((counts > 0).sum()))
common.record("crane.table", common.md_table(["pattern", "candidates"], rows))
