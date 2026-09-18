"""The shipped solver against the reference that needs no fit, game by game.

`results_tables.py` puts both on one line each; this asks where the difference
between those lines actually comes from.  Both policies are deterministic, so
playing all 12,947 words once is the whole sample space rather than a draw from
it: the paired column has no sampling error in it at all, and a word either
changes hands or it does not.

The reference is `IG(g) + a*p_win(g)` with `a` pinned by two exact facts rather
than fitted -- see README *Results*.  It is not the baseline; greedy entropy is,
and `--baseline` puts that arm in instead.

    python analysis/head_to_head.py               shipped vs the reference
    python analysis/head_to_head.py --baseline    shipped vs greedy entropy
    python analysis/head_to_head.py --all-priors  every prior combination

Backs README *Where the estimator's guesses come from*.  Moves with the fit and
with the answer log, being a replay of both policies.
"""

import argparse

import numpy as np

import common
from priors import PriorConfig
from solver import DERIVED_WIN_BONUS, ENTROPY, WordleSolver

MAX_GUESSES = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", action="store_true",
                    help="compare against greedy entropy rather than the "
                         "reference policy")
parser.add_argument("--all-priors", action="store_true")
parser.add_argument("--losses", type=int, default=10,
                    help="heaviest lost words to name per arm")
common.add_prior_filter(parser)
args = parser.parse_args()

configs = common.filter_priors(
    [PriorConfig(repeat=r, curation=c)
     for r in (False, True) for c in (False, True)]
    if args.all_priors else [PriorConfig()], args)

other = ("greedy entropy" if args.baseline
         else f"IG + {DERIVED_WIN_BONUS:g}*p_win")
other_kwargs = (dict(mode=ENTROPY) if args.baseline
                else dict(mode=ENTROPY, win_bonus=DERIVED_WIN_BONUS))


def opening_of(solver):
    guess, _ = solver.best_guess(solver.all_candidates(), MAX_GUESSES)
    return solver.data.all_words[guess]


for config in configs:
    common.header(f"shipped against {other}: {config.describe()}")
    print("  every word as the answer, weighted by this prior, a loss counted "
          "as 7\n")

    arms = {}
    for label, kwargs in (("shipped", {}), (other, other_kwargs)):
        solver = WordleSolver(config=config, **kwargs)
        arms[label] = (common.every_answer(solver), opening_of(solver))
    weights = solver.weights
    total = float(weights.sum())

    def mass(mask):
        return float(weights[mask].sum()) / total * 100

    print(f"  {'':22}" + "".join(f"{label:>16}" for label in arms))
    lines = [
        ("opening", lambda run, opening: opening),
        ("mean guesses", lambda run, _: f"{float(weights @ run) / total:.4f}"),
        ("loss probability", lambda run, _: f"{mass(run > MAX_GUESSES):.4f}%"),
        ("words lost", lambda run, _: f"{int((run > MAX_GUESSES).sum())}"),
    ]
    for name, fn in lines:
        print(f"  {name:22}"
              + "".join(f"{fn(*arms[label]):>16}" for label in arms))

    print("\n  guesses taken, as prior mass and as a count of words")
    # recorded for the shipped prior only, which is the table README carries
    tag = "" if config == PriorConfig() else f"{config.slug}."
    if args.baseline:
        tag += "baseline."
    table = []
    for k in range(1, MAX_GUESSES + 2):
        cells, shares = "", []
        for label in arms:
            run = arms[label][0]
            hit = run == k if k <= MAX_GUESSES else run > MAX_GUESSES
            cells += f"{mass(hit):9.2f}%{int(hit.sum()):6d}"
            shares.append(mass(hit))
        print(f"  {('lost' if k > MAX_GUESSES else k):>22}" + cells)
        if k > 1:
            digits = 4 if k > MAX_GUESSES else 2
            shown = [f"{x:.{digits}f}%" for x in shares]
            # more mass is good through three guesses and bad from four on
            better = shares.index(max(shares) if k <= 3 else min(shares))
            shown[better] = f"**{shown[better]}**"
            table.append(("lost" if k > MAX_GUESSES else str(k), *shown))
    common.record(tag + "table", common.md_table(
        ["guesses", "shipped", f"`{common.pretty(other)}`"], table))
    common.record(tag + "two.shipped", mass(first_two := (arms["shipped"][0] == 2)))
    common.record(tag + "two.other", mass(arms[other][0] == 2))
    common.record(tag + "lost.shipped", int((arms["shipped"][0] > MAX_GUESSES).sum()))
    common.record(tag + "lost.other", int((arms[other][0] > MAX_GUESSES).sum()))
    common.record(tag + "loss_ratio", mass(arms["shipped"][0] > MAX_GUESSES)
                  / mass(arms[other][0] > MAX_GUESSES))

    # the whole point of enumerating: a word either changes hands or it does not
    first, second = (arms[label][0] for label in arms)
    diff = second - first
    print("\n  head to head")
    for name, hit, key in (("shipped sooner", diff > 0, "sooner.shipped"),
                           (f"{other} sooner", diff < 0, "sooner.other"),
                           ("the same", diff == 0, "same")):
        print(f"    {name:<20}{mass(hit):9.2f}%{int(hit.sum()):6d} words")
        common.record(tag + key, mass(hit))
    print(f"    {'mean paired gap':<20}{float(weights @ diff) / total:+9.4f} guesses")
    common.record(tag + "gap", float(weights @ diff) / total)

    if args.losses:
        for label in arms:
            lost = np.nonzero(arms[label][0] > MAX_GUESSES)[0]
            heaviest = lost[np.argsort(-weights[lost])][:args.losses]
            side = "shipped" if label == "shipped" else "other"
            common.record(tag + f"heaviest.{side}", solver.data.all_words[heaviest[0]])
            common.record(tag + f"heaviest.{side}.mass",
                          weights[heaviest[0]] / total * 100)
            print(f"\n  {label} loses {len(lost)} words, heaviest first:")
            print("    " + ", ".join(
                f"{solver.data.all_words[w]} ({weights[w] / total * 100:.4f}%)"
                for w in heaviest))
    print()
