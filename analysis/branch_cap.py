"""What the branch cap prunes, counted rather than timed.

SOLVER_EXPLAINED *The endgame* says the cap keeps the fifty best splits by the
bound and that it has never yet cost the winner.  Both halves are checkable
without a stopwatch, which is the point of doing it this way: a count is the
same on a loaded machine as on an idle one, where a wall clock is not.

Two things are counted over real positions, capped against uncapped:

1.  WHETHER THE CAP EVER COSTS THE ANSWER.  Solve each position both ways and
    compare the guess and its value.  This is the claim the cap actually rests
    on, and it is a claim about answers rather than about speed.
2.  HOW OFTEN THE CAP IS WHAT PRUNED ANYTHING.  `_solve` walks the splits in
    bound order and breaks the moment one cannot beat the incumbent, so the cap
    only ever binds at a node that reaches fifty splits before that break.
    Recording the splits evaluated per node says how often that happens.

The answer to the second is "almost never", which is worth knowing before
reaching for a wider cap: what prunes the endgame is the admissible bound, and
the cap is trimming a tail the bound had already reached.
"""

import argparse

import numpy as np

import common
from exact import ExactEndgame, SearchBudgetExceeded
from priors import PriorConfig, build_weights
from solver import EXPECTED, WordleSolver
from wordle_data import load_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--positions", type=int, default=150,
                    help="real positions to ablate the cap over")
parser.add_argument("--cap", type=int, default=50,
                    help="the branch cap to test against no cap at all")
args = parser.parse_args()

data = load_data()
config = PriorConfig()
weights = build_weights(data, config)
solver = WordleSolver(data=data, config=config, mode=EXPECTED)
table = data.feedback_table


class Counting(ExactEndgame):
    """Records the splits each node evaluated before the incumbent broke the
    loop, and how many subtree evaluations the whole search took."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.per_node = []
        self.evaluations = 0
        self._here = 0

    def _solve(self, candidates, turns):
        outer, self._here = self._here, 0
        try:
            return super()._solve(candidates, turns)
        finally:
            if self._here:
                self.per_node.append(self._here)
            self._here = outer

    def _evaluate(self, *a, **kw):
        self._here += 1
        self.evaluations += 1
        return super()._evaluate(*a, **kw)


def real_positions(limit):
    """Positions the endgame search really meets, taken from played games."""
    out, seen = [], set()
    for answer in range(0, 9000, 7):
        cands = data.all_candidates()
        for turn in range(4):
            guess, _ = solver.best_guess(cands, 5 - turn)
            if guess == answer:
                break
            cands = solver.filter(guess, solver.feedback(guess, answer), cands)
            key = cands.tobytes()
            if 3 <= len(cands) <= 60 and key not in seen:
                seen.add(key)
                out.append((cands, max(2, 5 - turn - 1)))
            if len(out) >= limit:
                return out
    return out


positions = real_positions(args.positions)
common.header(f"what the branch cap prunes, over {len(positions)} real positions")


def run(max_branch):
    search = Counting(table, weights, loss_penalty=5.0, max_branch=max_branch)
    answers, bailed, nodes = [], 0, 0
    for cands, turns in positions:
        try:
            answers.append(search.solve(cands, turns))
        except SearchBudgetExceeded:
            answers.append(None)
            bailed += 1
        nodes += search.nodes      # solve() zeroes it per call, so total here
    search.total_nodes = nodes
    return search, answers, bailed


capped, capped_answers, capped_bailed = run(args.cap)
full, full_answers, full_bailed = run(None)

print(f"  {'':28s} {'cap ' + str(args.cap):>14s} {'no cap':>14s}")
print("  " + "-" * 58)
print(f"  {'interior nodes':28s} {capped.total_nodes:>14,} {full.total_nodes:>14,}")
print(f"  {'subtree evaluations':28s} {capped.evaluations:>14,} "
      f"{full.evaluations:>14,}")
print(f"  {'node-budget bailouts':28s} {capped_bailed:>14,} {full_bailed:>14,}")

same_guess = sum(a is not None and b is not None and a[0] == b[0]
                 for a, b in zip(capped_answers, full_answers))
same_value = sum(a is not None and b is not None and abs(a[1] - b[1]) < 1e-9
                 for a, b in zip(capped_answers, full_answers))
scored = sum(a is not None and b is not None
             for a, b in zip(capped_answers, full_answers))
print("\n  does the cap ever cost the answer?")
print(f"    same guess as the uncapped search : {same_guess} of {scored}")
print(f"    same value                        : {same_value} of {scored}")

print("\n  splits a node evaluated before the bound broke the loop:")
for label, search in ((f"cap {args.cap}", capped), ("no cap", full)):
    ev = np.array(search.per_node)
    print(f"    {label:>8s}: median {int(np.median(ev)):>3}, 90th pct "
          f"{int(np.percentile(ev, 90)):>4}, max {int(ev.max()):>6}, "
          f"over {len(ev)} nodes")
ev = np.array(capped.per_node)
binds = int((ev >= args.cap).sum())
print(f"    nodes that reached the cap of {args.cap} at all: {binds} "
      f"of {len(ev)}")
common.record("positions", scored)
common.record("nodes", capped.total_nodes)
common.record("same_nodes", capped.total_nodes == full.total_nodes)
common.record("evaluations.cap", capped.evaluations)
common.record("evaluations.none", full.evaluations)
common.record("busiest.cap", int(np.array(capped.per_node).max()))
common.record("busiest.none", int(np.array(full.per_node).max()))
common.record("same_guess", same_guess)
common.record("same_value", same_value)
common.record("binding_nodes", binds)
common.record("unbound_nodes", len(ev) - binds)
common.record("table", common.md_table(
    [f"over {scored} real positions", f"cap {args.cap}", "no cap"],
    [("interior nodes searched", f"{capped.total_nodes:,}", f"{full.total_nodes:,}"),
     ("subtrees evaluated", f"{capped.evaluations:,}", f"{full.evaluations:,}"),
     ("splits evaluated at the busiest single node",
      f"{int(np.array(capped.per_node).max()):,}",
      f"{int(np.array(full.per_node).max()):,}")]))
print("\n  Where the two node counts agree, the cap pruned no subtree the")
print("  bound had not already reached; what it removes past that is the")
print("  tail of leaf evaluations, which are scans and cost nothing here.")
