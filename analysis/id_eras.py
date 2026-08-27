"""What the NYT's `id` field indexes, and the two eras it does it in.

`solutions.json` carries the NYT's own `id` per puzzle.  It is not the puzzle
number and it is not stable per word: it is a position in the list the editor
draws from, and that list was re-sorted at the regime boundary.  Backs
SOLVER_EXPLAINED *The off-list group*, whose block argument rests on it.

Nothing here feeds the solver.  It is the evidence for one paragraph, kept
runnable because the answer log moves under it.
"""

import bisect
import math

import numpy as np

import common

sol = common.solutions()
curated = common.load("wordles.json")          # stored in original play order
curated_set = set(curated)
alpha = sorted(curated)
play_pos = {w: i + 1 for i, w in enumerate(curated)}

# where the appended off-list block sits, and where the re-sorted list starts
BLOCK_LO, BLOCK_HI = len(curated) + 1, 2397
ERA2_LO = BLOCK_HI + 1


def ascending_runs(pairs):
    """Split (id, word) in id order into maximal alphabetically sorted runs."""
    runs, run = [], [pairs[0]]
    for pair in pairs[1:]:
        if pair[1] > run[-1][1]:
            run.append(pair)
        else:
            runs.append(run)
            run = [pair]
    runs.append(run)
    return runs


common.header("the id field: two numbering eras")

# ---------------------------------------------------------------- era 1
era1 = sorted((r["id"], r["solution"], r["print_date"]) for r in sol
              if r["solution"] in curated_set and r["id"] < BLOCK_LO)
exact = [i for i, w, _ in era1 if play_pos[w] == i]

print(f"Era 1 -- id is a position in the list as first played.")
print(f"  wordles.json is stored in that order, not alphabetically "
      f"(sorted: {curated == alpha})")
print(f"  id equals the file position for the first {max(exact)} puzzles")
print()
print("  after that the two drift, one step at a time:")
seen, drift = None, []
for i, w, date in era1:
    off = play_pos[w] - i
    # the handful of words that moved wholesale are not part of the climb
    if off != seen and abs(off) < 100:
        drift.append((i, w, date, off))
        seen = off
for i, w, date, off in drift[:4]:
    print(f"    id {i:5d}  {w}  {date}   position - id = {off:3d}")
print("      ...")
for i, w, date, off in drift[-2:]:
    print(f"    id {i:5d}  {w}  {date}   position - id = {off:3d}")
print(f"    climbing to {drift[-1][3]} by {drift[-1][2]}, in "
      f"{len(drift) - 1} moves")
moved = [(i, w) for i, w, _ in era1 if abs(play_pos[w] - i) >= 100]
print(f"  {len(moved)} words sit far out of place, which is a substitution "
      f"rather than a step: " + ", ".join(w for _, w in moved))

# ------------------------------------------------------- the off-list block
print()
block = sorted((r["id"], r["solution"]) for r in sol
               if r["solution"] not in curated_set and BLOCK_LO <= r["id"] <= BLOCK_HI)
inside = [r for r in sol if r["solution"] in curated_set
          and BLOCK_LO <= r["id"] <= BLOCK_HI]
print(f"  Off-list words are appended past the curated {len(curated)}, at ids "
      f"{BLOCK_LO}-{BLOCK_HI}:")
print(f"    {len(block)} of the {BLOCK_HI - BLOCK_LO + 1} slots have been an "
      f"answer; curated words in that range: {len(inside)}")
runs = ascending_runs(block)
print("    the batches come out alphabetically sorted:")
for r in runs:
    if len(r) >= 4:
        print(f"      {len(r):2d} words, ids {r[0][0]}-{r[-1][0]}: "
              + " ".join(w for _, w in r))
longest = max(len(r) for r in runs)
print(f"    longest run {longest}; by chance that ordering is 1 in "
      f"{math.factorial(longest):,}")

# ---------------------------------------------------------------- era 2
print()
era2 = sorted((r["id"], r["solution"], r["print_date"]) for r in sol
              if r["id"] >= ERA2_LO)
first_date = min(d for _, _, d in era2)
print(f"Era 2 -- from {first_date}, the list is re-sorted alphabetically.")

# the sorted stretch is the pool itself; what sits either side was appended
pool = max(ascending_runs([(i, w) for i, w, _ in era2]), key=len)
before = [(i, w) for i, w, _ in era2 if i < pool[0][0]]
rank = np.array([bisect.bisect_left(alpha, w) for _, w in pool], dtype=float)
ids = np.array([i for i, _ in pool], dtype=float)
slope, intercept = np.polyfit(rank, ids, 1)
resid = ids - (slope * rank + intercept)
r = np.corrcoef(rank, ids)[0, 1]
print(f"  {len(before)} answer(s) sit below the sorted stretch: "
      + ", ".join(f"{w} at {i}" for i, w in before))
print(f"  fitted against alphabetical rank in the curated list, over the "
      f"{len(pool)} answers in that stretch ({pool[0][1]} to {pool[-1][1]}):")
print()
print(f"      id = {slope:.5f} x rank + {intercept:.2f}")
print(f"      r = {r:.5f}, largest residual {np.abs(resid).max():.1f} "
      f"over a span of {ids.max() - ids.min():.0f} ids")
print(f"      implied pool: {slope * len(curated):,.0f} ids for the "
      f"{len(curated)} curated words, so about "
      f"{len(curated) - slope * len(curated):,.0f} are absent")

repeats = common.repeat_events()
second = {common.answers()[i]: sol[i] for i, _, _ in repeats}
in_era2 = sum(1 for row in second.values() if row["id"] >= ERA2_LO)
print()
print(f"  all {len(repeats)} repeats in Wordle's history have their second use "
      f"in era 2:")
print(f"    {in_era2} of {len(repeats)} carry an era-2 id, the earliest being "
      f"{min(second.values(), key=lambda r: r['id'])['solution']} at "
      f"{min(r['id'] for r in second.values())} on "
      f"{min(second.values(), key=lambda r: r['id'])['print_date']}")

tail = [(i, w) for i, w, _ in era2 if i > pool[-1][0]]
print()
print(f"  past the sorted stretch the order breaks again, which is the next "
      f"batch appended:")
for i, w in tail:
    kind = "curated" if w in curated_set else "off-list"
    print(f"    {i:5d}  {w}  ({kind})")
print("  curated and off-list share that range, so an era-2 id no longer says")
print("  which list a word came from.")
