"""What the NYT's `id` field indexes, and the two eras it does it in.

The id is a position in one list that has only ever been appended to, and
reading it that way accounts for every id in the record.  README *What the id
field indexes* has the reconstruction; SOLVER_EXPLAINED *The off-list group*
takes what its block argument needs.  Nothing here feeds the solver.
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
common.record("exact_positions", max(exact))
print()
print("  after that the id runs behind it, by a gap that widens with position:")
seen, drift = None, []
for i, w, date in era1:
    off = play_pos[w] - i
    # the two words that moved wholesale are not part of the step function
    if off != seen and abs(off) < 100:
        drift.append((i, w, date, off))
        seen = off
for i, w, date, off in drift[:3]:
    print(f"    id {i:5d}  {w}  {date}   position - id = {off:3d}")
print("      ...")
for i, w, date, off in drift[-2:]:
    print(f"    id {i:5d}  {w}  {date}   position - id = {off:3d}")

# invert the step function: the file slots no era-1 id ever lands on
step, cursor = {}, 0
known = {i: play_pos[w] - i for i, w, _ in era1 if abs(play_pos[w] - i) < 100}
last_id = max(i for i, w, _ in era1 if abs(play_pos[w] - i) < 100)
for i in range(1, last_id + 1):
    cursor = known.get(i, cursor)
    step[i] = cursor
covered = {i + step[i] for i in step}
era1_slot = {curated[i + step[i] - 1]: i for i in step}
gone = [p for p in range(1, len(curated) + 1) if p not in covered]
played = {r["solution"] for r in sol}
print(f"  the gap is a step function of position, not of date: {len(gone)} "
      f"file slots are skipped,")
print(f"  and the last curated id is {last_id} = {len(curated)} - {len(gone)}. "
      f"The skipped words are")
print("  a single edit to the list, met one at a time as play reached them:")
for row in range(0, len(gone), 6):
    print("    " + " ".join(curated[p - 1] for p in gone[row:row + 6]))
back = [curated[p - 1] for p in gone if curated[p - 1] in played]
print(f"  {len(gone) - len(back)} of them were never played at all; "
      f"{len(back)} came back elsewhere in the numbering: "
      + ", ".join(back))
first_id = {}
for row in sol:
    first_id.setdefault(row["solution"], row["id"])
requeued = sorted(w for w in back if first_id[w] > last_id)
forward = sorted(w for w in back if first_id[w] <= last_id)
common.record("curated", len(curated))
common.record("last_curated_id", last_id)
common.record("back_lo", last_id + 1)
purge = [p for p in gone if curated[p - 1] not in forward]
common.record("skipped", len(gone))
common.record("purge", len(purge))
common.record("skip_first", min(purge))
common.record("skip_last", max(purge))
common.record("dropped", len(gone) - len(back))
common.record("requeued", " and ".join(f"`{w}`" for w in requeued))
common.record("forward", " and ".join(f"`{w}`" for w in forward))
common.record("forward_positions", " and ".join(str(play_pos[w]) for w in forward))
common.record("block_lo", BLOCK_LO)
common.record("block_hi", BLOCK_HI)
common.record("era2_lo", ERA2_LO)

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
for k, r in enumerate(sorted(runs, key=len, reverse=True)[:2], start=1):
    common.record(f"run{k}.words", len(r))
    common.record(f"run{k}.lo", r[0][0])
    common.record(f"run{k}.hi", r[-1][0])
    common.record(f"run{k}.list", " ".join(w for _, w in r))
    # the unseen ids between the run's first two words
    gap = range(r[0][0] + 1, r[1][0])
    common.record(f"run{k}.gap", " and ".join(str(i) for i in gap))
    common.record(f"run{k}.gap_words", f"`{r[0][1]}` and `{r[1][1]}`")
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
common.record("tail_lo", tail[0][0])
common.record("tail_hi", tail[-1][0])
print("  curated and off-list share that range, so an era-2 id no longer says")
print("  which list a word came from.")

# --------------------------------------------- what takes an era-2 id, and why
print()
start = common.regime_start(sol)
first_use = {}
for row in sol:
    first_use.setdefault(row["solution"], row)
table = {}
for row in sol[start:]:
    word = row["solution"]
    if first_use[word] is not row:
        kind = "repeat"
    elif word not in curated_set:
        kind = "off-list"
    else:
        kind = "curated, first use"
    era = "era-2" if row["id"] >= ERA2_LO else "era-1"
    table[(kind, era)] = table.get((kind, era), 0) + 1
print(f"  answers since {common.REGIME_DATE}, by kind and numbering:")
print(f"    {'':22s} {'era-1':>7s} {'era-2':>7s}")
for kind in ("repeat", "curated, first use", "off-list"):
    print(f"    {kind:22s} {table.get((kind, 'era-1'), 0):7d} "
          f"{table.get((kind, 'era-2'), 0):7d}")
common.record("era1_first_uses", table.get(("curated, first use", "era-1"), 0))
common.record("era2_first_uses", table.get(("curated, first use", "era-2"), 0))
common.record("era1_offlist", table.get(("off-list", "era-1"), 0))
common.record("era2_offlist", table.get(("off-list", "era-2"), 0))

# a repeat has no era-1 slot left to use; the rest of era 2 had one and did not
taken = {row["id"] for row in sol}
chose = [(row["id"], w) for w, row in first_use.items()
         if row["id"] >= ERA2_LO and w in era1_slot and era1_slot[w] not in taken]
print()
print(f"  a repeat is forced into era 2, its own slot being spent.  The other")
print(f"  {len(chose)} era-2 first-uses had a free era-1 slot and took a new "
      f"number anyway:")
for i, w in sorted(chose):
    print(f"    {w:6s} era-2 id {i:5d}, era-1 slot {era1_slot[w]} never played")
