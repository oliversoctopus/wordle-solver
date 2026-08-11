"""The regime table and the pool-exhaustion arithmetic.

Backs README *Both defaults describe a regime six months old* and
SOLVER_EXPLAINED 4.1.
"""

import datetime

import common

sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
N = len(words)

seen, is_repeat = set(), []
for w in words:
    is_repeat.append(w in seen)
    seen.add(w)
off_list = [w not in curated for w in words]

common.header("regime table")
print(f"{N} answers, through Wordle #{common.puzzle_no(sol[-1], N-1)} "
      f"({sol[-1]['print_date']})")
print()
print(f"  {'window':38s} {'puzzles':>8s} {'repeat':>9s} {'offlist|not repeat':>20s}")


def row(lo, hi, label):
    n = hi - lo
    rep = sum(is_repeat[lo:hi])
    nonrep = [j for j in range(lo, hi) if not is_repeat[j]]
    off = sum(off_list[j] for j in nonrep)
    print(f"  {label:38s} {n:8d} {100*rep/n:8.2f}% {100*off/len(nonrep):19.2f}%")


start = common.regime_start(sol)
end2025 = sum(1 for r in sol if r["print_date"] < "2026-01-01")
row(0, end2025, "2021-2025")
row(start, N, f"since #{common.puzzle_no(sol[start], start)} ({common.REGIME_DATE})")
row(N - 50, N, "last 50")
row(0, N, "all time")

common.header("the curated pool is running out")
used = {w for w in words if w in curated}
first_date = {}
for r in sol:
    first_date.setdefault(r["solution"], r["print_date"])
last = datetime.date.fromisoformat(sol[-1]["print_date"])
window = last - datetime.timedelta(days=730)
fresh = sum(1 for r in sol
            if r["solution"] in curated
            and datetime.date.fromisoformat(r["print_date"]) > window
            and first_date[r["solution"]] == r["print_date"])
rate = fresh / 2.0
remaining = len(curated) - len(used)
print(f"curated used            {len(used)} of {len(curated)}")
print(f"still unused            {remaining}")
print(f"fresh consumed / year   {rate:.0f}   (over the last two years)")
print(f"years of pool left      {remaining/rate:.1f}")
print(f"distinct words used     {len(seen)}   <- the repeat group's size")

common.header("arm sizes and the regime mixture")
reps = common.repeat_events(words)
offs = common.offlist_events(words, curated)
rep_rate = sum(1 for i, _, _ in reps if i >= start) / (N - start)
off_rate = sum(1 for i, _ in offs if i >= start) / (N - start)
print(f"repeat events           {len(reps)}")
print(f"off-list events         {len(offs)}")
print(f"still-unused curated    {remaining}")
print(f"regime mixture          {100*rep_rate:.1f}% / {100*off_rate:.1f}% / "
      f"{100*(1-rep_rate-off_rate):.1f}%")
