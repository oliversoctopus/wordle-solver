"""The regime table and the pool-exhaustion arithmetic.

Backs README *The regime the defaults describe* and SOLVER_EXPLAINED
*Not all words are equally likely*.  The closing block is what each prior
implies about the repeat rate, which is why the two priors turn out to
need each other -- SOLVER_EXPLAINED *Do the two priors need each other?* --
it needs the weights but no solver, so it stays a seconds-long script.
"""

import datetime

import numpy as np

import common
from priors import PriorConfig, build_weights
from wordle_data import load_data

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

common.header("what each prior implies about the two rates")
print("  the mass it puts on the already-used words, and on the non-curated\n"
      "  ones given not a repeat -- the same conditioning the two rates are\n"
      "  defined with.  No solver in this, just the weights.\n")

data = load_data(mask_last=0, with_feedback_table=False)
is_used = np.array([w in seen for w in data.all_words])
is_curated = np.array([w in curated for w in data.all_words])
print(f"  {'prior':14s} {'repeat':>9s} {'offlist|not repeat':>21s}")
for label, config in [
        ("none", PriorConfig(repeat=False, curation=False)),
        ("curation only", PriorConfig(repeat=False)),
        ("repeat only", PriorConfig(curation=False)),
        ("both", PriorConfig())]:
    w = build_weights(data, config)
    unused = w[~is_used].sum()
    print(f"  {label:14s} {100*w[is_used].sum():8.1f}% "
          f"{100*w[~is_used & ~is_curated].sum()/unused:20.1f}%")

# the observed pair: the regime row of the table above, restated
nonrep = [j for j in range(start, N) if not is_repeat[j]]
obs_off = sum(off_list[j] for j in nonrep) / len(nonrep)
print(f"\n  {'observed':14s} {100*rep_rate:8.1f}% {100*obs_off:20.1f}%   over the "
      f"regime's {N - start} puzzles")
print("\n  Each prior is wildly wrong about the rate the *other* one governs,")
print("  which is why neither is worth much alone.")
