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
wordlebot = common.load("wordlebot_weights.json")
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
    return n, 100 * rep / n, 100 * off / len(nonrep)


start = common.regime_start(sol)
end2025 = sum(1 for r in sol if r["print_date"] < "2026-01-01")
regime_no = common.puzzle_no(sol[start], start)
table = []
for key, (lo, hi, label, shown) in {
        "early": (0, end2025, "2021-2025", "2021\u20132025"),
        "since": (start, N, f"since #{regime_no} ({common.REGIME_DATE})",
                  f"**since Wordle #{regime_no} ({common.REGIME_DATE})**"),
        "last50": (N - 50, N, "last 50", "last 50"),
        "all": (0, N, "all time", "all time")}.items():
    n, rep, off = row(lo, hi, label)
    common.record(f"{key}.puzzles", n)
    common.record(f"{key}.repeat", rep)
    common.record(f"{key}.offlist", off)
    bold = key == "since"
    cells = [f"{n:,}", f"{rep:.2f}%" if key == "early" else f"{rep:.1f}%",
             f"{off:.2f}%" if key == "early" else f"{off:.1f}%"]
    if bold:
        cells = [f"**{c}**" for c in cells]
    table.append((shown, *cells))
common.record("table", common.md_table(
    ["window", "puzzles", "repeat rate", "off-list \\| not a repeat"], table))
common.record("answers", N)
common.record("last_no", common.puzzle_no(sol[-1], N - 1))
common.record("last_date", sol[-1]["print_date"])
common.record("regime_no", regime_no)

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
common.record("curated_used", len(used))
common.record("curated_unused", remaining)
common.record("fresh_per_year", rate)
common.record("years_left", remaining / rate)
common.record("distinct_used", len(seen))

common.header("arm sizes and the regime mixture")
reps = common.repeat_events(words)
offs = common.offlist_events(words, curated)
rep_rate = sum(1 for i, _, _ in reps if i >= start) / (N - start)
off_rate = sum(1 for i, _ in offs if i >= start) / (N - start)
print(f"repeat events           {len(reps)}")
print(f"off-list events         {len(offs)}")
common.record("repeat_events", len(reps))
common.record("offlist_events", len(offs))
common.record("mixture", f"{100*rep_rate:.1f}% / {100*off_rate:.1f}% / "
                         f"{100*(1-rep_rate-off_rate):.1f}%")
common.record("mixture.repeat", 100 * rep_rate)
common.record("mixture.offlist", 100 * off_rate)
common.record("mixture.fresh", 100 * (1 - rep_rate - off_rate))
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
implied = []
for label, config in [
        ("none", PriorConfig(repeat=False, curation=False)),
        ("curation only", PriorConfig(repeat=False)),
        ("repeat only", PriorConfig(curation=False)),
        ("both", PriorConfig())]:
    w = build_weights(data, config)
    unused = w[~is_used].sum()
    print(f"  {label:14s} {100*w[is_used].sum():8.1f}% "
          f"{100*w[~is_used & ~is_curated].sum()/unused:20.1f}%")
    key = label.replace(" ", "_")
    common.record(f"implied.{key}.repeat", 100 * w[is_used].sum())
    common.record(f"implied.{key}.offlist",
                  100 * w[~is_used & ~is_curated].sum() / unused)
    implied.append((label, f"{100*w[is_used].sum():.1f}%",
                    f"{100*w[~is_used & ~is_curated].sum()/unused:.1f}%"))

# the observed pair: the regime row of the table above, restated
nonrep = [j for j in range(start, N) if not is_repeat[j]]
obs_off = sum(off_list[j] for j in nonrep) / len(nonrep)
print(f"\n  {'observed':14s} {100*rep_rate:8.1f}% {100*obs_off:20.1f}%   over the "
      f"regime's {N - start} puzzles")
implied.append(("*observed*", f"*{100*rep_rate:.1f}%*", f"*{100*obs_off:.1f}%*"))
common.record("implied_table", common.md_table(
    ["prior", "implied repeat rate", "implied off-list rate, given not a repeat"],
    implied))
common.record("used_share", 100 * is_used.sum() / len(is_used))
common.record("used_words", int(is_used.sum()))

# the per-word weights the calibration paragraphs quote, under the shipped
# prior and under curation alone
listed = np.array([w in wordlebot for w in data.all_words])
for label, config in (("", PriorConfig()), ("curation_only.", PriorConfig(repeat=False))):
    w = build_weights(data, config)
    common.record(f"weight.{label}used", float(w[is_used].mean()))
    common.record(f"weight.{label}fresh", float(w[is_curated & ~is_used].mean()))
    common.record(f"weight.{label}offlist_listed",
                  float(w[~is_curated & listed].mean()))
print("\n  Each prior is wildly wrong about the rate the *other* one governs,")
print("  which is why neither is worth much alone.")
