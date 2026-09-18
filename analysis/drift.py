"""Is the block the editor recycles from moving forward through the archive?

Two stories fit: a *static window* over the early answers, or a *fixed replay
lag* -- reuse a constant time after first use.  The drift rate separates them,
in source-index per puzzle:

    static window     0.0     the block never moves
    fixed replay lag  1.0     one puzzle a day, so a constant lag drags the
                              source forward one position a day

Backs the closing paragraph of README *The repeat prior's shape*.
"""

import numpy as np

import common

sol = common.solutions()
words = [r["solution"] for r in sol]
events = common.repeat_events(words)

common.header("drift in the source window")
print(f"  {'events':>8s} {'rank corr':>11s} {'p':>7s} "
      f"{'drift / puzzle':>16s} {'drift / event':>16s}")
print("  " + "-" * 64)
for n in (18, len(events)):
    ev = events[:n]
    src = np.array([f for _, _, f in ev], dtype=float)
    when = np.array([t for t, _, _ in ev], dtype=float)
    ordinal = np.arange(len(ev), dtype=float)
    rho, p = common.rank_correlation(ordinal, src)
    per_puzzle = common.slope_se(when, src)
    per_event = common.slope_se(ordinal, src)
    print(f"  {n:8d} {rho:+11.2f} {p:7.3f} "
          f"{per_puzzle[0]:+9.2f} +- {per_puzzle[1]:4.2f} "
          f"{per_event[0]:+9.2f} +- {per_event[1]:5.2f}")
    tag = "first18." if n == 18 else ""
    common.record(tag + "events", n)
    common.record(tag + "rho", float(rho))
    common.record(tag + "p", float(p))
    common.record(tag + "drift", common.neg(f"{per_puzzle[0]:+.2f} \u00b1 {per_puzzle[1]:.2f}"))

print()
print("  0.0 = static window, 1.0 = fixed replay lag; an interval covering both")
print("  settles nothing, which is the state this question has stayed in.")

common.header("how long a word waits before it comes back")
gaps = np.array([(t - f) / 365.0 for t, _, f in events])
print(f"replay lag   mean {gaps.mean():.2f} y, sd {gaps.std(ddof=1):.2f}, "
      f"range {gaps.min():.2f}-{gaps.max():.2f}")
band = ((gaps >= 4.4) & (gaps <= 5.1)).sum()
print(f"{band} of {len(gaps)} events sit between 4.4 and 5.1 years")
common.record("lag_mean", float(gaps.mean()))
common.record("lag_sd", float(gaps.std(ddof=1)))
common.record("in_band", int(band))
latest = events[-2:]
common.record("latest_two", " and ".join(f"`{w}`" for _, w, _ in latest))
common.record("latest_two_sources", " and ".join(
    f"#{common.puzzle_no(sol[f], f)}" for _, _, f in latest))
# the shape's own summary: how many came from the first three hundred
early = sum(1 for _, _, f in events if common.puzzle_no(sol[f], f) <= 300)
common.record("from_first_300", early)
common.record("median_source", float(np.median(
    [common.puzzle_no(sol[f], f) for _, _, f in events])))
common.record("outliers", ", ".join(
    f"`{w}` (#{common.puzzle_no(sol[f], f)})" for _, w, f in events
    if common.puzzle_no(sol[f], f) > 300))
common.record("outlier_rate", 100 * (len(events) - early) / len(events))
print()
print("source puzzle numbers, in the order the repeats happened:")
for t, w, f in events:
    print(f"  {w:8s} first #{common.puzzle_no(sol[f], f):5d}  "
          f"reused {sol[t]['print_date']}  after {(t-f)/365.0:.2f} y")
