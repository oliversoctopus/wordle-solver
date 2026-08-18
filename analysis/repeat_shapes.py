"""Prequential comparison of repeat-prior shapes.

Every model predicts repeat *i* from repeats 1..i-1 only, so no parameter ever
sees the event it is scored on.  Backs README *The repeat prior's shape* and
SOLVER_EXPLAINED 4.2.

Pass --events 18 to reproduce the numbers the README carried before the
2026-08-10 update; that is the check that the method has not drifted.
"""

import argparse

import numpy as np

import common

BIN_WIDTH = 200
# uniform-mixture weight for the window models: with a hard cutoff instead, a
# single repeat landing outside the window costs ~7 nats and swamps everything
WINDOW_OUTSIDE = 0.01
AGE_GRID = np.linspace(-2.0, 6.0, 81)


def uniform(first, prior, t, target):
    return 1.0 / len(first)


def binned(alpha):
    """Laplace-smoothed histogram of repeats over the first-use index."""
    def fn(first, prior, t, target):
        words = list(first)
        starts = np.array([first[w] for w in words]) // BIN_WIDTH
        src = np.array([f // BIN_WIDTH for _, f in prior], dtype=int)
        n = int(max(starts.max(), src.max() if len(src) else 0)) + 1
        size = np.bincount(starts, minlength=n).astype(float)
        hits = np.bincount(src, minlength=n).astype(float) if len(src) else np.zeros(n)
        rel = ((hits + alpha) / np.maximum(size, 1.0))[starts]
        return rel[words.index(target)] / rel.sum()
    return fn


def windowed(last_k, half, outside=WINDOW_OUTSIDE):
    """Uniform over a band around where the last `last_k` repeats came from."""
    def fn(first, prior, t, target):
        words = list(first)
        idx = np.array([first[w] for w in words], dtype=float)
        if not prior:
            return 1.0 / len(words)
        recent = [f for _, f in prior][-last_k:] if last_k else [f for _, f in prior]
        inside = np.abs(idx - np.mean(recent)) <= half
        if not inside.any():
            return 1.0 / len(words)
        w = np.empty(len(words))
        w[inside] = (1 - outside) / inside.sum()
        w[~inside] = outside / max(1, (~inside).sum())
        return w[words.index(target)] / w.sum()
    return fn


def age_fixed(rate):
    """Weight rising with time since last use, at a fixed rate per year."""
    def fn(first, prior, t, target):
        words = list(first)
        ages = np.array([t - first[w] for w in words], dtype=float) / 365.0
        lo = rate * ages
        lo -= lo.max()
        w = np.exp(lo)
        return w[words.index(target)] / w.sum()
    return fn


def age_refit(all_words):
    """The age shape with its rate refit from the repeats seen so far.

    This is the only version that never peeks; any fixed rate that scores well
    was chosen by looking at the events it is then scored on.
    """
    def fn(first, prior, t, target):
        rate, best = 0.0, -np.inf
        for r in AGE_GRID:
            ll = 0.0
            for when, started in prior:
                hist, _ = common.state_at(all_words, when)
                ages = np.array([when - hist[w] for w in hist], float) / 365.0
                lo = r * ages
                m = lo.max()
                ll += (r * (when - started) / 365.0 - m) - np.log(np.exp(lo - m).sum())
            if ll > best:
                best, rate = ll, r
        return age_fixed(rate)(first, prior, t, target)
    return fn


def score(fn, events, all_words):
    return np.array([np.log(fn(*common.state_at(all_words, t), t, w))
                     for t, w, _ in events])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=0,
                        help="score only the first N repeats (18 reproduces the "
                             "pre-2026-08-10 README)")
    args = parser.parse_args()

    words = common.answers()
    events = common.repeat_events(words)
    if args.events:
        events = events[:args.events]
    common.header(f"repeat shapes, prequential over {len(events)} events")

    u = score(uniform, events, words)
    b05 = score(binned(0.5), events, words)
    b10 = score(binned(1.0), events, words)
    wins = [score(windowed(k, h), events, words).sum()
            for k in (3, 5, 8, None) for h in (25, 50, 100, 200)]
    refit = score(age_refit(words), events, words)

    print(f"  {'model':44s} {'logL':>9s} {'vs uniform':>11s}")
    print("  " + "-" * 68)
    print(f"  {'uniform over used (current)':44s} {u.sum():9.1f} {'':>11s}")
    print(f"  {'binned, width 200, a = 0.5':44s} {b05.sum():9.1f} "
          f"{b05.sum()-u.sum():+11.1f}")
    print(f"  {'binned, width 200, a = 1.0':44s} {b10.sum():9.1f} "
          f"{b10.sum()-u.sum():+11.1f}")
    print(f"  {'16 window variants (K in 3/5/8/all, +-25..200)':44s} "
          f"{min(wins):9.1f} {min(wins)-u.sum():+11.1f}")
    print(f"  {'':44s} {max(wins):9.1f} {max(wins)-u.sum():+11.1f}")
    print(f"  {'age, rate refit each event':44s} {refit.sum():9.1f} "
          f"{refit.sum()-u.sum():+11.1f}")
    for rate in (0.574, 1.0, 2.0):
        s = score(age_fixed(rate), events, words)
        print(f"  {f'age, rate = {rate}/year (chosen with hindsight)':44s} "
              f"{s.sum():9.1f} {s.sum()-u.sum():+11.1f}")

    m, se, d = common.paired(b05, u)
    print()
    print(f"binned vs uniform : {m:+.3f} +- {se:.3f} nats/event "
          f"({m/se:.1f} sigma), wins {int((d > 0).sum())}/{len(d)}")
    for label, s in (("age refit", refit), ("age rate=2.0",
                                            score(age_fixed(2.0), events, words))):
        m2, se2, d2 = common.paired(s, b05)
        print(f"{label:17s} vs binned : {m2:+.3f} +- {se2:.3f} nats/event, "
              f"wins {int((d2 > 0).sum())}/{len(d2)}")

    # in-sample: the same histogram fitted on every event, then scored on them
    all_src = [f // BIN_WIDTH for _, _, f in events]

    def insample(first, prior, t, target):
        return binned(0.5)(first, [(0, s * BIN_WIDTH) for s in all_src], t, target)

    ins = score(insample, events, words)
    mi, _, _ = common.paired(ins, u)
    print(f"\nin-sample would claim {mi:+.3f} nats/event, "
          f"inflating the prequential {m:+.3f} by {100*(mi/m-1):.0f}%")


if __name__ == "__main__":
    main()
