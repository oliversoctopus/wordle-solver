"""Prequential scoring of the curation shapes over the off-list answers.

Each shape is scored at the constant that actually ships, not the one it was
fitted at.  Backs SOLVER_EXPLAINED 4.3 and the opening of README *Shaping the
two priors*.
"""

import numpy as np

import common
from priors import FREQ_TEMPERATURE, OFFLIST_HEDGE

sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
noncurated = common.load("nonwordles.json")
answer_like = common.load("answer_like.json")
wordlebot = common.load("wordlebot_weights.json")

events = common.offlist_events(words, curated)
post = [(i, w) for i, w in events if sol[i]["print_date"] > common.SNAPSHOT_DATE]


def w_uniform(pool):
    return np.ones(len(pool))


def w_answer_like(pool):
    score = np.array([answer_like.get(w, -50.0) for w in pool])
    return np.exp(FREQ_TEMPERATURE * (score - score.max()))


def w_wordlebot(pool):
    listed = np.array([wordlebot.get(w, 0.0) for w in pool])
    inside = listed > 0.0
    out = np.zeros(len(pool))
    if inside.any():
        out[inside] = (1 - OFFLIST_HEDGE) * listed[inside] / listed[inside].sum()
    if (~inside).any():
        out[~inside] = OFFLIST_HEDGE / int((~inside).sum())
    return out


def score(weight_fn, wanted):
    """Walk every event so the pool shrinks correctly, score only `wanted`."""
    used, out = set(), []
    for t, word in events:
        pool = [w for w in noncurated if w not in used]
        if (t, word) in wanted:
            w = weight_fn(pool)
            w = w / w.sum()
            out.append(np.log(w[pool.index(word)]))
        used.add(word)
    return np.array(out)


SHAPES = (("uniform", w_uniform),
          (f"answer-like frequency, t = {FREQ_TEMPERATURE}", w_answer_like),
          (f"WordleBot, {OFFLIST_HEDGE:.0%} hedge", w_wordlebot))

common.header("curation shapes, nats per off-list event")
print(f"{len(events)} off-list events; the {common.SNAPSHOT_DATE} WordleBot "
      f"snapshot predates {len(post)} of them")
print()
print(f"  {'shape':36s} {'all events':>12s} {'post-snapshot':>15s} "
      f"{'effective support':>19s}")
print("  " + "-" * 86)
for name, fn in SHAPES:
    everything = score(fn, set(events)).mean()
    after = score(fn, set(post)).mean()
    support = common.effective_support(fn(list(noncurated)))
    print(f"  {name:36s} {everything:12.2f} {after:15.2f} {support:19,.0f}")

print()
print("  Restricting to post-snapshot events barely moves WordleBot, so it is")
print("  not winning by having quietly absorbed the events it is scored on.")

common.header("why the hedge is spread flat")
unlisted = [w for w in noncurated if wordlebot.get(w, 0.0) <= 0.0]
absent = [w for _, w in events if wordlebot.get(w, 0.0) <= 0.0]
score = np.array([answer_like.get(w, -50.0) for w in unlisted])
top = score.max()


def multiple(word):
    """Share of the hedge under an answer-like shape, against a flat one."""
    weights = np.append(np.exp(FREQ_TEMPERATURE * (score - top)),
                        np.exp(FREQ_TEMPERATURE * (answer_like.get(word, -50.0) - top)))
    return weights[-1] / weights.sum() * len(weights)


print(f"  The hedge covers the {len(unlisted):,} words the scrape does not have.")
print(f"  {len(absent)} of the {len(events)} off-list answers are among them, so no "
      f"event that has\n  happened can distinguish one hedge shape from another.")
print()
print("  Shaping it by answer-likeness looks attractive on the wrong sample. Over")
mult = np.array([multiple(w) for _, w in events])
print(f"  all {len(mult)} off-list answers, scored as if the hedge had had to cover "
      f"them, it is\n  worth {np.log(mult).mean():+.2f} nats/event (median "
      f"{np.median(mult):.2f}x flat, {int((mult > 1).sum())} of {len(mult)} better).")
print("  But those are words WordleBot *does* list. The hedge only ever fires on a")
print("  word the list is missing, and a missing word is missing because it is odd:")
print()
for word in ("loris", "kefir"):   # the two the 2023 snapshot missed
    m = multiple(word)
    pct = (score < answer_like.get(word, -50.0)).mean() * 100
    print(f"    {word}  {m:5.2f}x flat  ({np.log(m):+.2f} nats)   "
          f"answer-likeness percentile {pct:.0f}")
pair = np.log([multiple(w) for w in ("loris", "kefir")]).mean()
print(f"\n  Over the only two events that ever needed it: {pair:+.2f} nats/event.")
print("  Flat is what 'no idea which word' should look like, so flat it stays.")

common.header("sensitivity to the two shipped constants")
for t in (0.20, FREQ_TEMPERATURE, 0.50, 1.00):
    s = np.array([answer_like.get(w, -50.0) for w in noncurated])
    support = common.effective_support(np.exp(t * (s - s.max())))
    mark = "  <- shipped" if t == FREQ_TEMPERATURE else ""
    print(f"  answer-like t = {t:.2f}   effective support {support:8,.0f}{mark}")
print()
listed = np.array([wordlebot.get(w, 0.0) for w in noncurated])
inside = listed > 0.0
rest = w_answer_like([w for w, i in zip(noncurated, inside) if not i])
for hedge in (0.05, 0.10, 0.20):
    out = np.zeros(len(noncurated))
    out[inside] = (1 - hedge) * listed[inside] / listed[inside].sum()
    out[~inside] = hedge * rest / rest.sum()
    mark = "  <- shipped" if hedge == OFFLIST_HEDGE else ""
    print(f"  WordleBot hedge {hedge:.0%}   effective support "
          f"{common.effective_support(out):8,.0f}{mark}")
