"""Prequential scoring of the curation shapes over the off-list answers.

Each shape is scored at the constant that actually ships, not the one it was
fitted at.  Backs SOLVER_EXPLAINED *The off-list group* and the
opening of README *The curation prior's shape*.
"""

import os

import numpy as np

import common
from priors import FREQ_TEMPERATURE, OFFLIST_HEDGE, WORDLEBOT_SNAPSHOT_PATH

sol = common.solutions()
words = [r["solution"] for r in sol]
curated = set(common.load("wordles.json"))
noncurated = common.load("nonwordles.json")
answer_like = common.load("answer_like.json")
wordlebot = common.load("wordlebot_weights.json")
snapshot = common.load(os.path.basename(WORDLEBOT_SNAPSHOT_PATH))

events = common.offlist_events(words, curated)
post = [(i, w) for i, w in events if sol[i]["print_date"] > common.SNAPSHOT_DATE]
# the words the frozen list never had, which are the only ones the hedge covers
missed = [w for _, w in post if w not in snapshot]


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

common.header(f"does the {common.SNAPSHOT_DATE} list predict what came after it?")
print("  A list that absorbed answers after they were used would look just as")
print("  good scored against them. Only an old copy predicting later answers")
print("  separates the two, which is what the frozen scrape is for.\n")
added = sorted(set(wordlebot) - set(snapshot))
ever = [w for w in added if w in set(words)]
print(f"  off-list answers used after it   {len(post):5d}")
print(f"    already in the snapshot        {len(post) - len(missed):5d}"
      f"   ({(len(post) - len(missed)) / len(post):.0%})")
print(f"    the misses                           {', '.join(missed) or 'none'}")
print(f"  words added to the list since    {len(added):5d}")
print(f"    of those, ever an answer       {len(ever):5d}"
      f"   {', '.join(ever) or ''}")
print()
print("  It predicts forward at the rate above. Whether the two it missed were")
print("  backfilled the archive cannot settle: both are absent on 2025-05-25 and")
print("  present on 2025-09-16, with no capture in between, and that gap is the")
print("  window in which both were used.")

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
for word in missed:
    m = multiple(word)
    pct = (score < answer_like.get(word, -50.0)).mean() * 100
    print(f"    {word}  {m:5.2f}x flat  ({np.log(m):+.2f} nats)   "
          f"answer-likeness percentile {pct:.0f}")
pair = np.log([multiple(w) for w in missed]).mean()
print(f"\n  Over the only {len(missed)} events that ever needed it: "
      f"{pair:+.2f} nats/event.")
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
