"""Prequential comparison of shapes for the fresh-curated group.

`priors.build_weights` gives the used group the `binned` shape and the off-list
group the `wordlebot` one, and passes ``None`` for the group between them: the
curated words not yet used, which carry 81% of the prior's mass spread flat
over ~490 words.  It is the largest group and the only one whose spread has
never been measured, so this measures it.

Every model predicts pick *i* from picks 1..i-1 only, and is normalized over
the words still unused on that day, so no parameter sees the event it is
scored on.  The last block does the opposite deliberately: a conditional logit
fitted on every event at once, whose score is therefore a ceiling rather than
an estimate -- if the best a model can do with hindsight is small, nothing
honest does better.

The off-list group is scored the same way in the same run, because the finding
is the contrast: the group holding 9% of the mass has structure worth
recovering and the group holding 81% has almost none.  Which off-list shape
wins is `analysis/curation_shapes.py`'s question, not this one.

Needs `wordfreq`, which `answer_like.py` also needs and nothing else does.

Backs README *The group with no shape*.

Pass --events 300 for a fixed point: it scores a frozen prefix, so what it
prints should not move as the answer record grows.
"""

import argparse

import numpy as np
from wordfreq import zipf_frequency

import common
from priors import DEFAULT_NONCURATED_PROB, DEFAULT_REPEAT_PROB, OFFLIST_HEDGE

# what the shipped defaults leave the fresh-curated group, which is the whole
# reason its spread is worth measuring
FRESH_MASS = (1.0 - DEFAULT_REPEAT_PROB) * (1.0 - DEFAULT_NONCURATED_PROB)

BIN_WIDTHS = (100, 200, 400)
BIN_ALPHA = 0.5
VOWELS = set("aeiou")
RARE = set("jqxzvwk")

# gradient ascent for the ceiling fit; no scipy in the environment, and the
# conditional-logit gradient is just observed minus expected features
FIT_ITERS = 1200
FIT_STEP = 0.5
FIT_MOMENTUM = 0.9


def editor_start(sol):
    """The first puzzle the NYT credits an editor on.

    Before it the answers are the curated list walked in its stored order, so
    the pick is not a choice and there is nothing to model.  The record says
    when that changed and the play order agrees: the same puzzle is where the
    source index stops tracking the count of curated answers used.
    """
    return next(i for i, row in enumerate(sol) if row.get("editor"))


def offlist_rate(words, curated, start):
    """Off-list answers and non-repeats since `start`, the order the prior
    draws in: off-list is conditional on not being a repeat."""
    seen, off, fresh = set(words[:start]), 0, 0
    for word in words[start:]:
        if word not in seen:
            fresh += 1
            off += word not in curated
        seen.add(word)
    return off, fresh


def pools_and_picks(words, pool_words, events):
    """The words still unused at each event, and which one was picked.

    Walks the whole record rather than the events, so a pool shrinks on every
    answer that leaves it and not only on the ones being scored.
    """
    index = {w: i for i, w in enumerate(pool_words)}
    used = np.zeros(len(pool_words), dtype=bool)
    pools, picks, ptr = [], [], 0
    for t, word in events:
        while ptr < t:
            if words[ptr] in index:
                used[index[words[ptr]]] = True
            ptr += 1
        pools.append(np.flatnonzero(~used))
        picks.append(index[word])
    return pools, picks


def score(weight, pools, picks):
    """Nats per event, each pick normalized over the pool it came from."""
    out = np.empty(len(picks))
    for k, (pool, pick) in enumerate(zip(pools, picks)):
        w = weight[pool]
        out[k] = np.log(w[np.searchsorted(pool, pick)]) - np.log(w.sum())
    return out


def uniform(pool_words, _):
    return np.ones(len(pool_words))


def wordlebot(hedge):
    """WordleBot's own weight, with `hedge` of the mass spread over the words
    the scrape does not list.

    The shipped hedge is sized for the off-list group, where 9,746 of 10,638
    words are unlisted.  It does not transfer: only 7 curated words are
    unlisted, so the same 5% hands each of them several times what a listed
    word gets.  Hence the two rows.
    """
    def fn(pool_words, tables):
        table = tables["wordlebot"]
        listed = np.array([table.get(w, 0.0) for w in pool_words])
        inside = listed > 0.0
        out = np.zeros(len(pool_words))
        if inside.any():
            out[inside] = (1 - hedge) * listed[inside] / listed[inside].sum()
        if (~inside).any():
            out[~inside] = (hedge / int((~inside).sum()) if hedge
                            else listed[inside].min() / listed[inside].sum())
        return out
    return fn


def source_binned(width):
    """`priors._repeat_shape`'s idea, applied to where the word sits in the
    curated list rather than to where it was first used."""
    def fn(pool_words, tables):
        pos = np.array([tables["position"].get(w, 0) for w in pool_words])
        bins = pos // width
        size = np.bincount(bins, minlength=bins.max() + 1).astype(float)
        return ((BIN_ALPHA + size) / np.maximum(size, 1.0))[bins]
    return fn


def features(pool_words, tables):
    """The columns the ceiling fit is allowed to use, standardized."""
    cols, names = [], []
    for name, fn in (
            ("word frequency", lambda w: zipf_frequency(w, "en")),
            ("log WordleBot",
             lambda w: np.log(max(tables["wordlebot"].get(w, 1e-4), 1e-4))),
            ("list position", lambda w: tables["position"].get(w, 0)),
            ("vowels", lambda w: sum(c in VOWELS for c in w)),
            ("rare letters", lambda w: sum(c in RARE for c in w)),
            ("distinct letters", lambda w: len(set(w)))):
        cols.append([fn(w) for w in pool_words])
        names.append(name)
    x = np.array(cols, dtype=float).T
    return (x - x.mean(0)) / x.std(0), names


def design_matrix(x, pools, picks):
    """Every pool padded to one rectangle, so all the fits share one gather.

    The pad columns are pushed to zero probability rather than masked out of
    the arithmetic, which keeps the whole fit as three dense products.
    """
    wide = max(len(p) for p in pools)
    take = np.zeros((len(pools), wide), dtype=np.int64)
    live = np.zeros((len(pools), wide), dtype=bool)
    at = np.zeros(len(pools), dtype=np.int64)
    for k, (pool, pick) in enumerate(zip(pools, picks)):
        take[k, :len(pool)] = pool
        live[k, :len(pool)] = True
        at[k] = np.searchsorted(pool, pick)
    return x[take], np.where(live, 0.0, -np.inf), at


def fit_logit(full, dead, at, cols):
    """Conditional logit fitted on every event at once, by gradient ascent
    with momentum; the gradient is observed minus expected features."""
    design = np.ascontiguousarray(full[:, :, cols])
    rows = np.arange(design.shape[0])
    beta, step = np.zeros(len(cols)), np.zeros(len(cols))
    for _ in range(FIT_ITERS):
        u = design @ beta + dead
        p = np.exp(u - u.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        grad = (design[rows, at] - np.einsum("ek,ekj->ej", p, design)).mean(0)
        step = FIT_MOMENTUM * step + FIT_STEP * grad
        beta = beta + step
    u = design @ beta + dead
    ll = float((u[rows, at] - np.logaddexp.reduce(u, axis=1)).mean())
    return beta, ll, float(np.abs(grad).max())


def group(name, pool_words, events, words, tables, shapes, limit):
    """Score one group's shapes, and return its uniform baseline and events."""
    if limit:
        events = events[:limit]
    pools, picks = pools_and_picks(words, pool_words, events)
    base = score(uniform(pool_words, tables), pools, picks)
    print(f"\n  {name}: {len(events):,} events, "
          f"pool {len(pools[0]):,} -> {len(pools[-1]):,}")
    # a frozen prefix is the method check, so it records under its own prefix
    tag = (f"first{limit}." if limit else "") + name.replace(" ", "_") + "."
    rec = lambda key, value: common.record(tag + key, value)  # noqa: E731
    rec("events", len(events))
    rec("pool_from", len(pools[0]))
    rec("pool_to", len(pools[-1]))
    table, positions = [], []
    print(f"    {'shape':34s} {'nats/event':>11s} {'vs uniform':>12s} "
          f"{'support':>10s}")
    print("    " + "-" * 70)
    for label, fn in shapes:
        weight = fn(pool_words, tables)
        s = score(weight, pools, picks)
        mean, se, d = common.paired(s, base)
        first = label.startswith("uniform")
        vs = "" if first else f"{mean:+.4f}"
        # support over the pool as it stands now, not over every word the
        # group ever held: what the shape would spread today's mass across
        print(f"    {label:34s} {s.mean():11.4f} {vs:>12s} "
              f"{common.effective_support(weight[pools[-1]]):10,.0f}")
        key = ("uniform" if first else "hedge" if "hedge" in label
               else "wordlebot" if label.startswith("WordleBot")
               else "position_" + label.split("= ")[-1])
        support = common.effective_support(weight[pools[-1]])
        rec(f"{key}.nats", float(s.mean()))
        rec(f"{key}.support", float(support))
        if not first:
            wins = int((d > 0).sum())
            print(f"    {'':34s} {'':11s} {'+- ' + format(se, '.4f'):>12s}"
                  f"   {mean / se:.1f} sigma, wins {wins}/{len(d)}")
            rec(f"{key}.gain", float(mean))
            rec(f"{key}.se", float(se))
            rec(f"{key}.sigma", float(mean / se))
            rec(f"{key}.wins", wins)
        cells = (common.neg(f"{s.mean():.4f}"),
                 "\u2014" if first else common.neg(f"{mean:+.4f}"),
                 f"{support:,.0f}")
        # the one list-position row README carries is the best of the widths
        if label.startswith("list position"):
            positions.append((float(s.mean()), cells))
            continue
        if label.startswith("WordleBot weight"):
            cells = (cells[0], f"**{cells[1]} ± {se:.4f}**", cells[2])
        table.append((label.replace("(as shipped)", "(as shipped off-list)"),
                      *cells))
    if positions:
        table.append(("list position, binned " + "/".join(map(str, BIN_WIDTHS)),
                      *max(positions)[1]))
    rec("table", common.md_table(
        ["shape", "nats/event", "vs. uniform", "effective support"], table))
    return pools, picks, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=0,
                        help="score only the first N fresh-curated picks "
                             "(300 is a fixed point the answer record cannot "
                             "move)")
    args = parser.parse_args()

    sol = common.solutions()
    words = [row["solution"] for row in sol]
    curated = common.load("wordles.json")
    tables = {"wordlebot": common.load("wordlebot_weights.json"),
              "position": {w: i for i, w in enumerate(curated)}}
    start = editor_start(sol)

    common.header("the group with no shape")
    print(f"  Fresh-curated picks are counted from puzzle "
          f"{common.puzzle_no(sol[start], start)} "
          f"({sol[start]['print_date']}), "
          f"the first the record credits\n  to an editor.  Before it the "
          f"answers walk the curated list in its stored order.")
    print(f"\n  `priors.build_weights` shapes two of its three groups and "
          f"passes None for this one,\n  which carries {FRESH_MASS:.0%} "
          f"of the mass at the shipped probabilities.")

    fresh = common.fresh_events(words, set(curated), start)
    offlist = common.offlist_events(words, set(curated))
    like = common.load("answer_like.json")
    print(f"\n  answer-like frequency cannot reach this group at all: "
          f"answer_like.json covers\n  {len(set(like) & set(curated))} of the "
          f"{len(curated):,} curated words, so --curation-shape frequency "
          f"shapes\n  only the off-list group.")

    shapes = [("uniform over the pool (current)", uniform),
              ("WordleBot weight", wordlebot(0.0)),
              (f"WordleBot, {OFFLIST_HEDGE:.0%} hedge (as shipped)",
               wordlebot(OFFLIST_HEDGE))]
    shapes += [(f"list position, binned W = {w}", source_binned(w))
               for w in BIN_WIDTHS]
    pools, picks, _ = group("fresh curated", curated, fresh, words, tables,
                            shapes, args.events)
    group("off-list", common.load("nonwordles.json"), offlist, words, tables,
          shapes[:3], 0)
    print("\n  The shape worth 2-3 nats on the group holding 9% of the mass "
          "is worth\n  hundredths on the group holding 81%.")

    common.header("the ceiling: how much structure is there at all")
    print("  A conditional logit over the same pools, fitted on every event "
          "at once.  It is\n  scored on what it was fitted on, so it bounds "
          "any model linear in these features.\n")
    x, names = features(curated, tables)
    base = score(np.ones(len(curated)), pools, picks).mean()
    full, dead, at = design_matrix(x, pools, picks)
    singles = {}
    for k, name in enumerate(names):
        beta, ll, _ = fit_logit(full, dead, at, [k])
        print(f"    {name:22s} alone   {ll:9.4f}   gain {ll - base:+.4f}   "
              f"b = {beta[0]:+.3f}")
        singles[name] = ll - base
    beta, ll, resid = fit_logit(full, dead, at, list(range(len(names))))
    print(f"\n    {f'all {len(names)} together':22s}         {ll:9.4f}   "
          f"CEILING {ll - base:+.4f} nats/event")
    tag = f"first{args.events}." if args.events else ""
    common.record(tag + "ceiling", float(ll - base))
    best_single = max(singles, key=singles.get)
    common.record(tag + "best_feature", best_single)
    common.record(tag + "best_feature_gain", float(singles[best_single]))
    common.record(tag + "features", len(names))
    print(f"    {'':22s}         (max |gradient| {resid:.2e})")
    print("    " + ", ".join(f"{n} {b:+.3f}" for n, b in zip(names, beta)))

    common.header("why the group is not pooled with the off-list one")
    print("  WordleBot weights curated and off-list words alike, so one pool "
          "weighted by it\n  is the obvious simplification.  It is not taken, "
          "because the weight ranks words\n  and does not set a rate:\n")
    unused = [w for w in curated if w not in set(words)]
    offlist_pool = [w for w in common.load("nonwordles.json")
                    if w not in set(words)]
    fw = np.array([tables["wordlebot"].get(w, 0.0) for w in unused])
    ow = np.array([tables["wordlebot"].get(w, 0.0) for w in offlist_pool])
    listed = fw.sum() + ow.sum()
    unlisted = int((fw == 0).sum() + (ow == 0).sum())
    share = ((1 - OFFLIST_HEDGE) * ow.sum() / listed
             + OFFLIST_HEDGE * int((ow == 0).sum()) / unlisted)
    total = share + ((1 - OFFLIST_HEDGE) * fw.sum() / listed
                     + OFFLIST_HEDGE * int((fw == 0).sum()) / unlisted)
    print(f"    fresh curated {len(unused):6,d} words, "
          f"{int((fw > 0).sum()):5,d} listed, weight {fw.sum():9,.1f}")
    print(f"    off-list      {len(offlist_pool):6,d} words, "
          f"{int((ow > 0).sum()):5,d} listed, weight {ow.sum():9,.1f}")
    off, fresh_n = offlist_rate(words, set(curated), common.regime_start(sol))
    print(f"\n    pooled and weighted by WordleBot -> off-list rate "
          f"{share / total:6.1%}")
    common.record("pool.fresh_words", len(unused))
    common.record("pool.fresh_listed", int((fw > 0).sum()))
    common.record("pool.fresh_unlisted", int((fw == 0).sum()))
    common.record("pool.offlist_listed", int((ow > 0).sum()))
    common.record("pool.offlist_unlisted", int((ow == 0).sum()))
    common.record("pool.offlist_words", len(offlist_pool))
    common.record("pool.pooled_rate", 100 * share / total)
    print(f"    the shipped split                -> off-list rate "
          f"{DEFAULT_NONCURATED_PROB:6.1%}")
    print(f"    the regime actually ran at          {off / fresh_n:6.1%}"
          f"   ({off} of {fresh_n} non-repeats)")
    common.record("pool.regime_rate", 100 * off / fresh_n)
    print("\n  Pooling misprices the rate by six times, which is the same "
          "failure the `curation\n  only` arm makes on repeats.  Keeping the "
          "rate and shaping inside the group is\n  what the table above "
          "measures, and it is worth hundredths of a nat.")


if __name__ == "__main__":
    main()
