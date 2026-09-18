"""Should the feedback table be indexed [answer, guess] instead, or as well?

The table is square over the same word list either way, so the choice is not
about size but about which axis the inner loop walks.  Every shipped kernel
fixes a guess and reads scattered candidates, which `[guess, answer]` makes a
row.  Written for `[answer, guess]` the loops invert -- fix a candidate, walk a
contiguous run of guesses -- and the table reads become perfect: this measures
them as insensitive to whether the candidates are scattered or adjacent, which
the shipped layout is not.

That is not the whole cost, and the rest of it is the answer to the question.
Visiting every guess on every candidate means every guess's buckets have to be
live at once, so the accumulator grows from 243 slots shared by a block of 64
guesses to 243 per guess in flight.  The scatter does not go away; it moves off
the table and onto the accumulator, which is touched exactly as often.

So the comparison turns on how many accumulators the reduce needs.  Entropy
keeps one per bucket and expected guesses keeps four, and both are swept here
because they do not give the same answer.  Both answer-major kernels have had
the same optimization effort as the shipped ones -- the accumulator is
allocated once per thread rather than once per block, which was worth about 2x
at the low end -- so this is not a draft measured against a tuned kernel.

Backs SOLVER_EXPLAINED *Storing the table the other
way round.  It is a timing probe, so it moves with the machine.
"""

import argparse
import time

import numpy as np
from numba import get_num_threads, njit, prange

import common
import estimator
import kernels
from estimator import f_eval_x
from priors import PriorConfig, build_weights
from solver import WordleSolver
from wordle_data import N_PATTERNS, WIN_PATTERN, load_data

# candidate counts swept, spanning the endgame search at the low end and the
# opening at the high one
SIZES = (3, 6, 12, 25, 60, 130, 400, 1200, 2315)

# guesses whose accumulators are held at once.  The answer-major kernels' one
# tuning knob, and it pulls both ways: a block of G reads G contiguous bytes per
# candidate, so below 64 it leaves part of a cache line unused, while the
# accumulator it must keep live is G x 243 x 8 bytes per quantity accumulated.
BLOCKS = (4, 8, 16, 32, 64)

# the fused kernel's accumulator does not grow with the block, so it can take
# much wider ones; what bounds it is the G x m tile staying inside the cache
FUSED_BLOCKS = (16, 32, 64, 128, 256)


@njit(parallel=True, cache=True)
def ig_answer_major(tt, candidates, weights, pool, block, rescan_max, threads):
    """`kernels.ig_all` written for a table indexed [answer, guess].

    The parallel loop runs over threads rather than over blocks so the buckets
    can be allocated once for the whole scan: both reduces leave every bucket
    they touch at zero, so the array comes back clean.  Allocating per block
    instead costs 243 slots per guess, which is the cost `_RESCAN_MAX` exists to
    avoid, and it dominates every small scan.
    """
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        mass = np.zeros((block, N_PATTERNS))

        for b in range(t, n_blocks, threads):
            start = b * block
            stop = min(start + block, n_out)
            here = stop - start

            for i in range(m):
                c = candidates[i]
                w = weights[i]
                for j in range(here):
                    mass[j, tt[c, pool[start + j]]] += w

            if m < rescan_max:
                for j in range(here):
                    out[start + j] = 0.0
                for i in range(m):
                    c = candidates[i]
                    for j in range(here):
                        r = tt[c, pool[start + j]]
                        v = mass[j, r]
                        if v > 0.0:
                            p = v / total
                            out[start + j] -= p * np.log2(p)
                            mass[j, r] = 0.0
            else:
                for j in range(here):
                    h = 0.0
                    for r in range(N_PATTERNS):
                        v = mass[j, r]
                        if v > 0.0:
                            p = v / total
                            h -= p * np.log2(p)
                            mass[j, r] = 0.0
                    out[start + j] = h
    return out


@njit(cache=True, inline="always")
def _bucket(mass, count, wlw, heaviest, params, codes, knots, base):
    if count == 1.0:
        return 1.0
    if count == 2.0:
        return 2.0 - heaviest / mass
    return f_eval_x(np.log2(mass) - wlw / mass, params, codes, knots, base)


@njit(parallel=True, cache=True)
def eg_answer_major(tt, candidates, weights, wlw, params, codes, knots, base,
                    pool, block, rescan_max, threads):
    """`kernels.expected_guesses_all` for a table indexed [answer, guess].

    The four quantities are packed into one array so that a bucket is 32
    contiguous bytes rather than four cache lines in four arrays.  That is worth
    a third at the widest positions here and nothing at all to the shipped
    kernel, whose accumulator never leaves L1 either way.
    """
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        # [mass, count, w_log_w, heaviest] per pattern, per guess in flight
        acc = np.zeros((block, N_PATTERNS, 4))

        for b in range(t, n_blocks, threads):
            start = b * block
            stop = min(start + block, n_out)
            here = stop - start

            for i in range(m):
                c = candidates[i]
                w = weights[i]
                lw = wlw[i]
                for j in range(here):
                    r = tt[c, pool[start + j]]
                    acc[j, r, 0] += w
                    acc[j, r, 1] += 1.0
                    acc[j, r, 2] += lw
                    if w > acc[j, r, 3]:
                        acc[j, r, 3] = w

            if m < rescan_max:
                for j in range(here):
                    out[start + j] = 1.0
                for i in range(m):
                    c = candidates[i]
                    for j in range(here):
                        r = tt[c, pool[start + j]]
                        if acc[j, r, 1] > 0.0:
                            if r != WIN_PATTERN:
                                out[start + j] += acc[j, r, 0] / total * _bucket(
                                    acc[j, r, 0], acc[j, r, 1], acc[j, r, 2],
                                    acc[j, r, 3], params, codes, knots, base)
                            acc[j, r, 0] = 0.0
                            acc[j, r, 1] = 0.0
                            acc[j, r, 2] = 0.0
                            acc[j, r, 3] = 0.0
            else:
                for j in range(here):
                    expected = 1.0
                    for r in range(N_PATTERNS):
                        if acc[j, r, 1] > 0.0:
                            if r != WIN_PATTERN:
                                expected += acc[j, r, 0] / total * _bucket(
                                    acc[j, r, 0], acc[j, r, 1], acc[j, r, 2],
                                    acc[j, r, 3], params, codes, knots, base)
                            acc[j, r, 0] = 0.0
                            acc[j, r, 1] = 0.0
                            acc[j, r, 2] = 0.0
                            acc[j, r, 3] = 0.0
                    out[start + j] = expected
    return out


@njit(parallel=True, cache=True)
def eg_fused(tt, candidates, weights, wlw, params, codes, knots, base,
             pool, block, rescan_max, threads):
    """Answer-major with the transpose fused in, which is the strongest form.

    The two kernels above each give something up: answer-major reads
    contiguously but holds G guesses' accumulators, the shipped one holds a
    single small accumulator but reads scattered.  Neither trade is forced.  Read
    the G x m tile for a block of guesses out of the answer-major table --
    contiguous, G bytes per candidate -- transposing it into an L1-sized buffer
    on the way in.  Each guess's feedback vector is then a contiguous row of that
    tile, so the reduce is the shipped one, over a single 243-bucket accumulator
    reused across the whole block.

    What it pays instead is writing and re-reading the tile, G*m bytes, which is
    free while that stays in L1 and is why it turns over near 2,000 candidates.
    """
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)

    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for t in prange(threads):
        tile = np.empty((block, m), dtype=np.uint8)
        mass = np.zeros(N_PATTERNS)
        count = np.zeros(N_PATTERNS)
        w_log_w = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)

        for b in range(t, n_blocks, threads):
            start = b * block
            stop = min(start + block, n_out)
            here = stop - start

            for i in range(m):
                c = candidates[i]
                for j in range(here):
                    tile[j, i] = tt[c, pool[start + j]]

            for j in range(here):
                for i in range(m):
                    r = tile[j, i]
                    w = weights[i]
                    mass[r] += w
                    count[r] += 1.0
                    w_log_w[r] += wlw[i]
                    if w > heaviest[r]:
                        heaviest[r] = w

                expected = 1.0
                if m < rescan_max:
                    for i in range(m):
                        r = tile[j, i]
                        if count[r] > 0.0:
                            if r != WIN_PATTERN:
                                expected += mass[r] / total * _bucket(
                                    mass[r], count[r], w_log_w[r], heaviest[r],
                                    params, codes, knots, base)
                            mass[r] = 0.0
                            count[r] = 0.0
                            w_log_w[r] = 0.0
                            heaviest[r] = 0.0
                else:
                    for r in range(N_PATTERNS):
                        if count[r] > 0.0:
                            if r != WIN_PATTERN:
                                expected += mass[r] / total * _bucket(
                                    mass[r], count[r], w_log_w[r], heaviest[r],
                                    params, codes, knots, base)
                            mass[r] = 0.0
                            count[r] = 0.0
                            w_log_w[r] = 0.0
                            heaviest[r] = 0.0
                out[start + j] = expected
    return out


def best_of(fn, reps):
    """Fastest of ``reps`` runs, the warm-up not counted."""
    fn()
    best = np.inf
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def sweep(fit, table, transposed, pool, rng, label, mode):
    """Both layouts over a range of candidate counts, on one pool."""
    slots = 1 if mode == "entropy" else 4
    common.header(f"{label}, {mode} ({pool.size} guesses scored)")
    print(f"  {slots} accumulator{'' if slots == 1 else 's'} per bucket; "
          f"answer-major must hold "
          f"{BLOCKS[0]}-{BLOCKS[-1]} guesses' worth, "
          f"{slots * BLOCKS[0] * N_PATTERNS * 8 / 1024:.0f}-"
          f"{slots * BLOCKS[-1] * N_PATTERNS * 8 / 1024:.0f} KB, against the "
          f"shipped {slots * N_PATTERNS * 8 / 1024:.1f} KB")
    header = (f"\n  {'candidates':>11} {'[guess, answer]':>17} "
              f"{'[answer, guess]':>17} {'block':>6}")
    if mode != "entropy":
        header += f" {'fused':>15} {'block':>6}"
    print(header + f" {'best ratio':>10}")

    sizes, ratios = [], []
    for m in SIZES + (table.shape[0],):
        if m == table.shape[0]:
            candidates = np.arange(m, dtype=np.int64)
        else:
            candidates = np.sort(
                rng.choice(table.shape[0], size=m, replace=False)
            ).astype(np.int64)
        weights = fit.weights[candidates].copy()
        # a small scan finishes in tens of microseconds, where the thread pool
        # alone is most of the variance, so small sizes are timed hardest
        reps = 5 if m > 1000 else 40 if m > 100 else 200

        if mode == "entropy":
            def shipped():
                return kernels.ig_all(table, candidates, weights, pool=pool)

            def other(block):
                return ig_answer_major(transposed, candidates, weights, pool,
                                       block, kernels._RESCAN_MAX,
                                       get_num_threads())
        else:
            wlw = kernels._w_log_w(weights)

            def shipped():
                return kernels.expected_guesses_all(
                    table, candidates, weights, fit.params, fit.codes,
                    fit.knots, fit.base, pool=pool)

            def other(block):
                return eg_answer_major(
                    transposed, candidates, weights, wlw, fit.params, fit.codes,
                    fit.knots, fit.base, pool, block, kernels._RESCAN_MAX,
                    get_num_threads())

        row = shipped()
        guess_major = best_of(shipped, reps)

        def fastest(build, blocks):
            best, chosen = np.inf, None
            for block in blocks:
                got = build(block)
                if not np.allclose(row, got):
                    raise AssertionError(
                        f"layouts disagree at m={m}, block={block}: "
                        f"{np.abs(row - got).max():.3e}")
                taken = best_of(lambda: build(block), reps)
                if taken < best:
                    best, chosen = taken, block
            return best, chosen

        answer_major, chosen = fastest(other, BLOCKS)
        if mode == "entropy":
            fused, fused_block = np.inf, None
        else:
            fused, fused_block = fastest(
                lambda block: eg_fused(
                    transposed, candidates, weights, wlw, fit.params, fit.codes,
                    fit.knots, fit.base, pool, block, kernels._RESCAN_MAX,
                    get_num_threads()),
                [g for g in FUSED_BLOCKS if g * m <= 4 << 20])

        best_other = min(answer_major, fused)
        sizes.append(m)
        ratios.append(guess_major / best_other)
        line = (f"  {m:>11,} {guess_major * 1e3:>15.2f}ms "
                f"{answer_major * 1e3:>15.2f}ms {chosen:>6}")
        if mode != "entropy":
            line += f" {fused * 1e3:>13.2f}ms {fused_block:>6}"
        print(line + f" {ratios[-1]:>7.2f}x")

    print("\n  ratio above 1 favors [answer, guess] at its best, below 1 the "
          "shipped layout.")
    return np.array(sizes, dtype=float), np.array(ratios)


def profile(fit, data, games, seed):
    """Candidate count and elapsed time of every scoring call in real play.

    Every kernel is compiled before the clock starts.  numba caches to
    __pycache__ keyed on the source file, so any edit to `kernels.py` throws the
    cache away and the first call to reach a kernel pays its compile -- which
    lands as a single several-hundred-millisecond call and swamps the band it
    falls in.  `two_turn_scores` is the one that hides, being reached only from
    the endgame search rather than from the sweeps above.
    """
    warm = np.arange(4, dtype=np.int64)
    weights = fit.weights[warm].copy()
    kernels.ig_all(data.feedback_table, warm, weights)
    kernels.two_turn_scores(data.feedback_table, warm, weights, 1.0)
    kernels.expected_guesses_all(data.feedback_table, warm, weights, fit.params,
                                 fit.codes, fit.knots, fit.base)

    calls = []
    harvest = []
    originals = {}

    def instrument(name):
        original = getattr(kernels, name)
        originals[name] = original

        def wrapped(table, candidates, *rest, **kwargs):
            started = time.perf_counter()
            out = original(table, candidates, *rest, **kwargs)
            calls.append((len(candidates), time.perf_counter() - started))
            if kwargs.get("pool") is None and len(candidates) >= 25:
                harvest.append(np.array(candidates, dtype=np.int64))
            return out

        setattr(kernels, name, wrapped)

    for name in ("ig_all", "expected_guesses_all", "two_turn_scores"):
        instrument(name)
    try:
        solver = WordleSolver(data=data)
        answers = solver.all_candidates()
        rng = np.random.default_rng(seed)
        for answer in rng.choice(answers, size=games, replace=False):
            live = solver.all_candidates()
            for turn in range(6):
                guess, _ = solver.best_guess(live, 6 - turn)
                pattern = solver.feedback(guess, int(answer))
                if pattern == WIN_PATTERN:
                    break
                live = solver.filter(guess, pattern, live)
    finally:
        for name, original in originals.items():
            setattr(kernels, name, original)
    return calls, harvest


def on_real_sets(fit, table, transposed, harvest, rng):
    """Both layouts on the candidate sets real play actually produces.

    The sweeps draw candidates at random, which assumes the worst about how they
    fall across a row.  They do not fall that way: the word list is
    alphabetical and a surviving set shares letters, so a real set of a given
    size touches about half the distinct cache lines a random one does, and the
    shipped layout's penalty is correspondingly smaller than a random sweep
    makes it look.  This measures the ratio on the real sets instead of assuming
    the random one carries over.
    """
    common.header(f"the {len(harvest)} candidate sets that play produced")
    print(f"  {'size band':>12} {'sets':>6} {'lines touched':>15} "
          f"{'if random':>11} {'[guess, answer]':>17} {'fused':>13} "
          f"{'ratio':>8}")

    pool = np.arange(table.shape[0], dtype=np.int64)
    for low, high in ((25, 130), (131, 400), (401, 1200), (1201, 3000)):
        picked = [c for c in harvest if low <= len(c) <= high][:12]
        if not picked:
            continue
        real_lines, random_lines = [], []
        shipped_total, fused_total = 0.0, 0.0
        for candidates in picked:
            m = len(candidates)
            real_lines.append(len(np.unique(candidates // 64)))
            drawn = rng.choice(table.shape[0], size=m, replace=False)
            random_lines.append(len(np.unique(drawn // 64)))

            weights = fit.weights[candidates].copy()
            wlw = kernels._w_log_w(weights)
            reps = 10 if m > 400 else 60
            row = kernels.expected_guesses_all(
                table, candidates, weights, fit.params, fit.codes, fit.knots,
                fit.base, pool=pool)
            shipped_total += best_of(lambda: kernels.expected_guesses_all(
                table, candidates, weights, fit.params, fit.codes, fit.knots,
                fit.base, pool=pool), reps)

            best = np.inf
            for block in [g for g in FUSED_BLOCKS if g * m <= 4 << 20]:
                got = eg_fused(transposed, candidates, weights, wlw, fit.params,
                               fit.codes, fit.knots, fit.base, pool, block,
                               kernels._RESCAN_MAX, get_num_threads())
                if not np.allclose(row, got):
                    raise AssertionError(f"layouts disagree at m={m}")
                best = min(best, best_of(lambda: eg_fused(
                    transposed, candidates, weights, wlw, fit.params, fit.codes,
                    fit.knots, fit.base, pool, block, kernels._RESCAN_MAX,
                    get_num_threads()), reps))
            fused_total += best

        print(f"  {f'{low}-{high}':>12} {len(picked):>6} "
              f"{np.mean(real_lines):>15.0f} {np.mean(random_lines):>11.0f} "
              f"{shipped_total * 1e3:>15.2f}ms {fused_total * 1e3:>11.2f}ms "
              f"{shipped_total / fused_total:>7.2f}x")


def project(calls, sizes, ratios, games):
    """What those same calls would have cost under the other layout.

    Each call's ratio is interpolated in log-log from the sweep, the two being
    measured on the same machine in the same run.
    """
    common.header(f"where the time is, over {games} played-out games")

    counts = np.array([m for m, _ in calls], dtype=float)
    times = np.array([dt for _, dt in calls])
    predicted = np.exp(np.interp(np.log(counts), np.log(sizes), np.log(ratios)))
    other = times / predicted

    bands = ((1, 12), (13, 24), (25, 400), (401, 1100), (1101, 10 ** 9))
    names = {(1, 12): "1-12", (13, 24): "13-24", (25, 400): "25-400",
             (401, 1100): "401-1,100", (1101, 10 ** 9): "over 1,100"}
    print(f"  {'candidates':>11} {'calls':>7} {'[guess, answer]':>17} "
          f"{'[answer, guess]':>17} {'share':>7}")
    for low, high in bands:
        pick = (counts >= low) & (counts <= high)
        if not pick.any():
            continue
        print(f"  {names[(low, high)]:>11} {int(pick.sum()):>7} "
              f"{times[pick].sum() * 1e3:>15.0f}ms "
              f"{other[pick].sum() * 1e3:>15.0f}ms "
              f"{100 * times[pick].sum() / times.sum():>6.1f}%")

    print(f"\n  {'every call':>11} {len(calls):>7} {times.sum() * 1e3:>15.0f}ms "
          f"{other.sum() * 1e3:>15.0f}ms")

    # the opening is one call per process, not per game -- the memo answers it
    # thereafter -- so how much it weighs here is a fact about the length of the
    # run and not about the layouts, and it is quoted apart
    opening = counts >= 10000
    print(f"  {'less opening':>11} {int((~opening).sum()):>7} "
          f"{times[~opening].sum() * 1e3:>15.0f}ms "
          f"{other[~opening].sum() * 1e3:>15.0f}ms")

    print("\n  heaviest single calls:")
    for i in np.argsort(-times)[:5]:
        print(f"    {int(counts[i]):>6,} candidates  {times[i] * 1e3:>8.2f}ms")

    return (times.sum(), other.sum(),
            times[~opening].sum(), other[~opening].sum())


class Fit:
    """Weights and the cached `f`, which the expected-guesses kernel needs."""

    def __init__(self, data, config):
        self.weights = build_weights(data, config)
        self.params, form = estimator.load_active(config)
        self.codes, self.knots, self.base = form.arrays()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", type=int, default=60,
                    help="games played out for the profile")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--skip-profile", action="store_true",
                    help="the sweeps alone, without playing anything")
args = parser.parse_args()

data = load_data()
table = data.feedback_table
n_words = table.shape[0]
fit = Fit(data, PriorConfig())

built = time.perf_counter()
transposed = np.ascontiguousarray(table.T)
common.header("the two layouts")
print(f"  {n_words:,} x {n_words:,} uint8, {table.nbytes / 1e6:.0f} MB each")
print(f"  answer-major copy built in {time.perf_counter() - built:.1f}s")
print(f"  {get_num_threads()} threads, rescan below "
      f"{kernels._RESCAN_MAX} candidates")

rng = np.random.default_rng(args.seed)
whole = np.arange(n_words, dtype=np.int64)
sweep(fit, table, transposed, whole, rng, "every guess scored", "entropy")
sizes, ratios = sweep(fit, table, transposed, whole, rng,
                      "every guess scored", "expected")
# the per-band figures the documents quote, from the shipped mode's grid
for m, ratio in zip(sizes, ratios):
    if int(m) in (60, 130, 400):
        common.record(f"expected.at{int(m)}", float(ratio))

# what exact.py actually asks for: one representative per distinct split, which
# is a scattered quarter of the list and gives up the contiguous run
scattered = np.sort(rng.choice(n_words, size=n_words // 4,
                               replace=False)).astype(np.int64)
sweep(fit, table, transposed, scattered, rng,
      "a scattered quarter of the guesses", "expected")

if not args.skip_profile:
    calls, harvest = profile(fit, data, args.games, args.seed)
    on_real_sets(fit, table, transposed, harvest, rng)
    everything, other, less, other_less = project(calls, sizes, ratios,
                                                  args.games)
    common.header("the verdict")
    print("  expected guesses being the shipped mode, its ratios are the ones "
          "projected.\n  As in the sweeps, above 1 favors [answer, guess]:\n")
    print(f"    every call            {everything / other:>6.2f}x")
    print(f"    less the opening      {less / other_less:>6.2f}x")
    print("""
  Neither layout is a mistake, and the reason is structural.  Every pair of
  guess and candidate has one table byte to read and one bucket to update.
  Reducing a guess needs that guess's whole feedback vector, which the shipped
  layout holds as a row and the other as a strided column -- so answer-major
  must either keep every guess in the block accumulating at once, which puts
  the accumulator out of L1, or rebuild the vector by transposing a tile, which
  is what the fused kernel does and costs writing and re-reading it.  Both are
  additive on top of what the shipped layout already pays, and at the widest
  positions the shipped layout is reading the table at full cache-line
  efficiency anyway.  So the opening cannot be won back, and it is the most
  expensive single call in the game.

  Below that the shipped layout is the one paying, a line fetched per scattered
  candidate, and the fused kernel takes the band from about 25 to 1,200
  candidates.  The two are complementary rather than ranked, and which total
  comes out ahead depends on how many positions the one opening is amortized
  over.""")
    print(f"\n  Keeping both layouts would buy that band for another "
          f"{table.nbytes / 1e6:.0f} MB resident\n  and a doubled startup "
          f"read.")
