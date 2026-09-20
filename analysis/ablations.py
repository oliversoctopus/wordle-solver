"""What each optimization is worth in wall clock, one knob off at a time.

Each naive kernel is the shipped one with one idea removed, so what is timed
is the removal; `--check` scores them against the shipped kernel.  The three
levels are separate operations, and the naive whole is not their product.
NEEDS AN IDLE MACHINE: every figure is a ratio of wall clocks, and contention
costs the sixteen-core run more than the one-core run it is compared against.
Backs the ablation table in SOLVER_EXPLAINED *Why any of this finishes*.
"""

import argparse
import json
import os
import time

import numba
import numpy as np
from numba import njit, prange, set_num_threads

import common
import estimator
from estimator import f_eval_x
from exact import ExactEndgame
from kernels import _bucket_guesses, _RESCAN_MAX, _w_log_w, expected_guesses_all
from memo import Memo
from priors import ALL_CONFIGS, PriorConfig, build_weights
from solver import EXPECTED, WordleSolver
from wordle_data import N_PATTERNS, WIN_PATTERN, load_data, score_guess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sweep-games", type=int, default=400,
                    help="answers played in the sweep level; the memo is worth "
                         "more the longer this is")
parser.add_argument("--python-pool", type=int, default=40,
                    help="guesses timed for the pure-Python reading, which is "
                         "then scaled to the whole pool")
parser.add_argument("--positions", type=int, default=150,
                    help="real positions in the endgame level")
parser.add_argument("--check", action="store_true",
                    help="score the naive kernels against the shipped one")
parser.add_argument("--projection", action="store_true",
                    help="carry the naive scan up to a sweep and a whole fit")
args = parser.parse_args()

DEFAULT_THREADS = numba.get_num_threads()
BLOCK = 64


# --------------------------------------------------------- the naive scorers

@njit(cache=True)
def _color(gl, al):
    """The coloring rule applied instead of looked up, as the browser build
    does, having no table."""
    out = 0
    seen = np.zeros(26, np.uint8)
    mark = np.zeros(5, np.uint8)
    for i in range(5):
        if gl[i] == al[i]:
            mark[i] = 2
        else:
            seen[al[i]] += 1
    for i in range(5):
        if mark[i] == 0 and seen[gl[i]] > 0:
            mark[i] = 1
            seen[gl[i]] -= 1
    for i in range(5):
        out = out * 3 + mark[i]
    return out


@njit(parallel=True, cache=True)
def _eg_no_precompute(feedback_table, candidates, weights, params, codes,
                      knots, base, pool, block):
    """`w log w` put back inside the inner loop.

    The shipped kernel takes one logarithm per candidate for the whole scan;
    this one takes a logarithm per (guess, candidate) pair, because a
    candidate's normalized share is not known until its bucket's mass is.
    """
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)
    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for b in prange(n_blocks):
        mass = np.zeros(N_PATTERNS)
        count = np.zeros(N_PATTERNS)
        ent = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)
        stop = min((b + 1) * block, n_out)
        for j in range(b * block, stop):
            guess = pool[j]
            for i in range(m):
                r = feedback_table[guess, candidates[i]]
                w = weights[i]
                mass[r] += w
                count[r] += 1.0
                if w > heaviest[r]:
                    heaviest[r] = w
            # the normalization exists only now, so each candidate costs its
            # own logarithm here
            for i in range(m):
                r = feedback_table[guess, candidates[i]]
                p = weights[i] / mass[r]
                ent[r] -= p * np.log2(p)

            expected = 1.0
            if m < _RESCAN_MAX:
                for i in range(m):
                    r = feedback_table[guess, candidates[i]]
                    if count[r] > 0.0:
                        if r != WIN_PATTERN:
                            if count[r] == 1.0:
                                v = 1.0
                            elif count[r] == 2.0:
                                v = 2.0 - heaviest[r] / mass[r]
                            else:
                                v = f_eval_x(ent[r], params, codes, knots, base)
                            expected += mass[r] / total * v
                        mass[r] = 0.0
                        count[r] = 0.0
                        ent[r] = 0.0
                        heaviest[r] = 0.0
            else:
                for r in range(N_PATTERNS):
                    if count[r] > 0.0:
                        if r != WIN_PATTERN:
                            if count[r] == 1.0:
                                v = 1.0
                            elif count[r] == 2.0:
                                v = 2.0 - heaviest[r] / mass[r]
                            else:
                                v = f_eval_x(ent[r], params, codes, knots, base)
                            expected += mass[r] / total * v
                        mass[r] = 0.0
                        count[r] = 0.0
                        ent[r] = 0.0
                        heaviest[r] = 0.0
            out[j] = expected
    return out


@njit(parallel=True, cache=True)
def _eg_no_table(letters, candidates, weights, wlw, params, codes, knots,
                 base, pool, block):
    """The shipped kernel with every table read replaced by the rule itself."""
    n_out = pool.shape[0]
    m = candidates.shape[0]
    out = np.empty(n_out)
    total = 0.0
    for i in range(m):
        total += weights[i]

    n_blocks = (n_out + block - 1) // block
    for b in prange(n_blocks):
        mass = np.zeros(N_PATTERNS)
        count = np.zeros(N_PATTERNS)
        w_log_w = np.zeros(N_PATTERNS)
        heaviest = np.zeros(N_PATTERNS)
        stop = min((b + 1) * block, n_out)
        for j in range(b * block, stop):
            guess = pool[j]
            for i in range(m):
                r = _color(letters[guess], letters[candidates[i]])
                w = weights[i]
                mass[r] += w
                count[r] += 1.0
                w_log_w[r] += wlw[i]
                if w > heaviest[r]:
                    heaviest[r] = w
            expected = 1.0
            if m < _RESCAN_MAX:
                for i in range(m):
                    r = _color(letters[guess], letters[candidates[i]])
                    if count[r] > 0.0:
                        if r != WIN_PATTERN:
                            expected += mass[r] / total * _bucket_guesses(
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
                            expected += mass[r] / total * _bucket_guesses(
                                mass[r], count[r], w_log_w[r], heaviest[r],
                                params, codes, knots, base)
                        mass[r] = 0.0
                        count[r] = 0.0
                        w_log_w[r] = 0.0
                        heaviest[r] = 0.0
            out[j] = expected
    return out


# ------------------------------------------------------------------ harness

def timed(fn, repeat=3):
    """Best of `repeat`: the run least interrupted by anything else.  It is a
    defence against a busy machine and not a cure -- see the module docstring."""
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


RATIOS = {}


def row(label, seconds, base, extra="", key=None):
    factor = f"{seconds / base:>9.1f}x" if base else " " * 10
    print(f"  {label:<46s} {seconds:>9.3f}s {factor}  {extra}")
    if key and base:
        RATIOS[key] = common.record(key, seconds / base)


data = load_data()
config = PriorConfig()
weights = build_weights(data, config)
solver = WordleSolver(data=data, config=config, mode=EXPECTED)
params, codes, knots, base = (solver.params, solver._codes, solver._knots,
                              solver._base)
table = data.feedback_table
all_rows = np.arange(data.n_words, dtype=np.int64)
letters = np.frombuffer("".join(data.all_words).encode(), dtype=np.uint8)
letters = np.ascontiguousarray(
    (letters.reshape(data.n_words, 5) - ord("a")).astype(np.uint8))


def scan_level(candidates, name):
    w = weights[candidates]
    wlw = _w_log_w(w)

    def shipped():
        return expected_guesses_all(table, candidates, w, params, codes,
                                    knots, base)

    def no_pre():
        return _eg_no_precompute(table, candidates, w, params, codes, knots,
                                 base, all_rows, BLOCK)

    def no_tab():
        return _eg_no_table(letters, candidates, w, wlw, params, codes, knots,
                            base, all_rows, BLOCK)

    for fn in (shipped, no_pre, no_tab):        # compile before timing
        fn()

    print(f"\n  one scoring scan: {len(candidates):,} candidates against all "
          f"{data.n_words:,} guesses ({name})")
    t_base, ref = timed(shipped)
    row("shipped", t_base, t_base)

    tag = "opening" if len(candidates) == data.n_words else "midgame"
    for label, fn, key in (("no w-log-w precompute (a log per pair)", no_pre, "wlogw"),
                           ("no table (the rule applied per pair)", no_tab, "table")):
        t, got = timed(fn)
        diff = f"max diff {np.abs(got - ref).max():.1e}" if args.check else ""
        row(label, t, t_base, diff, key=f"{tag}.{key}")

    set_num_threads(1)
    t, _ = timed(shipped)
    set_num_threads(DEFAULT_THREADS)
    row(f"single-threaded (of {DEFAULT_THREADS} cores)", t, t_base,
        key=f"{tag}.threads")

    # the pure-Python reading, timed on a slice and scaled: the loop is uniform
    # in the pool, so this stretches one measurement rather than composing two
    pool = all_rows[:args.python_pool]
    t0 = time.perf_counter()
    for g in pool:
        gw = data.all_words[g]
        buckets = {}
        for idx in range(len(candidates)):
            r = score_guess(gw, data.all_words[candidates[idx]])
            buckets[r] = buckets.get(r, 0.0) + w[idx]
    t_py = (time.perf_counter() - t0) * data.n_words / len(pool)
    row("pure Python, no table, no kernels", t_py, t_base,
        f"(scaled from {len(pool)} guesses)", key=f"{tag}.python")
    return t_base, t_py


common.header("what each optimization is worth, one at a time")
print("  Wall clock throughout.  On a machine that is doing anything else,")
print("  these figures are measuring that instead.")

T_OPENING, T_OPENING_NAIVE = scan_level(data.all_candidates(),
                                       "the opening")

opening = data.word2ind["salet"]
answer = data.word2ind["month"]
mid = solver.filter(opening, solver.feedback(opening, answer),
                    data.all_candidates())
common.record("midgame.candidates", len(mid))
scan_level(mid, f"after salet against {data.all_words[answer]}")
common.record("table", common.md_table(
    ["taking one idea back out", "at the opening", "at a mid-game position"],
    [(label, f"{RATIOS['opening.' + key]:,.{digits}f}×",
      f"{RATIOS['midgame.' + key]:,.{digits}f}×")
     for label, key, digits in (
         ("`w log w` back inside the inner loop", "wlogw", 1),
         ("no table — the coloring rule applied per pair", "table", 0),
         (f"one core instead of {DEFAULT_THREADS}", "threads", 1),
         ("**all three at once, in plain Python**", "python", 0))]))


# ------------------------------------------------------- the endgame search

def real_positions(limit):
    """Positions the endgame search really meets, taken from played games.
    `branch_cap.py` collects the same set; kept separate so neither script
    imports the other's top-level run."""
    out, seen = [], set()
    for a in range(0, 9000, 7):
        cands = data.all_candidates()
        for turn in range(4):
            guess, _ = solver.best_guess(cands, 5 - turn)
            if guess == a:
                break
            cands = solver.filter(guess, solver.feedback(guess, a), cands)
            key = cands.tobytes()
            if 3 <= len(cands) <= 60 and key not in seen:
                seen.add(key)
                out.append((cands, max(2, 5 - turn - 1)))
            if len(out) >= limit:
                return out
    return out


positions = real_positions(args.positions)
print(f"\n  the exact search over {len(positions)} real positions, from a cold "
      f"memo each time")


def sweep_search(branch):
    t0 = time.perf_counter()
    for cands, turns in positions:
        s = ExactEndgame(table, weights, loss_penalty=5.0, max_branch=branch)
        try:
            s.solve(cands, turns)
        except Exception:
            pass
    return time.perf_counter() - t0


t_ship = sweep_search(50)
row("shipped (cap 50)", t_ship, t_ship)
row("no branch cap", sweep_search(None), t_ship,
    "branch_cap.py counts the same thing", key="branch_cap")
common.record("search_positions", len(positions))


# --------------------------------------------------------------- the sweep

print(f"\n  a sweep of {args.sweep_games} answers, played out")
answers = np.arange(args.sweep_games, dtype=np.int64)


def sweep(memo_limit):
    s = WordleSolver(data=data, config=config, mode=EXPECTED)
    if memo_limit is not None:
        s._memo = Memo(memo_limit)
        s._lookahead_memo = Memo(memo_limit)
        s.exact.memo = Memo(memo_limit)

    def run():
        for a in answers:
            s.play(int(a), max_guesses=12, deadline=6)
    return run


t_warm, _ = timed(sweep(None), repeat=1)
row("shipped", t_warm, t_warm)
t_cold, _ = timed(sweep(1), repeat=1)
row("no memo (all three caches held to one entry)", t_cold, t_warm,
    "grows with --sweep-games", key=f"memo.at{args.sweep_games}")


# ---------------------------------------------------- what none of it finishes

def train_games():
    """Games one `train` plays, from the rounds each cached fit recorded."""
    total = 0
    for cfg in ALL_CONFIGS:
        path = estimator.cache_path(cfg)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            fit = json.load(fh)
        # a rollout per round, one more for the RMSE, then the holdout
        total += ((len(fit["history"]) + 1) * fit["n_games"]
                  + fit.get("holdout", {}).get("games", 0))
    return total


def spelled(seconds):
    """A duration the way the documents write one."""
    if seconds >= 86400:
        return f"{seconds / 86400:,.0f} days"
    if seconds >= 60:
        return f"{seconds / 60:.1f} minutes"
    if seconds >= 1:
        return f"{seconds:.1f} s"
    return f"{seconds * 1e3:.0f} ms"


def duration(seconds):
    for unit, size in (("days", 86400.0), ("hours", 3600.0), ("min", 60.0)):
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:.3f} s"


if args.projection:
    pairs = data.n_words * data.n_words
    common.header("what the project would cost with none of it")
    print("  Projections, not measurements: the naive column is one measured")
    print("  opening scan times the number of them the operation performs.")
    print(f"  One opening scan is {pairs:,} (guess, answer) pairs.\n")
    print(f"  {'':34s} {'shipped':>14s} {'naive':>16s} {'ratio':>10s}")
    print("  " + "-" * 78)

    fits = train_games()
    rows = [("choose the opening guess", T_OPENING, 1),
            ("play every answer once", None, data.n_words)]
    if fits:
        rows.append((f"train, all {len(ALL_CONFIGS)} shipped priors", None, fits))

    for label, shipped, scans in rows:
        naive = T_OPENING_NAIVE * scans
        # every game opens at the full list, so a memo-less run rescans it once
        # per game; the later turns are a rounding error beside that
        have = f"{duration(shipped):>16s}" if shipped else f"{'--':>16s}"
        ratio = f"{naive / shipped:>9,.0f}x" if shipped else " " * 10
        print(f"  {label:<34s} {have} {duration(naive):>16s} {ratio}")
        key = {"choose the opening guess": "opening", "play every answer once": "sweep"}.get(label, "train")
        common.record(f"projection.{key}.naive", spelled(naive))
        if shipped:
            common.record(f"projection.{key}.shipped", spelled(shipped))

    common.record("projection.pairs", int(pairs))
    common.record("projection.ns_per_pair", T_OPENING / pairs * 1e9)
    common.record("projection.us_per_pair_python", T_OPENING_NAIVE / pairs * 1e6)
    print(f"\n  per (guess, answer) pair, at the opening:")
    print(f"    shipped     {T_OPENING / pairs * 1e9:>8.2f} ns "
          f"across {DEFAULT_THREADS} cores")
    print(f"    pure Python {T_OPENING_NAIVE / pairs * 1e6:>8.2f} us")
