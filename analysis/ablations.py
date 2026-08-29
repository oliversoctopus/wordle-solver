"""What each optimization is worth in wall clock, one knob off at a time.

SOLVER_EXPLAINED *Why any of this finishes* collects five ideas.  What each is
worth can be measured by taking it out and leaving everything else alone, which
is what this does -- and it is not a thing the speedups quoted elsewhere can
answer between them: they are measured on different operations against
different baselines, so a product of them would be a number no run has ever
produced.

THIS NEEDS AN IDLE MACHINE, and unlike most of `analysis/` it cannot be made
not to.  Every figure is a ratio of wall clocks, so background load does not
average out of it: contention costs the sixteen-core run far more than the
one-core run it is being compared against, and the single-threaded row goes
first.  A reading taken on a busy machine measures the contention.  Where a
question can be settled by counting instead, count it -- `branch_cap.py` is the
same ablation of the endgame's cap done in subtree evaluations, and its answer
holds whatever else the machine is doing.

The naive kernels below are the shipped ones with exactly one idea removed --
same accumulators, same blocks, same prange -- so what is timed is the removal
rather than a rewrite.  `--check` scores them against the shipped kernel, since
a wrong ablation is usually a fast one.

The three levels are separate operations.  A factor from one says nothing about
another, and the naive whole is not the product of these parts.
"""

import argparse
import time

import numba
import numpy as np
from numba import njit, prange, set_num_threads

import common
from estimator import f_eval_x
from exact import ExactEndgame
from kernels import _bucket_guesses, _RESCAN_MAX, _w_log_w, expected_guesses_all
from memo import Memo
from priors import PriorConfig, build_weights
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
args = parser.parse_args()

DEFAULT_THREADS = numba.get_num_threads()
BLOCK = 64


# --------------------------------------------------------- the naive scorers

@njit(cache=True)
def _color(gl, al):
    """The coloring rule applied rather than looked up, which is what the
    browser build does, having no table to read."""
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


def row(label, seconds, base, extra=""):
    factor = f"{seconds / base:>9.1f}x" if base else " " * 10
    print(f"  {label:<46s} {seconds:>9.3f}s {factor}  {extra}")


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

    for label, fn in (("no w-log-w precompute (a log per pair)", no_pre),
                      ("no table (the rule applied per pair)", no_tab)):
        t, got = timed(fn)
        diff = f"max diff {np.abs(got - ref).max():.1e}" if args.check else ""
        row(label, t, t_base, diff)

    set_num_threads(1)
    t, _ = timed(shipped)
    set_num_threads(DEFAULT_THREADS)
    row(f"single-threaded (of {DEFAULT_THREADS} cores)", t, t_base)

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
        f"(scaled from {len(pool)} guesses)")


common.header("what each optimization is worth, one at a time")
print("  Wall clock throughout.  On a machine that is doing anything else,")
print("  these figures are measuring that instead.")

scan_level(data.all_candidates(), "the opening")

opening = data.word2ind["salet"]
answer = data.word2ind["month"]
mid = solver.filter(opening, solver.feedback(opening, answer),
                    data.all_candidates())
scan_level(mid, f"after salet against {data.all_words[answer]}")


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
    "branch_cap.py counts the same thing")


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
    "grows with --sweep-games")
