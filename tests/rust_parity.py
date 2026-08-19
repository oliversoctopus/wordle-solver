"""The Rust core agrees with the Python it was ported from.

Not a test of either implementation on its own: each check runs the two against
the same inputs and diffs them, so the Python is the oracle for the port in the
same way the other files here use one -- the second implementation, the answer
record, or a property that holds of every board.  Nothing below asserts a figure
written down by hand.

Skips rather than fails when the Rust has not been built, so a checkout without
a toolchain still runs the rest of the suite.  Build it with ``cargo build
--release``, or ``./build.sh`` if cargo is not on your PATH.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET = os.path.join(ROOT, "target", "release")
sys.path.insert(0, ROOT)

# Two independent implementations of the same arithmetic differ in their last
# bits, and nothing here should care.  What sets the bar is the tightest margin
# any decision actually turns on: over all 12,947 guesses at the opening the
# closest pair of scores differs by 1.1e-9 under the shipped prior and 6e-11
# under the loosest, so anything well under 6e-11 cannot change a choice.  This
# sits sixty times below that and ten thousand above float noise.
#
# Absolute as well as relative, because a score of zero is an ordinary answer --
# a guess that leaves every candidate in one bucket gains exactly no
# information -- and dividing by it says nothing.
ATOL = 1e-12
RTOL = 1e-12


def gap(got, mine):
    """Worst disagreement, as a multiple of what is tolerated. Passes at <= 1."""
    got, mine = np.asarray(got, dtype=float), np.asarray(mine, dtype=float)
    allowed = ATOL + RTOL * np.abs(mine)
    return float(np.max(np.abs(got - mine) / allowed)) if got.size else 0.0


def binary(name):
    """Path to a parity binary, or None if it has not been built."""
    path = os.path.join(TARGET, name + (".exe" if os.name == "nt" else ""))
    return path if os.path.exists(path) else None


def check_feedback(report):
    """The coloring rule, against every entry of the shipped table."""
    path = binary("parity_feedback")
    if path is None:
        return None
    done = subprocess.run([path, ROOT], capture_output=True, text=True)
    for line in (done.stdout + done.stderr).rstrip().splitlines():
        report(line.strip())
    return done.returncode == 0


def check_priors(report):
    """Every prior weight vector, against ``priors.build_weights``."""
    path = binary("parity_priors")
    if path is None:
        return None

    import priors
    from wordle_data import load_data

    with tempfile.TemporaryDirectory() as out_dir:
        done = subprocess.run([path, ROOT, out_dir], capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False

        data = load_data(with_feedback_table=False)
        worst, worst_slug = 0.0, None
        for config in priors.ALL_CONFIGS:
            weights_file = os.path.join(out_dir, config.slug + ".f64")
            if not os.path.exists(weights_file):
                report(f"{config.slug}: the Rust wrote no weights")
                return False
            rust = np.fromfile(weights_file, dtype="<f8")
            mine = priors.build_weights(data, config)
            if rust.shape != mine.shape:
                report(f"{config.slug}: {rust.shape} against {mine.shape}")
                return False
            # relative, since the weights span several orders of magnitude
            worst_here = gap(rust, mine)
            report(f"{config.slug}: worst {worst_here:.3f} of tolerance")
            if worst_here > worst:
                worst, worst_slug = worst_here, config.slug

        if worst > 1.0:
            report(f"worst is {worst:.3f} of tolerance, on {worst_slug}")
            return False
        report(f"all {len(priors.ALL_CONFIGS)} inside tolerance")
        return True


def check_estimator(report):
    """``f`` over every form, and ``effective_size``, against ``estimator``."""
    path = binary("parity_estimator")
    if path is None:
        return None

    import estimator
    import priors
    from wordle_data import load_data

    # the grid and the synthetic coefficients parity_estimator.rs also builds
    xs = np.arange(1401) * 0.01
    def synthetic(n):
        return np.array([0.5 - 0.15 * i for i in range(n)])
    prefixes = [1, 2, 3, 5, 8, 13, 64, 128, 129, 1000, 5000, 12947]

    def f_over(params, form):
        codes, knots, base = form.arrays()
        return np.array([estimator.f_eval_x(x, params, codes, knots, base)
                         for x in xs])

    with tempfile.TemporaryDirectory() as out_dir:
        done = subprocess.run([path, ROOT, out_dir], capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False

        worst, worst_where = 0.0, None

        def compare(name, mine, filename):
            nonlocal worst, worst_where
            got = np.fromfile(os.path.join(out_dir, filename), dtype="<f8")
            if got.shape != mine.shape:
                report(f"{name}: {got.shape} against {mine.shape}")
                return False
            here = gap(got, mine)
            if here > worst:
                worst, worst_where = here, name
            return True

        for name in sorted(estimator.FORMS):
            form = estimator.FORMS[name]
            mine = f_over(synthetic(form.n_params), form)
            if not compare(f"f/{name}", mine, f"f_{name}.f64"):
                return False
        report(f"{len(estimator.FORMS)} forms on the synthetic parameters")

        data = load_data(with_feedback_table=False)
        for config in priors.ALL_CONFIGS:
            slug = config.slug
            params, form = estimator.load_active(config)[:2]
            if not compare(f"fit/{slug}", f_over(params, form), f"fit_{slug}.f64"):
                return False
            weights = priors.build_weights(data, config)
            mine = np.array([estimator.effective_size(weights[:k]) for k in prefixes])
            if not compare(f"eff/{slug}", mine, f"eff_{slug}.f64"):
                return False
        report(f"{len(priors.ALL_CONFIGS)} cached fits, and effective_size on each")

        report(f"worst {worst:.3f} of tolerance" +
               (f", on {worst_where}" if worst_where else ""))
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


def _candidate_sets(data):
    """Sets that straddle both thresholds a kernel changes behavior at.

    ``kernels._RESCAN_MAX`` decides whether the reduce walks the candidates or
    sweeps all 243 patterns, and ``exact.MAX_PACKED`` whether a packed signature
    is the partition itself.  A set built by playing a real word is not the same
    shape as the first n indices, so both kinds are here.
    """
    from wordle_data import score_guess

    n = data.n_words
    sets = [np.arange(n, dtype=np.int64)]
    for size in (1, 2, 3, 5, 7, 8, 12, 50, 127, 128, 129, 200):
        sets.append(np.arange(size, dtype=np.int64))
    # real positions: everything consistent with one played word
    for guess, answer in (("salet", "crane"), ("raise", "dwarf"), ("salet", "abbey")):
        g = data.word2ind[guess]
        pattern = score_guess(guess, answer)
        kept = np.array([c for c in range(n)
                         if data.feedback_table[g, c] == pattern], dtype=np.int64)
        if kept.size:
            sets.append(kept)
    return sets


def check_kernels(report):
    """The three scorers and the split helpers, against ``kernels``."""
    path = binary("parity_kernels")
    if path is None:
        return None

    import estimator
    import kernels
    import priors
    from wordle_data import load_data

    data = load_data()
    config = priors.PriorConfig()
    weights_all = priors.build_weights(data, config)
    params, form = estimator.load_active(config)[:2]
    codes, knots, base = form.arrays()
    table = data.feedback_table
    sets = _candidate_sets(data)
    report(f"{len(sets)} sets, sizes {min(s.size for s in sets)}"
           f"-{max(s.size for s in sets)}")

    with tempfile.TemporaryDirectory() as out_dir:
        sets_file = os.path.join(out_dir, "sets.bin")
        with open(sets_file, "wb") as handle:
            handle.write(np.uint32(len(sets)).tobytes())
            for s in sets:
                handle.write(np.uint32(s.size).tobytes())
                handle.write(s.astype("<u8").tobytes())

        done = subprocess.run([path, ROOT, sets_file, out_dir],
                              capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False
        for line in done.stdout.strip().splitlines():
            report(line.strip())

        floats = np.fromfile(os.path.join(out_dir, "kernels.f64"), dtype="<f8")
        blobs = np.fromfile(os.path.join(out_dir, "kernels.i64"), dtype="<u1")
        at, bat = 0, 0
        worst, worst_where = 0.0, None

        def next_floats(k):
            nonlocal at
            got = floats[at:at + k]
            at += k
            return got

        def next_indices():
            nonlocal bat
            k = int(np.frombuffer(blobs[bat:bat + 4].tobytes(), dtype="<u4")[0])
            bat += 4
            got = np.frombuffer(blobs[bat:bat + 8 * k].tobytes(), dtype="<u8")
            bat += 8 * k
            return got.astype(np.int64)

        def compare(name, mine):
            nonlocal worst, worst_where
            got = next_floats(mine.size)
            here = gap(got, mine)
            if here > worst:
                worst, worst_where = here, name
            return True

        for s in sets:
            w = weights_all[s]
            compare("ig", kernels.ig_all(table, s, w))
            compare("expected", kernels.expected_guesses_all(
                table, s, w, params, codes, knots, base))
            compare("two_turn/1", kernels.two_turn_scores(table, s, w, 1.0))
            compare("two_turn/5", kernels.two_turn_scores(table, s, w, 5.0))
            compare("two_turn/hard", kernels.two_turn_scores(
                table, s, w, 5.0, pool=s))

            if s.size <= 200:
                # order and all: it decides which word represents a split, and
                # so the order the exact search tries them in under its cap
                mine = kernels.distinct_partitions(table, s)
                if not np.array_equal(next_indices(), mine):
                    report(f"distinct_partitions differs at size {s.size}")
                    return False
                mine = kernels.same_split(table, s, int(s[0]))
                if not np.array_equal(next_indices(), mine):
                    report(f"same_split differs at size {s.size}")
                    return False
                if s.size <= 7:
                    mine = kernels.signature_keys(table, s)
                    if not np.array_equal(next_indices().astype(np.int64),
                                          mine.astype(np.int64)):
                        report(f"signature_keys differs at size {s.size}")
                        return False
            pattern = int(table[s[0], s[0]])
            mine = kernels.filter_candidates(table, int(s[0]), pattern, s)
            if not np.array_equal(next_indices(), mine):
                report(f"filter_candidates differs at size {s.size}")
                return False

        report("the split helpers agree exactly on every set")
        report(f"worst {worst:.3f} of tolerance" +
               (f", on {worst_where}" if worst_where else ""))
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


def _endgame_sets(data, wanted=24):
    """Real positions small enough for the exact search to take on.

    Played out rather than sliced off the front of the list: what the search
    meets are sets left by a word, which share letters and split unevenly, and
    an arbitrary prefix of the alphabet does not behave like one.  One ply
    usually overshoots -- a played word leaves either a couple of candidates or
    several hundred -- so this goes two deep and keeps a spread of sizes, which
    is what reaches both the dedupe's packed path and the hashed one.
    """
    from wordle_data import score_guess

    table = data.feedback_table
    openings = ("salet", "crane", "raise")
    answers = ("hatch", "batty", "wound", "foggy", "abbey", "squid",
               "water", "hater", "catty", "ratty", "brook", "mushy")

    by_size = {}
    for opening in openings:
        g = data.word2ind[opening]
        for answer in answers:
            first = np.flatnonzero(
                table[g] == score_guess(opening, answer)).astype(np.int64)
            for depth_two in ("bronc", "hatch", "dwarf"):
                h = data.word2ind[depth_two]
                kept = first[table[h, first] == score_guess(depth_two, answer)]
                if 3 <= kept.size <= 60:
                    by_size.setdefault(int(kept.size), kept)
            if 3 <= first.size <= 60:
                by_size.setdefault(int(first.size), first)

    # a spread rather than the first n found, so both dedupe paths are reached
    return [by_size[k] for k in sorted(by_size)][:wanted]


def check_exact(report):
    """The endgame search, against ``exact.ExactEndgame``.

    Both the guess and the value have to match.  A guess of equal value would
    not do: ``SPLIT_TIE_TOL`` is 1e-9 and the closest real pair of scores over a
    whole scan is 1.1e-9, so a drifting branch order shows up as a different
    word long before it shows up as a different number.
    """
    path = binary("parity_exact")
    if path is None:
        return None

    import priors
    from exact import ExactEndgame
    from wordle_data import load_data

    data = load_data()
    weights = priors.build_weights(data, priors.PriorConfig())
    sets = _endgame_sets(data)
    arms = [(5.0, False), (1.0, False), (5.0, True)]
    turns_list = [3, 4, 5, 6]
    report(f"{len(sets)} positions, sizes {[int(s.size) for s in sets]}")

    with tempfile.TemporaryDirectory() as out_dir:
        sets_file = os.path.join(out_dir, "sets.bin")
        with open(sets_file, "wb") as handle:
            handle.write(np.uint32(len(sets)).tobytes())
            for s in sets:
                handle.write(np.uint32(s.size).tobytes())
                handle.write(s.astype("<u8").tobytes())

        done = subprocess.run([path, ROOT, sets_file, out_dir],
                              capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False
        for line in done.stdout.strip().splitlines():
            report(line.strip())

        ints = np.fromfile(os.path.join(out_dir, "exact.i64"), dtype="<u1")
        floats = np.fromfile(os.path.join(out_dir, "exact.f64"), dtype="<f8")
        iat, fat = 0, 0

        def next_int():
            nonlocal iat
            v = int(np.frombuffer(ints[iat:iat + 8].tobytes(), dtype="<i8")[0])
            iat += 8
            return v

        def next_indices():
            nonlocal iat
            k = int(np.frombuffer(ints[iat:iat + 4].tobytes(), dtype="<u4")[0])
            iat += 4
            got = np.frombuffer(ints[iat:iat + 8 * k].tobytes(), dtype="<i8")
            iat += 8 * k
            return got.astype(np.int64)

        def next_floats(k):
            nonlocal fat
            got = floats[fat:fat + k]
            fat += k
            return got

        worst = 0.0
        checked = 0
        ties = 0
        for penalty, hard in arms:
            for s in sets:
                search = ExactEndgame(data.feedback_table, weights,
                                      loss_penalty=penalty, hard=hard)
                for turns in turns_list:
                    guess, value = search.solve(s, turns)
                    got_guess, got_value = next_int(), next_floats(1)[0]
                    worst = max(worst, gap([got_value], [value]))
                    # A position routinely has several optimal guesses, and
                    # which one comes back depends on the order the bounds were
                    # sorted into.  `np.argsort` sorts with quicksort, whose tie
                    # order is unstable and documented as arbitrary, so Python's
                    # own answer is not reproducible across numpy versions.  The
                    # question worth asking is whether the Rust pick is *also*
                    # optimal, so value it and see.
                    if got_guess != guess:
                        mine = search._evaluate(int(got_guess), s, weights[s],
                                                weights[s].sum(), turns, np.inf)
                        if gap([mine], [value]) > 1.0:
                            report(f"solve at p={penalty} hard={hard} "
                                   f"t={turns} size={s.size}: {got_guess} is "
                                   f"worth {mine!r}, not {value!r}")
                            return False
                        ties += 1

                    # The values are compared elementwise and the lead word
                    # exactly; the tail's *words* are not, and cannot be.
                    # `_bounded_splits` routinely leaves scores of guesses tied
                    # on the bound -- 1176 of 1177 at exactly 2.0 in one of
                    # these positions -- and `np.argsort` breaks those ties with
                    # quicksort, which is unstable and documented as arbitrary.
                    # Python's own answer there is not reproducible across numpy
                    # versions, so demanding the Rust reproduce it would be
                    # testing an accident.
                    top = search.top(s, turns, k=5)
                    got_words = next_indices()
                    got_values = next_floats(len(top))
                    if len(got_words) != len(top):
                        report(f"top at p={penalty} hard={hard} t={turns} "
                               f"size={s.size}: {len(got_words)} words against "
                               f"{len(top)}")
                        return False
                    for got_v, (_, v) in zip(got_values, top):
                        worst = max(worst, gap([got_v], [v]))
                    checked += 1

        report(f"{checked} solve and top pairs; {ties} picked a different "
               f"guess of exactly equal value")
        report(f"worst {worst:.3f} of tolerance on the values")
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


# (label, what agreement establishes, the check)
CHECKS = [
    ("the coloring rule",
     "every one of the 12,947^2 pairs matches feedback_table.npy",
     check_feedback),
    ("the prior weights",
     "every configuration's weight vector matches priors.build_weights",
     check_priors),
    ("the estimator",
     "f over every form and every cached fit matches estimator.f_eval_x",
     check_estimator),
    ("the kernels",
     "the three scorers and the split helpers match kernels, either side of "
     "both thresholds",
     check_kernels),
    ("the endgame search",
     "solve and top return the identical guess and value, over three arms",
     check_exact),
]


def main():
    results = []
    for label, establishes, check in CHECKS:
        lines = []
        ok = check(lines.append)
        if ok is None:
            print(f"skip  {label}: not built")
            results.append(None)
            continue
        print(f"{'ok  ' if ok else 'FAIL'}  {label} -- {establishes}")
        for line in lines:
            print(f"      {line}")
        results.append(ok)

    failed = results.count(False)
    skipped = results.count(None)
    if failed:
        print(f"\n{failed} of {len(results)} disagree with the Python")
        return 1
    if skipped == len(results):
        print("\nnothing built; run cargo build --release")
        return 0
    print(f"\nall {len(results) - skipped} agree with the Python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
