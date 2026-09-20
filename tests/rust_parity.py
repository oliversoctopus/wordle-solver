"""The Rust core agrees with the Python it was ported from.

Each check runs the two against the same inputs and diffs them; the Python is
the oracle and nothing asserts a figure written by hand.  Skips when the Rust
is not built: ``cargo build --release``, or ``./build.sh`` when cargo is not
on PATH.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET = os.path.join(ROOT, "target", "release")
sys.path.insert(0, ROOT)

# between the noise a BLAS reduction opens and the closest pair of distinct
# scores; absolute as well as relative, since a score of exactly zero is ordinary
ATOL = 1e-11
RTOL = 1e-11


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
    not do: distinct scores over a whole scan sit far apart, so a drifting
    branch order shows up as a different word long before it shows up as a
    different number.
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
                    # a position routinely has several optimal guesses; both sides break the tie by
                    # word index, so this finds nothing and stays to price a divergence
                    if got_guess != guess:
                        mine = search._evaluate(int(got_guess), s, weights[s],
                                                weights[s].sum(), turns, np.inf)
                        if gap([mine], [value]) > 1.0:
                            report(f"solve at p={penalty} hard={hard} "
                                   f"t={turns} size={s.size}: {got_guess} is "
                                   f"worth {mine!r}, not {value!r}")
                            return False
                        ties += 1

                    # words as well as values, all five, since the branch order
                    # is the same rule on both sides and the bounds tie in bulk
                    top = search.top(s, turns, k=5)
                    got_words = next_indices()
                    got_values = next_floats(len(top))
                    if len(got_words) != len(top):
                        report(f"top at p={penalty} hard={hard} t={turns} "
                               f"size={s.size}: {len(got_words)} words against "
                               f"{len(top)}")
                        return False
                    for got_w, got_v, (w, v) in zip(got_words, got_values, top):
                        worst = max(worst, gap([got_v], [v]))
                        if got_w != w:
                            report(f"top at p={penalty} hard={hard} t={turns} "
                                   f"size={s.size}: {got_w} where the Python "
                                   f"ranks {w}, at {got_v!r} against {v!r}")
                            return False
                    checked += 1

        report(f"{checked} solve and top pairs; {ties} picked a different "
               f"guess of exactly equal value")
        report(f"worst {worst:.3f} of tolerance on the values")
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


def check_solver(report):
    """Which branch of the dispatch answers, and what it answers with.

    This covers the dispatch: it has five ways out, a closed form at one or two
    candidates, the exact search,
    the heaviest candidate on the last turn, lookahead, and the heuristic
    scorer -- and the arms below reach each of them.
    """
    path = binary("parity_solver")
    if path is None:
        return None

    import priors
    from solver import ENTROPY, EXPECTED, WordleSolver
    from wordle_data import load_data

    data = load_data()
    config = priors.PriorConfig()
    arms = [
        (EXPECTED, False, True, 0),
        (EXPECTED, True, True, 0),
        (EXPECTED, False, False, 0),
        (EXPECTED, False, True, 1),
        (ENTROPY, False, True, 0),
        (ENTROPY, True, True, 0),
    ]
    turns_list = [1, 2, 4, 6]

    # one candidate through the whole list, so every branch is reached
    sets = [np.arange(n, dtype=np.int64) for n in (1, 2, 3, 40, 61, 500)]
    sets.append(np.arange(data.n_words, dtype=np.int64))
    sets.extend(_endgame_sets(data, wanted=4))

    # which arms visit each set, one bit per arm: everything narrow goes to all six,
    # the opening to one arm per mode.  The Rust reads these, so the rule lives here.
    every_arm = (1 << len(arms)) - 1
    wide_arms = (1 << 0) | (1 << 4)
    masks = [wide_arms if s.size == data.n_words else every_arm for s in sets]
    report(f"{len(sets)} positions, sizes {[int(s.size) for s in sets]}")

    with tempfile.TemporaryDirectory() as out_dir:
        sets_file = os.path.join(out_dir, "sets.bin")
        with open(sets_file, "wb") as handle:
            handle.write(np.uint32(len(sets)).tobytes())
            for s in sets:
                handle.write(np.uint32(s.size).tobytes())
                handle.write(s.astype("<u8").tobytes())

        done = subprocess.run([path, ROOT, sets_file, out_dir,
                               ",".join(str(m) for m in masks)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False

        ints = np.fromfile(os.path.join(out_dir, "solver.i64"), dtype="<u1")
        floats = np.fromfile(os.path.join(out_dir, "solver.f64"), dtype="<f8")
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

        worst, ties, checked = 0.0, 0, 0
        for arm, (mode, hard, use_exact, lookahead) in enumerate(arms):
            for mask, s in zip(masks, sets):
                if not mask >> arm & 1:
                    continue
                solver = WordleSolver(data=data, config=config, mode=mode,
                                      hard=hard, use_exact=use_exact,
                                      lookahead=lookahead, fused=False)
                for turns in turns_list:
                    guess, score = solver.best_guess(s, guesses_left=turns)
                    got_guess, got_score = next_int(), floats[fat]
                    fat += 1
                    got_exact = bool(next_int())
                    worst = max(worst, gap([got_score], [score]))
                    if got_exact != solver.solved_exactly(s, guesses_left=turns):
                        report(f"solved_exactly at {mode} hard={hard} "
                               f"t={turns} size={s.size}: {got_exact}")
                        return False
                    if got_guess != guess:
                        # equal scores mean equally good guesses; the search's
                        # own tie order is arbitrary, as check_exact explains
                        mine = solver.score_one(int(got_guess), s)
                        if gap([mine], [score]) > 1.0:
                            report(f"best_guess at {mode} hard={hard} "
                                   f"t={turns} size={s.size}: {got_guess} "
                                   f"scores {mine!r}, not {score!r}")
                            return False
                        ties += 1
                    checked += 1

                guess, _ = solver.best_guess(s, guesses_left=6)
                mine = solver.split_siblings(guess, s)
                got = next_indices()
                if not np.array_equal(np.sort(got), np.sort(mine)):
                    report(f"split_siblings at {mode} hard={hard} "
                           f"size={s.size}: {got.size} against {mine.size}")
                    return False

        report(f"{checked} guesses over {len(arms)} arms and "
               f"{len(turns_list)} turn counts")
        report(f"{ties} picked a different guess of equal score")
        report(f"worst {worst:.3f} of tolerance on the scores")
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


def check_every_answer(report):
    """The whole sweep: every word played as the answer, both sides.

    This is the acceptance test for the port rather than one more comparison.
    The solver is deterministic, so playing all 12,947 words once and weighting
    by the prior *computes* the mean -- there is no sampling error to hide in,
    and README *Results* is the number to hit.  Compared per answer rather than
    on the mean, so two games that differ in opposite directions cannot cancel.
    """
    path = binary("every_answer")
    if path is None:
        return None

    import priors
    from analysis.common import every_answer, exact_stats
    from solver import WordleSolver
    from wordle_data import load_data

    with tempfile.TemporaryDirectory() as out_dir:
        dump = os.path.join(out_dir, "lengths.f64")
        done = subprocess.run([path, ROOT, "--dump", dump],
                              capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False
        for line in done.stdout.strip().splitlines():
            report(line.strip())

        data = load_data()
        config = priors.PriorConfig()
        solver = WordleSolver(data=data, config=config, fused=False)
        # the oracle has to be the Python loop: `common.every_answer` hands the sweep to
        # this very binary where it can
        os.environ["WORDLE_NO_RUST"] = "1"
        try:
            mine = every_answer(solver)
        finally:
            os.environ.pop("WORDLE_NO_RUST", None)
        got = np.fromfile(dump, dtype="<f8")
        if got.shape != mine.shape:
            report(f"{got.shape} against {mine.shape}")
            return False

        differ = int((got != mine).sum())
        longer = int((got > mine).sum())
        mean, loss = exact_stats(mine, solver.weights)
        got_mean, got_loss = exact_stats(got, solver.weights)
        report(f"python mean {mean:.4f}, loss {loss * 100:.4f}%")
        report(f"{differ} of {got.size} answers differ, {longer} longer "
               f"and {differ - longer} shorter")
        report(f"mean apart by {abs(got_mean - mean):.2e}, "
               f"loss by {abs(got_loss - loss):.2e}")

        # a looser tolerance: with several optimal guesses the two take different ones and
        # land on the same value by another route.  Nothing the documents quote may move.
        if abs(got_mean - mean) > 1e-5 * mean or abs(got_loss - loss) > 1e-5:
            report("the weighted figures moved by more than the documents' precision")
            return False
        if f"{got_mean:.4f}" != f"{mean:.4f}":
            report(f"mean rounds to {got_mean:.4f}, not {mean:.4f}")
            return False
        if f"{got_loss * 100:.4f}" != f"{loss * 100:.4f}":
            report(f"loss rounds to {got_loss * 100:.4f}%, not {loss * 100:.4f}%")
            return False
        report("both round to the figures README quotes")
        return True


def _probe(requests):
    """Run api_probe over ``[(path, body), ...]``, returning the parsed replies.

    One process for the whole list, because a game's session lives in it: the
    module holds the answer the way web.py's Store does, and a second process
    would not know about it.
    """
    import json

    path = binary("api_probe")
    if path is None:
        return None
    argv = [path, ROOT]
    for endpoint, body in requests:
        argv += [endpoint, json.dumps(body)]
    done = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8")
    replies = [json.loads(line) for line in done.stdout.splitlines()
               if line.startswith("{")]
    return replies if len(replies) == len(requests) else None


def _diff(got, mine, where, report):
    """Every leaf of two replies, compared. Floats go through ``gap``; anything
    else has to be equal.  Returns the worst float disagreement, or None once
    something has been reported."""
    if isinstance(mine, dict):
        if set(got) - set(mine) or set(mine) - set(got):
            report(f"{where}: keys {sorted(set(got) ^ set(mine))} on one side only")
            return None
        worst = 0.0
        for key in mine:
            here = _diff(got[key], mine[key], f"{where}.{key}", report)
            if here is None:
                return None
            worst = max(worst, here)
        return worst
    if isinstance(mine, list):
        if len(got) != len(mine):
            report(f"{where}: {len(got)} entries against {len(mine)}")
            return None
        worst = 0.0
        for i, (a, b) in enumerate(zip(got, mine)):
            here = _diff(a, b, f"{where}[{i}]", report)
            if here is None:
                return None
            worst = max(worst, here)
        return worst
    if isinstance(mine, float) and isinstance(got, (int, float)):
        return gap([got], [mine])
    if got != mine:
        report(f"{where}: {got!r}, not {mine!r}")
        return None
    return 0.0


def check_messages(report):
    """The two implementations word an error the same way.

    Every message the page can show is written twice, once in ``web.py`` and
    once in the Rust the static build answers from, so a rewrite has to be
    made twice; this notices when it was made once.  The Python side is driven
    in process: a handler raises before it writes a reply, so a bare instance
    with the pool on it reaches the message.
    """
    import web
    from wordle_data import score_guess

    pool = web.SolverPool()
    data = pool.data()
    settings = web.parse_settings({})

    handler = object.__new__(web.Handler)
    handler.pool = pool
    handler.store = web.Store("board")
    handler.games = web.Store("game")
    handler.send_json = lambda payload, status=200: payload
    board = handler.store.create(web.Session(settings))
    over = web.Game(data.word2ind["crane"], web.PriorConfig(), "the one you chose")
    over.rows = [(data.word2ind["crane"], score_guess("crane", "crane"))]
    finished = handler.games.create(over)

    def row(word, answer):
        return (data.word2ind[word], score_guess(word, answer))

    # (what is wrong, the request, what the Python does with the same thing)
    cases = [
        ("a row that already won, with rows under it",
         ("/api/board", {"session": board,
                         "rows": [{"word": "crane", "pattern": "GGGGG"},
                                  {"word": "salet", "pattern": "NYNYN"}]}),
         lambda: web.build_state(pool, board, settings,
                                 [row("crane", "crane"), row("salet", "crane")])),
        ("rows that cannot all be true",
         ("/api/board", {"session": board,
                         "rows": [{"word": "crane", "pattern": "NNNNN"},
                                  {"word": "salet", "pattern": "NYNYN"}]}),
         lambda: web.build_state(pool, board, settings,
                                 [row("crane", "proxy"), row("salet", "crane")])),
        ("a row whose word is not a word",
         ("/api/board", {"session": board,
                         "rows": [{"word": "zzzzz", "pattern": "NNNNN"}]}),
         lambda: handler.api_board({"session": board,
                                    "rows": [{"word": "zzzzz",
                                              "pattern": "NNNNN"}]})),
        ("a row with no word in it",
         ("/api/board", {"session": board,
                         "rows": [{"word": "", "pattern": "NNNNN"}]}),
         lambda: handler.api_board({"session": board,
                                    "rows": [{"word": "", "pattern": "NNNNN"}]})),
        ("a session that is not there",
         ("/api/game/guess", {"session": "nope", "word": "crane"}),
         lambda: handler.games.get("nope")),
        # g1 because the setup below deals it first, in the one process the
        # whole list runs in; a game's session cannot outlive that
        ("guessing after the game is over",
         ("/api/game/guess", {"session": "g1", "word": "crane"}),
         lambda: handler.api_game_guess({"session": finished, "word": "crane"})),
        ("an answer that is not a word",
         ("/api/game/new", {"answer": "zzzzz"}),
         lambda: handler.api_game_new({"answer": "zzzzz"})),
        ("finding against a word that is not one",
         ("/api/find", {"answer": "zzzzz", "pattern": "NNNNN"}),
         lambda: handler.api_find({"answer": "zzzzz", "pattern": "NNNNN"})),
        ("finding with no answer typed",
         ("/api/find", {"answer": "", "pattern": "NNNNN"}),
         lambda: handler.api_find({"answer": "", "pattern": "NNNNN"})),
    ]

    # the over-game case needs a game the module has finished too, so it is
    # played out first and the session it lands in is the one asked about
    setup = [("/api/game/new", {"answer": "crane"}),
             ("/api/game/guess", {"session": "g1", "word": "crane"})]
    replies = _probe(setup + [request for _, request, _ in cases])
    if replies is None:
        return None
    replies = replies[len(setup):]

    apart = 0
    for (what, (path, _), python), got in zip(cases, replies):
        try:
            python()
            report(f"{what}: the Python allowed it")
            return False
        except web.SolverError as error:
            mine = str(error)
        theirs = got.get("error")
        if theirs is None:
            report(f"{what}: the Rust allowed it, {path} said {got}")
            return False
        if theirs != mine:
            report(f"{what}:")
            report(f"    web.py     {mine!r}")
            report(f"    the Rust   {theirs!r}")
            apart += 1

    report(f"{len(cases)} bad requests, both implementations refusing each")
    if apart:
        report(f"{apart} worded differently -- rewrite both, or neither")
        return False
    return True


def _drop_tied_words(got, mine):
    """Two guesses of equal score are equally good, and which of them a ranking
    names is arbitrary -- check_exact says why.  Where the two sides picked
    differently at the same score, everything that follows from the word goes,
    and the score stays.  Returns how many entries that was.
    """
    ties = 0
    for key in ("recommendation", "alternatives"):
        pairs = zip(got[key] or [], mine[key] or []) if key == "alternatives" \
            else [(got[key], mine[key])] if got[key] and mine[key] else []
        for a, b in pairs:
            if a["word"] == b["word"] or gap([a["score"]], [b["score"]]) > 1.0:
                continue
            ties += 1
            for entry in (a, b):
                for field in ("word", "same_split", "same_split_count",
                              "possible", "win_probability"):
                    entry.pop(field, None)
    return ties


def check_api(report):
    """The endpoints the page calls, against the ones web.py answers with.

    Every leaf of every reply, not just the numbers: the page reads the shape
    as much as the values, and a field that is missing or spelled differently
    breaks it just as thoroughly as a wrong score would.  The switches are
    covered here too, since nothing above this reads a request body.
    """

    import priors
    import web

    settings = [
        {},
        {"repeat": False, "curation": False},
        {"repeat": False},
        {"mode": "entropy"},
        {"hard": True},
        {"form": "cubic"},
        {"form": "invonly", "top": 3},
        # against a known answer the rows are recolored by the server, so the
        # patterns sent below are overridden and the two have to agree on that
        {"answer": "fogey"},
        {"answer": "pouch", "mode": "entropy"},
    ]
    # two rows of each opening the prior switches can produce, since a board starting
    # on that word is served from `openings.json`: one wide pattern and one narrow
    played = [[],
              [("salet", "NYNYN")], [("salet", "GYNNY")],
              [("tares", "NYNYN")], [("tares", "NYYNY")],
              [("trace", "NYNYN")], [("trace", "GNNYN")],
              [("crane", "NNNNN"), ("moist", "NYNNY")]]
    boards = [("crane", "NNNNN", "left"), ("crane", "NNNNN", "bits"),
              ("salet", "GNYNN", "weight"), ("proxy", "NYNNG", "alpha"),
              ("fuzzy", "GGGGY", "left")]      # the last gives no match at all
    plays = [("crane", ["salet", "brand"]), ("proxy", ["adieu", "proxy"])]

    requests = []
    for switches in settings:
        for rows in played:
            requests.append(("/api/board", {
                **{"repeat": True, "curation": True, "mode": "expected",
                   "hard": False, "top": 5}, **switches,
                "rows": [{"word": word, "pattern": pattern}
                         for word, pattern in rows]}))
    requests += [("/api/find", {"answer": answer, "pattern": pattern,
                                "sort": sort, "repeat": True, "curation": True})
                 for answer, pattern, sort in boards]
    for n, (answer, words) in enumerate(plays, start=1):
        requests.append(("/api/game/new", {"answer": answer, "repeat": True,
                                           "curation": True}))
        requests += [("/api/game/guess", {"session": f"g{n}", "word": word})
                     for word in words]

    replies = _probe(requests)
    if replies is None:
        return None

    pool = web.SolverPool()
    data = pool.data()
    config = priors.PriorConfig()
    worst, at, ties = 0.0, "", 0

    at_reply = 0
    for switches in settings:
        for rows in played:
            got = replies[at_reply]
            body = {"repeat": True, "curation": True, "mode": "expected",
                    "hard": False, "top": 5, **switches}
            at_reply += 1
            parsed = web.parse_settings(body)
            mine = web.build_state(
                pool, got["session"], parsed,
                [(data.word2ind[word], web.parse_pattern(pattern))
                 for word, pattern in rows],
                web.parse_answer(body, data))
            with pool.lock:
                solver_weights = pool.solver_for(parsed).weights
            # neither is meaningful across the two: one has no network at all,
            # and the session is the module's own name for the board
            for key in ("session", "elapsed_ms"):
                mine.pop(key, None)
                got.pop(key, None)
            # "the 80 likeliest" is ambiguous when more than 80 words tie, and numpy's argsort
            # is not stable; what has to match is the weights
            for reply in (got, mine):
                reply["possible_answers"] = [float(solver_weights[data.word2ind[w]])
                                             for w in reply["possible_answers"]]
            ties += _drop_tied_words(got, mine)
            here = _diff(got, mine, f"board {switches} after {len(rows)}", report)
            if here is None:
                return False
            if here > worst:
                worst, at = here, f"the board under {switches or 'the defaults'}"
    report(f"{len(settings)} sets of switches over {len(played)} boards, "
           "every field of every reply")
    report(f"{ties} entries named a different guess of equal score")

    found = replies[at_reply:at_reply + len(boards)]
    at_reply += len(boards)
    for (answer, pattern, sort), got in zip(boards, found):
        mine = web.find_guesses(pool, data.word2ind[answer],
                                web.parse_pattern(pattern), config, sort)
        for reply in (got, mine):
            reply.pop("elapsed_ms", None)
        here = _diff(got, mine, f"find {answer}/{pattern}/{sort}", report)
        if here is None:
            return False
        if here > worst:
            worst, at = here, f"find {answer}/{pattern} by {sort}"
    report(f"{len(boards)} finder queries, "
           f"{sum(r['total'] for r in found)} matches in all")

    for answer, words in plays:
        game = web.Game(data.word2ind[answer], config, "the one you chose")
        for step in range(len(words) + 1):
            got = replies[at_reply]
            at_reply += 1
            mine = web.game_state(pool, got["session"], game)
            here = _diff(got, mine, f"play {answer} after {step}", report)
            if here is None:
                return False
            if here > worst:
                worst, at = here, f"the letter odds of {answer} after {step}"
            if step < len(words):
                guess = data.word2ind[words[step]]
                game.rows.append((guess, web.score_guess(words[step], answer)))

    report(f"{len(plays)} games played out, share grids and all")
    report(f"worst {worst:.3f} of tolerance, on {at}")
    return worst <= 1.0


# (label, what agreement establishes, the check)
def check_rng(report):
    """The training draw, against ``np.random.default_rng``.

    Exactly, not to a tolerance: a fit is only comparable with the Python's if
    the two played the same answers, so one differing index is a failure and
    not a rounding difference.
    """
    path = binary("parity_rng")
    if path is None:
        return None

    import priors
    from solver import sample_answers
    from wordle_data import load_data

    seeds = (0, 1, 12345)
    with tempfile.TemporaryDirectory() as out_dir:
        done = subprocess.run([path, ROOT, out_dir], capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False

        for seed in seeds:
            mine = np.random.default_rng(seed).random(64)
            got = np.fromfile(os.path.join(out_dir, f"doubles_{seed}.f64"), dtype="<f8")
            if not np.array_equal(mine, got):
                report(f"PCG64 doubles differ at seed {seed}")
                return False
        report(f"{len(seeds)} seeds of PCG64 doubles, bit for bit")

        data = load_data(with_feedback_table=False)
        for config in priors.ALL_CONFIGS:
            weights = priors.build_weights(data, config)
            for seed in seeds:
                mine = sample_answers(weights, 2000, seed=seed)
                got = np.fromfile(
                    os.path.join(out_dir, f"answers_{config.slug}_{seed}.f64"),
                    dtype="<f8").astype(np.int64)
                if not np.array_equal(mine, got):
                    differ = int((mine != got).sum())
                    report(f"{config.slug} seed {seed}: {differ} of 2000 answers differ")
                    return False
        report(f"{len(priors.ALL_CONFIGS) * len(seeds)} draws of 2,000 answers, "
               f"every index identical")
        return True


def check_fit(report):
    """The least-squares solve, against ``estimator.fit``.

    Compared on a tolerance rather than exactly, and it is the second place in
    the port that has to be: LAPACK reduces in an order that is neither written
    down nor stable across builds, so ``core/src/fit.rs`` solves by its own SVD.
    """
    path = binary("parity_fit")
    if path is None:
        return None

    import estimator

    # the datasets parity_fit.rs builds: one wide, one too narrow for a spline's
    # upper hinges, one falling with n so the guard drops every term
    def dataset(kind):
        i = np.arange(900)
        step = ((i * 7919) % 1000).astype(np.float64)
        n = 1.0 + step * (0.03 if kind == 1 else 12.9)
        x = np.log2(n)
        wobble = ((i * 37) % 13) / 13.0 - 0.5
        targets = (6.0 - 0.35 * x + wobble) if kind == 2 else (1.0 + 0.42 * x + wobble)
        return n, targets

    with tempfile.TemporaryDirectory() as out_dir:
        done = subprocess.run([path, ROOT, out_dir], capture_output=True, text=True)
        if done.returncode != 0:
            report((done.stdout + done.stderr).strip())
            return False

        worst, worst_where = 0.0, None
        for kind in range(3):
            n_eff, targets = dataset(kind)
            for name in sorted(estimator.FORMS):
                form = estimator.get_form(name)
                mine, terms = estimator.fit(n_eff, targets, form, n_max=12947.0)
                got = np.fromfile(os.path.join(out_dir, f"fit_{kind}_{name}.f64"),
                                  dtype="<f8")
                if int(got[-1]) != terms:
                    report(f"{name} on dataset {kind}: kept {int(got[-1])} terms "
                           f"against {terms}")
                    return False
                here = gap(got[:-1], mine)
                if here > worst:
                    worst, worst_where = here, f"{name}/dataset {kind}"
        report(f"{3 * len(estimator.FORMS)} fits, and every one kept the same terms")
        report(f"worst {worst:.3f} of tolerance" +
               (f", on {worst_where}" if worst_where else ""))
        if worst > 1.0:
            report("over tolerance")
            return False
        return True


def check_train(report):
    """The seed round of policy iteration, against ``train``.

    Everything up to and including the first fit: the answers drawn, the games
    played under greedy entropy, the states recorded and the parameters fitted
    to them.  Past the fit the two walk different paths: the least-squares
    solves differ in the last bits, and acting greedily on the result turns
    that into a different tie, rollout and next fit.  Even this round's holdout
    is played with the fitted parameters, so the comparison stops at the fit.
    """
    path = binary("train")
    if path is None:
        return None

    import estimator
    import train as training
    from priors import ALL_CONFIGS
    from wordle_data import load_data

    games = 200
    data = load_data()
    worst, worst_where = 0.0, None

    for config in ALL_CONFIGS:
        form = estimator.get_form(estimator.load_record(config)["form"])
        done = subprocess.run(
            [path, ROOT, "--prior", config.slug, "--form", form.name,
             "--games", str(games), "--seed", "0", "--max-iters", "0",
             "--exact-max", "0"],
            capture_output=True, text=True)
        if done.returncode != 0 or not done.stdout.strip():
            trouble = (done.stdout + done.stderr).strip().splitlines()
            report(trouble[-1] if trouble else "the trainer wrote nothing")
            return False
        theirs = json.loads(done.stdout)[0]

        _, meta = training.policy_iteration(
            config, data=data, form=form, n_games=games, seed=0, max_iters=0,
            solver_options={"exact_max": 0}, save=False, verbose=False,
            # explicitly, since the default is now the native trainer and this
            # is the side it is being compared against
            use_rust=False)

        mine_round, their_round = meta["history"][0], theirs["history"][0]
        if mine_round["mean_guesses"] != their_round["mean_guesses"]:
            report(f"{config.slug}: the seed round's means differ, "
                   f"{their_round['mean_guesses']} against "
                   f"{mine_round['mean_guesses']}")
            return False
        if mine_round["terms_used"] != their_round["terms_used"]:
            report(f"{config.slug}: the fit kept {their_round['terms_used']} terms "
                   f"against {mine_round['terms_used']}")
            return False
        here = gap(their_round["fitted_params"], mine_round["fitted_params"])
        if here > worst:
            worst, worst_where = here, config.slug

    report(f"{len(ALL_CONFIGS)} priors, {games} games each: the same answers "
           f"drawn, the same mean off the same rollouts, the same terms kept")
    report(f"worst {worst:.3f} of tolerance" +
           (f", on {worst_where}" if worst_where else ""))
    if worst > 1.0:
        report("over tolerance")
        return False
    return True


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
    ("the training draw",
     "every sampled answer matches np.random.default_rng, index for index",
     check_rng),
    ("the least-squares fit",
     "f fitted over three datasets and every form matches estimator.fit",
     check_fit),
    ("policy iteration",
     "the seed round agrees: the draw, the rollouts and the fit off them",
     check_train),
    ("the kernels",
     "the three scorers and the split helpers match kernels, either side of "
     "both thresholds",
     check_kernels),
    ("the endgame search",
     "solve and top return the identical guess and value, over three arms",
     check_exact),
    ("the solver dispatch",
     "the same branch answers, with the same guess, over six arms",
     check_solver),
    ("the endpoints",
     "the finder and play mode answer with what web.py answers with",
     check_api),
    ("the messages",
     "both implementations refuse a bad request in the same words",
     check_messages),
]


# the whole sweep, an acceptance test for the port: games may diverge and this
# fails only when the weighted mean moves.  A large share of the runtime, so asked for.
ACCEPTANCE = [
    ("every answer",
     "the whole sweep reproduces the mean and loss probability README quotes",
     check_every_answer),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--every-answer", action="store_true",
                        help="also play all 12,947 words through both sides")
    args = parser.parse_args()

    results = []
    for label, establishes, check in CHECKS + (ACCEPTANCE if args.every_answer
                                               else []):
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
