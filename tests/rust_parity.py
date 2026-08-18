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

# a weight of ~1e-4 against f64's ~1e-16 leaves eleven orders of headroom, and
# the two implementations only ever differ in the order they add things up
TOLERANCE = 1e-12


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
            gap = float(np.max(np.abs(rust - mine) / np.maximum(mine, 1e-300)))
            report(f"{config.slug}: worst relative gap {gap:.3e}")
            if gap > worst:
                worst, worst_slug = gap, config.slug

        if worst > TOLERANCE:
            report(f"worst is {worst:.3e} on {worst_slug}, over {TOLERANCE:.0e}")
            return False
        report(f"all {len(priors.ALL_CONFIGS)} within {TOLERANCE:.0e}")
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
            gap = float(np.max(np.abs(got - mine) / np.maximum(np.abs(mine), 1e-300)))
            if gap > worst:
                worst, worst_where = gap, name
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

        report(f"worst relative gap {worst:.3e}" +
               (f" on {worst_where}" if worst_where else ""))
        if worst > TOLERANCE:
            report(f"over {TOLERANCE:.0e}")
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
