"""Handing a whole workload to the native binaries, where they can take it.

The boundary that pays is a whole sweep: one subprocess, settings on the
command line, lengths back as a little-endian f64 file.  Every entry point
answers ``None`` where a binary is not built, a setting has no flag, or a run
fails, and the Python plays instead.  ``WORDLE_NO_RUST=1`` turns the dispatch
off, and the two are compared under it.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "target", "release")


def binary(name):
    """Path to a native binary, or None where it has not been built."""
    if os.environ.get("WORDLE_NO_RUST") == "1":
        return None
    path = os.path.join(TARGET, name + (".exe" if os.name == "nt" else ""))
    return path if os.path.exists(path) else None


def _settings(solver, max_guesses):
    """Flags reproducing this solver, or None where they cannot.

    Strict: anything not listed here is a setting the binary has
    no flag for, and a sweep run without it would answer a different question.
    """
    from exact import DEFAULT_MAX_BRANCH, DEFAULT_ROOT_BRANCH
    from solver import ENTROPY, EXPECTED

    # the native sweep plays to six and has no flag for anything else
    if max_guesses != 6:
        return None
    # lookahead has no counterpart on the native side at all
    if solver.lookahead:
        return None
    # nor does a fixed answer list, which replaces the prior's weights
    if solver.answers is not None:
        return None
    # the caps are read from the environment by both, so they already agree
    if solver.exact.max_branch != DEFAULT_MAX_BRANCH:
        return None
    if solver.exact.root_branch != DEFAULT_ROOT_BRANCH:
        return None

    flags = ["--prior", solver.config.slug]
    flags += ["--mode", "entropy" if solver.mode == ENTROPY else "expected"]
    if solver.hard:
        flags.append("--hard")
    if not solver.use_exact:
        flags.append("--no-exact")
    if solver.win_bonus:
        flags += ["--win-bonus", repr(float(solver.win_bonus))]
    flags += ["--exact-max", str(int(solver.exact_max))]
    flags += ["--loss-penalty", repr(float(solver.exact.loss_penalty))]
    # the fit is passed outright rather than looked up, since a script may be
    # replaying an archived one the cache no longer holds
    if solver.mode == EXPECTED:
        flags += ["--form", solver.form.name]
        flags += ["--params", ",".join(repr(float(p)) for p in solver.params)]
    return flags


def every_answer(solver, max_guesses):
    """Lengths for every answer under this solver, or None to do it in Python."""
    path = binary("every_answer")
    if path is None:
        return None
    flags = _settings(solver, max_guesses)
    if flags is None:
        return None

    with tempfile.TemporaryDirectory() as out_dir:
        dump = os.path.join(out_dir, "lengths.f64")
        done = subprocess.run([path, ROOT, "--json", "--dump", dump] + flags,
                              capture_output=True, text=True)
        if done.returncode != 0 or not os.path.exists(dump):
            message = (done.stdout + done.stderr).strip().splitlines()
            print(f"  (native sweep declined: {message[-1] if message else '?'};"
                  f" running it here instead)", file=sys.stderr)
            return None
        lengths = np.fromfile(dump, dtype="<f8")

    if lengths.shape != solver.weights.shape:
        return None
    return lengths
