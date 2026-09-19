"""Hands a fit to the native trainer where it can reproduce the settings.

The Rust does the work and the Python keeps the reporting; a setting the flags
cannot express declines, and `estimator.save_params` still writes the cache.
The two least-squares solves differ in their last bits and policy iteration
amplifies that, so the native fit is a different valid fit and not the
Python's.  README *Fitting in Rust* has the measurement.
"""

import json
import os
import subprocess
import sys

import numpy as np

import estimator
from priors import DEFAULT_NONCURATED_PROB, DEFAULT_REPEAT_PROB, ALL_CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "target", "release")


def binary():
    """Path to the native trainer, or None where it has not been built."""
    if os.environ.get("WORDLE_NO_RUST") == "1":
        return None
    path = os.path.join(TARGET, "train" + (".exe" if os.name == "nt" else ""))
    return path if os.path.exists(path) else None


def _flags(config, form, n_games, seed, max_iters, tol, patience,
           holdout_games, solver_options):
    """Flags reproducing this fit, or None where they cannot.

    Strict, and strictest about the prior: the binary builds its
    own weights from a slug, so a configuration the slug does not fully
    describe would fit a different prior under the right name.
    """
    options = dict(solver_options or {})
    if options.get("hard") or options.get("lookahead"):
        return None
    if options.get("win_bonus"):
        return None
    # the caps are read from the environment by both, so they already agree
    from exact import DEFAULT_MAX_BRANCH, DEFAULT_ROOT_BRANCH
    branch = options.get("exact_max_branch", DEFAULT_MAX_BRANCH)
    if branch is not None and branch != DEFAULT_MAX_BRANCH:
        return None
    if options.get("exact_root_branch", DEFAULT_ROOT_BRANCH) != DEFAULT_ROOT_BRANCH:
        return None
    if config.repeat_prob != DEFAULT_REPEAT_PROB:
        return None
    if config.noncurated_prob != DEFAULT_NONCURATED_PROB:
        return None
    if not any(c.slug == config.slug for c in ALL_CONFIGS):
        return None

    flags = ["--prior", config.slug, "--form", estimator.get_form(form).name,
             "--games", str(int(n_games)), "--seed", str(int(seed)),
             "--max-iters", str(int(max_iters)), "--tol", repr(float(tol)),
             "--patience", str(int(patience))]
    if holdout_games is not None:
        flags += ["--holdout-games", str(int(holdout_games))]
    flags += ["--exact-max", str(int(options.get("exact_max", 60)))]
    flags += ["--loss-penalty", repr(float(options.get("loss_penalty", 5.0)))]
    if not options.get("use_exact", True):
        flags.append("--no-exact")
    return flags


def policy_iteration(config, form, n_games, seed, max_iters, tol, patience,
                     holdout_games, solver_options, verbose=True):
    """``(params, meta)`` from the native trainer, or None to fit here."""
    path = binary()
    if path is None:
        return None
    flags = _flags(config, form, n_games, seed, max_iters, tol, patience,
                   holdout_games, solver_options)
    if flags is None:
        if verbose:
            print("  (native trainer declined: a setting it has no flag for)",
                  file=sys.stderr)
        return None

    done = subprocess.run([path, HERE] + flags, capture_output=True, text=True)
    if done.returncode != 0 or not done.stdout.strip():
        message = (done.stdout + done.stderr).strip().splitlines()
        print(f"  (native trainer declined: {message[-1] if message else '?'};"
              f" fitting here instead)", file=sys.stderr)
        return None
    if verbose and done.stderr:
        print(done.stderr.rstrip())

    outcome = json.loads(done.stdout)[0]
    params = np.array(outcome["params"], dtype=np.float64)
    meta = {
        "n_games": int(n_games),
        "seed": int(seed),
        "iterations": outcome["iterations"],
        "converged": outcome["converged"],
        "terms_used": outcome["terms_used"],
        "entropy_baseline_mean_guesses": outcome["entropy_baseline_mean_guesses"],
        # the loop writes NaN where only the seed round ran, and JSON has no NaN
        "best_train_mean_guesses": _nan(outcome["best_train_mean_guesses"]),
        "mean_guesses_last_eval": outcome["mean_guesses_last_eval"],
        "fit_rmse": outcome["fit_rmse"],
        "holdout": _holdout(outcome["holdout"]),
        "opening_word": outcome["opening_word"],
        "opening_expected_guesses": outcome["opening_expected_guesses"],
        "history": outcome["history"],
        "fitted_by": "rust",
    }
    return params, meta


def _nan(value):
    return float("nan") if value is None else value


def _holdout(record):
    return {
        "games": record["games"],
        "mean_guesses_solved": _nan(record["mean_guesses_solved"]),
        "mean_guesses_penalized": record["mean_guesses_penalized"],
        "solve_rate": record["solve_rate"],
        "distribution": {int(k): v for k, v in record["distribution"].items()},
    }
