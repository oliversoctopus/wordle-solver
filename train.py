"""Policy iteration: fit f, act greedily on it, refit, until the fit settles.

One round plays a batch of games with the current policy, records each
position's effective size and the guesses the policy needed from there, and
least-squares fits f to those pairs.  Round 0 is seeded with greedy entropy.
Training games have no deadline, so the targets are unbiased estimates of
expected guesses; evaluation applies the six-guess limit.
"""

import time

import numpy as np

import estimator
import rust_train
from priors import ALL_CONFIGS
from solver import DEFAULT_DEADLINE, ENTROPY, EXPECTED, WordleSolver, sample_answers
from wordle_data import load_data

DEFAULT_GAMES = 2000
DEFAULT_MAX_ITERS = 8
DEFAULT_TOL = 1e-3
# rounds without an improvement in mean guesses before giving up on convergence
DEFAULT_PATIENCE = 3

# f is only ever queried on buckets of three or more candidates (one and two are
# handled exactly), so it is only fitted on those.
MIN_COUNT_FOR_FIT = 3


def collect_rollouts(solver, answers, max_guesses=12, deadline=None, progress_every=0):
    """Play every answer and return the (state, guesses-still-needed) samples."""
    n_eff, counts, targets, lengths = [], [], [], []
    for i, answer in enumerate(answers):
        guesses, states = solver.play(int(answer), max_guesses=max_guesses, deadline=deadline)
        total = len(guesses)
        lengths.append(total)
        # states[0] is the full word list, identical in every game and never an
        # argument to f, so it is not used as a fitting sample
        for depth, (size, count) in enumerate(states[1:], start=1):
            n_eff.append(size)
            counts.append(count)
            targets.append(total - depth)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(answers)} games, running mean {np.mean(lengths):.4f}")
    return (
        np.array(n_eff, dtype=np.float64),
        np.array(counts, dtype=np.int64),
        np.array(targets, dtype=np.float64),
        np.array(lengths, dtype=np.int64),
    )


def fit_from_rollouts(n_eff, counts, targets, form, n_max=None):
    """``n_max`` is the largest effective size f must stay monotone up to; pass
    the full word count so a bucket bigger than anything seen in training can
    never be scored as *cheaper* than a smaller one."""
    form = estimator.get_form(form)
    keep = counts >= MIN_COUNT_FOR_FIT
    if keep.sum() < form.n_params + 1:
        raise ValueError("too few multi-candidate states to fit the estimator")
    n_kept, t_kept = n_eff[keep], targets[keep]
    if n_max is None:
        n_max = float(n_kept.max())
    return estimator.fit(n_kept, t_kept, form, n_max=n_max)


def fit_rmse(n_eff, counts, targets, params, form):
    """How far f is from the guesses actually needed, on the states f is used on."""
    keep = counts >= MIN_COUNT_FOR_FIT
    if not keep.any():
        return float("nan")
    predicted = estimator.f_value(n_eff[keep], params, form)
    return float(np.sqrt(np.mean((predicted - targets[keep]) ** 2)))


def policy_iteration(config, data=None, form=None, n_games=DEFAULT_GAMES,
                     max_iters=DEFAULT_MAX_ITERS, tol=DEFAULT_TOL, seed=0,
                     patience=DEFAULT_PATIENCE, holdout_games=None,
                     solver_options=None, verbose=True, save=True, active=True,
                     use_rust=True):
    """Fit f for one prior combination and (optionally) cache it.

    Exact convergence is not guaranteed: the policy is greedy, so a small
    change in f can flip a guess, and the evaluation is a finite sample.  The
    loop therefore also stops when the measured mean stops improving, and what
    gets cached is the parameter set whose *policy* measured best, not simply
    the last fit.
    """
    data = data if data is not None else load_data()
    form = estimator.get_form(form)
    options = dict(solver_options or {})
    if verbose:
        print(f"\n=== {config.slug} / {form.name}: {config.describe()} ===")
        print(f"  form: {form.description}")

    if use_rust:
        handed = rust_train.policy_iteration(
            config, form, n_games, seed, max_iters, tol, patience,
            holdout_games, options, verbose=verbose)
        if handed is not None:
            params, meta = handed
            if save:
                path = estimator.save_params(config, params, form, meta, active=active)
                if verbose:
                    print(f"  saved -> {path}")
            return params, meta

    seed_solver = WordleSolver(data=data, config=config, mode=ENTROPY, **options)
    answers = sample_answers(seed_solver.weights, n_games, seed=seed)

    solver = None
    params = None        # latest fit, i.e. the policy used by the next round
    history = []
    converged = False
    best_params = None
    best_mean = float("inf")
    stale_rounds = 0

    for iteration in range(max_iters + 1):
        current = seed_solver if iteration == 0 else solver
        label = "entropy seed" if iteration == 0 else f"iter {iteration}"

        started = time.perf_counter()
        n_eff, counts, targets, lengths = collect_rollouts(current, answers)
        elapsed = time.perf_counter() - started

        mean_guesses = float(lengths.mean())
        new_params, n_terms = fit_from_rollouts(n_eff, counts, targets, form,
                                                n_max=float(data.n_words))
        history.append({
            "label": label,
            "mean_guesses": mean_guesses,
            "policy_params": None if params is None else [float(p) for p in params],
            "fitted_params": [float(p) for p in new_params],
            "terms_used": n_terms,
            "seconds": elapsed,
        })

        if verbose:
            preview = "  ".join(
                f"f({n})={v:.3f}"
                for n, v in zip((5, 50, 500), estimator.f_value([5, 50, 500], new_params, form))
            )
            dropped = "" if n_terms == form.n_params else f"  [{n_terms}/{form.n_params} terms]"
            print(f"  {label:>12}: mean {mean_guesses:.4f} guesses   {preview}{dropped}"
                  f"   ({elapsed:.1f}s)")

        if params is not None:
            # `params` is what produced these rollouts, so this is its measured value
            if mean_guesses < best_mean - 1e-12:
                best_mean, best_params, stale_rounds = mean_guesses, params, 0
            else:
                stale_rounds += 1

            delta = float(np.abs(new_params - params).max())
            if delta < tol:
                converged = True
                if verbose:
                    print(f"  converged (max parameter change {delta:.2e} < {tol:g})")
                break
        if stale_rounds >= patience:
            if verbose:
                print(f"  stopping: no improvement in {patience} rounds "
                      f"(best mean {best_mean:.4f})")
            break

        params = new_params
        if solver is None:
            solver = WordleSolver(data=data, config=config, mode=EXPECTED,
                                  params=params, form=form, **options)
        else:
            solver.set_params(params)

    if best_params is None:  # only the seed round ran
        best_params = params
        best_mean = float("nan")
    params = best_params
    terms_used = sum(1 for p in params if p != 0.0)

    final = WordleSolver(data=data, config=config, mode=EXPECTED, params=params,
                         form=form, **options)
    opening, opening_score = final.best_guess(final.all_candidates())

    # how well f predicts the guesses its own policy needs
    n_eff, counts, targets, lengths = collect_rollouts(final, answers)
    rmse = fit_rmse(n_eff, counts, targets, params, form)

    holdout = evaluate(final, n_games=holdout_games or max(200, n_games // 2),
                       seed=seed + 10_000, verbose=False)
    if verbose:
        print(f"  fit RMSE {rmse:.4f} guesses; holdout ({holdout['games']} unseen answers, "
              f"{DEFAULT_DEADLINE} guess limit): solved {holdout['solve_rate'] * 100:.1f}%, "
              f"mean {holdout['mean_guesses_penalized']:.4f}")

    meta = {
        "n_games": int(n_games),
        "seed": int(seed),
        "iterations": len(history),
        "converged": converged,
        "terms_used": terms_used,
        "entropy_baseline_mean_guesses": history[0]["mean_guesses"],
        "best_train_mean_guesses": best_mean,
        "mean_guesses_last_eval": history[-1]["mean_guesses"],
        "fit_rmse": rmse,
        "holdout": holdout,
        "opening_word": data.all_words[opening],
        "opening_expected_guesses": float(opening_score),
        "history": history,
    }
    if save:
        path = estimator.save_params(config, params, form, meta, active=active)
        if verbose:
            print(f"  saved -> {path}")
    if verbose:
        print(f"  opening word: {data.all_words[opening]} (E = {opening_score:.4f} guesses)")
        print("  " + estimator.describe(params, form).replace("\n", "\n  "))
    return params, meta


def train_all(configs=None, **kwargs):
    """Fit and cache one parameter set per prior combination (4 of them).

    Rollouts use whatever solver options are passed, exact endgame search
    included, so f is fitted to the returns of the policy that will actually
    be played.
    """
    data = kwargs.pop("data", None) or load_data()
    configs = configs if configs is not None else ALL_CONFIGS
    results = {}
    for config in configs:
        params, meta = policy_iteration(config, data=data, **kwargs)
        results[config.slug] = (params, meta)
    return results


def evaluate(solver, n_games=200, seed=12345, max_guesses=DEFAULT_DEADLINE,
             deadline=True, verbose=True):
    """Play games under the real guess limit and report the distribution.

    ``deadline`` (the default) makes the final guess a possible answer.
    """
    answers = sample_answers(solver.weights, n_games, seed=seed)
    limit = max_guesses if deadline else None
    lengths = []
    for answer in answers:
        guesses, _ = solver.play(int(answer), max_guesses=max_guesses, deadline=limit)
        # play() stops after max_guesses whether or not it has won
        solved = guesses[-1] == int(answer)
        lengths.append(len(guesses) if solved else max_guesses + 1)
    lengths = np.array(lengths)
    solved = lengths <= max_guesses
    stats = {
        "games": int(n_games),
        "mean_guesses_solved": float(lengths[solved].mean()) if solved.any() else float("nan"),
        # failures counted as max_guesses + 1, so policies stay comparable
        "mean_guesses_penalized": float(lengths.mean()),
        "solve_rate": float(solved.mean()),
        "distribution": {int(k): int((lengths == k).sum()) for k in np.unique(lengths)},
    }
    if verbose:
        print(f"  mode={solver.mode}  {solver.config.describe()}")
        print(f"  solved {stats['solve_rate'] * 100:.1f}% of {n_games} games, "
              f"mean {stats['mean_guesses_solved']:.4f} guesses when solved "
              f"({stats['mean_guesses_penalized']:.4f} counting a failure as "
              f"{max_guesses + 1})")
        for k in sorted(stats["distribution"]):
            tag = f"{k}" if k <= max_guesses else f"{max_guesses}+ (failed)"
            print(f"    {tag:>12}: {stats['distribution'][k]}")
    return stats


def compare_forms(config, forms=None, data=None, n_games=DEFAULT_GAMES,
                  holdout_games=2000, seed=0, solver_options=None,
                  verbose=True, **kwargs):
    """Run policy iteration once per functional form and tabulate the results.

    Every form sees the same training answers and the same holdout answers, so
    the comparison is paired.
    """
    data = data if data is not None else load_data()
    forms = [estimator.get_form(f) for f in (forms or estimator.FORMS)]

    rows = []
    baseline = WordleSolver(data=data, config=config, mode=ENTROPY,
                            **(solver_options or {}))
    stats = evaluate(baseline, n_games=holdout_games, seed=seed + 10_000, verbose=False)
    rows.append({
        "form": "(greedy entropy)",
        "expression": "no estimator",
        "terms": "-",
        "rmse": float("nan"),
        "opening": data.all_words[baseline.best_guess(baseline.all_candidates())[0]],
        "holdout_mean": stats["mean_guesses_penalized"],
        "holdout_solved_mean": stats["mean_guesses_solved"],
        "solve_rate": stats["solve_rate"],
    })

    for form in forms:
        params, meta = policy_iteration(
            config, data=data, form=form, n_games=n_games, seed=seed,
            holdout_games=holdout_games, solver_options=solver_options,
            verbose=verbose, active=False, **kwargs)
        rows.append({
            "form": form.name,
            "expression": form.description,
            "terms": f"{meta['terms_used']}/{form.n_params}",
            "rmse": meta["fit_rmse"],
            "opening": meta["opening_word"],
            "holdout_mean": meta["holdout"]["mean_guesses_penalized"],
            "holdout_solved_mean": meta["holdout"]["mean_guesses_solved"],
            "solve_rate": meta["holdout"]["solve_rate"],
        })

    rows.sort(key=lambda r: r["holdout_mean"])
    if verbose:
        print_comparison(config, rows, holdout_games)
    return rows


def print_comparison(config, rows, holdout_games):
    print(f"\n=== forms compared on {config.slug}: {config.describe()} ===")
    print(f"    {holdout_games} unseen answers, {DEFAULT_DEADLINE} guess limit, "
          f"failures counted as {DEFAULT_DEADLINE + 1}\n")
    header = f"{'form':<18}{'expression':<34}{'terms':>7}{'RMSE':>8}{'open':>8}{'mean':>9}{'solved%':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        rmse = "     -  " if row["rmse"] != row["rmse"] else f"{row['rmse']:8.4f}"
        print(f"{row['form']:<18}{row['expression']:<34}{row['terms']:>7}{rmse}"
              f"{row['opening']:>8}{row['holdout_mean']:>9.4f}"
              f"{row['solve_rate'] * 100:>9.2f}")
