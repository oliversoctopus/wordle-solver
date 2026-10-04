"""f(n): a fitted estimate of how many guesses are still needed from a state
whose candidate set has effective size n = 2 ** H, with H the entropy of the
candidates' normalized weights.

Every form in ``FORMS`` is linear in its parameters and written as
f(n) = base + sum_i params[i] * term_i(n) with x = log2(n), base 0 or 1, and
floored at 1.  Parameters come from policy iteration (train.py) and are cached
under estimator_cache/: one active fit per prior, plus a per-form archive from
``wordle_solver.py compare``.
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from wordle_data import write_json

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "estimator_cache")
FORM_CACHE_DIR = os.path.join(CACHE_DIR, "forms")

# term codes, shared by the numpy and numba evaluations
TERM_CONST = 0    # 1
TERM_X = 1        # log2(n)
TERM_X2 = 2       # log2(n)^2
TERM_X3 = 3       # log2(n)^3
TERM_INVN = 4     # 1 - 1/n
TERM_LOG1PX = 5   # log2(1 + log2(n))
TERM_SQRTX = 6    # sqrt(log2(n))
TERM_HINGE = 7    # max(0, log2(n) - knot)

TERM_LABELS = {
    TERM_CONST: "1",
    TERM_X: "x",
    TERM_X2: "x^2",
    TERM_X3: "x^3",
    TERM_INVN: "1-1/n",
    TERM_LOG1PX: "log2(1+x)",
    TERM_SQRTX: "sqrt(x)",
}


@dataclass(frozen=True)
class Form:
    """A basis for f.  Terms are ordered so that dropping from the end removes
    the most flexible term first (see :func:`fit`)."""

    name: str
    codes: tuple
    knots: tuple
    base: float
    description: str

    @property
    def n_params(self):
        return len(self.codes)

    @property
    def term_labels(self):
        return tuple(
            f"(x-{k:g})+" if c == TERM_HINGE else TERM_LABELS[c]
            for c, k in zip(self.codes, self.knots)
        )

    def arrays(self):
        return (np.array(self.codes, dtype=np.int64),
                np.array(self.knots, dtype=np.float64),
                float(self.base))


def _form(name, terms, base, description):
    codes, knots = zip(*terms)
    return Form(name=name, codes=tuple(codes), knots=tuple(knots),
                base=float(base), description=description)


FORMS = {f.name: f for f in [
    _form("loglinear", [(TERM_CONST, 0.0), (TERM_X, 0.0)], 0.0,
          "a + b*log2(n)"),
    _form("anchored", [(TERM_X, 0.0), (TERM_INVN, 0.0), (TERM_X2, 0.0)], 1.0,
          "1 + a*x + b*(1-1/n) + c*x^2"),
    _form("anchored2", [(TERM_X, 0.0), (TERM_INVN, 0.0)], 1.0,
          "1 + a*x + b*(1-1/n)"),
    _form("quadratic", [(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_X2, 0.0)], 0.0,
          "a + b*x + c*x^2"),
    _form("cubic", [(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_X2, 0.0), (TERM_X3, 0.0)], 0.0,
          "a + b*x + c*x^2 + d*x^3"),
    _form("saturating", [(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_LOG1PX, 0.0)], 0.0,
          "a + b*x + c*log2(1+x)"),
    _form("sqrt", [(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_SQRTX, 0.0)], 0.0,
          "a + b*x + c*sqrt(x)"),
    _form("spline", [(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_HINGE, 2.0),
                     (TERM_HINGE, 4.0), (TERM_HINGE, 6.0), (TERM_HINGE, 8.0)], 0.0,
          "piecewise linear in x, knots at 2/4/6/8"),
    _form("invonly", [(TERM_CONST, 0.0), (TERM_INVN, 0.0)], 0.0,
          "a + b*(1-1/n)"),
]}

# the form every prior is fitted in (README *Which form is the default*)
DEFAULT_FORM = "spline"


def get_form(form):
    """Accept a Form, a form name, or None (meaning the default)."""
    if isinstance(form, Form):
        return form
    if form is None:
        return FORMS[DEFAULT_FORM]
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; known: {', '.join(sorted(FORMS))}")
    return FORMS[form]


@njit(cache=True)
def f_eval_x(x, params, codes, knots, base):
    """Scalar f, as a function of ``x = log2(n)``.

    x rather than n is the primitive because every caller reaches f through an
    entropy, so it has x in hand already; going via n would raise 2**x only for
    this to take the logarithm straight back off.  Kept in sync with
    :func:`design_matrix` by hand.
    """
    val = base
    for i in range(codes.shape[0]):
        code = codes[i]
        if code == 0:
            term = 1.0
        elif code == 1:
            term = x
        elif code == 2:
            term = x * x
        elif code == 3:
            term = x * x * x
        elif code == 4:
            term = 1.0 - 2.0 ** (-x)  # 1 - 1/n
        elif code == 5:
            term = np.log2(1.0 + x)
        elif code == 6:
            term = np.sqrt(x)
        else:
            term = x - knots[i]
            if term < 0.0:
                term = 0.0
        val += params[i] * term
    if val < 1.0:
        val = 1.0
    return val


@njit(cache=True)
def f_eval(n, params, codes, knots, base):
    """Scalar f(n)."""
    return f_eval_x(np.log2(n), params, codes, knots, base)


def design_matrix(n, form):
    """Basis functions evaluated at effective sizes ``n``."""
    form = get_form(form)
    n = np.asarray(n, dtype=np.float64)
    x = np.log2(n)
    columns = []
    for code, knot in zip(form.codes, form.knots):
        if code == TERM_CONST:
            columns.append(np.ones_like(x))
        elif code == TERM_X:
            columns.append(x)
        elif code == TERM_X2:
            columns.append(x * x)
        elif code == TERM_X3:
            columns.append(x * x * x)
        elif code == TERM_INVN:
            columns.append(1.0 - 1.0 / n)
        elif code == TERM_LOG1PX:
            columns.append(np.log2(1.0 + x))
        elif code == TERM_SQRTX:
            columns.append(np.sqrt(x))
        else:
            columns.append(np.maximum(0.0, x - knot))
    return np.column_stack(columns)


def f_value(n, params, form):
    """Vectorized f(n), matching :func:`f_eval`."""
    form = get_form(form)
    params = np.asarray(params, dtype=np.float64)
    return np.maximum(1.0, form.base + design_matrix(n, form) @ params)


def effective_size(weights):
    """2 ** (entropy of the normalized weights)."""
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0.0 or weights.size == 0:
        return 1.0
    p = weights / total
    p = p[p > 0.0]
    h = float(-(p * np.log2(p)).sum())
    return float(2.0 ** h)


def _lstsq(a, b, sample_weights=None):
    if sample_weights is not None:
        root = np.sqrt(np.asarray(sample_weights, dtype=np.float64))
        a = a * root[:, None]
        b = b * root
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    return solution


def _is_monotone(params, form, n_max):
    grid = np.geomspace(1.0, max(4.0, float(n_max)), 600)
    vals = f_value(grid, params, form)
    return bool(np.all(np.diff(vals) >= -1e-9))


def fit(n_eff, targets, form, sample_weights=None, n_max=None):
    """Least-squares fit of f to (effective size, guesses-still-needed) samples.

    f must not decrease anywhere in [1, n_max]: a value function that rewards a
    larger candidate set would let the search prefer uninformative guesses.  If
    the full basis violates that, terms are dropped from the end (the most
    flexible first) until it holds.  Returns the full-length parameter vector
    (dropped terms are zero) and how many terms survived.
    """
    form = get_form(form)
    n_eff = np.asarray(n_eff, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if n_eff.size < form.n_params + 1:
        raise ValueError("not enough samples to fit the estimator")

    basis = design_matrix(n_eff, form)
    residual = targets - form.base
    if n_max is None:
        n_max = float(n_eff.max())

    for k in range(form.n_params, 0, -1):
        params = np.zeros(form.n_params, dtype=np.float64)
        params[:k] = _lstsq(basis[:, :k], residual, sample_weights)
        if _is_monotone(params, form, n_max):
            return params, k

    # nothing monotone: fall back to a flat f at the mean, which is at least safe
    params = np.zeros(form.n_params, dtype=np.float64)
    if form.codes[0] == TERM_CONST:
        params[0] = max(1.0, float(targets.mean()))
    return params, 0


# --------------------------------------------------------------------- caching

def cache_path(config):
    """The active fit for a prior combination (one file per combination)."""
    return os.path.join(CACHE_DIR, f"params_{config.slug}.json")


def form_cache_path(config, form):
    """The archived fit for one (prior combination, form) pair."""
    return os.path.join(FORM_CACHE_DIR, f"{config.slug}__{get_form(form).name}.json")


def save_params(config, params, form, meta=None, active=True):
    form = get_form(form)
    path = cache_path(config) if active else form_cache_path(config, form)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "slug": config.slug,
        "prior": config.fingerprint(),
        "form": form.name,
        "form_expression": form.description,
        "terms": list(form.term_labels),
        "params": [float(p) for p in params],
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fitted_by": "python",
    }
    if meta:
        payload.update(meta)
    return write_json(path, payload)


def load_record(config, form=None):
    """Cached record: the active one if ``form`` is None, else the archived
    fit for that form.  None if there isn't one."""
    path = cache_path(config) if form is None else form_cache_path(config, form)
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def _record_params(record, form):
    params = np.array(record["params"], dtype=np.float64)
    if params.shape != (form.n_params,):
        return None
    return params


def _prior_matches(record, config):
    return record.get("prior") == config.fingerprint()


class NotFitted(RuntimeError):
    """There is no fit for this prior, so there is no f to play with."""


def _not_fitted(config, reason, form=None):
    """The refusal, with the command that ends it.

    Inventing an f here would hand back a policy nobody has measured, and it
    would not be saved, so the next run would invent it again.
    """
    parts = ["python wordle_solver.py train --this-config-only", *config.flags()]
    if form is not None:
        parts.append(f"--form {form.name}")
    return NotFitted(f"{reason} for {config.slug}. Fit and cache it with:\n"
                     f"    {' '.join(parts)}")


def active_form_name(config):
    """The form the active cache holds for this prior combination, or None if
    there is no usable fit and ``load_active`` would refuse to play."""
    record = load_record(config)
    name = record.get("form") if record else None
    return name if name in FORMS else None


def load_active(config):
    """``(params, form)`` from the active cache, or ``NotFitted``."""
    record = load_record(config)
    if record is None:
        raise _not_fitted(config, "No cached fit")
    name = record.get("form")
    if name not in FORMS:  # written before forms existed, so the term order is unknown
        raise _not_fitted(config, "The cached fit predates the form registry")
    form = get_form(name)
    params = _record_params(record, form)
    if params is None:
        raise _not_fitted(config, f"The cached fit does not fit the '{name}' form",
                          form)
    if not _prior_matches(record, config):
        raise _not_fitted(config, "The cached fit was made with other prior "
                                  "probabilities", form)
    return params, form


def load_form_params(config, form):
    """Parameters for a specific form: its archive, else the active cache if it
    happens to use that form, else ``NotFitted``."""
    form = get_form(form)
    for record in (load_record(config, form), load_record(config)):
        if record is None or record.get("form") != form.name:
            continue
        params = _record_params(record, form)
        if params is not None and _prior_matches(record, config):
            return params
    raise _not_fitted(config, f"No cached fit in the '{form.name}' form", form)


def describe(params, form):
    form = get_form(form)
    params = np.asarray(params, dtype=np.float64)
    samples = [2, 5, 20, 100, 500, 2000]
    terms = ", ".join(f"{label}: {val:+.4f}"
                      for label, val in zip(form.term_labels, params))
    preview = "  ".join(f"f({n})={v:.3f}"
                        for n, v in zip(samples, f_value(samples, params, form)))
    prefix = "1 + " if form.base else ""
    return f"f(n) = {prefix}[{terms}]   ({form.name})\n    {preview}"
