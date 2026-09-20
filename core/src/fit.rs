//! Least squares for `f`, and the monotonicity guard that trims a form.
//!
//! The basis is routinely rank deficient: a spline hinge beyond the widest
//! position seen is a column of zeros. numpy answers with an SVD and a
//! minimum-norm solution, landing that column on zero. This drops such
//! columns and solves the full-rank remainder by Householder QR. It agrees
//! with numpy closely and not bitwise, so the fit is compared on a tolerance
//! in `tests/rust_parity.py`.

use crate::estimator::{term_value, Form};
use crate::sum::pairwise_of;

/// numpy's `rcond=None`: a direction below `max(m, n) * eps * scale` carries
/// nothing, and a dead column becomes a zero coefficient.
const EPS: f64 = f64::EPSILON;

/// Least-squares solution of `a x = b`, `a` given as columns.
///
/// Columns carrying no information are dropped before the solve and come back
/// with a coefficient of exactly zero, so the system QR sees is full rank --
/// which a plain QR needs, its back-substitution dividing by a diagonal that a
/// dead column would leave at zero.
pub fn lstsq(columns: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = columns.len();
    if n == 0 {
        return Vec::new();
    }
    let m = columns[0].len();

    let norms: Vec<f64> = columns
        .iter()
        .map(|column| pairwise_of(m, |i| column[i] * column[i]).sqrt())
        .collect();
    let scale = norms.iter().cloned().fold(0.0, f64::max).max(1.0);
    let cutoff = scale * (m.max(n) as f64) * EPS;
    let live: Vec<usize> = (0..n).filter(|&j| norms[j] > cutoff).collect();

    let mut x = vec![0.0; n];
    if live.is_empty() {
        return x;
    }
    let reduced: Vec<Vec<f64>> = live.iter().map(|&j| columns[j].clone()).collect();
    for (slot, value) in live.iter().zip(qr_solve(reduced, b.to_vec())) {
        x[*slot] = value;
    }
    x
}

/// `a x = b` in the least-squares sense, by Householder QR with `a` full rank.
///
/// The reflections are applied to `b` as they are formed rather than
/// accumulated into a `q`, so what is left to do at the end is one
/// back-substitution against the triangle in the first rows.
fn qr_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = a.len();
    let m = a[0].len();

    for j in 0..k {
        let norm = pairwise_of(m - j, |t| a[j][t + j] * a[j][t + j]).sqrt();
        if norm == 0.0 {
            continue;
        }
        // reflect onto the axis pointing away from the pivot, which is the
        // choice that avoids cancelling the leading entry against itself
        let alpha = if a[j][j] > 0.0 { -norm } else { norm };
        let mut v = vec![0.0; m];
        v[j..m].copy_from_slice(&a[j][j..m]);
        v[j] -= alpha;
        let vv = pairwise_of(m - j, |t| v[t + j] * v[t + j]);
        if vv == 0.0 {
            continue;
        }

        for column in a.iter_mut().take(k).skip(j) {
            let dot = pairwise_of(m - j, |t| v[t + j] * column[t + j]);
            let factor = 2.0 * dot / vv;
            for i in j..m {
                column[i] -= factor * v[i];
            }
        }
        let dot = pairwise_of(m - j, |t| v[t + j] * b[t + j]);
        let factor = 2.0 * dot / vv;
        for i in j..m {
            b[i] -= factor * v[i];
        }
    }

    let mut x = vec![0.0; k];
    for j in (0..k).rev() {
        if a[j][j] == 0.0 {
            continue;
        }
        let mut total = b[j];
        for c in (j + 1)..k {
            total -= a[c][j] * x[c];
        }
        x[j] = total / a[j][j];
    }
    x
}

/// `np.geomspace(1.0, max(4.0, n_max), 600)`, the grid monotonicity is read on.
fn geomspace(n_max: f64, count: usize) -> Vec<f64> {
    let stop = n_max.max(4.0).log10();
    let step = stop / (count as f64 - 1.0);
    let mut grid: Vec<f64> = (0..count).map(|i| 10f64.powf(step * i as f64)).collect();
    grid[0] = 1.0;
    grid[count - 1] = n_max.max(4.0);
    grid
}

/// Does `f` never decrease across `[1, n_max]`?
///
/// A value function that rewarded a larger candidate set would let the search
/// prefer a guess that tells it nothing, so a form that cannot hold this is
/// trimmed rather than played.
pub fn is_monotone(params: &[f64], form: &Form, n_max: f64) -> bool {
    let grid = geomspace(n_max, 600);
    let mut previous = f64::NEG_INFINITY;
    for n in grid {
        let value = crate::estimator::f_eval(n, params, form);
        if value - previous < -1e-9 {
            return false;
        }
        previous = value;
    }
    true
}

/// Fit `f` to `(effective size, guesses still needed)` pairs.
///
/// Terms are dropped from the end, most flexible first, until what is left is
/// monotone; the returned vector is full length with the dropped terms zero,
/// alongside how many survived. With nothing monotone available it falls back
/// to a flat `f` at the mean, which is at least safe.
pub fn fit(n_eff: &[f64], targets: &[f64], form: &Form, n_max: f64) -> (Vec<f64>, usize) {
    let n_params = form.codes.len();
    let residual: Vec<f64> = targets.iter().map(|t| t - form.base).collect();
    let xs: Vec<f64> = n_eff.iter().map(|n| n.log2()).collect();

    let full: Vec<Vec<f64>> = (0..n_params)
        .map(|i| {
            xs.iter()
                .map(|&x| term_value(form.codes[i], form.knots[i], x))
                .collect()
        })
        .collect();

    for k in (1..=n_params).rev() {
        let solution = lstsq(&full[..k], &residual);
        let mut params = vec![0.0; n_params];
        params[..k].copy_from_slice(&solution);
        if is_monotone(&params, form, n_max) {
            return (params, k);
        }
    }

    let mut params = vec![0.0; n_params];
    if form.codes.first() == Some(&crate::estimator::TERM_CONST) {
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        params[0] = if mean > 1.0 { mean } else { 1.0 };
    }
    (params, 0)
}
