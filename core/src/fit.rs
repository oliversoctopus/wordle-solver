//! Least squares for `f`, and the monotonicity guard that trims a form.
//!
//! `estimator.fit` is one `np.linalg.lstsq` per candidate term count, and that
//! is an SVD with a minimum-norm solution when the basis is rank deficient --
//! which it routinely is, since a spline hinge beyond the widest position seen
//! is a column of zeros. A plain QR would return nonsense there rather than the
//! zero coefficient numpy returns, so this solves by SVD too: one-sided Jacobi,
//! which is accurate on the tall thin matrices a fit actually has and short
//! enough to read.
//!
//! It agrees with numpy to about 1e-12 rather than to the last bit. LAPACK
//! reduces in an order neither documented nor stable across builds, so this is
//! the second place in the port compared on a tolerance, the letter odds being
//! the first.

use crate::estimator::{term_value, Form};

/// numpy's `rcond=None`: singular values below `max(m, n) * eps * s_max` are
/// treated as zero, which is what turns a dead column into a zero coefficient.
const EPS: f64 = f64::EPSILON;

/// Columns of `a` orthogonalized in place by Givens rotations, accumulating the
/// right factor. Returns `(singular values, v)` with `a` left holding `u * s`.
fn jacobi_svd(a: &mut [Vec<f64>], v: &mut [Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    let m = if n == 0 { 0 } else { a[0].len() };
    for (j, row) in v.iter_mut().enumerate() {
        for (i, slot) in row.iter_mut().enumerate() {
            *slot = if i == j { 1.0 } else { 0.0 };
        }
    }

    // 30 sweeps is far past convergence for the six columns a form can have;
    // the loop exits as soon as a whole sweep rotates nothing.
    for _ in 0..30 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    alpha += a[p][i] * a[p][i];
                    beta += a[q][i] * a[q][i];
                    gamma += a[p][i] * a[q][i];
                }
                if gamma == 0.0 || gamma.abs() <= EPS * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (ap, aq) = (a[p][i], a[q][i]);
                    a[p][i] = c * ap - s * aq;
                    a[q][i] = s * ap + c * aq;
                }
                for i in 0..n {
                    let (vp, vq) = (v[p][i], v[q][i]);
                    v[p][i] = c * vp - s * vq;
                    v[q][i] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    a.iter()
        .map(|column| column.iter().map(|z| z * z).sum::<f64>().sqrt())
        .collect()
}

/// Minimum-norm least-squares solution of `a x = b`, `a` given as columns.
///
/// Mirrors `np.linalg.lstsq(a, b, rcond=None)`: singular directions below the
/// cutoff contribute nothing, so a column carrying no information comes back
/// with a coefficient of exactly zero instead of an arbitrary large one.
pub fn lstsq(columns: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = columns.len();
    if n == 0 {
        return Vec::new();
    }
    let m = columns[0].len();
    let mut a: Vec<Vec<f64>> = columns.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    let sigma = jacobi_svd(&mut a, &mut v);

    let s_max = sigma.iter().cloned().fold(0.0, f64::max);
    let cutoff = s_max * (m.max(n) as f64) * EPS;

    let mut x = vec![0.0; n];
    for j in 0..n {
        if sigma[j] <= cutoff || sigma[j] == 0.0 {
            continue;
        }
        // a[j] is u_j * sigma_j, so this is (u_j . b) / sigma_j in one pass
        let mut projection = 0.0;
        for i in 0..m {
            projection += a[j][i] * b[i];
        }
        let coefficient = projection / (sigma[j] * sigma[j]);
        for (i, slot) in x.iter_mut().enumerate() {
            *slot += coefficient * v[j][i];
        }
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
