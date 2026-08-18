//! `f(n)`, the estimated guesses still needed from a position of effective
//! size `n`, and the basis functions it is fitted in.
//!
//! Only the runtime half is here. Fitting stays in Python, where
//! `estimator.fit` reaches for a real least-squares solve and runs a handful of
//! times per training run; what crosses over is evaluating a fit that already
//! exists. The parameters arrive as JSON the Python wrote.

use std::error::Error;

use crate::json;
use crate::sum::pairwise;

// term codes, shared with the Python by value
pub const TERM_CONST: i64 = 0; // 1
pub const TERM_X: i64 = 1; // log2(n)
pub const TERM_X2: i64 = 2; // log2(n)^2
pub const TERM_X3: i64 = 3; // log2(n)^3
pub const TERM_INVN: i64 = 4; // 1 - 1/n
pub const TERM_LOG1PX: i64 = 5; // log2(1 + log2(n))
pub const TERM_SQRTX: i64 = 6; // sqrt(log2(n))
pub const TERM_HINGE: i64 = 7; // max(0, log2(n) - knot)

/// A basis for `f`. Terms are ordered so that dropping from the end removes the
/// most flexible first, which is how the monotonicity guard trims a form; a
/// dropped term arrives with a coefficient of exactly zero, so nothing here has
/// to know it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct Form {
    pub name: &'static str,
    pub codes: Vec<i64>,
    pub knots: Vec<f64>,
    pub base: f64,
    pub description: &'static str,
}

fn form(name: &'static str, terms: &[(i64, f64)], base: f64, description: &'static str) -> Form {
    Form {
        name,
        codes: terms.iter().map(|t| t.0).collect(),
        knots: terms.iter().map(|t| t.1).collect(),
        base,
        description,
    }
}

/// Every basis `estimator.FORMS` holds, in the same order.
pub fn forms() -> Vec<Form> {
    vec![
        form(
            "loglinear",
            &[(TERM_CONST, 0.0), (TERM_X, 0.0)],
            0.0,
            "a + b*log2(n)",
        ),
        form(
            "anchored",
            &[(TERM_X, 0.0), (TERM_INVN, 0.0), (TERM_X2, 0.0)],
            1.0,
            "1 + a*x + b*(1-1/n) + c*x^2",
        ),
        form(
            "anchored2",
            &[(TERM_X, 0.0), (TERM_INVN, 0.0)],
            1.0,
            "1 + a*x + b*(1-1/n)",
        ),
        form(
            "quadratic",
            &[(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_X2, 0.0)],
            0.0,
            "a + b*x + c*x^2",
        ),
        form(
            "cubic",
            &[
                (TERM_CONST, 0.0),
                (TERM_X, 0.0),
                (TERM_X2, 0.0),
                (TERM_X3, 0.0),
            ],
            0.0,
            "a + b*x + c*x^2 + d*x^3",
        ),
        form(
            "saturating",
            &[(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_LOG1PX, 0.0)],
            0.0,
            "a + b*x + c*log2(1+x)",
        ),
        form(
            "sqrt",
            &[(TERM_CONST, 0.0), (TERM_X, 0.0), (TERM_SQRTX, 0.0)],
            0.0,
            "a + b*x + c*sqrt(x)",
        ),
        form(
            "spline",
            &[
                (TERM_CONST, 0.0),
                (TERM_X, 0.0),
                (TERM_HINGE, 2.0),
                (TERM_HINGE, 4.0),
                (TERM_HINGE, 6.0),
                (TERM_HINGE, 8.0),
            ],
            0.0,
            "piecewise linear in x, knots at 2/4/6/8",
        ),
        form(
            "invonly",
            &[(TERM_CONST, 0.0), (TERM_INVN, 0.0)],
            0.0,
            "a + b*(1-1/n)",
        ),
    ]
}

/// Fitted when no form is asked for.
pub const DEFAULT_FORM: &str = "saturating";

pub fn get_form(name: &str) -> Option<Form> {
    forms().into_iter().find(|f| f.name == name)
}

/// Scalar `f`, as a function of `x = log2(n)`.
///
/// `x` rather than `n` is the primitive because every caller reaches `f`
/// through an entropy and so has `x` in hand; going via `n` would raise `2^x`
/// only for this to take the logarithm straight back off.
#[inline]
pub fn f_eval_x(x: f64, params: &[f64], form: &Form) -> f64 {
    let mut val = form.base;
    for (i, &code) in form.codes.iter().enumerate() {
        let term = match code {
            TERM_CONST => 1.0,
            TERM_X => x,
            TERM_X2 => x * x,
            TERM_X3 => x * x * x,
            TERM_INVN => 1.0 - 2.0f64.powf(-x), // 1 - 1/n
            TERM_LOG1PX => (1.0 + x).log2(),
            TERM_SQRTX => x.sqrt(),
            _ => {
                let hinge = x - form.knots[i];
                if hinge < 0.0 {
                    0.0
                } else {
                    hinge
                }
            }
        };
        val += params[i] * term;
    }
    if val < 1.0 {
        1.0
    } else {
        val
    }
}

/// Scalar `f(n)`.
#[inline]
pub fn f_eval(n: f64, params: &[f64], form: &Form) -> f64 {
    f_eval_x(n.log2(), params, form)
}

/// `2 ** (entropy of the normalized weights)`: how many equally likely words
/// this set would be as hard as.
pub fn effective_size(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 1.0;
    }
    let total = pairwise(weights);
    if total <= 0.0 {
        return 1.0;
    }
    let terms: Vec<f64> = weights
        .iter()
        .map(|&w| w / total)
        .filter(|&p| p > 0.0)
        .map(|p| -(p * p.log2()))
        .collect();
    pairwise(&terms).exp2()
}

/// A fit the Python cached, as `(form, params)`.
pub fn load_fit(root: &str, slug: &str) -> Result<(Form, Vec<f64>), Box<dyn Error>> {
    let path = format!("{root}/estimator_cache/params_{slug}.json");
    let value = json::parse_file(&path)?;
    let name = value
        .get("form")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{path} names no form"))?;
    let form = get_form(name).ok_or_else(|| format!("unknown form {name} in {path}"))?;
    let params: Vec<f64> = value
        .get("params")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{path} has no params"))?
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();
    if params.len() != form.codes.len() {
        return Err(format!(
            "{path}: {} params for a {}-term form",
            params.len(),
            form.codes.len()
        )
        .into());
    }
    Ok((form, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_matches_the_python_names() {
        let names: Vec<&str> = forms().iter().map(|f| f.name).collect();
        assert_eq!(
            names,
            vec![
                "loglinear",
                "anchored",
                "anchored2",
                "quadratic",
                "cubic",
                "saturating",
                "sqrt",
                "spline",
                "invonly"
            ]
        );
    }

    #[test]
    fn f_is_floored_at_one() {
        // one candidate is worth exactly one guess, and nothing may score under
        let f = get_form("loglinear").unwrap();
        assert_eq!(f_eval_x(0.0, &[-5.0, 0.0], &f), 1.0);
    }

    #[test]
    fn a_hinge_is_flat_below_its_knot() {
        let f = get_form("spline").unwrap();
        let params = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(f_eval_x(1.0, &params, &f), 1.0); // hinge at 2 not yet active
        assert_eq!(f_eval_x(3.0, &params, &f), 2.0); // 1 + (3 - 2)
    }

    #[test]
    fn a_flat_set_is_its_own_count() {
        assert!((effective_size(&[0.25; 4]) - 4.0).abs() < 1e-12);
        assert!((effective_size(&[1.0]) - 1.0).abs() < 1e-12);
    }
}
