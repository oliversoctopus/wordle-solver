//! What the module holds between requests.
//!
//! The page fetches the data files once and posts them to `/api/init`; after
//! that everything lives here. Weights are cached per prior, since building one
//! walks the whole word list and the switches change priors far more often than
//! they change anything else.
//!
//! A `Solver` borrows its source and weights, so one cannot be kept in a static
//! without a self-referential struct. It is built per request instead, which
//! costs an empty memo -- fine for a page that solves one position at a time,
//! where the memo only ever pays inside a single request anyway.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::estimator::{get_form, Form};
use crate::json;
use crate::priors::{build_weights, PriorConfig, ShapeTables};
use crate::source::Live;
use crate::WordleData;

pub struct Loaded {
    pub data: WordleData,
    pub tables: ShapeTables,
    /// Fitted `f` per prior slug, as the Python's `estimator_cache` holds it.
    pub fits: HashMap<String, (Form, Vec<f64>)>,
    /// Precomputed answers for the opening position, per settings key. The
    /// browser cannot afford to score 12,947 guesses against 12,947 candidates.
    pub openings: HashMap<String, json::Value>,
    weights: Mutex<HashMap<String, Vec<f64>>>,
}

impl Loaded {
    /// The prior over answers for this configuration, built once and kept.
    pub fn weights_for(&self, config: &PriorConfig) -> Result<Vec<f64>, String> {
        let slug = config.slug();
        let mut cache = self.weights.lock().map_err(|_| "weights lock poisoned")?;
        if let Some(found) = cache.get(&slug) {
            return Ok(found.clone());
        }
        let built =
            build_weights(&self.data, config, &self.tables).map_err(|error| error.to_string())?;
        cache.insert(slug, built.clone());
        Ok(built)
    }

    pub fn fit_for(&self, config: &PriorConfig) -> Result<(Form, Vec<f64>), String> {
        let slug = config.slug();
        self.fits.get(&slug).cloned().ok_or_else(|| {
            format!(
                "No fitted f for this prior ({slug}). Entropy mode needs none and \
                 runs on anything; expected guesses needs one."
            )
        })
    }

    pub fn source(&self) -> Live<'_> {
        Live::new(&self.data.words)
    }
}

static LOADED: OnceLock<Loaded> = OnceLock::new();

pub fn loaded() -> Option<&'static Loaded> {
    LOADED.get()
}

pub fn is_ready() -> bool {
    LOADED.get().is_some()
}

/// Takes the data files the page fetched. Ignored if already loaded, so a
/// double init is harmless rather than an error.
pub fn init(body: &json::Value) -> Result<(), String> {
    if LOADED.get().is_some() {
        return Ok(());
    }

    let text = |key: &str| -> Result<&str, String> {
        body.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("init is missing {key}"))
    };

    let mask_last = body
        .get("mask_last")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as usize;

    let data = WordleData::from_json(
        text("ind2word")?,
        text("wordles")?,
        text("solutions")?,
        mask_last,
    )
    .map_err(|error| error.to_string())?;

    let tables = ShapeTables::from_json(text("answer_like")?, text("wordlebot")?)
        .map_err(|error| error.to_string())?;

    let mut fits = HashMap::new();
    if let Some(object) = body.get("fits").and_then(|v| v.as_object()) {
        for (slug, value) in object {
            let name = value
                .get("form")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("fit {slug} names no form"))?;
            let form =
                get_form(name).ok_or_else(|| format!("fit {slug} names unknown form {name}"))?;
            let params: Vec<f64> = value
                .get("params")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("fit {slug} has no params"))?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            fits.insert(slug.clone(), (form, params));
        }
    }

    let mut openings = HashMap::new();
    if let Some(object) = body.get("openings").and_then(|v| v.as_object()) {
        for (key, value) in object {
            openings.insert(key.clone(), value.clone());
        }
    }

    let _ = LOADED.set(Loaded {
        data,
        tables,
        fits,
        openings,
        weights: Mutex::new(HashMap::new()),
    });
    Ok(())
}
