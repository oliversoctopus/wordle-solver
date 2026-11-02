//! What the module holds between requests.
//!
//! The page fetches the data files once and posts them to `/api/init`; after
//! that everything lives here. Weights are cached per prior. A `Solver`
//! borrows its source and weights and is built per request; what it caches
//! is kept here and lent back, since the page re-sends the whole board on
//! every submit and a board solved once arrives again on the next keystroke.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::estimator::{get_form, Form};
use crate::data::parse_table;
use crate::json;
use crate::priors::{build_weights, PriorConfig, ShapeTables};
use crate::solver::Memos;
use crate::source::Live;
use crate::WordleData;

/// Every fit the Python cached for one prior: the active one, the meaning of
/// *Whatever is cached*, and the archive it keeps per form.
pub struct Fits {
    pub active: (Form, Vec<f64>),
    pub by_form: HashMap<String, Vec<f64>>,
}

pub struct Loaded {
    pub data: WordleData,
    pub tables: ShapeTables,
    /// Fitted `f` per prior slug, as the Python's `estimator_cache` holds it.
    pub fits: HashMap<String, Fits>,
    /// Precomputed answers for the opening position, per settings key. The
    /// browser cannot afford to score 12,947 guesses against 12,947 candidates.
    pub openings: HashMap<String, json::Value>,
    /// Each word's Zipf frequency in English, by word index; zero if unseen.
    pub word_freq: Vec<f64>,
    weights: Mutex<HashMap<String, Vec<f64>>>,
    /// Positions already worked out, per settings key, least recent first.
    memos: Mutex<Vec<(String, Memos)>>,
    games: Mutex<Games>,
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

    /// The fit to play with. `form` is the name the *Shape of f* menu names, or
    /// None for whichever one the Python left active.
    pub fn fit_for(
        &self,
        config: &PriorConfig,
        form: Option<&str>,
    ) -> Result<(Form, Vec<f64>), String> {
        let slug = config.slug();
        let fits = self.fits.get(&slug).ok_or_else(|| {
            format!(
                "No fitted f for this prior ({slug}). Entropy mode needs none and \
                 runs on anything; expected guesses needs one."
            )
        })?;
        let Some(name) = form else {
            return Ok(fits.active.clone());
        };
        let shape = get_form(name).ok_or_else(|| format!("Unknown form {name}"))?;
        let params = fits
            .by_form
            .get(name)
            .ok_or_else(|| format!("No {name} fit for this prior ({slug})"))?;
        Ok((shape, params.clone()))
    }

    /// The name of the form `form` selects, as the estimator panel shows it.
    pub fn form_name(&self, config: &PriorConfig, form: Option<&str>) -> Result<String, String> {
        match form {
            Some(name) => Ok(name.to_string()),
            None => Ok(self
                .fits
                .get(&config.slug())
                .map(|fits| fits.active.0.name.to_string())
                .unwrap_or_else(|| crate::estimator::DEFAULT_FORM.to_string())),
        }
    }

    /// Lends this setting whatever it has already worked out. The caller
    /// hands it back with `keep_memos`, which is also what bounds how many
    /// settings are held at once.
    pub fn take_memos(&self, key: &str) -> Memos {
        let Ok(mut held) = self.memos.lock() else {
            return Memos::new();
        };
        match held.iter().position(|(name, _)| name == key) {
            Some(at) => held.remove(at).1,
            None => Memos::new(),
        }
    }

    pub fn keep_memos(&self, key: String, memos: Memos) {
        if let Ok(mut held) = self.memos.lock() {
            held.retain(|(name, _)| name != &key);
            held.push((key, memos));
            while held.len() > MAX_MEMO_SETS {
                held.remove(0);
            }
        }
    }

    pub fn source(&self) -> Live<'_> {
        Live::new(&self.data.words)
    }

    /// Runs `body` over the game store, which holds play mode's answers for as
    /// long as the tab holds the page.
    pub fn with_games<T>(&self, body: impl FnOnce(&mut Games) -> T) -> Result<T, String> {
        let mut games = self.games.lock().map_err(|_| "game lock poisoned")?;
        Ok(body(&mut games))
    }
}

/// One game of play mode: the answer, the prior it was drawn from, and the
/// rows played so far.
#[derive(Clone)]
pub struct Game {
    pub answer: usize,
    pub config: PriorConfig,
    pub source: String,
    pub rows: Vec<(usize, u8)>,
    pub gave_up: bool,
}

/// The sessions, oldest dropped first, as `web.Sessions` does it.
#[derive(Default)]
pub struct Games {
    order: Vec<String>,
    games: HashMap<String, Game>,
    next: u64,
}

/// Beyond this many, the oldest goes. One tab plays one game at a time, so the
/// cap is only there to stop a long sitting from growing without bound.
const MAX_GAMES: usize = 32;

/// How many settings keep their solved positions, so flipping a switch and
/// flipping it back re-solves nothing.
const MAX_MEMO_SETS: usize = 4;

impl Games {
    pub fn create(&mut self, game: Game) -> String {
        self.next += 1;
        let id = format!("g{}", self.next);
        if self.order.len() >= MAX_GAMES {
            let oldest = self.order.remove(0);
            self.games.remove(&oldest);
        }
        self.order.push(id.clone());
        self.games.insert(id.clone(), game);
        id
    }

    pub fn get(&mut self, id: &str) -> Result<&mut Game, String> {
        self.games
            .get_mut(id)
            .ok_or_else(|| "That game has expired; start a new one".to_string())
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
        .unwrap_or(0.0) as usize;

    let data = WordleData::from_json(
        text("ind2word")?,
        text("wordles")?,
        text("solutions")?,
        mask_last,
    )
    .map_err(|error| error.to_string())?;

    let tables = ShapeTables::from_json(text("answer_like")?, text("wordlebot")?)
        .map_err(|error| error.to_string())?;

    let freq_table = parse_table(text("word_freq")?).map_err(|error| error.to_string())?;
    let word_freq = (0..data.words.len())
        .map(|i| freq_table.get(data.words.word(i)).copied().unwrap_or(0.0))
        .collect();

    let numbers = |value: Option<&json::Value>| -> Option<Vec<f64>> {
        Some(
            value?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
        )
    };

    let mut fits = HashMap::new();
    if let Some(object) = body.get("fits").and_then(|v| v.as_object()) {
        for (slug, value) in object {
            let name = value
                .get("form")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("fit {slug} names no form"))?;
            let form =
                get_form(name).ok_or_else(|| format!("fit {slug} names unknown form {name}"))?;
            let params =
                numbers(value.get("params")).ok_or_else(|| format!("fit {slug} has no params"))?;

            let mut by_form = HashMap::new();
            if let Some(archive) = value.get("forms").and_then(|v| v.as_object()) {
                for (other, params) in archive {
                    if let Some(params) = numbers(Some(params)) {
                        by_form.insert(other.clone(), params);
                    }
                }
            }
            fits.insert(
                slug.clone(),
                Fits {
                    active: (form, params),
                    by_form,
                },
            );
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
        word_freq,
        weights: Mutex::new(HashMap::new()),
        memos: Mutex::new(Vec::new()),
        games: Mutex::new(Games::default()),
    });
    Ok(())
}
