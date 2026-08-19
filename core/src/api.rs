//! The endpoints `web/app.js` already calls, answered here instead of by a
//! server.
//!
//! The shapes are `web.py`'s, field for field, because the page is unchanged:
//! whatever it read off a JSON body before, it reads off the same body now. An
//! error comes back as `{"error": "..."}` with the same wording, which is what
//! the page shows the user.

use crate::feedback::{N_PATTERNS, WIN_PATTERN};
use crate::json::{self, Object, Value};
use crate::priors::{CurationShape, PriorConfig, RepeatShape};
use crate::solver::{Mode, Solver};

use crate::state;

pub const MAX_GUESSES: usize = 6;
const MAX_LISTED_CANDIDATES: usize = 60;
const MAX_LISTED_SIBLINGS: usize = 12;

/// Routes one request. Never panics into the page: anything that goes wrong
/// comes back as an error object the page already knows how to display.
pub fn handle(path: &str, body: &str) -> String {
    let parsed = if body.trim().is_empty() {
        Value::Object(Default::default())
    } else {
        match json::parse(body) {
            Ok(value) => value,
            Err(error) => return error_json(&format!("Malformed request: {error}")),
        }
    };

    let result = match path {
        "/api/status" => Ok(status()),
        "/api/init" => state::init(&parsed).map(|()| status()),
        "/api/new" => board(&parsed, true),
        "/api/board" => board(&parsed, false),
        other => Err(format!("No such endpoint: {other}")),
    };

    match result {
        Ok(reply) => reply,
        Err(message) => error_json(&message),
    }
}

fn error_json(message: &str) -> String {
    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object.str("error", message);
    object.end();
    out
}

/// What the page polls before assuming the solver is usable, and where it gets
/// the contents of its dropdowns.
///
/// There is no table to load and no kernel to compile here, so readiness is
/// only about whether the data files have arrived -- but the rest of the fields
/// are not optional: the page builds the *Shape of f* menu straight out of
/// `forms`, and throws if it is missing.
fn status() -> String {
    let ready = state::is_ready();
    let mut names: Vec<&str> = crate::estimator::forms().iter().map(|f| f.name).collect();
    names.sort_unstable();

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object
        .bool("ready", ready)
        .str(
            "progress",
            if ready {
                "ready"
            } else {
                "Fetching the word lists."
            },
        )
        .bool("error", false)
        .strings("forms", names)
        .strings("modes", ["entropy", "expected"])
        .int("max_guesses", MAX_GUESSES as i64)
        .strings("sorts", ["left", "bits", "weight", "alpha"]);
    object.end();
    out
}

// ------------------------------------------------------------------ settings

/// The switches, parsed the way `web.parse_settings` parses them.
struct Settings {
    config: PriorConfig,
    mode: Mode,
    hard: bool,
    top: usize,
}

fn parse_settings(body: &Value) -> Result<Settings, String> {
    let flag = |key: &str, fallback: bool| match body.get(key) {
        Some(Value::Bool(value)) => *value,
        _ => fallback,
    };
    let text = |key: &str| body.get(key).and_then(|v| v.as_str()).unwrap_or("");

    let mode = match text("mode") {
        "entropy" => Mode::Entropy,
        "" | "expected" => Mode::Expected,
        other => return Err(format!("Unknown mode: {other}")),
    };
    let repeat_shape = match text("repeat_shape") {
        "uniform" => RepeatShape::Uniform,
        "" | "binned" => RepeatShape::Binned,
        other => return Err(format!("Unknown repeat shape: {other}")),
    };
    let curation_shape = match text("curation_shape") {
        "uniform" => CurationShape::Uniform,
        "frequency" => CurationShape::Frequency,
        "" | "wordlebot" => CurationShape::Wordlebot,
        other => return Err(format!("Unknown curation shape: {other}")),
    };

    Ok(Settings {
        config: PriorConfig {
            repeat: flag("repeat_prior", true),
            curation: flag("curation_prior", true),
            repeat_shape,
            curation_shape,
            ..Default::default()
        },
        mode,
        hard: flag("hard", false),
        top: body.get("top").and_then(|v| v.as_f64()).unwrap_or(5.0) as usize,
    })
}

// --------------------------------------------------------------------- board

/// `/api/new` and `/api/board` differ only in whether rows are read: the page
/// sends the whole board every time, so setting it and starting a fresh one are
/// the same operation with an empty list.
fn board(body: &Value, fresh: bool) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let settings = parse_settings(body)?;

    let mut rows: Vec<(usize, u8)> = Vec::new();
    if !fresh {
        let raw = body
            .get("rows")
            .and_then(|v| v.as_array())
            .ok_or("Malformed board")?;
        if raw.len() > MAX_GUESSES {
            return Err(format!("A board is at most {MAX_GUESSES} rows"));
        }
        for (index, row) in raw.iter().enumerate() {
            let word = row
                .get("word")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            let guess = loaded.data.words.index_of(&word).ok_or_else(|| {
                if word.is_empty() {
                    format!("Row {}: needs a five letter word", index + 1)
                } else {
                    format!("Row {}: '{word}' is not in the word list", index + 1)
                }
            })?;
            let pattern = parse_pattern(row.get("pattern"))
                .map_err(|why| format!("Row {}: {why}", index + 1))?;
            rows.push((guess, pattern));
        }
    }

    build_state(loaded, &settings, &rows)
}

/// The page sends a pattern as `"NYGNN"` or as `[0, 1, 2, 0, 0]`, and both have
/// to be taken; it sends whichever it has to hand.
fn parse_pattern(value: Option<&Value>) -> Result<u8, String> {
    let tiles: Vec<u8> = match value {
        Some(Value::String(text)) => text
            .trim()
            .to_uppercase()
            .chars()
            .map(|c| match c {
                'N' => Ok(0),
                'Y' => Ok(1),
                'G' => Ok(2),
                other => Err(format!("'{other}' is not a tile color")),
            })
            .collect::<Result<Vec<u8>, String>>()?,
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item.as_f64() {
                Some(n) if (0.0..=2.0).contains(&n) => Ok(n as u8),
                _ => Err("a tile is 0, 1 or 2".to_string()),
            })
            .collect::<Result<Vec<u8>, String>>()?,
        _ => return Err("needs a pattern".to_string()),
    };
    if tiles.len() != 5 {
        return Err("a pattern is five tiles".to_string());
    }
    Ok(tiles.iter().fold(0u8, |acc, &tile| acc * 3 + tile))
}

/// The key `openings.json` is indexed by. Must match `build_site`'s.
fn settings_key(settings: &Settings) -> String {
    format!(
        "{}|{}|{}",
        settings.config.slug(),
        if settings.mode == Mode::Entropy {
            "entropy"
        } else {
            "expected"
        },
        if settings.hard { "hard" } else { "free" }
    )
}

/// The opening ranking, if this is the opening and one was precomputed.
fn precomputed_opening(
    loaded: &state::Loaded,
    settings: &Settings,
    rows: &[(usize, u8)],
) -> Option<Vec<(usize, f64)>> {
    if !rows.is_empty() {
        return None;
    }
    let listed = loaded.openings.get(&settings_key(settings))?.as_array()?;
    let mut out = Vec::with_capacity(listed.len());
    for entry in listed.iter().take(settings.top.max(1)) {
        let word = entry.get("word")?.as_str()?;
        let score = entry.get("score")?.as_f64()?;
        out.push((loaded.data.words.index_of(word)?, score));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn build_state(
    loaded: &state::Loaded,
    settings: &Settings,
    rows: &[(usize, u8)],
) -> Result<String, String> {
    let weights = loaded.weights_for(&settings.config)?;
    let (form, params) = match settings.mode {
        Mode::Expected => loaded.fit_for(&settings.config)?,
        // entropy never consults f, so an unfitted prior is fine here
        Mode::Entropy => loaded.fit_for(&settings.config).unwrap_or_else(|_| {
            (
                crate::estimator::get_form("loglinear").unwrap(),
                vec![0.0, 1.0],
            )
        }),
    };

    let solver_form = form.clone();
    let solver_params = params.clone();
    let source = loaded.source();
    let mut solver = Solver::new(&source, &weights, params, form, settings.mode)
        .with_hard(settings.hard)
        .with_loss_penalty(crate::solver::DEFAULT_LOSS_PENALTY);

    let mut candidates: Vec<usize> = (0..loaded.data.n_words()).collect();
    for (index, &(guess, pattern)) in rows.iter().enumerate() {
        if pattern == WIN_PATTERN && index != rows.len() - 1 {
            return Err(format!(
                "Row {} already solves it; clear the rows under it",
                index + 1
            ));
        }
        candidates = solver.filter(guess, pattern, &candidates);
        if candidates.is_empty() {
            return Err(format!(
                "Row {} contradicts the rows above it -- no word matches all of them",
                index + 1
            ));
        }
    }

    let turn = rows.len() + 1;
    let solved = rows.last().map(|&(_, p)| p == WIN_PATTERN).unwrap_or(false);
    let over = solved || (!solved && turn > MAX_GUESSES);

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object
        .str("session", "wasm")
        .int("turn", turn as i64)
        .int(
            "guesses_left",
            MAX_GUESSES.saturating_sub(rows.len()) as i64,
        )
        .int("max_guesses", MAX_GUESSES as i64)
        .bool("solved", solved)
        .bool("over", over)
        .str(
            "mode",
            match settings.mode {
                Mode::Entropy => "entropy",
                Mode::Expected => "expected",
            },
        )
        .bool("hard", settings.hard)
        .str(
            "score_label",
            match settings.mode {
                Mode::Entropy => "bits",
                Mode::Expected => "expected guesses",
            },
        )
        .int("candidates_left", candidates.len() as i64)
        .num("effective_size", solver.effective_size(&candidates))
        // there is no network here, so the page's timing readout is the solve
        .int("elapsed_ms", 0);

    object.raw("prior", |out| describe_prior(out, loaded, &settings.config));
    if settings.mode == Mode::Expected {
        object.raw("estimator", |out| {
            describe_estimator(out, &solver_form, &solver_params)
        });
    } else {
        object.null("estimator");
    }

    object.raw("rows", |out| {
        out.push('[');
        for (i, &(guess, pattern)) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let mut row = Object::new(out);
            row.str("word", loaded.data.words.word(guess))
                .str("pattern", &pattern_to_text(pattern));
            row.end();
        }
        out.push(']');
    });

    // likeliest first, which is the order the page lists them in
    let mut by_weight = candidates.clone();
    by_weight.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    object.strings(
        "possible_answers",
        by_weight
            .iter()
            .take(MAX_LISTED_CANDIDATES)
            .map(|&i| loaded.data.words.word(i)),
    );

    if over {
        object.null("recommendation").raw("alternatives", |out| {
            out.push_str("[]");
        });
        object.end();
        return Ok(out);
    }

    let guesses_left = MAX_GUESSES - rows.len();
    let exact = solver.solved_exactly(&candidates, Some(guesses_left));
    // The opening is the one position a browser cannot afford: 12,947 guesses
    // against 12,947 candidates. It is also a constant per setting, so
    // `build_site` works it out once and it arrives with the data.
    let top = precomputed_opening(loaded, settings, rows).unwrap_or_else(|| {
        solver.top_guesses(&candidates, settings.top.max(1), Some(guesses_left))
    });
    let (pick, score) = top[0];

    object.raw("recommendation", |out| {
        let mut entry = Object::new(out);
        describe_guess(
            &mut entry,
            loaded,
            &solver,
            &weights,
            pick,
            score,
            &candidates,
        );
        entry
            .bool("exact", exact)
            // a guess that cannot be the answer is still often the right one:
            // worth saying which it is
            .bool("possible", candidates.contains(&pick))
            .num("win_probability", solver.win_probability(&candidates, pick));
        entry.end();
    });

    object.raw("alternatives", |out| {
        out.push('[');
        for (i, &(guess, value)) in top.iter().skip(1).enumerate() {
            if i > 0 {
                out.push(',');
            }
            let mut entry = Object::new(out);
            describe_guess(
                &mut entry,
                loaded,
                &solver,
                &weights,
                guess,
                value,
                &candidates,
            );
            entry.end();
        }
        out.push(']');
    });

    object.end();
    Ok(out)
}

/// The prior as the page explains it: which halves are on, how much mass each
/// takes, how that mass is shaped, and the rates it is set from.
fn describe_prior(out: &mut String, loaded: &state::Loaded, config: &PriorConfig) {
    let shape_name = |shape: RepeatShape| match shape {
        RepeatShape::Uniform => "uniform",
        RepeatShape::Binned => "binned",
    };
    let curation_name = |shape: CurationShape| match shape {
        CurationShape::Uniform => "uniform",
        CurationShape::Frequency => "frequency",
        CurationShape::Wordlebot => "wordlebot",
    };
    let on = |flag: bool| if flag { "ON" } else { "OFF" };
    let summary = format!(
        "repeat prior {}{}, curation prior {}{}",
        on(config.repeat),
        if config.repeat {
            format!(
                " (p={}, {})",
                config.repeat_prob,
                shape_name(config.repeat_shape)
            )
        } else {
            String::new()
        },
        on(config.curation),
        if config.curation {
            format!(
                " (p={}, {})",
                config.noncurated_prob,
                curation_name(config.curation_shape)
            )
        } else {
            String::new()
        }
    );

    let stats = crate::priors::regime_stats(&loaded.data);
    let mut object = Object::new(out);
    object
        .str("summary", &summary)
        .bool("repeat", config.repeat)
        .num("repeat_prob", config.repeat_prob)
        .bool("curation", config.curation)
        .num("noncurated_prob", config.noncurated_prob);
    if config.repeat {
        object.str("repeat_shape", shape_name(config.repeat_shape));
    } else {
        object.null("repeat_shape");
    }
    if config.curation {
        object.str("curation_shape", curation_name(config.curation_shape));
    } else {
        object.null("curation_shape");
    }
    object.raw("measured", |out| {
        let mut measured = Object::new(out);
        measured
            .int("start_puzzle", stats.start_puzzle)
            .int("puzzles", stats.puzzles as i64)
            .int("words", stats.words as i64)
            .int("curated", stats.curated as i64)
            .int("used", stats.used as i64)
            .raw("regime", |out| {
                let mut rates = Object::new(out);
                rates
                    .num("repeat", stats.regime_repeat)
                    .num("noncurated", stats.regime_noncurated);
                rates.end();
            })
            .raw("all_time", |out| {
                let mut rates = Object::new(out);
                rates
                    .num("repeat", stats.all_time_repeat)
                    .num("noncurated", stats.all_time_noncurated);
                rates.end();
            });
        measured.end();
    });
    object.end();
}

/// The fitted `f` in pieces, so the page can set it as an equation rather than
/// as a one-line summary.
fn describe_estimator(out: &mut String, form: &crate::estimator::Form, params: &[f64]) {
    const SAMPLES: [f64; 6] = [2.0, 5.0, 20.0, 100.0, 500.0, 2000.0];
    let labels = form.term_labels();

    let mut object = Object::new(out);
    object
        .str("form", form.name)
        // there is one fit per prior here, so what is cached is what is fitted
        .str("fitted", form.name)
        .str("description", form.description)
        .num("base", form.base)
        .raw("terms", |out| {
            out.push('[');
            for (i, (label, value)) in labels.iter().zip(params).enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let mut term = Object::new(out);
                term.str("label", label).num("coefficient", *value);
                term.end();
            }
            out.push(']');
        })
        .raw("samples", |out| {
            out.push('[');
            for (i, &n) in SAMPLES.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let mut sample = Object::new(out);
                sample
                    .num("n", n)
                    .num("f", crate::estimator::f_eval(n, params, form));
                sample.end();
            }
            out.push(']');
        });
    object.end();
}

/// One entry of the ranking: a word, its score, and the words that play
/// identically from here.
fn describe_guess(
    entry: &mut Object,
    loaded: &state::Loaded,
    solver: &Solver<crate::source::Live>,
    weights: &[f64],
    guess: usize,
    score: f64,
    candidates: &[usize],
) {
    let mut siblings = solver.split_siblings(guess, candidates);
    // likeliest first, so the familiar spelling of a split is the one on show
    siblings.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entry
        .str("word", loaded.data.words.word(guess))
        .num("score", score)
        .int("same_split_count", siblings.len() as i64)
        .strings(
            "same_split",
            siblings
                .iter()
                .take(MAX_LISTED_SIBLINGS)
                .map(|&i| loaded.data.words.word(i)),
        );
}

fn pattern_to_text(pattern: u8) -> String {
    let mut tiles = [b'N'; 5];
    let mut left = pattern as usize;
    for slot in tiles.iter_mut().rev() {
        *slot = match left % 3 {
            0 => b'N',
            1 => b'Y',
            _ => b'G',
        };
        left /= 3;
    }
    debug_assert!((pattern as usize) < N_PATTERNS);
    String::from_utf8_lossy(&tiles).into_owned()
}
