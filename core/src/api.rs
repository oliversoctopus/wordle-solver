//! The endpoints `web/app.js` already calls, answered here instead of by a
//! server.
//!
//! The shapes are `web.py`'s, field for field, because the page is unchanged:
//! whatever it read off a JSON body before, it reads off the same body now. An
//! error comes back as `{"error": "..."}` with the same wording, and the page
//! shows it to the user.

use crate::feedback::{pattern_text, N_PATTERNS, WIN_PATTERN};
use crate::game::MAX_GUESSES;
use crate::json::{self, Object, Value};
use crate::priors::PriorConfig;
use crate::solver::{Mode, Solver};

use crate::state;
use crate::sum::{pairwise, pairwise_of};

const MAX_LISTED_CANDIDATES: usize = 80;
const MAX_LISTED_SIBLINGS: usize = 40;

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
        "/api/draw" => draw(&parsed),
        "/api/find" => find(&parsed),
        "/api/game/new" => game_new(&parsed),
        "/api/game/guess" => game_guess(&parsed),
        "/api/game/hint" => game_hint(&parsed),
        "/api/game/reveal" => game_reveal(&parsed),
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

    let progress = if ready {
        "ready"
    } else {
        "Fetching the word lists."
    };
    let record_through = state::loaded().map_or("", |l| l.data.record_through.as_str());

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object
        .bool("ready", ready)
        .str("progress", progress)
        .bool("error", false)
        .strings("forms", names)
        .strings("modes", ["entropy", "expected"])
        .int("max_guesses", MAX_GUESSES as i64)
        .strings("sorts", ["left", "bits", "weight", "alpha"])
        .str("record_through", record_through);
    object.end();
    out
}

// ------------------------------------------------------------------ settings

/// The switches, parsed the way `web.parse_settings` parses them.
struct Settings {
    config: PriorConfig,
    mode: Mode,
    /// The *Shape of f* menu, or None for whichever fit the Python left active.
    form: Option<String>,
    hard: bool,
    top: usize,
}

/// The fit the mode plays with.  Entropy never reads `f`, so a prior with no
/// fit gets zeros there, as `WordleSolver` does.
fn fit_for_mode(
    loaded: &state::Loaded,
    settings: &Settings,
) -> Result<(crate::estimator::Form, Vec<f64>), String> {
    match loaded.fit_for(&settings.config, settings.form.as_deref()) {
        Err(_) if settings.mode == Mode::Entropy => {
            let form = crate::estimator::get_form(crate::estimator::DEFAULT_FORM).unwrap();
            let params = vec![0.0; form.codes.len()];
            Ok((form, params))
        }
        outcome => outcome,
    }
}

fn flag(body: &Value, key: &str, fallback: bool) -> bool {
    match body.get(key) {
        Some(Value::Bool(value)) => *value,
        _ => fallback,
    }
}

/// Just the two prior switches, as `web.parse_config` reads them. The shapes
/// are not on the page, so they stay at their defaults on both sides.
fn parse_config(body: &Value) -> PriorConfig {
    PriorConfig {
        repeat: flag(body, "repeat", true),
        curation: flag(body, "curation", true),
        ..Default::default()
    }
}

fn parse_settings(body: &Value) -> Result<Settings, String> {
    let text = |key: &str| body.get(key).and_then(|v| v.as_str()).unwrap_or("");

    let mode = match text("mode") {
        "entropy" => Mode::Entropy,
        "" | "expected" => Mode::Expected,
        other => return Err(format!("Unknown mode: {other}")),
    };
    let form = match text("form") {
        "" => None,
        name if crate::estimator::get_form(name).is_some() => Some(name.to_string()),
        other => return Err(format!("Unknown form {other}")),
    };

    Ok(Settings {
        config: parse_config(body),
        mode,
        form,
        hard: flag(body, "hard", false),
        top: body
            .get("top")
            .and_then(|v| v.as_f64())
            .unwrap_or(5.0)
            .clamp(1.0, 10.0) as usize,
    })
}

// --------------------------------------------------------------------- board

/// `/api/new` and `/api/board` differ only in whether rows are read: the page
/// sends the whole board every time, so setting it and starting a fresh one are
/// the same operation with an empty list.
fn board(body: &Value, fresh: bool) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let settings = parse_settings(body)?;
    let answer = parse_answer(body, loaded)?;

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

    build_state(loaded, &settings, &rows, answer)
}

/// The answer the board is being played against, as `web.parse_answer` reads
/// it: set, it colors every row itself and the page's tiles become a display.
fn parse_answer(body: &Value, loaded: &state::Loaded) -> Result<Option<usize>, String> {
    let word = body
        .get("answer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if word.is_empty() {
        return Ok(None);
    }
    loaded
        .data
        .words
        .index_of(&word)
        .map(Some)
        .ok_or_else(|| format!("'{word}' is not in the word list"))
}

/// One answer drawn from the prior, for the board to be played against.
fn draw(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let weights = loaded.weights_for(&parse_config(body))?;
    let seed = body.get("seed").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object.str("word", loaded.data.words.word(draw_answer(&weights, seed)));
    object.end();
    Ok(out)
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

/// The key `openings.json` is indexed by; `build_site` writes it with this.
///
/// The form is in the key because it changes `f`, and `f` is what expected
/// mode ranks by. Entropy never consults it, so its ten entries per prior are
/// one ranking written ten times over.
pub fn settings_key(config: &PriorConfig, mode: Mode, hard: bool, form: &str) -> String {
    format!(
        "{}|{}|{}|{}",
        config.slug(),
        if mode == Mode::Entropy {
            "entropy"
        } else {
            "expected"
        },
        if hard { "hard" } else { "free" },
        form
    )
}

/// The ranking for this position, if one was precomputed.
///
/// Two positions are: the opening, and turn two behind the word that setting
/// opens with. Both are constants, so `build_site` works them out ahead of
/// time. `README.md` § *Without a server* has which settings get a turn two and
/// why nothing deeper is worth the trouble.
fn precomputed_ranking(
    loaded: &state::Loaded,
    settings: &Settings,
    rows: &[(usize, u8)],
) -> Option<Vec<(usize, f64)>> {
    let form = loaded
        .form_name(&settings.config, settings.form.as_deref())
        .ok()?;
    let opening = settings_key(&settings.config, settings.mode, settings.hard, &form);

    let key = match rows {
        [] => opening,
        // and only behind the word this setting actually opens with, since
        // that is the board the rankings were worked out for
        [(guess, pattern)] => {
            let first = loaded.openings.get(&opening)?.as_array()?.first()?;
            if loaded.data.words.index_of(first.get("word")?.as_str()?)? != *guess {
                return None;
            }
            format!("{opening}|{}", pattern_text(*pattern))
        }
        _ => return None,
    };

    let listed = loaded.openings.get(&key)?.as_array()?;
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
    answer: Option<usize>,
) -> Result<String, String> {
    // against a known answer the rows' patterns are the true ones, whatever
    // the page sent
    let truthful: Vec<(usize, u8)>;
    let rows = match answer {
        Some(answer) => {
            truthful = rows
                .iter()
                .map(|&(guess, _)| (guess, loaded.data.words.score(guess, answer)))
                .collect();
            &truthful[..]
        }
        None => rows,
    };
    let weights = loaded.weights_for(&settings.config)?;
    let (form, params) = fit_for_mode(loaded, settings)?;

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
    let over = solved || turn > MAX_GUESSES;

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
        .str("mode", mode_name(settings.mode))
        .bool("hard", settings.hard)
        .str("score_label", score_label(settings.mode))
        .int("candidates_left", candidates.len() as i64)
        .num("effective_size", solver.effective_size(&candidates))
        // there is no clock here; `wasm-api.js` times the call and fills it in
        .int("elapsed_ms", 0);

    object.raw("prior", |out| describe_prior(out, loaded, &settings.config));
    match answer {
        Some(answer) => object.str("answer", loaded.data.words.word(answer)),
        None => object.null("answer"),
    };
    if settings.mode == Mode::Expected {
        // which form the menu's empty choice resolves to, whatever is selected
        let fitted = loaded.form_name(&settings.config, None)?;
        object.raw("estimator", |out| {
            describe_estimator(out, &solver_form, &solver_params, &fitted)
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
                .str("pattern", &pattern_text(pattern));
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

    // lent for the length of this request and handed back at the end; it only decides
    // whether a position already seen is worked out again
    let held = settings_key(
        &settings.config,
        settings.mode,
        settings.hard,
        &loaded.form_name(&settings.config, settings.form.as_deref())?,
    );
    let mut memos = loaded.take_memos(&held);
    solver.swap_memos(&mut memos);
    let exact = solver.solved_exactly(&candidates, Some(guesses_left));
    // Precomputed where it can be; see `precomputed_ranking`.
    let top = precomputed_ranking(loaded, settings, rows).unwrap_or_else(|| {
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
            // whether the pick could be the answer; a non-candidate is often the
            // right play
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

    solver.swap_memos(&mut memos);
    loaded.keep_memos(held, memos);

    object.end();
    Ok(out)
}

fn score_label(mode: Mode) -> &'static str {
    match mode {
        Mode::Entropy => "bits of expected information",
        Mode::Expected => "expected guesses to finish",
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Entropy => "entropy",
        Mode::Expected => "expected",
    }
}

/// The prior in a sentence, as `PriorConfig.describe` writes it.
fn prior_summary(config: &PriorConfig) -> String {
    let on = |flag: bool| if flag { "ON" } else { "OFF" };
    format!(
        "repeat prior {}{}, curation prior {}{}",
        on(config.repeat),
        if config.repeat {
            format!(
                " (p={}, {})",
                config.repeat_prob,
                config.repeat_shape.name()
            )
        } else {
            String::new()
        },
        on(config.curation),
        if config.curation {
            format!(
                " (p={}, {})",
                config.noncurated_prob,
                config.curation_shape.name()
            )
        } else {
            String::new()
        }
    )
}

/// The prior as the page explains it: which halves are on, how much mass each
/// takes, how that mass is shaped, and the rates it is set from.
fn describe_prior(out: &mut String, loaded: &state::Loaded, config: &PriorConfig) {
    let stats = crate::priors::regime_stats(&loaded.data);
    let mut object = Object::new(out);
    object
        .str("summary", &prior_summary(config))
        .bool("repeat", config.repeat)
        .num("repeat_prob", config.repeat_prob)
        .bool("curation", config.curation)
        .num("noncurated_prob", config.noncurated_prob);
    if config.repeat {
        object.str("repeat_shape", config.repeat_shape.name());
    } else {
        object.null("repeat_shape");
    }
    if config.curation {
        object.str("curation_shape", config.curation_shape.name());
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
fn describe_estimator(
    out: &mut String,
    form: &crate::estimator::Form,
    params: &[f64],
    fitted: &str,
) {
    const SAMPLES: [f64; 6] = [2.0, 5.0, 20.0, 100.0, 500.0, 2000.0];
    let labels = form.term_labels();

    let mut object = Object::new(out);
    object
        .str("form", form.name)
        // what "whatever is cached" resolves to for this prior, so the page can
        // say which form is the fitted one instead of naming one in advance
        .str("fitted", fitted)
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


/// The tiles of a pattern, least significant last, as `num_to_feedback` gives
/// them.
fn pattern_tiles(pattern: u8) -> [u8; 5] {
    let mut tiles = [0u8; 5];
    let mut left = pattern;
    for slot in tiles.iter_mut().rev() {
        *slot = left % 3;
        left /= 3;
    }
    tiles
}

// ----------------------------------------------------------------- play mode

/// The emoji the share grid is drawn in, indexed by tile: black, yellow, green.
const TILE_BLOCK: [&str; 3] = ["\u{2B1B}", "\u{1F7E8}", "\u{1F7E9}"];

/// The one word the page reads back and the one it never gets to see, held
/// together for as long as the tab is open.
///
/// The Python keeps the answer on the server so that no reply can spoil it.
/// There is no server here, so that property is gone -- a determined player
/// can read the answer out of memory. It is still not in any reply until the
/// game is over, and the page relies on that.
fn game_new(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let config = parse_config(body);

    let wanted = body
        .get("answer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let (answer, source) = if wanted.is_empty() {
        let weights = loaded.weights_for(&config)?;
        let seed = body.get("seed").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let source = format!("drawn from {}", prior_summary(&config));
        (draw_answer(&weights, seed), source)
    } else {
        let index = loaded
            .data
            .words
            .index_of(&wanted)
            .ok_or_else(|| format!("'{wanted}' is not in the word list"))?;
        let source = "the one you chose".to_string();
        (index, source)
    };

    let game = state::Game {
        answer,
        config,
        source,
        rows: Vec::new(),
        gave_up: false,
    };
    let (id, game) = loaded.with_games(|games| {
        let id = games.create(game);
        (id.clone(), games.get(&id).expect("just created").clone())
    })?;
    game_state(loaded, &id, &game)
}

/// One draw from the prior. `seed` is a uniform sample the page passes in --
/// the module has no clock and no entropy source of its own, and the page has
/// `Math.random`.
fn draw_answer(weights: &[f64], seed: f64) -> usize {
    let total = pairwise(weights);
    let mut want = seed.clamp(0.0, 1.0) * total;
    let mut last = 0;
    for (index, &weight) in weights.iter().enumerate() {
        if weight <= 0.0 {
            continue; // a word the prior rules out is never drawn, at any seed
        }
        last = index;
        want -= weight;
        if want <= 0.0 {
            return index;
        }
    }
    last
}

fn game_guess(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let id = session_id(body);
    let word = body
        .get("word")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();

    let game = loaded.with_games(|games| {
        let game = games.get(&id)?;
        if game_over(game) {
            return Err("This game is over".to_string());
        }
        let guess = loaded.data.words.index_of(&word).ok_or_else(|| {
            if word.is_empty() {
                "Type a five letter word".to_string()
            } else {
                format!("'{word}' is not in the word list")
            }
        })?;
        let pattern = loaded.data.words.score(guess, game.answer);
        game.rows.push((guess, pattern));
        Ok(game.clone())
    })??;
    game_state(loaded, &id, &game)
}

fn game_reveal(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let id = session_id(body);
    let game = loaded.with_games(|games| {
        let game = games.get(&id)?;
        game.gave_up = !solved(game);
        Ok::<_, String>(game.clone())
    })??;
    game_state(loaded, &id, &game)
}

fn game_hint(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let settings = parse_settings(body)?;
    let id = session_id(body);
    let game = loaded.with_games(|games| {
        let game = games.get(&id)?;
        if game_over(game) {
            return Err("This game is over".to_string());
        }
        Ok(game.clone())
    })??;

    // every setting comes with the request, so changing one rethinks the advice for
    // this position instead of dealing a new answer
    let weights = loaded.weights_for(&settings.config)?;
    let (form, params) = fit_for_mode(loaded, &settings)?;
    let source = loaded.source();
    let mut solver = Solver::new(&source, &weights, params, form, settings.mode)
        .with_hard(settings.hard)
        .with_loss_penalty(crate::solver::DEFAULT_LOSS_PENALTY);

    let mut candidates: Vec<usize> = (0..loaded.data.n_words()).collect();
    for &(guess, pattern) in &game.rows {
        candidates = solver.filter(guess, pattern, &candidates);
    }

    let guesses_left = MAX_GUESSES.saturating_sub(game.rows.len()).max(1);

    // lent for this request and handed back below, as `build_state` does it:
    // asking for a hint twice on one board should only think once
    let held = settings_key(
        &settings.config,
        settings.mode,
        settings.hard,
        &loaded.form_name(&settings.config, settings.form.as_deref())?,
    );
    let mut memos = loaded.take_memos(&held);
    solver.swap_memos(&mut memos);

    let exact = solver.solved_exactly(&candidates, Some(guesses_left));
    // precomputed where it can be; see `precomputed_ranking`
    let top = precomputed_ranking(loaded, &settings, &game.rows).unwrap_or_else(|| {
        solver.top_guesses(&candidates, settings.top.max(1), Some(guesses_left))
    });
    let (pick, score) = top[0];

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    describe_guess(
        &mut object,
        loaded,
        &solver,
        &weights,
        pick,
        score,
        &candidates,
    );
    object
        .str("mode", mode_name(settings.mode))
        .bool("hard", settings.hard)
        .str("score_label", score_label(settings.mode))
        .int("candidates_left", candidates.len() as i64)
        .bool("exact", exact)
        .raw("alternatives", |out| {
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

    solver.swap_memos(&mut memos);
    loaded.keep_memos(held, memos);

    object.end();
    Ok(out)
}

fn session_id(body: &Value) -> String {
    body.get("session")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn solved(game: &state::Game) -> bool {
    game.rows.last().map(|&(_, p)| p == WIN_PATTERN) == Some(true)
}

fn game_over(game: &state::Game) -> bool {
    solved(game) || game.gave_up || game.rows.len() >= MAX_GUESSES
}

fn game_state(loaded: &state::Loaded, id: &str, game: &state::Game) -> Result<String, String> {
    let solved = solved(game);
    let over = game_over(game);

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object.str("session", id).raw("rows", |out| {
        out.push('[');
        for (i, &(guess, pattern)) in game.rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let mut row = Object::new(out);
            row.str("word", loaded.data.words.word(guess))
                .str("pattern", &pattern_text(pattern));
            row.end();
        }
        out.push(']');
    });
    object
        .int("turn", game.rows.len() as i64 + 1)
        .int(
            "guesses_left",
            MAX_GUESSES.saturating_sub(game.rows.len()) as i64,
        )
        .int("max_guesses", MAX_GUESSES as i64)
        .bool("solved", solved)
        .bool("gave_up", game.gave_up)
        .bool("over", over)
        .str("source", &game.source);
    object.raw("prior", |out| describe_prior(out, loaded, &game.config));

    if over {
        object
            .str("answer", loaded.data.words.word(game.answer))
            .str("share", &share_grid(&game.rows, solved));
    } else {
        object.null("answer").null("share");
        let odds = letter_odds(loaded, game)?;
        object.raw("letters", |out| {
            out.push('[');
            for (i, &(letter, chance)) in odds.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let mut entry = Object::new(out);
                entry
                    .str("letter", &((b'a' + letter as u8) as char).to_string())
                    .num("chance", chance);
                entry.end();
            }
            out.push(']');
        });
    }
    object.end();
    Ok(out)
}

/// The emoji grid, exactly as `wordle_game.py --share` prints it.
fn share_grid(rows: &[(usize, u8)], solved: bool) -> String {
    let count = if solved {
        rows.len().to_string()
    } else {
        "X".to_string()
    };
    let mut lines = vec![format!("Wordle {count}/{MAX_GUESSES}")];
    for &(_, pattern) in rows {
        lines.push(
            pattern_tiles(pattern)
                .iter()
                .map(|&tile| TILE_BLOCK[tile as usize])
                .collect(),
        );
    }
    lines.join("\n")
}

/// This close to zero and no candidate holds the letter at all.
const LETTER_ABSENT: f64 = 1e-9;

/// The chance the answer contains each letter, likeliest first.
///
/// The weighting is the prior the answer was drawn from and not the settings
/// on the request, so turning a prior off mid-game rethinks the hint but
/// leaves these numbers alone. A letter no remaining candidate holds sits at
/// 0.0 and is left out; the gray keys have said that already. The ones at 1.0
/// stay, and they sort to the front.
fn letter_odds(loaded: &state::Loaded, game: &state::Game) -> Result<Vec<(usize, f64)>, String> {
    let weights = loaded.weights_for(&game.config)?;
    let words = &loaded.data.words;

    let mut candidates: Vec<usize> = (0..words.len()).collect();
    for &(guess, pattern) in &game.rows {
        candidates.retain(|&c| words.score(guess, c) == pattern);
    }

    // the odds themselves are a matrix product on the Python's side and BLAS
    // reduces in an order of its own; the mass at least is the same float
    let total = pairwise_of(candidates.len(), |i| weights[candidates[i]]);
    let mut odds = [0.0f64; 26];
    for &c in &candidates {
        let weight = weights[c];
        let mut seen = [false; 26];
        for &letter in words.letters(c) {
            let slot = (letter - b'a') as usize;
            if !seen[slot] {
                seen[slot] = true;
                odds[slot] += weight;
            }
        }
    }
    if candidates.is_empty() || total <= 0.0 {
        return Ok(Vec::new());
    }

    let mut listed: Vec<(usize, f64)> = odds
        .iter()
        .enumerate()
        .map(|(letter, &mass)| (letter, mass / total))
        .filter(|&(_, chance)| chance > LETTER_ABSENT)
        .collect();
    // likeliest first; alphabetical among ties, which is the order they arrive in
    listed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(listed)
}

// --------------------------------------------------------------- the finder

/// Words the finder lists, best first by its sort.
const MAX_LISTED_MATCHES: usize = 120;

/// Which words could have produced this pattern against this answer.
///
/// Backwards from the rest of the page: the answer is known and the guess is
/// what is being looked for. The Python reads one column of the feedback
/// table; here the column is scored, which is 12,947 comparisons and cheap.
///
/// Each match is then measured three ways, none of which depend on the pattern
/// having been seen -- `left` is how many words would still be possible after
/// it, `bits` is what the guess reveals on average over every answer, and
/// `weight` is the prior's own chance that the guess is itself the answer.
fn find(body: &Value) -> Result<String, String> {
    let loaded = state::loaded().ok_or("The word lists have not loaded yet")?;
    let config = parse_config(body);
    let sort = match body.get("sort").and_then(|v| v.as_str()).unwrap_or("left") {
        name @ ("left" | "bits" | "weight" | "alpha") => name.to_string(),
        other => {
            return Err(format!(
                "Sort must be one of left, bits, weight, alpha: {other}"
            ))
        }
    };

    let answer = body
        .get("answer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let answer = loaded.data.words.index_of(&answer).ok_or_else(|| {
        if answer.is_empty() {
            "Type the answer, a five letter word".to_string()
        } else {
            format!("'{answer}' is not in the word list")
        }
    })?;
    let pattern = parse_pattern(body.get("pattern"))?;

    let weights = loaded.weights_for(&config)?;
    let words = &loaded.data.words;
    let matches: Vec<usize> = (0..words.len())
        .filter(|&guess| words.score(guess, answer) == pattern)
        .collect();

    // `left` is one bucket's size and `bits` the entropy over all of them, one sweep
    // per match; the guess's letters are hoisted as the kernels hoist theirs
    let mut measured: Vec<(usize, f64, usize)> = Vec::with_capacity(matches.len());
    // the sweep below stands in for `kernels.ig_all`, so the mass is summed the
    // way that kernel sums it -- left to right, not through `sum::pairwise`
    let total: f64 = weights.iter().sum();
    for &guess in &matches {
        let letters = words.letters(guess);
        let mut mass = [0.0f64; N_PATTERNS];
        let mut left = 0usize;
        for (candidate, &weight) in weights.iter().enumerate() {
            let bucket = crate::feedback::score_guess(letters, words.letters(candidate));
            mass[bucket as usize] += weight;
            if bucket == pattern {
                left += 1;
            }
        }
        let mut bits = 0.0;
        for share in mass {
            if share > 0.0 {
                let p = share / total;
                bits -= p * p.log2();
            }
        }
        measured.push((guess, bits, left));
    }

    let mut order: Vec<usize> = (0..matches.len()).collect();
    // stable throughout, so ties stay in word-list order as the Python's do
    match sort.as_str() {
        "left" => order.sort_by_key(|&i| measured[i].2),
        "bits" => order.sort_by(|&a, &b| {
            measured[b]
                .1
                .partial_cmp(&measured[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        "weight" => order.sort_by(|&a, &b| {
            weights[matches[b]]
                .partial_cmp(&weights[matches[a]])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => order.sort_by(|&a, &b| words.word(matches[a]).cmp(words.word(matches[b]))),
    }

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object
        .str("answer", words.word(answer))
        .str("pattern", &pattern_text(pattern))
        .str("sort", &sort)
        .int("total", matches.len() as i64)
        .int("shown", matches.len().min(MAX_LISTED_MATCHES) as i64)
        .raw("words", |out| {
            out.push('[');
            for (n, &i) in order.iter().take(MAX_LISTED_MATCHES).enumerate() {
                if n > 0 {
                    out.push(',');
                }
                let (guess, bits, left) = measured[i];
                let mut entry = Object::new(out);
                entry
                    .str("word", words.word(guess))
                    .int("left", left as i64)
                    .num("bits", bits)
                    .num("weight", weights[guess]);
                entry.end();
            }
            out.push(']');
        })
        // as on the board: no clock here, and `wasm-api.js` fills it in
        .int("elapsed_ms", 0);
    object.end();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Over a sweep of seeds every word comes up in proportion to its weight,
    /// and a word the prior rules out never does.
    #[test]
    fn the_draw_follows_the_prior() {
        let weights = [0.5, 0.0, 0.25, 0.25];
        let mut counts = [0usize; 4];
        let steps = 10_000;
        for step in 0..steps {
            counts[draw_answer(&weights, step as f64 / steps as f64)] += 1;
        }
        assert_eq!(counts[1], 0);
        for (index, share) in [0.5, 0.0, 0.25, 0.25].iter().enumerate() {
            let seen = counts[index] as f64 / steps as f64;
            assert!((seen - share).abs() < 0.01, "{index} came up {seen} times");
        }
    }

    /// The endpoints of the range are inside it, however the arithmetic rounds.
    #[test]
    fn no_seed_falls_off_either_end() {
        let weights = [1.0, 2.0, 3.0, 0.0];
        for seed in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            assert!(draw_answer(&weights, seed) < 3);
        }
    }

    #[test]
    fn a_lost_game_shares_an_x() {
        // three rows, none of them a win: "Wordle X/6" and three lines
        let rows = [(0usize, 0u8), (1, 1), (2, 3)];
        let grid = share_grid(&rows, false);
        assert!(grid.starts_with("Wordle X/6\n"));
        assert_eq!(grid.lines().count(), 4);
        assert_eq!(share_grid(&rows, true).lines().next(), Some("Wordle 3/6"));
    }
}
