//! Drives `wordle_core::api` from a shell, the way the page drives it from
//! `fetch`.
//!
//! The endpoints answer the browser, but they are ordinary functions and
//! debugging them through a page is guesswork with a slow loop. This loads the
//! same files the page fetches, then replays whatever requests are asked for.
//!
//!     api_probe <repo root> <path> [json body] [<path> [json body] ...]
//!
//! `/api/init` is done for you before anything else runs.

use std::env;
use std::fs;
use std::process::ExitCode;

use wordle_core::api;
use wordle_core::json::{write_string, Object};

fn init_body(root: &str) -> Result<String, Box<dyn std::error::Error>> {
    let read = |path: &str| fs::read_to_string(format!("{root}/{path}"));

    let mut out = String::new();
    let mut object = Object::new(&mut out);
    object
        .str("ind2word", &read("data/derived/ind2word.json")?)
        .str("wordles", &read("data/wordles.json")?)
        .str("solutions", &read("data/solutions.json")?)
        .str("answer_like", &read("data/answer_like.json")?)
        .str("wordlebot", &read("data/wordlebot_weights.json")?)
        .int("mask_last", 1);

    // the two files build_site writes, if they have been built
    for (key, path) in [
        ("fits", "site/fits.json"),
        ("openings", "site/openings.json"),
    ] {
        if let Ok(text) = read(path) {
            object.raw(key, |out| out.push_str(&text));
        }
    }
    object.end();
    let _ = write_string;
    Ok(out)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".to_string());

    let body = match init_body(&root) {
        Ok(body) => body,
        Err(error) => {
            println!("  FAIL: could not read the data files: {error}");
            return ExitCode::FAILURE;
        }
    };
    let reply = api::handle("/api/init", &body);
    if reply.contains("\"error\":\"") {
        println!("  FAIL: init said {reply}");
        return ExitCode::FAILURE;
    }

    let rest: Vec<String> = args.collect();
    if rest.is_empty() {
        println!("{}", api::handle("/api/status", ""));
        return ExitCode::SUCCESS;
    }

    let mut failed = false;
    let mut i = 0;
    while i < rest.len() {
        let path = &rest[i];
        let next_is_body = rest.get(i + 1).map(|s| !s.starts_with("/api/")) == Some(true);
        let body = if next_is_body {
            rest[i + 1].as_str()
        } else {
            ""
        };
        i += if next_is_body { 2 } else { 1 };

        let reply = api::handle(path, body);
        println!("{path}\n{reply}\n");
        failed |= reply.contains("\"error\":\"");
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
