//! Checks the coloring rule against the shipped `feedback_table.npy`, entry by
//! entry.
//!
//! The table is the oracle and it is a complete one: every pair the solver can
//! ever ask about is in it, so agreeing on all 12,947^2 leaves nothing about
//! the rule untested.
//!
//!     parity_feedback [repo root]

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::process::ExitCode;

use wordle_core::WordList;

/// Reads a version 1.0 `.npy` of `uint8`, returning `(rows, cols, bytes)`.
/// Only the shape this repo writes is supported: C order, one dtype, 2-D.
fn read_npy_u1(path: &str) -> Result<(usize, usize, Vec<u8>), String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic[1..6] != b"NUMPY" {
        return Err(format!("not a .npy file: {path}"));
    }
    let mut len_bytes = [0u8; 2];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| e.to_string())?;
    let header_len = u16::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;
    let header = String::from_utf8_lossy(&header).into_owned();

    if !header.contains("'|u1'") {
        return Err(format!("expected uint8 data in {path}"));
    }
    if !header.contains("'fortran_order': False") {
        return Err(format!("expected C order in {path}"));
    }
    let shape = header
        .split('(')
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| "no shape in header".to_string())?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!("expected a 2-D shape, got ({shape})"));
    }

    let mut data = vec![0u8; dims[0] * dims[1]];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("short read on {path}: {e}"))?;
    Ok((dims[0], dims[1], data))
}

fn check(root: &str) -> Result<bool, String> {
    let words =
        WordList::load(&format!("{root}/data/derived/ind2word.json")).map_err(|e| e.to_string())?;
    let (rows, cols, table) = read_npy_u1(&format!("{root}/data/derived/feedback_table.npy"))?;

    println!("  words {}, table {rows} x {cols}", words.len());
    if rows != words.len() || cols != words.len() {
        println!("  FAIL: table does not match the word list");
        return Ok(false);
    }

    let mut mismatches = 0usize;
    for guess in 0..rows {
        let row = &table[guess * cols..(guess + 1) * cols];
        for (answer, &expected) in row.iter().enumerate() {
            let got = words.score(guess, answer);
            if got != expected {
                if mismatches < 5 {
                    println!(
                        "  {} vs {}: got {got}, table says {expected}",
                        words.word(guess),
                        words.word(answer)
                    );
                }
                mismatches += 1;
            }
        }
    }

    let total = rows * cols;
    if mismatches > 0 {
        println!("  FAIL: {mismatches} of {total} entries disagree");
        return Ok(false);
    }
    println!("  ok: all {total} entries agree");
    Ok(true)
}

fn main() -> ExitCode {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    match check(&root) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            println!("  FAIL: {message}");
            ExitCode::FAILURE
        }
    }
}
