//! Scaffolding the parity binaries share.
//!
//! Candidate sets are built on the Python side and handed over, so that a
//! disagreement is always about the code under test and never about the inputs.
//! This is the reading end of that, plus the little-endian writers Python picks
//! the results back up with.

use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// Reads `[u32 count]` then, per set, `[u32 len][len x u64]`.
pub fn read_sets(path: &str) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    let mut at = 0usize;
    let u32_at =
        |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());

    let count = u32_at(&bytes, at) as usize;
    at += 4;

    let mut sets = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u32_at(&bytes, at) as usize;
        at += 4;
        let mut set = Vec::with_capacity(len);
        for _ in 0..len {
            set.push(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize);
            at += 8;
        }
        sets.push(set);
    }
    Ok(sets)
}

/// Little-endian f64 and i64 streams, which is what `np.fromfile` wants.
pub struct Sink {
    pub floats: BufWriter<File>,
    pub ints: BufWriter<File>,
}

impl Sink {
    pub fn create(out_dir: &str, stem: &str) -> std::io::Result<Sink> {
        Ok(Sink {
            floats: BufWriter::new(File::create(format!("{out_dir}/{stem}.f64"))?),
            ints: BufWriter::new(File::create(format!("{out_dir}/{stem}.i64"))?),
        })
    }

    pub fn float(&mut self, value: f64) -> std::io::Result<()> {
        self.floats.write_all(&value.to_le_bytes())
    }

    pub fn all_floats(&mut self, values: &[f64]) -> std::io::Result<()> {
        for &v in values {
            self.float(v)?;
        }
        Ok(())
    }

    pub fn int(&mut self, value: i64) -> std::io::Result<()> {
        self.ints.write_all(&value.to_le_bytes())
    }

    /// A variable-length index list, length-prefixed so Python can split it.
    pub fn indices(&mut self, values: &[usize]) -> std::io::Result<()> {
        self.ints.write_all(&(values.len() as u32).to_le_bytes())?;
        for &v in values {
            self.int(v as i64)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.floats.flush()?;
        self.ints.flush()
    }
}
