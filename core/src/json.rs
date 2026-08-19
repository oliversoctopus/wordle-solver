//! A small JSON reader, so the core keeps its promise of no dependencies.
//!
//! Everything it parses is machine-written by this repository -- the word
//! lists, the answer record, the two weight tables -- so it aims at being
//! correct on that rather than at being a general library. It does handle the
//! standard escapes, because being quietly wrong about a string is worse than
//! being slow.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Field of an object, or `None` if this is not an object or has no such
    /// key. `solutions.json`'s first entry has no `days_since_launch`, so a
    /// missing field is ordinary rather than exceptional.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object().and_then(|o| o.get(key))
    }
}

#[derive(Debug)]
pub struct JsonError {
    pub at: usize,
    pub what: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.what)
    }
}

impl std::error::Error for JsonError {}

pub fn parse(text: &str) -> Result<Value, JsonError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    parser.skip_space();
    let value = parser.value()?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(parser.fail("trailing content"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Parser<'a> {
    fn fail(&self, what: &str) -> JsonError {
        JsonError {
            at: self.at,
            what: what.to_string(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while matches!(
            self.peek(),
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
        ) {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.fail(&format!("expected {}", byte as char)))
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value, JsonError> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.fail("unknown literal"))
        }
    }

    fn value(&mut self) -> Result<Value, JsonError> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(_) => self.number(),
            None => Err(self.fail("unexpected end of input")),
        }
    }

    fn object(&mut self) -> Result<Value, JsonError> {
        self.expect(b'{')?;
        let mut out = HashMap::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(out));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.expect(b':')?;
            self.skip_space();
            let value = self.value()?;
            out.insert(key, value);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(out));
                }
                _ => return Err(self.fail("expected , or } in object")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, JsonError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(out));
        }
        loop {
            self.skip_space();
            out.push(self.value()?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(out));
                }
                _ => return Err(self.fail("expected , or ] in array")),
            }
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(self.fail("unterminated string")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.peek().ok_or_else(|| self.fail("dangling escape"))?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.at..self.at + 4)
                                .ok_or_else(|| self.fail("short \\u escape"))?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| self.fail("bad \\u"))?,
                                16,
                            )
                            .map_err(|_| self.fail("bad \\u"))?;
                            self.at += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.fail("unknown escape")),
                    }
                }
                Some(_) => {
                    // copy the whole UTF-8 sequence, whatever its length
                    let start = self.at;
                    self.at += 1;
                    while matches!(self.peek(), Some(b) if b & 0xc0 == 0x80) {
                        self.at += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.at])
                            .map_err(|_| self.fail("invalid UTF-8"))?,
                    );
                }
            }
        }
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        while matches!(self.peek(), Some(b) if b.is_ascii_digit()
                       || b == b'.' || b == b'e' || b == b'E'
                       || b == b'+' || b == b'-')
        {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Value::Number)
            .ok_or_else(|| self.fail("not a number"))
    }
}

pub fn parse_file(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_shapes_this_repo_writes() {
        let array = parse(r#"["cigar","rebut"]"#).unwrap();
        assert_eq!(array.as_array().unwrap().len(), 2);
        assert_eq!(array.as_array().unwrap()[0].as_str(), Some("cigar"));

        let table = parse(r#"{"aback": 1.0, "abase": 0.911774}"#).unwrap();
        assert_eq!(table.get("abase").unwrap().as_f64(), Some(0.911774));

        // solutions.json's first entry has no days_since_launch
        let rows =
            parse(r#"[{"id":1,"solution":"cigar"},{"id":2,"days_since_launch":1}]"#).unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows[0].get("solution").unwrap().as_str(), Some("cigar"));
        assert!(rows[0].get("days_since_launch").is_none());
        assert_eq!(
            rows[1].get("days_since_launch").unwrap().as_f64(),
            Some(1.0)
        );
    }

    #[test]
    fn reads_negatives_and_exponents() {
        assert_eq!(parse("-15.851").unwrap().as_f64(), Some(-15.851));
        assert_eq!(parse("1e-3").unwrap().as_f64(), Some(0.001));
    }

    #[test]
    fn rejects_trailing_junk() {
        assert!(parse("[1,2] extra").is_err());
    }
}

// ------------------------------------------------------------------ writing

/// Appends `text` as a JSON string, escapes and all.
pub fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Appends a number, or `null` where JSON has no way to say it.
pub fn write_f64(out: &mut String, value: f64) {
    if value.is_finite() {
        // enough digits to round-trip, without the trailing noise of {:?}
        out.push_str(&format!("{value}"));
    } else {
        out.push_str("null");
    }
}

/// Builds one JSON object, comma placement handled.
pub struct Object<'a> {
    out: &'a mut String,
    first: bool,
}

impl<'a> Object<'a> {
    pub fn new(out: &'a mut String) -> Object<'a> {
        out.push('{');
        Object { out, first: true }
    }

    fn key(&mut self, name: &str) {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
        write_string(self.out, name);
        self.out.push(':');
    }

    pub fn str(&mut self, name: &str, value: &str) -> &mut Self {
        self.key(name);
        write_string(self.out, value);
        self
    }

    pub fn num(&mut self, name: &str, value: f64) -> &mut Self {
        self.key(name);
        write_f64(self.out, value);
        self
    }

    pub fn int(&mut self, name: &str, value: i64) -> &mut Self {
        self.key(name);
        self.out.push_str(&value.to_string());
        self
    }

    pub fn bool(&mut self, name: &str, value: bool) -> &mut Self {
        self.key(name);
        self.out.push_str(if value { "true" } else { "false" });
        self
    }

    pub fn null(&mut self, name: &str) -> &mut Self {
        self.key(name);
        self.out.push_str("null");
        self
    }

    /// Writes the value with `body`, which must append exactly one JSON value.
    pub fn raw(&mut self, name: &str, body: impl FnOnce(&mut String)) -> &mut Self {
        self.key(name);
        body(self.out);
        self
    }

    pub fn strings(
        &mut self,
        name: &str,
        values: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> &mut Self {
        self.raw(name, |out| {
            out.push('[');
            for (i, value) in values.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, value.as_ref());
            }
            out.push(']');
        })
    }

    pub fn end(&mut self) {
        self.out.push('}');
    }
}

#[cfg(test)]
mod write_tests {
    use super::*;

    // The control characters have to leave as two-character escapes, not as
    // themselves: a raw newline inside a JSON string is invalid, and every
    // parser but this one's would reject it.
    #[test]
    fn escapes_survive_a_round_trip() {
        let awkward = "a\"b\\c\nd\te\r\u{1}f";
        let mut out = String::new();
        write_string(&mut out, awkward);
        assert!(!out.contains('\n'), "a raw newline escaped into the output");
        assert_eq!(parse(&out).unwrap().as_str(), Some(awkward));
    }

    #[test]
    fn round_trips_through_the_parser() {
        let mut out = String::new();
        let mut object = Object::new(&mut out);
        object
            .str("word", "sa\"let")
            .int("turn", 3)
            .num("score", 3.077)
            .bool("solved", false)
            .null("estimator")
            .strings("rows", ["abbey", "crane"]);
        object.end();

        let parsed = parse(&out).expect("valid JSON");
        assert_eq!(parsed.get("word").unwrap().as_str(), Some("sa\"let"));
        assert_eq!(parsed.get("turn").unwrap().as_f64(), Some(3.0));
        assert_eq!(parsed.get("score").unwrap().as_f64(), Some(3.077));
        assert_eq!(parsed.get("estimator"), Some(&Value::Null));
        assert_eq!(parsed.get("rows").unwrap().as_array().unwrap().len(), 2);
    }
}
