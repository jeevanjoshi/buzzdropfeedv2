//! Minimal dependency-free JSON parser/serializer (RFC 8259 subset).
//!
//! Only what the dashboard needs: objects, arrays, strings, numbers, booleans,
//! null — deep enough to walk the pipeline's state / budget / channel JSON files
//! without pulling in external JSON crates (keeps the binary trivially portable
//! to the Raspberry Pi edge node where no extra build deps are desired).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    pub fn get_str(&self, key: &str) -> Option<String> {
        match self {
            Value::Obj(m) => match m.get(key) {
                Some(Value::Str(s)) => Some(s.clone()),
                Some(Value::Num(n)) => Some(fmt_num(*n)),
                Some(Value::Bool(b)) => Some(b.to_string()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn get_obj(&self, key: &str) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Obj(m) => match m.get(key) {
                Some(Value::Obj(o)) => Some(o),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Navigate a dotted path into a parsed JSON tree; returns the referenced Value.
pub fn get_path<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = v;
    for k in path {
        if let Value::Obj(m) = cur {
            cur = m.get(*k)?;
        } else {
            return None;
        }
    }
    Some(cur)
}

// ── parser ────────────────────────────────────────────────────────────────

pub fn parse(s: &str) -> Option<Value> {
    let mut c = Cursor {
        b: s.as_bytes(),
        i: 0,
    };
    let v = parse_value(&mut c)?;
    c.skip_ws();
    if c.i < c.b.len() {
        return None; // trailing garbage
    }
    Some(v)
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }
    fn next(&mut self) -> Option<u8> {
        let c = self.b.get(self.i).copied();
        if c.is_some() {
            self.i += 1;
        }
        c
    }
}

fn parse_value(c: &mut Cursor) -> Option<Value> {
    match c.peek()? {
        b'{' => parse_object(c),
        b'[' => parse_array(c),
        b'"' => Some(Value::Str(parse_string(c)?)),
        b't' => lit(c, b"true").map(|_| Value::Bool(true)),
        b'f' => lit(c, b"false").map(|_| Value::Bool(false)),
        b'n' => lit(c, b"null").map(|_| Value::Null),
        _ if is_num_start(c.peek().unwrap_or(0)) => parse_number(c),
        _ => None,
    }
}

fn lit(c: &mut Cursor, word: &[u8]) -> Option<()> {
    for w in word {
        if c.next()? != *w {
            return None;
        }
    }
    Some(())
}

fn parse_object(c: &mut Cursor) -> Option<Value> {
    assert_eq!(c.next()?, b'{');
    let mut map = BTreeMap::new();
    c.skip_ws();
    if c.next()? == b'}' {
        return Some(Value::Obj(map));
    }
    c.i -= 1; // put back the byte we tentatively read (it wasn't '}')
    loop {
        c.skip_ws();
        let k = parse_string(c)?;
        c.skip_ws();
        if c.next()? != b':' {
            return None;
        }
        let v = parse_value(c)?;
        map.insert(k, v);
        c.skip_ws();
        match c.next()? {
            b',' => continue,
            b'}' => break,
            _ => return None,
        }
    }
    Some(Value::Obj(map))
}

fn parse_array(c: &mut Cursor) -> Option<Value> {
    assert_eq!(c.next()?, b'[');
    let mut arr = Vec::new();
    c.skip_ws();
    if c.next()? == b']' {
        return Some(Value::Arr(arr));
    }
    c.i -= 1;
    loop {
        arr.push(parse_value(c)?);
        c.skip_ws();
        match c.next()? {
            b',' => continue,
            b']' => break,
            _ => return None,
        }
    }
    Some(Value::Arr(arr))
}

fn parse_string(c: &mut Cursor) -> Option<String> {
    if c.next()? != b'"' {
        return None;
    }
    let mut out = String::new();
    loop {
        let ch = c.next()?;
        match ch {
            b'"' => break,
            b'\\' => {
                let esc = c.next()?;
                out.push(match esc {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{0008}',
                    b'f' => '\u{000C}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let hi = parse_hex(c, 4)?;
                        // handle surrogate pairs minimally
                        let cp: u32 = if (0xD800..0xDC00).contains(&hi) {
                            let lo = c.next()?;
                            if lo != b'\\' || c.next()? != b'u' {
                                return None;
                            }
                            let lo = parse_hex(c, 4)?;
                            if (0xDC00..0xE000).contains(&lo) {
                                0x10000 + (((hi - 0xD800) << 10) | (lo - 0xDC00))
                            } else {
                                return None;
                            }
                        } else {
                            hi
                        };
                        out.push(char::from_u32(cp)?);
                        continue;
                    }
                    _ => return None,
                });
            }
            _ => out.push(ch as char),
        }
    }
    Some(out)
}

fn parse_hex(c: &mut Cursor, n: usize) -> Option<u32> {
    let mut v: u32 = 0;
    for _ in 0..n {
        let ch = c.next()? as char;
        let d = ch.to_digit(16)?;
        v = v * 16 + d;
    }
    Some(v)
}

fn is_num_start(b: u8) -> bool {
    b.is_ascii_digit() || b == b'-'
}

fn parse_number(c: &mut Cursor) -> Option<Value> {
    let start = c.i;
    // consume number chars
    while c.i < c.b.len() {
        let cc = c.b[c.i] as char;
        if cc.is_ascii_digit() || cc == '-' || cc == '+' || cc == '.' || cc == 'e' || cc == 'E' {
            c.i += 1;
        } else {
            break;
        }
    }
    let raw = std::str::from_utf8(&c.b[start..c.i]).ok()?;
    raw.parse::<f64>().ok().map(Value::Num)
}

// ── serializer ────────────────────────────────────────────────────────────

pub fn serialize(v: &Value) -> String {
    let mut out = String::new();
    write_value(v, &mut out);
    out
}

fn write_value(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(n) => out.push_str(&fmt_num(*n)),
        Value::Str(s) => write_string(s, out),
        Value::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(x, out);
            }
            out.push(']');
        }
        Value::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_value(x, out);
            }
            out.push('}');
        }
    }
}

pub fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
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

pub fn fmt_num(n: f64) -> String {
    if n == n.trunc() && n.is_finite() && n.abs() < 1e15 {
        return format!("{}", n as i64);
    }
    let s = format!("{:.4}", n);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}
