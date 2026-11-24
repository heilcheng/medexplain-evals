//! JSON text in the layout Python's `json.dumps(..., ensure_ascii=False)` uses
//! (`", "` between items and `": "` after keys), so JSONL files written by the Rust
//! code stay byte-compatible with the files the Python version wrote.

use serde::Serialize;
use serde_json::ser::Formatter;
use std::io;

#[derive(Default)]
struct PyFormatter;

impl Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + io::Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// Serialise on one line in the Python layout.
pub fn to_string<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(buf).expect("serde_json emits UTF-8"))
}
