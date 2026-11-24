#![allow(dead_code)]
use serde_json::Value;
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

pub fn golden(name: &str) -> Value {
    let p = root().join("tests/golden").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

/// Structural equality with floats within 1e-9 (relative above 1).
pub fn close(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let kx: Vec<_> = x.keys().collect();
            let ky: Vec<_> = y.keys().collect();
            assert_eq!(kx, ky, "{path}: keys");
            for k in x.keys() {
                close(&x[k], &y[k], &format!("{path}.{k}"));
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: length");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                close(p, q, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() <= 1e-9 * x.abs().max(1.0), "{path}: {x} vs {y}");
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}
