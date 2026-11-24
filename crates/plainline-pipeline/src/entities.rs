//! `html.unescape` as Python implements it, including named entities written without a
//! semicolon (`&amp`, `&copy`) and the legacy numeric ranges. The tables in
//! `data/entities.tsv` come from the Python standard library (see `data/gen_entities.py`).

use regex::{Captures, Regex};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

struct Tables {
    named: HashMap<String, String>,
    replaced: HashMap<u32, char>,
    invalid: HashSet<u32>,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let mut t = Tables { named: HashMap::new(), replaced: HashMap::new(), invalid: HashSet::new() };
    let cps = |s: &str| -> String {
        s.split(' ').map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap()).collect()
    };
    for line in include_str!("../data/entities.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "E" => {
                t.named.insert(f[1].to_string(), cps(f[2]));
            }
            "R" => {
                t.replaced.insert(u32::from_str_radix(f[1], 16).unwrap(), cps(f[2]).chars().next().unwrap());
            }
            "X" => {
                t.invalid.insert(u32::from_str_radix(f[1], 16).unwrap());
            }
            _ => {}
        }
    }
    t
});

static CHARREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\x0c <&#;]{1,32};?)").unwrap());

fn replace(caps: &Captures) -> String {
    let t = &*TABLES;
    let s = &caps[1];
    if let Some(num) = s.strip_prefix('#') {
        let (digits, radix) = match num.strip_prefix(['x', 'X']) {
            Some(h) => (h, 16),
            None => (num, 10),
        };
        let digits = digits.trim_end_matches(';');
        // Python integers do not overflow; anything this large is out of range anyway.
        let n = u64::from_str_radix(digits, radix).unwrap_or(u64::MAX);
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        if let Some(c) = t.replaced.get(&n) {
            return c.to_string();
        }
        if (0xD800..=0xDFFF).contains(&n) || n > 0x10FFFF {
            return "\u{fffd}".into();
        }
        if t.invalid.contains(&n) {
            return String::new();
        }
        return char::from_u32(n).map(String::from).unwrap_or_default();
    }
    if let Some(v) = t.named.get(s) {
        return v.clone();
    }
    // The longest prefix that is a legacy entity, as the HTML standard does.
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    for x in (2..chars.len()).rev() {
        let cut = chars[x].0;
        if let Some(v) = t.named.get(&s[..cut]) {
            return format!("{v}{}", &s[cut..]);
        }
    }
    format!("&{s}")
}

/// Replace character references in `text`, like `html.unescape`.
pub fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    CHARREF.replace_all(text, |c: &Captures| replace(c)).into_owned()
}
