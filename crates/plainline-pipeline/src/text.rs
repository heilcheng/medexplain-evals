//! Small text helpers shared by the pipeline.

use fancy_regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static SENT_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?<=[.!?])\s+(?=[A-Z0-9"(\[])"#).unwrap());
static TAG: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"</?[A-Za-z][^<>]*>").unwrap());
static WS: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\s+").unwrap());
static NONWORD: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\W+").unwrap());

/// Decode HTML entities, drop tags, fold non-breaking spaces and collapse whitespace.
pub fn clean(text: &str) -> String {
    let unescaped = crate::entities::unescape(text);
    let no_tags = TAG.replace_all(&unescaped, " ");
    let spaced = no_tags.replace('\u{a0}', " ");
    WS.replace_all(&spaced, " ").trim().to_string()
}

/// Split `text` on every match of `re`, like Python's `re.split` for a pattern without groups.
pub(crate) fn regex_split<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text).flatten() {
        out.push(&text[last..m.start()]);
        last = m.end();
    }
    out.push(&text[last..]);
    out
}

pub fn sentences(text: &str) -> Vec<String> {
    let cleaned = clean(text);
    regex_split(&SENT_SPLIT, &cleaned)
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Cut at a sentence boundary at or under `max_words` (at least one sentence).
pub fn truncate_words(text: &str, max_words: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut n = 0;
    for s in sentences(text) {
        let w = s.split_whitespace().count();
        if !out.is_empty() && n + w > max_words {
            break;
        }
        n += w;
        out.push(s);
    }
    out.join(" ")
}

/// A lowercase key with every non-word character removed.
pub(crate) fn squash(s: &str) -> String {
    NONWORD.replace_all(&s.to_lowercase(), "").into_owned()
}

/// Drop sentences that repeat an earlier one (SPL highlights are often repeated in the full text).
pub fn dedupe_sentences(text: &str) -> String {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for s in sentences(text) {
        if seen.insert(squash(&s)) {
            out.push(s);
        }
    }
    out.join(" ")
}
