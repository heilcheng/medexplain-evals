//! Readability: Flesch-Kincaid grade and SMOG, with a pass/fail band.
//!
//! Syllables are counted with a vowel-group heuristic, so values differ slightly from
//! dictionary-based tools. The band is a gate, not a quality score.

use regex::Regex;
use std::sync::LazyLock;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z'-]*").unwrap());
static SENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?]+(?:\s|$)|\n{2,}").unwrap());
static VOWELS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[aeiouy]+").unwrap());
static ED_OK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(t|d)ed$|(s|x|z|ch|sh|ce|ge)s$").unwrap());

/// Syllables in one word (vowel groups with a few silent-ending corrections), at least 1.
pub fn syllables(word: &str) -> usize {
    let lower = word.to_lowercase();
    let w = lower.trim_matches(|c| c == '\'' || c == '-');
    if w.is_empty() {
        return 0;
    }
    let mut n = VOWELS.find_iter(w).count();
    if w.ends_with('e') && !w.ends_with("le") && !w.ends_with("ee") && n > 1 {
        n -= 1;
    }
    if (w.ends_with("es") || w.ends_with("ed")) && n > 1 && !ED_OK.is_match(w) {
        n -= 1;
    }
    n.max(1)
}

/// (words, sentences, syllables, polysyllables)
fn counts(text: &str) -> (usize, usize, usize, usize) {
    let words: Vec<&str> = WORD.find_iter(text).map(|m| m.as_str()).collect();
    let sents = split_nonblank(text).max(1);
    let syl: Vec<usize> = words.iter().map(|w| syllables(w)).collect();
    let poly = syl.iter().filter(|&&s| s >= 3).count();
    (words.len(), sents, syl.iter().sum(), poly)
}

/// Number of non-blank pieces when `text` is split at sentence ends, as Python's `re.split`.
fn split_nonblank(text: &str) -> usize {
    let mut n = 0;
    let mut last = 0;
    for m in SENT.find_iter(text) {
        if !text[last..m.start()].trim().is_empty() {
            n += 1;
        }
        last = m.end();
    }
    if !text[last..].trim().is_empty() {
        n += 1;
    }
    n
}

pub fn fk_grade(text: &str) -> f64 {
    let (w, s, syl, _) = counts(text);
    if w == 0 {
        return 0.0;
    }
    0.39 * (w as f64 / s as f64) + 11.8 * (syl as f64 / w as f64) - 15.59
}

pub fn smog_grade(text: &str) -> f64 {
    let (w, s, _, poly) = counts(text);
    if w == 0 {
        return 0.0;
    }
    1.043 * (poly as f64 * (30.0 / s as f64)).sqrt() + 3.1291
}

pub fn word_count(text: &str) -> usize {
    WORD.find_iter(text).count()
}

/// A grade band; either side may be unbounded.
pub type Band = (Option<f64>, Option<f64>);

/// `None` when the condition has no band.
pub fn in_band(grade: f64, band: Option<Band>) -> Option<bool> {
    let (lo, hi) = band?;
    Some(lo.is_none_or(|l| grade >= l) && hi.is_none_or(|h| grade <= h))
}

/// Round half to even on the exact binary value, like Python's `round(x, ndigits)`.
pub fn round_py(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}
