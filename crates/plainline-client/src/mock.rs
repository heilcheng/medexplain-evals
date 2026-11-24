use crate::{ChatClient, ClientError, Message};
use async_trait::async_trait;
use fancy_regex::Regex;
use serde_json::json;
use std::collections::HashSet;
use std::sync::LazyLock;

static SENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?<=[.!?])\s+").unwrap());
static WORD: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"[a-z0-9]+").unwrap());
static SOURCE: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"(?s)<source>\n(.*?)\n</source>").unwrap());
static EXPL: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?s)<explanation>\n(.*?)\n</explanation>").unwrap());
static ITEM: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"(?m)^\[(\S+)\] (.*)$").unwrap());

/// Deterministic offline stand-in. Not a measurement; used for tests and smoke runs.
///
/// Writer role: echoes a share of the source sentences (fewer for lay readers).
/// Judge role: marks a fact stated when most of its content words occur in the explanation.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockClient;

fn split_sentences(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in SENT.find_iter(src).flatten() {
        out.push(src[last..m.start()].to_string());
        last = m.end();
    }
    out.push(src[last..].to_string());
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

impl MockClient {
    fn write(user: &str) -> String {
        let src = SOURCE.captures(user).map_or(user, |c| c.get(1).unwrap().as_str());
        let sents = split_sentences(src);
        let div = if user.to_lowercase().contains("grade") { 3 } else { 2 };
        let keep = (sents.len() / div).max(2);
        sents.into_iter().take(keep).collect::<Vec<_>>().join(" ")
    }

    fn judge(user: &str) -> String {
        let expl_text = EXPL
            .captures(user)
            .map_or(String::new(), |c| c.get(1).unwrap().as_str().to_lowercase());
        let expl: HashSet<&str> = WORD.find_iter(&expl_text).map(|m| m.as_str()).collect();
        let tail = user.rsplit("</explanation>").next().unwrap_or("");
        let mut out = Vec::new();
        for c in ITEM.captures_iter(tail) {
            let (fid, text) = (&c[1], c[2].to_lowercase());
            if fid.starts_with('n') {
                out.push(json!({"id": fid, "verdict": "not_asserted"}));
                continue;
            }
            let words: Vec<&str> = WORD
                .find_iter(&text)
                .map(|m| m.as_str())
                .filter(|w| w.len() > 3)
                .collect();
            let hit = words.iter().filter(|w| expl.contains(*w)).count() as f64 / words.len().max(1) as f64;
            out.push(json!({"id": fid, "verdict": if hit >= 0.6 { "stated" } else { "absent" }}));
        }
        json!({"verdicts": out}).to_string()
    }
}

#[async_trait]
impl ChatClient for MockClient {
    fn name(&self) -> &str {
        "mock"
    }

    async fn complete(&self, messages: &[Message], _temperature: f64, _max_tokens: u32) -> Result<String, ClientError> {
        let user = messages.last().map_or("", |m| m.content.as_str());
        if messages.first().is_some_and(|m| m.content.contains("PLAINLINE_JUDGE")) {
            Ok(Self::judge(user))
        } else {
            Ok(Self::write(user))
        }
    }
}
