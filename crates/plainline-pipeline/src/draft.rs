//! Draft atomic facts. Every output is marked draft and needs clinician review.
//!
//! Two generators:
//!   `llm_draft`   a model call with `EXTRACT_SYSTEM` (the intended generator).
//!   `rule_draft`  a deterministic section-and-keyword heuristic, used for the pilot when no
//!                 model budget was spent. Its facts are whole source sentences, not truly
//!                 atomic, and its must_not_say list is a short generic template.
//!
//! Neither is evidence of clinical validity.

use crate::text::{sentences, squash};
use crate::{PipelineError, Result};
use fancy_regex::Regex;
use plainline_client::ChatClient;
use plainline_core::judge::extract_json;
use plainline_core::prompts::extract_messages;
use plainline_core::schema::{Fact, FactSet, Source, SourceKind, Status, Tier};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

pub const RULE_VERSION: &str = "rule-v0";

static HARD: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)\b(not exceed|maximum|do not|must not|should not|never|contraindicated|avoid|discontinue|stop|fatal|death|life-threatening|seek|call|emergency|911|immediately)\b",
    )
    .unwrap()
});
static SERIOUS: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)\b(risk|serious|monitor|bleeding|toxicity|overdose|suicid|pregnan|reaction|symptoms?|warning)\b",
    )
    .unwrap()
});
static DOSE: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\b\d+\s?(mg|mcg|g|mL|units?)\b").unwrap());
static URGENT: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?i)\b(call|seek|emergency|911|immediately|right away|go to)\b").unwrap());
static INFO: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?i)\b(symptoms?|cause[sd]?|treat|treatment|risk|prevent|diagnos)").unwrap());
static SECTION_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\n([A-Z][A-Z ,/&\-]+)\n").unwrap());
static HEAD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(WARNING:\s+[A-Z0-9 ,/&;\-]+\s*[\u{2022}\-]?\s*|\d+(\.\d+)*\s+[A-Z][A-Z &,/\-]{3,}\s+|\d+(\.\d+)*\s+(?=[A-Z][a-z]+ ))(?=[A-Z][a-z])",
    )
    .unwrap()
});

/// Split the source text on the upper-case headings written by the fetchers.
fn split_sections(text: &str) -> HashMap<String, String> {
    let text = format!("\n\n{text}");
    // Python's re.split with one capture group: pieces and captured headings alternate.
    let mut parts: Vec<String> = Vec::new();
    let mut last = 0;
    for caps in SECTION_SPLIT.captures_iter(&text).flatten() {
        let m = caps.get(0).unwrap();
        parts.push(text[last..m.start()].to_string());
        parts.push(caps.get(1).unwrap().as_str().to_string());
        last = m.end();
    }
    parts.push(text[last..].to_string());
    let mut secs = HashMap::new();
    secs.insert(String::new(), parts[0].clone());
    let mut i = 1;
    while i + 1 < parts.len() {
        secs.insert(parts[i].trim().to_lowercase(), parts[i + 1].clone());
        i += 2;
    }
    secs
}

/// Drop leftover section headings glued to the start of a sentence.
fn norm(sent: &str) -> String {
    let s = sent.trim();
    // `^` anchors the pattern, so at most one heading is removed.
    match HEAD.find(s) {
        Ok(Some(m)) => s[m.end()..].trim().to_string(),
        _ => s.to_string(),
    }
}

fn usable(s: &str) -> bool {
    let n = s.split_whitespace().count();
    (5..=55).contains(&n) && !s.trim_end().ends_with('?')
}

fn caps(tier: Tier) -> usize {
    match tier {
        Tier::MustKeep => 8,
        Tier::ShouldKeep => 7,
        Tier::Optional => 3,
    }
}

struct Drafter {
    facts: Vec<Fact>,
    seen: HashSet<String>,
    counts: HashMap<&'static str, usize>,
}

impl Drafter {
    fn add(&mut self, sent: &str, tier: Tier, section: &str) {
        let sent = norm(sent);
        let key: String = squash(&sent).chars().take(80).collect();
        if self.seen.contains(&key) || !usable(&sent) {
            return;
        }
        self.seen.insert(key);
        let fact_id = format!("f{}", self.facts.len() + 1);
        self.facts.push(Fact {
            fact_id,
            text: sent,
            tier,
            section: section.to_string(),
        });
    }

    fn add_cap(&mut self, sent: &str, tier: Tier, section: &str) {
        let name = match tier {
            Tier::MustKeep => "must_keep",
            Tier::ShouldKeep => "should_keep",
            Tier::Optional => "optional",
        };
        if self.counts.get(name).copied().unwrap_or(0) < caps(tier) {
            let before = self.facts.len();
            self.add(sent, tier, section);
            *self.counts.entry(name).or_insert(0) += self.facts.len() - before;
        }
    }
}

/// Deterministic section-and-keyword fact draft. Marked as a draft.
pub fn rule_draft(src: &Source) -> FactSet {
    let secs = split_sections(&src.text);
    let sec = |k: &str| secs.get(k).map_or("", String::as_str);
    let mut d = Drafter {
        facts: Vec::new(),
        seen: HashSet::new(),
        counts: HashMap::new(),
    };

    if src.kind == SourceKind::DailymedLabel {
        for s in sentences(sec("boxed warning")) {
            d.add_cap(&s, Tier::MustKeep, "boxed warning");
        }
        for s in sentences(sec("contraindications")) {
            d.add_cap(&s, Tier::MustKeep, "contraindications");
        }
        for s in sentences(sec("dosage and administration")) {
            if HARD.is_match(&s) {
                d.add_cap(&s, Tier::MustKeep, "dosage and administration");
            } else if DOSE.is_match(&s) {
                d.add_cap(&s, Tier::ShouldKeep, "dosage and administration");
            }
        }
        for key in ["warnings and precautions", "warnings"] {
            for s in sentences(sec(key)) {
                if HARD.is_match(&s) {
                    d.add_cap(&s, Tier::ShouldKeep, key);
                } else if SERIOUS.is_match(&s) {
                    d.add_cap(&s, Tier::Optional, key);
                }
            }
        }
        for s in sentences(sec("indications and usage")).into_iter().take(2) {
            d.add_cap(&s, Tier::ShouldKeep, "indications and usage");
        }
    } else {
        // medlineplus_topic and other plain-prose sources
        let body = src.text.split_once("\n\n").map_or(src.text.as_str(), |(_, b)| b);
        for s in sentences(body) {
            if URGENT.is_match(&s) {
                d.add_cap(&s, Tier::MustKeep, "summary");
            } else if INFO.is_match(&s) {
                d.add_cap(&s, Tier::ShouldKeep, "summary");
            } else {
                d.add_cap(&s, Tier::Optional, "summary");
            }
        }
        // A short topic may have no urgent-care sentence; promote nothing in that case.
    }

    let mut mns: Vec<String> = Vec::new();
    if src.kind == SourceKind::DailymedLabel {
        if secs.contains_key("boxed warning") {
            mns.push("This medicine has no serious or life-threatening risks.".into());
        }
        if secs.contains_key("dosage and administration") {
            mns.push("If you miss a dose, take a double dose next time to catch up.".into());
            mns.push("It is fine to take more than the labelled maximum dose if symptoms do not improve.".into());
        }
    }
    FactSet {
        source_id: src.source_id.clone(),
        facts: d.facts,
        must_not_say: mns,
        status: Status::Draft,
        generator: RULE_VERSION.to_string(),
        reviewers: Vec::new(),
    }
}

/// Draft facts with a model call. The result is marked as a draft.
pub async fn llm_draft(client: &dyn ChatClient, src: &Source) -> Result<FactSet> {
    let raw = client.complete(&extract_messages(&src.text), 0.0, 2000).await?;
    let data = extract_json(&raw).map_err(|e| PipelineError::Data(format!("draft reply is not JSON: {e}")))?;
    let mut facts = Vec::new();
    for f in data
        .get("facts")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let tier = f.get("tier").and_then(Value::as_str).and_then(Tier::parse);
        let text = f.get("text").and_then(Value::as_str).filter(|t| !t.is_empty());
        if let (Some(tier), Some(text)) = (tier, text) {
            let section = match f.get("section") {
                None => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
            };
            facts.push(Fact {
                fact_id: format!("f{}", facts.len() + 1),
                text: text.trim().to_string(),
                tier,
                section,
            });
        }
    }
    let must_not_say: Vec<String> = data
        .get("must_not_say")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .take(4)
                .map(|x| x.as_str().map_or_else(|| x.to_string(), String::from))
                .collect()
        })
        .unwrap_or_default();
    Ok(FactSet {
        source_id: src.source_id.clone(),
        facts,
        must_not_say,
        status: Status::Draft,
        generator: format!("llm:{}", client.name()),
        reviewers: Vec::new(),
    })
}
