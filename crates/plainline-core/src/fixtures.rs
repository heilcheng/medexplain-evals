//! Small fixtures shared by the tests of the workspace crates.

use crate::schema::{CANARY, Fact, FactSet, Item, Source, SourceKind, Split, Status, Tier};

/// A one-source item with four facts (two must_keep) and one must_not_say statement.
pub fn item() -> Item {
    let source = Source {
        source_id: "t1".into(),
        kind: SourceKind::DailymedLabel,
        title: "Testdrug".into(),
        text: "Testdrug can cause liver damage. Do not exceed 4 grams per day. Take with food. It is used for pain."
            .into(),
        origin_url: "http://x".into(),
        license_note: "test".into(),
        fetched: "2026-01-01".into(),
        version: String::new(),
        split: Split::Dev,
        canary: CANARY.into(),
    };
    let fact = |id: &str, text: &str, tier| Fact {
        fact_id: id.into(),
        text: text.into(),
        tier,
        section: String::new(),
    };
    let facts = FactSet {
        source_id: "t1".into(),
        facts: vec![
            fact("f1", "Can cause liver damage", Tier::MustKeep),
            fact("f2", "Do not exceed 4 grams per day", Tier::MustKeep),
            fact("f3", "Take with food", Tier::ShouldKeep),
            fact("f4", "Used for pain", Tier::Optional),
        ],
        must_not_say: vec!["It is fine to take more than 4 grams per day.".into()],
        status: Status::Draft,
        generator: "test".into(),
        reviewers: vec![],
    };
    Item { source, facts }
}

/// `{"verdicts": [{"id": k, "verdict": v}, ...]}` as a model would reply.
pub fn verdict_json(pairs: &[(&str, &str)]) -> String {
    let v: Vec<_> = pairs
        .iter()
        .map(|(k, v)| serde_json::json!({"id": k, "verdict": v}))
        .collect();
    serde_json::json!({"verdicts": v}).to_string()
}
