mod common;

use plainline_core::pyjson;
use plainline_core::readers::conditions;
use plainline_core::schema::{DRAFT_STATUS, FactSet, Source, Status, Tier, load_items, read_jsonl};
use std::fs;
use std::path::Path;

#[test]
fn pilot_data() {
    let items = load_items(common::root().join("data"), None).unwrap();
    assert!((28..=40).contains(&items.len()));
    assert!(
        items.iter().all(|i| i.facts.status == Status::Draft),
        "no pilot fact set may claim review"
    );
    assert!(items.iter().all(|i| !i.facts.tier(Tier::MustKeep).is_empty()));
    assert!(items.iter().all(|i| i.source.canary.contains("PLAINLINE-CANARY")));
    let kinds: std::collections::BTreeSet<String> = items
        .iter()
        .map(|i| {
            serde_json::to_value(i.source.kind)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["dailymed_label", "medlineplus_topic"]
    );
    assert!(items.iter().all(|i| i.source.origin_url.starts_with("https://")));
    assert_eq!(DRAFT_STATUS, "draft, needs clinician review");
}

#[test]
fn readers() {
    let adapt = conditions("adapt", None).unwrap();
    assert_eq!(adapt[0].id, "neutral");
    assert_eq!(adapt.len(), 6);
    let cue = conditions("cue", None).unwrap();
    assert_eq!(cue[0].id, "neutral");
    assert!(cue.len() >= 6);
    assert!(cue[1..].iter().all(|c| !c.instruction.contains("physician")));
    assert!(conditions("other", None).is_err());
}

#[test]
fn split_filter_and_missing_facts() {
    let d = common::root().join("data");
    assert_eq!(load_items(&d, Some("test")).unwrap().len(), 0);
    assert_eq!(
        load_items(&d, Some("dev")).unwrap().len(),
        load_items(&d, None).unwrap().len()
    );
}

#[test]
fn data_files_round_trip_byte_for_byte() {
    // The Rust writer must keep the layout the Python writer produced, so rebuilding the
    // pilot does not churn the committed data.
    let d = common::root().join("data");
    let check = |rel: &str, rewrite: &dyn Fn(&str) -> String| {
        let text = fs::read_to_string(d.join(rel)).unwrap();
        for line in text.lines().filter(|l| !l.is_empty()) {
            assert_eq!(rewrite(line), line, "{rel}");
        }
    };
    check("sources/sources.jsonl", &|l| {
        pyjson::to_string(&serde_json::from_str::<Source>(l).unwrap()).unwrap()
    });
    check("facts/facts.jsonl", &|l| {
        pyjson::to_string(&serde_json::from_str::<FactSet>(l).unwrap()).unwrap()
    });
}

#[test]
fn write_jsonl_round_trips() {
    let items = load_items(common::root().join("data"), None).unwrap();
    let tmp = std::env::temp_dir().join(format!("plainline-rt-{}", std::process::id()));
    let path = tmp.join("nested").join("x.jsonl");
    let facts: Vec<FactSet> = items.iter().map(|i| i.facts.clone()).collect();
    plainline_core::schema::write_jsonl(&path, &facts).unwrap();
    let back: Vec<FactSet> = read_jsonl(&path).unwrap();
    assert_eq!(back, facts);
    fs::remove_dir_all(tmp).unwrap();
}

#[test]
fn source_requires_fields() {
    assert!(serde_json::from_str::<Source>(r#"{"source_id": "x"}"#).is_err());
}

#[test]
fn no_em_dashes_or_arrows() {
    let bad: Vec<char> = [0x2014u32, 0x2192, 0x2190, 0x21d2, 0x2194]
        .iter()
        .map(|&c| char::from_u32(c).unwrap())
        .collect();
    let root = common::root();
    let mut files = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if matches!(
                name.as_str(),
                "target" | ".git" | ".venv" | "node_modules" | "__pycache__" | "runs"
            ) {
                continue;
            }
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    walk(&root, &mut files);
    let exts = [
        "md", "py", "rs", "toml", "yaml", "yml", "jsonl", "json", "tsv", "cff", "bib", "tex",
    ];
    let mut scanned = 0;
    for p in files {
        if p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.contains(&e))
        {
            let text = fs::read_to_string(&p).unwrap_or_default();
            assert!(!text.contains(&bad[..]), "em dash or arrow in {}", p.display());
            scanned += 1;
        }
    }
    assert!(scanned > 20);
}
