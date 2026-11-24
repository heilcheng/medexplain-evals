use indexmap::IndexMap;
use plainline_core::fixtures::verdict_json;
use plainline_core::judge::{Label, agreement, extract_json, parse_verdicts};

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn map(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn parse_plain_and_fenced() {
    let raw = format!(
        "```json\n{}\n```",
        verdict_json(&[("f1", "stated"), ("n1", "asserted")])
    );
    let (f, s, err) = parse_verdicts(&raw, &ids(&["f1"]), &ids(&["n1"]));
    assert_eq!(f, map(&[("f1", "stated")]));
    assert_eq!(s, map(&[("n1", "asserted")]));
    assert_eq!(err, 0);
}

#[test]
fn missing_and_invalid_fall_back_conservatively() {
    let (f, s, err) = parse_verdicts(&verdict_json(&[("f1", "maybe")]), &ids(&["f1", "f2"]), &ids(&["n1"]));
    assert_eq!(f, map(&[("f1", "absent"), ("f2", "absent")]));
    assert_eq!(s, map(&[("n1", "not_asserted")]));
    assert_eq!(err, 3);
}

#[test]
fn garbage_output() {
    let (f, _, err) = parse_verdicts("I cannot do that", &ids(&["f1"]), &[]);
    assert_eq!(f, map(&[("f1", "absent")]));
    assert_eq!(err, 1);
}

#[test]
fn malformed_entry_stops_the_scan_but_keeps_earlier_ones() {
    let raw = r#"{"verdicts": [{"id": "f1", "verdict": "stated"}, 7, {"id": "f2", "verdict": "stated"}]}"#;
    let (f, _, err) = parse_verdicts(raw, &ids(&["f1", "f2"]), &[]);
    assert_eq!(f, map(&[("f1", "stated"), ("f2", "absent")]));
    assert_eq!(err, 1);
}

#[test]
fn verdicts_are_trimmed_and_case_folded_and_the_last_duplicate_wins() {
    let raw = r#"{"verdicts": [{"id": "f1", "verdict": " Stated "}, {"id": "f1", "verdict": "ABSENT"}]}"#;
    let (f, _, err) = parse_verdicts(raw, &ids(&["f1"]), &[]);
    assert_eq!(f, map(&[("f1", "absent")]));
    assert_eq!(err, 0);
}

#[test]
fn json_is_found_inside_prose() {
    let v = extract_json("Sure! {\"verdicts\": []} Hope that helps.").unwrap();
    assert!(v["verdicts"].is_array());
    assert!(extract_json("no braces here").is_err());
}

fn labels(v: &[u8]) -> Vec<Label> {
    v.iter()
        .enumerate()
        .map(|(i, &s)| Label {
            key: i.to_string(),
            stated: s == 1,
        })
        .collect()
}

#[test]
fn agreement_kappa() {
    let j = labels(&[1, 1, 0, 0, 1, 0]);
    assert_eq!(agreement(&j, &labels(&[1, 1, 0, 0, 1, 0])).kappa, Some(1.0));
    let r = agreement(&j, &labels(&[0, 0, 0, 0, 1, 0]));
    assert!(r.agreement.unwrap() > 0.0 && r.agreement.unwrap() < 1.0);
    assert!(r.kappa.unwrap() < 1.0);
    assert_eq!(agreement(&j, &[]).n, 0);
    assert_eq!(agreement(&j, &[]).kappa, None);
}
