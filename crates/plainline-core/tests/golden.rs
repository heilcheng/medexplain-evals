//! Checks against outputs recorded from the original Python implementation
//! (`tests/golden/`, produced by `tests/golden/generate_from_python.py` at the commit before
//! the Python logic was removed).

mod common;

use common::{close, golden};
use plainline_core::judge::{JudgeResult, Label, agreement, parse_verdicts};
use plainline_core::metrics::{ScoredRow, headline_points, length_control, score_output, summarize_model};
use plainline_core::readability::{fk_grade, smog_grade, syllables, word_count};
use plainline_core::readers::Condition;
use plainline_core::schema::{FactSet, load_items};
use serde_json::{Value, json};
use std::collections::HashMap;

fn corpus() -> HashMap<String, String> {
    load_items(common::root().join("data"), None)
        .unwrap()
        .into_iter()
        .map(|i| (i.source.source_id, i.source.text))
        .collect()
}

#[test]
fn readability_matches_python() {
    let g = golden("readability.json");
    let texts = corpus();
    let mut n = 0;
    for row in g["texts"].as_array().unwrap() {
        let text = match row.get("source_id") {
            Some(id) => texts[id.as_str().unwrap()].as_str(),
            None => row["text"].as_str().unwrap(),
        };
        assert_eq!(word_count(text) as u64, row["words"].as_u64().unwrap());
        assert!((fk_grade(text) - row["fk"].as_f64().unwrap()).abs() < 1e-9, "{text:?}");
        assert!(
            (smog_grade(text) - row["smog"].as_f64().unwrap()).abs() < 1e-9,
            "{text:?}"
        );
        n += 1;
    }
    for (w, s) in g["syllables"].as_object().unwrap() {
        assert_eq!(syllables(w) as u64, s.as_u64().unwrap(), "{w}");
        n += 1;
    }
    assert!(n > 500);
}

#[test]
fn verdict_parsing_matches_python() {
    for case in golden("verdicts.json").as_array().unwrap() {
        let ids = |k: &str| -> Vec<String> {
            case[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().into())
                .collect()
        };
        let (f, s, e) = parse_verdicts(case["raw"].as_str().unwrap(), &ids("fact_ids"), &ids("stmt_ids"));
        let got = json!({"facts": f, "stmts": s, "errors": e});
        let want = json!({"facts": case["facts"], "stmts": case["stmts"], "errors": case["errors"]});
        close(&want, &got, &format!("{:?}", case["raw"]));
    }
}

#[test]
fn agreement_matches_python() {
    for case in golden("agreement.json").as_array().unwrap() {
        let labels = |k: &str| -> Vec<Label> { serde_json::from_value(case[k].clone()).unwrap() };
        let got = serde_json::to_value(agreement(&labels("judged"), &labels("human"))).unwrap();
        close(&case["expected"], &got, "agreement");
    }
}

#[test]
fn metrics_match_python() {
    let cases = golden("metrics_cases.json");
    let cases = cases.as_array().unwrap();
    assert!(cases.len() >= 18);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let rows: Vec<ScoredRow> = serde_json::from_value(case["rows"].clone()).unwrap();
        let order: Vec<String> = serde_json::from_value(case["cond_order"].clone()).unwrap();
        let (n_boot, seed) = (
            case["n_boot"].as_u64().unwrap() as usize,
            case["seed"].as_u64().unwrap(),
        );
        let got = serde_json::to_value(summarize_model(&rows, &order, n_boot, seed)).unwrap();
        close(&case["summary"], &got, name);
        let hp = match headline_points(&rows, &order) {
            Some(h) => json!({"worst_reader_cfr": h.worst_reader_cfr, "mean_adaptation_tax": h.mean_adaptation_tax}),
            None => json!({}),
        };
        close(&case["headline_points"], &hp, &format!("{name}.headline_points"));
        let lc = length_control(&rows).map_or(json!({}), |l| serde_json::to_value(l).unwrap());
        close(&case["length_control"], &lc, &format!("{name}.length_control"));
    }
}

#[test]
fn score_output_matches_python() {
    for case in golden("score_output.json").as_array().unwrap() {
        let fs: FactSet = serde_json::from_value(case["factset"].clone()).unwrap();
        let c = &case["condition"];
        let band = c["band"].as_array().map(|b| (b[0].as_f64(), b[1].as_f64()));
        let cond = Condition {
            id: c["id"].as_str().unwrap().into(),
            kind: c["kind"].as_str().unwrap().into(),
            instruction: String::new(),
            band,
            group: String::new(),
        };
        let jr = JudgeResult {
            fact_verdicts: serde_json::from_value(case["fact_verdicts"].clone()).unwrap(),
            mns_verdicts: serde_json::from_value(case["mns_verdicts"].clone()).unwrap(),
            parse_errors: case["parse_errors"].as_u64().unwrap() as u32,
            raw: vec![],
        };
        let row = score_output(case["text"].as_str().unwrap(), &fs, &jr, &cond);
        let got = serde_json::to_value(&row).unwrap();
        let want = case["expected"].as_object().unwrap();
        let got: serde_json::Map<String, Value> = want.keys().map(|k| (k.clone(), got[k].clone())).collect();
        close(&Value::Object(want.clone()), &Value::Object(got), &fs.source_id);
    }
}
