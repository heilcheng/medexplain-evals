use plainline_core::fixtures::item;
use plainline_core::judge::JudgeResult;
use plainline_core::metrics::{ScoredRow, bootstrap, headline_points, length_control, score_output, summarize_model};
use plainline_core::readers::Condition;

fn jr(pairs: &[(&str, &str)]) -> JudgeResult {
    JudgeResult {
        fact_verdicts: pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        mns_verdicts: [("n1".to_string(), "not_asserted".to_string())].into_iter().collect(),
        ..JudgeResult::default()
    }
}

fn cond(id: &str, kind: &str, band: Option<(Option<f64>, Option<f64>)>) -> Condition {
    Condition {
        id: id.into(),
        kind: kind.into(),
        instruction: "x".into(),
        band,
        group: String::new(),
    }
}

#[test]
fn score_output_fractions() {
    let it = item();
    let c = cond("grade6_adult", "reader", Some((None, Some(8.0))));
    let r = score_output(
        "Take it with food.",
        &it.facts,
        &jr(&[
            ("f1", "absent"),
            ("f2", "stated"),
            ("f3", "stated"),
            ("f4", "contradicted"),
        ]),
        &c,
    );
    assert_eq!(r.cfr, Some(0.5));
    assert_eq!(r.sfr, Some(1.0));
    assert_eq!(r.distortion, Some(0.25));
    assert_eq!(r.mns, Some(0.0));
    assert_eq!(r.must_missing, ["f1"]);
    assert_eq!(r.band_ok, Some(true));
}

#[test]
fn score_output_mns_violation() {
    let it = item();
    let mut j = jr(&[("f1", "stated"), ("f2", "stated"), ("f3", "stated"), ("f4", "stated")]);
    j.mns_verdicts.insert("n1".into(), "asserted".into());
    let r = score_output("x", &it.facts, &j, &cond("neutral", "neutral", None));
    assert_eq!(r.mns, Some(1.0));
    assert_eq!(r.band_ok, None);
}

#[test]
fn bootstrap_deterministic_and_covers() {
    let x: Vec<f64> = (0..40).map(|i| (i % 2) as f64).collect();
    let stat = |ix: &[usize]| ix.iter().map(|&i| x[i]).sum::<f64>() / ix.len() as f64;
    let a = bootstrap(stat, x.len(), 500, 1, 0.95);
    let b = bootstrap(stat, x.len(), 500, 1, 0.95);
    assert_eq!(a, b);
    assert!(a.lo < 0.5 && 0.5 < a.hi);
    let spread = |ix: &[usize]| ix.iter().map(|&i| i as f64).sum::<f64>() / ix.len() as f64;
    let (p, q) = (
        bootstrap(spread, x.len(), 500, 1, 0.95),
        bootstrap(spread, x.len(), 500, 2, 0.95),
    );
    assert_ne!(p.lo, q.lo);
}

#[test]
fn bootstrap_degenerate_inputs() {
    let one = bootstrap(|_| 1.0, 1, 100, 0, 0.95);
    assert_eq!(one.point, 1.0);
    assert!(one.lo.is_nan() && one.hi.is_nan());
    let nan = bootstrap(|_| f64::NAN, 10, 100, 0, 0.95);
    assert!(nan.point.is_nan() && nan.lo.is_nan());
}

/// Deterministic stand-in for the original NumPy fixture.
fn rows(n: usize) -> Vec<ScoredRow> {
    let mut out = Vec::new();
    let mut state: u64 = 12345;
    for i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let base = 0.6 + 0.4 * ((state >> 11) as f64 / (1u64 << 53) as f64);
        for (cond, drop, words) in [
            ("neutral", 0.0, 200),
            ("grade6_adult", 0.2, 120),
            ("clinician", 0.0, 210),
        ] {
            let g6 = cond == "grade6_adult";
            out.push(ScoredRow {
                item_id: format!("i{i}"),
                condition: cond.into(),
                cfr: Some(base - drop),
                sfr: Some(0.5),
                distortion: Some(0.0),
                mns: Some(0.0),
                band_ok: if cond == "neutral" {
                    None
                } else {
                    Some(!g6 || i % 2 == 0)
                },
                words: if g6 {
                    (if i < 10 { 120 } else { 190 }) + i as u64
                } else {
                    words + i as u64
                },
                fk_grade: 7.0,
                ..ScoredRow::default()
            });
        }
    }
    out
}

fn order(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn adaptation_tax_is_paired_and_positive() {
    let s = summarize_model(&rows(20), &order(&["neutral", "clinician", "grade6_adult"]), 300, 0);
    let tax = s.conditions["grade6_adult"].adaptation_tax.unwrap();
    assert!((tax.point - 0.2).abs() < 1e-9);
    assert!(tax.lo <= 0.2 + 1e-9 && 0.2 - 1e-9 <= tax.hi);
    assert!(s.conditions["clinician"].adaptation_tax.unwrap().point.abs() < 1e-9);
    assert!(s.conditions["neutral"].adaptation_tax.is_none());
}

#[test]
fn worst_reader_and_band() {
    let s = summarize_model(&rows(20), &order(&["neutral", "clinician", "grade6_adult"]), 300, 0);
    let (clin, g6) = (
        s.conditions["clinician"].cfr.point,
        s.conditions["grade6_adult"].cfr.point,
    );
    assert_eq!(s.headline.worst_reader_cfr.unwrap().point, clin.min(g6));
    assert_eq!(s.conditions["grade6_adult"].band_pass, Some(0.5));
    assert_eq!(s.conditions["neutral"].band_pass, None);
}

#[test]
fn matched_length_excludes_far_outputs() {
    let s = summarize_model(&rows(20), &order(&["neutral", "grade6_adult"]), 100, 0);
    // The first ten items are far shorter than neutral (ratio about 0.6) and drop out.
    let m = s.conditions["grade6_adult"].matched_length_tax.unwrap().unwrap();
    assert_eq!(m.n_items, 10);
}

#[test]
fn headline_points_handles_missing_neutral() {
    assert!(headline_points(&[], &order(&["neutral", "x"])).is_none());
    let h = headline_points(&rows(20), &order(&["neutral", "clinician", "grade6_adult"])).unwrap();
    assert!((h.mean_adaptation_tax - 0.1).abs() < 1e-9);
}

#[test]
fn length_control_detects_length_confound() {
    let rows: Vec<ScoredRow> = (2..14)
        .map(|k| ScoredRow {
            cfr: Some((k * 5) as f64 / 100.0),
            words: k * 5,
            ..ScoredRow::default()
        })
        .collect();
    let lc = length_control(&rows).unwrap();
    assert!(lc.spearman_cfr_vs_words.unwrap() > 0.99);
    assert!(lc.by_length_tercile["long"].cfr > lc.by_length_tercile["short"].cfr);
    assert!(length_control(&rows[..5]).is_none());
}

#[test]
fn worst_reader_cfr_ignores_nan_condition() {
    let mut r = rows(20);
    for row in r.iter_mut().filter(|r| r.condition == "clinician") {
        row.cfr = None;
    }
    let o = order(&["neutral", "clinician", "grade6_adult"]);
    let s = summarize_model(&r, &o, 100, 0);
    let w = s.headline.worst_reader_cfr.unwrap().point;
    assert!(!w.is_nan() && (w - s.conditions["grade6_adult"].cfr.point).abs() < 1e-9);
    assert!(!headline_points(&r, &o).unwrap().worst_reader_cfr.is_nan());
}

#[test]
fn nan_serialises_as_null() {
    let s = summarize_model(&rows(1), &order(&["neutral", "grade6_adult"]), 10, 0);
    let v = serde_json::to_value(&s).unwrap();
    assert!(v["conditions"]["neutral"]["cfr"]["lo"].is_null());
}
