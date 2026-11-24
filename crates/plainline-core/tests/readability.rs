use plainline_core::metrics::{ScoredRow, length_control};
use plainline_core::readability::{fk_grade, in_band, round_py, smog_grade, syllables, word_count};

#[test]
fn syllable_counts() {
    assert_eq!(syllables("cat"), 1);
    assert_eq!(syllables("medicine"), 3);
    assert!(syllables("anticoagulant") >= 5);
}

#[test]
fn grade_orders_texts() {
    let plain = "Take one pill each day. Do not take more. Call your doctor if you bleed.";
    let dense = "Concomitant administration of anticoagulants necessitates meticulous international normalized ratio \
                 surveillance throughout therapeutic initiation.";
    assert!(fk_grade(plain) < 8.0 && 8.0 < fk_grade(dense));
    assert!(smog_grade(plain) < smog_grade(dense));
}

#[test]
fn empty_and_band() {
    assert_eq!(fk_grade(""), 0.0);
    assert_eq!(word_count("a b c"), 3);
    assert_eq!(in_band(5.0, None), None);
    assert_eq!(in_band(5.0, Some((None, Some(8.0)))), Some(true));
    assert_eq!(in_band(9.0, Some((None, Some(8.0)))), Some(false));
    assert_eq!(in_band(9.0, Some((Some(10.0), None))), Some(false));
}

#[test]
fn python_style_rounding() {
    assert_eq!(round_py(2.675, 2), 2.67); // the binary value is just below 2.675
    assert_eq!(round_py(0.125, 2), 0.12); // exact tie goes to even
    assert_eq!(round_py(7.0, 2), 7.0);
}

fn row(cfr: f64, words: u64) -> ScoredRow {
    ScoredRow {
        cfr: Some(cfr),
        words,
        ..ScoredRow::default()
    }
}

#[test]
fn spearman_handles_ties() {
    let rows: Vec<ScoredRow> = [(1.0, 10), (1.0, 20), (1.0, 30), (0.5, 40), (0.5, 50), (0.5, 60)]
        .iter()
        .map(|&(c, w)| row(c, w))
        .collect();
    let rho = length_control(&rows).unwrap().spearman_cfr_vs_words.unwrap();
    assert!((rho - (-0.8783101)).abs() < 1e-4, "{rho}");
}
