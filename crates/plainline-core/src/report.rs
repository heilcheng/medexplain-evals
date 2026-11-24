//! Turn scored.jsonl into summary.json and a markdown table.

use crate::error::{Error, Result};
use crate::metrics::{Est, ModelSummary, ScoredRow, summarize_model};
use crate::schema::read_jsonl;
use indexmap::IndexMap;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const DRAFT_BANNER: &str = "Fact lists are machine-drafted and have NOT been reviewed by clinicians. \
Numbers below are pilot values for pipeline checking, not validated results.";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Flags {
    pub mock: bool,
    pub all_facts_validated: bool,
    pub judge_versions: Vec<String>,
    pub judges: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub flags: Flags,
    pub models: IndexMap<String, ModelSummary>,
}

/// Python-style `format(x, ".{prec}f")`: NaN prints as "nan".
fn ff(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x:.prec$}")
    }
}

fn fmt_est(point: f64, lo: f64, hi: f64) -> String {
    let k = 100.0;
    if lo.is_nan() {
        return ff(point * k, 1);
    }
    format!("{} [{}, {}]", ff(point * k, 1), ff(lo * k, 1), ff(hi * k, 1))
}

fn fmt(d: Option<&Est>) -> String {
    d.map_or("n/a".into(), |d| fmt_est(d.point, d.lo, d.hi))
}

/// Summarise already-loaded rows, models in order of first appearance.
pub fn summarize_rows(rows: &[ScoredRow], cond_order: &[String], n_boot: usize, seed: u64) -> Summary {
    let mut by_model: IndexMap<String, Vec<ScoredRow>> = IndexMap::new();
    for r in rows {
        by_model.entry(r.model.clone()).or_default().push(r.clone());
    }
    let flags = Flags {
        mock: rows.iter().any(|r| r.mock),
        all_facts_validated: !rows.is_empty() && rows.iter().all(|r| r.facts_status == "clinician_reviewed"),
        judge_versions: rows
            .iter()
            .map(|r| r.judge_version.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        judges: rows
            .iter()
            .map(|r| r.judge.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    Summary {
        flags,
        models: by_model
            .into_iter()
            .map(|(m, rs)| (m, summarize_model(&rs, cond_order, n_boot, seed)))
            .collect(),
    }
}

pub fn build_summary(
    scored_path: impl AsRef<Path>,
    cond_order: &[String],
    n_boot: usize,
    seed: u64,
) -> Result<Summary> {
    let rows: Vec<ScoredRow> = read_jsonl(scored_path)?;
    Ok(summarize_rows(&rows, cond_order, n_boot, seed))
}

pub fn to_markdown(summary: &Summary) -> String {
    let f = &summary.flags;
    let mut lines: Vec<String> = vec!["# Plainline results".into(), String::new()];
    if !f.all_facts_validated {
        lines.extend([format!("> {DRAFT_BANNER}"), String::new()]);
    }
    if f.mock {
        lines.extend([
            "> These outputs came from the offline mock client. They measure nothing about any model.".to_string(),
            String::new(),
        ]);
    }
    lines.extend([
        format!(
            "Judge: {} ({}). Intervals are 95% bootstrap CIs over items.",
            f.judges.join(", "),
            f.judge_versions.join(", ")
        ),
        String::new(),
    ]);
    for (model, s) in &summary.models {
        let h = &s.headline;
        lines.extend([
            format!("## {model}"),
            String::new(),
            format!("Items: {}. Judge parse-error rate: {}%.", s.n_items, ff(s.parse_error_rate * 100.0, 1)),
            format!(
                "Worst-reader CFR: {}. Mean adaptation tax (points): {}. CFR spread across conditions (points): {}.",
                fmt(h.worst_reader_cfr.as_ref()),
                fmt(h.mean_adaptation_tax.as_ref()),
                fmt(Some(&h.cfr_spread))
            ),
            String::new(),
            "| Condition | CFR % | Tax (pts) | Tax at matched length (pts) | SFR % | Distortion % | Must-not-say % | In band % | Words | FK grade |".into(),
            "|---|---|---|---|---|---|---|---|---|---|".into(),
        ]);
        for (c, d) in &s.conditions {
            let band = d.band_pass.map_or("n/a".to_string(), |b| ff(b * 100.0, 0));
            let ml = d.matched_length_tax.as_ref().and_then(|m| m.as_ref()).map(|m| Est {
                point: m.point,
                lo: m.lo,
                hi: m.hi,
            });
            lines.push(format!(
                "| {c} | {} | {} | {} | {} | {} | {} | {band} | {} | {} |",
                fmt(Some(&d.cfr)),
                fmt(d.adaptation_tax.as_ref()),
                fmt(ml.as_ref()),
                fmt(Some(&d.sfr)),
                fmt(Some(&d.distortion)),
                fmt(Some(&d.mns)),
                ff(d.mean_words.unwrap_or(f64::NAN), 0),
                ff(d.mean_fk_grade, 1)
            ));
        }
        if let Some(lc) = &s.length_control {
            let parts: Vec<String> = lc
                .by_length_tercile
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{k} ({} words, n={}): {}%",
                        ff(v.mean_words, 0),
                        v.n,
                        ff(v.cfr * 100.0, 1)
                    )
                })
                .collect();
            lines.extend([
                String::new(),
                format!("CFR by output-length tercile: {}", parts.join("; ")),
            ]);
            if let Some(rho) = lc.spearman_cfr_vs_words {
                lines.push(format!(
                    "Spearman correlation of CFR with output length: {}.",
                    ff(rho, 2)
                ));
            }
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Write `summary.json` and `results.md` into `out_dir` and return the summary.
pub fn write_report(
    scored_path: impl AsRef<Path>,
    cond_order: &[String],
    out_dir: impl AsRef<Path>,
    n_boot: usize,
    seed: u64,
) -> Result<Summary> {
    let summary = build_summary(scored_path, cond_order, n_boot, seed)?;
    let out = out_dir.as_ref();
    let json = serde_json::to_string_pretty(&summary).map_err(|e| Error::json("serialise summary", e))?;
    fs::write(out.join("summary.json"), json).map_err(|e| Error::io(out.join("summary.json"), e))?;
    fs::write(out.join("results.md"), to_markdown(&summary)).map_err(|e| Error::io(out.join("results.md"), e))?;
    Ok(summary)
}
