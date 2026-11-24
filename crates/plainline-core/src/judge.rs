//! Judge-light fact checking: verdict parsing and judge-versus-human agreement.
//!
//! The judge never rates quality. It answers a narrow question per fact. The calls
//! themselves live in `plainline-client`; this module holds the pure parts.

use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

pub const FACT_VERDICTS: [&str; 3] = ["stated", "contradicted", "absent"];
pub const STMT_VERDICTS: [&str; 2] = ["asserted", "not_asserted"];

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JudgeResult {
    /// fact_id -> stated | contradicted | absent
    pub fact_verdicts: IndexMap<String, String>,
    /// n1.. -> asserted | not_asserted
    pub mns_verdicts: IndexMap<String, String>,
    #[serde(default)]
    pub parse_errors: u32,
    #[serde(default)]
    pub raw: Vec<String>,
}

static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^```(?:json)?|```$").unwrap());
static BRACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\{.*\}").unwrap());

/// Parse the first JSON object in a model reply, tolerating code fences and prose around it.
pub fn extract_json(text: &str) -> Result<Value, serde_json::Error> {
    let text = text.trim();
    let text = FENCE.replace_all(text, "");
    let text = text.trim();
    match serde_json::from_str::<Value>(text) {
        Ok(v) => Ok(v),
        Err(e) => match BRACES.find(text) {
            None => Err(e),
            Some(m) => serde_json::from_str(m.as_str()),
        },
    }
}

/// Python's `str()` of a JSON value, for the cases that can appear in a verdict list.
fn py_str(v: Option<&Value>, default: &str) -> String {
    match v {
        None => default.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(true)) => "True".into(),
        Some(Value::Bool(false)) => "False".into(),
        Some(other) => other.to_string(),
    }
}

/// Return (fact verdicts, statement verdicts, number of ids that were missing or invalid).
///
/// Missing or invalid ids fall back to the conservative reading: a fact counts as
/// absent, a statement as not asserted. The error count is reported so a high rate is visible.
pub fn parse_verdicts(
    raw: &str,
    fact_ids: &[String],
    stmt_ids: &[String],
) -> (IndexMap<String, String>, IndexMap<String, String>, u32) {
    let mut got: HashMap<String, String> = HashMap::new();
    if let Ok(data) = extract_json(raw) {
        if let Some(Value::Array(list)) = data.get("verdicts") {
            for v in list {
                // A malformed entry stops the scan; entries before it are kept.
                let Some(obj) = v.as_object() else { break };
                got.insert(
                    py_str(obj.get("id"), "None"),
                    py_str(obj.get("verdict"), "").trim().to_lowercase(),
                );
            }
        }
    }
    let mut errors = 0;
    let mut facts = IndexMap::new();
    for i in fact_ids {
        let v = match got.get(i) {
            Some(v) if FACT_VERDICTS.contains(&v.as_str()) => v.clone(),
            _ => {
                errors += 1;
                "absent".to_string()
            }
        };
        facts.insert(i.clone(), v);
    }
    let mut stmts = IndexMap::new();
    for i in stmt_ids {
        let v = match got.get(i) {
            Some(v) if STMT_VERDICTS.contains(&v.as_str()) => v.clone(),
            _ => {
                errors += 1;
                "not_asserted".to_string()
            }
        };
        stmts.insert(i.clone(), v);
    }
    (facts, stmts, errors)
}

/// One label of a (key, stated) pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Label {
    pub key: String,
    pub stated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agreement {
    pub n: usize,
    pub agreement: Option<f64>,
    pub kappa: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_stated_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human_stated_rate: Option<f64>,
}

/// Compare judge verdicts with human labels: raw agreement and Cohen's kappa on the overlap.
pub fn agreement(judged: &[Label], human: &[Label]) -> Agreement {
    let h: HashMap<&str, bool> = human.iter().map(|r| (r.key.as_str(), r.stated)).collect();
    let pairs: Vec<(bool, bool)> = judged
        .iter()
        .filter_map(|r| h.get(r.key.as_str()).map(|&y| (r.stated, y)))
        .collect();
    let n = pairs.len();
    if n == 0 {
        return Agreement {
            n: 0,
            agreement: None,
            kappa: None,
            judge_stated_rate: None,
            human_stated_rate: None,
        };
    }
    let nf = n as f64;
    let a = pairs.iter().filter(|(x, y)| x == y).count() as f64 / nf;
    let p_j = pairs.iter().filter(|(x, _)| *x).count() as f64 / nf;
    let p_h = pairs.iter().filter(|(_, y)| *y).count() as f64 / nf;
    let pe = p_j * p_h + (1.0 - p_j) * (1.0 - p_h);
    let kappa = if pe == 1.0 { 1.0 } else { (a - pe) / (1.0 - pe) };
    Agreement {
        n,
        agreement: Some(a),
        kappa: Some(kappa),
        judge_stated_rate: Some(p_j),
        human_stated_rate: Some(p_h),
    }
}
