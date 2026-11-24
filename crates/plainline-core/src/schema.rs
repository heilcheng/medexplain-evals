//! Data schema. Every machine-drafted fact set carries status "draft, needs clinician review".

use crate::error::{Error, Result};
use crate::pyjson;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const DRAFT_STATUS: &str = "draft, needs clinician review";
pub const CANARY: &str = "PLAINLINE-CANARY-7c1e5a52-do-not-train-on-this-file";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    #[serde(rename = "must_keep")]
    MustKeep,
    #[serde(rename = "should_keep")]
    ShouldKeep,
    #[serde(rename = "optional")]
    Optional,
}

impl Tier {
    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "must_keep" => Some(Tier::MustKeep),
            "should_keep" => Some(Tier::ShouldKeep),
            "optional" => Some(Tier::Optional),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    #[serde(rename = "draft, needs clinician review")]
    Draft,
    #[serde(rename = "clinician_reviewed")]
    ClinicianReviewed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Draft => DRAFT_STATUS,
            Status::ClinicianReviewed => "clinician_reviewed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    #[serde(rename = "dailymed_label")]
    DailymedLabel,
    #[serde(rename = "medlineplus_topic")]
    MedlineplusTopic,
    #[serde(rename = "cdc_page")]
    CdcPage,
    #[serde(rename = "vignette")]
    Vignette,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    #[serde(rename = "dev")]
    Dev,
    #[serde(rename = "test")]
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Dev => "dev",
            Split::Test => "test",
        }
    }
}

fn default_split() -> Split {
    Split::Dev
}
fn default_canary() -> String {
    CANARY.to_string()
}
fn default_status() -> Status {
    Status::Draft
}

/// One source document chunk given to the model under test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub source_id: String,
    pub kind: SourceKind,
    pub title: String,
    pub text: String,
    pub origin_url: String,
    pub license_note: String,
    pub fetched: String,
    #[serde(default)]
    pub version: String,
    #[serde(default = "default_split")]
    pub split: Split,
    #[serde(default = "default_canary")]
    pub canary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub fact_id: String,
    pub text: String,
    pub tier: Tier,
    /// Section of the source the fact was drawn from, when known.
    #[serde(default)]
    pub section: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactSet {
    pub source_id: String,
    pub facts: Vec<Fact>,
    /// Statements a wrong simplification might assert. Judged for violations.
    #[serde(default)]
    pub must_not_say: Vec<String>,
    #[serde(default = "default_status")]
    pub status: Status,
    /// How the draft was produced, e.g. "rule-v0" or a model id.
    pub generator: String,
    #[serde(default)]
    pub reviewers: Vec<String>,
}

impl FactSet {
    pub fn tier(&self, tier: Tier) -> Vec<&Fact> {
        self.facts.iter().filter(|f| f.tier == tier).collect()
    }

    pub fn is_validated(&self) -> bool {
        self.status == Status::ClinicianReviewed && self.reviewers.len() >= 2
    }
}

/// A source with its fact set, ready to evaluate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub source: Source,
    pub facts: FactSet,
}

/// Read a JSONL file into typed rows. Blank lines are skipped.
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line).map_err(|e| Error::json(format!("{}:{}", path.display(), n + 1), e))?);
    }
    Ok(out)
}

/// Write rows as JSONL in the Python layout, creating parent directories.
pub fn write_jsonl<T: Serialize>(path: impl AsRef<Path>, rows: &[T]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let mut buf = String::new();
    for r in rows {
        buf.push_str(&pyjson::to_string(r).map_err(|e| Error::json("serialise row", e))?);
        buf.push('\n');
    }
    fs::write(path, buf).map_err(|e| Error::io(path, e))
}

/// Load sources and fact sets from `data_dir`; both files are JSONL.
///
/// Items follow the order of the sources file and only include sources that have a fact
/// set. A repeated source id keeps its first position and the last record, as a Python
/// dict would.
pub fn load_items(data_dir: impl AsRef<Path>, split: Option<&str>) -> Result<Vec<Item>> {
    let d = data_dir.as_ref();
    let srcs: Vec<Source> = read_jsonl(d.join("sources").join("sources.jsonl"))?;
    let fsets: Vec<FactSet> = read_jsonl(d.join("facts").join("facts.jsonl"))?;
    let mut order: Vec<String> = Vec::new();
    let mut sources: HashMap<String, Source> = HashMap::new();
    for s in srcs {
        if !sources.contains_key(&s.source_id) {
            order.push(s.source_id.clone());
        }
        sources.insert(s.source_id.clone(), s);
    }
    let facts: HashMap<String, FactSet> = fsets.into_iter().map(|f| (f.source_id.clone(), f)).collect();
    let mut items = Vec::new();
    for sid in order {
        if let Some(fs) = facts.get(&sid) {
            let source = sources.remove(&sid).expect("source recorded in order");
            if split.is_none_or(|sp| source.split.as_str() == sp) {
                items.push(Item {
                    source,
                    facts: fs.clone(),
                });
            }
        }
    }
    Ok(items)
}

/// Structural problems with a fact set; empty means fine.
pub fn validate_factset(fs: &FactSet, require_must: bool) -> Vec<String> {
    let mut problems = Vec::new();
    let mut ids: Vec<&str> = fs.facts.iter().map(|f| f.fact_id.as_str()).collect();
    let n = ids.len();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != n {
        problems.push("duplicate fact ids".to_string());
    }
    if require_must && fs.tier(Tier::MustKeep).is_empty() {
        problems.push("no must_keep facts".to_string());
    }
    if fs.status != Status::Draft && !fs.is_validated() {
        problems.push("status claims review but fewer than 2 reviewers are recorded".to_string());
    }
    problems
}
