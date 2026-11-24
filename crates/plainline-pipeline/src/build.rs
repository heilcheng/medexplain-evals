//! Build the pilot dataset: fetch sources, draft facts, write JSONL.

use crate::dailymed;
use crate::draft::rule_draft;
use crate::medlineplus;
use crate::{PipelineError, Result};
use plainline_core::schema::{validate_factset, write_jsonl, FactSet, Source};
use serde::Serialize;
use std::path::Path;
use std::time::Duration;

pub const PILOT_DRUGS: [&str; 20] = [
    "warfarin", "methotrexate", "hydrocodone bitartrate and acetaminophen", "metformin", "lisinopril", "amiodarone",
    "digoxin", "levothyroxine", "isotretinoin", "lithium carbonate", "ciprofloxacin", "ibuprofen", "apixaban",
    "clozapine", "sertraline", "tramadol", "prednisone", "gabapentin", "zolpidem", "colchicine",
];
pub const PILOT_TOPICS: [&str; 10] = [
    "Anaphylaxis", "Appendicitis", "Heart Attack", "Hypothermia", "Meningitis", "Heat Illness", "Stroke", "Sepsis",
    "Seizures", "Concussion",
];
pub const USER_AGENT: &str = "plainline-research-pipeline/0.1 (open benchmark; contact via GitHub repo)";

/// Where to fetch from; the defaults are the public NLM endpoints.
#[derive(Debug, Clone)]
pub struct Endpoints {
    pub dailymed_api: String,
    pub medlineplus_index: String,
    pub medlineplus_xml_base: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Endpoints {
            dailymed_api: dailymed::API.into(),
            medlineplus_index: medlineplus::INDEX.into(),
            medlineplus_xml_base: medlineplus::XML_BASE.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PilotReport {
    pub sources: usize,
    pub facts: usize,
    pub skipped: Vec<String>,
}

/// Fetch the pilot sources, draft facts by rule and write `sources/sources.jsonl` and
/// `facts/facts.jsonl` under `out_dir`.
pub async fn build_pilot(
    out_dir: impl AsRef<Path>,
    drugs: &[&str],
    topics: &[&str],
    endpoints: &Endpoints,
    fetched: &str,
) -> Result<PilotReport> {
    let client = reqwest::Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(60)).build()?;
    let mut sources: Vec<Source> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for d in drugs {
        let s = match dailymed::fetch_label(&client, &endpoints.dailymed_api, d, fetched).await {
            Ok(s) => s,
            Err(PipelineError::Http(e)) => {
                skipped.push(format!("{d}: {e}"));
                None
            }
            Err(e) => return Err(e),
        };
        match s {
            Some(s) => sources.push(s),
            None => skipped.push(d.to_string()),
        }
    }
    let (got, missing) = medlineplus::fetch_topics(
        &client,
        &endpoints.medlineplus_index,
        &endpoints.medlineplus_xml_base,
        topics,
        fetched,
    )
    .await?;
    sources.extend(got);
    skipped.extend(missing);

    let (mut kept, mut facts): (Vec<Source>, Vec<FactSet>) = (Vec::new(), Vec::new());
    for s in sources {
        let fs = rule_draft(&s);
        let problems = validate_factset(&fs, true);
        if !problems.is_empty() {
            skipped.push(format!("{}: {}", s.source_id, py_list(&problems)));
            continue;
        }
        kept.push(s);
        facts.push(fs);
    }
    let out = out_dir.as_ref();
    write_jsonl(out.join("sources").join("sources.jsonl"), &kept)?;
    write_jsonl(out.join("facts").join("facts.jsonl"), &facts)?;
    Ok(PilotReport { sources: kept.len(), facts: facts.iter().map(|f| f.facts.len()).sum(), skipped })
}

/// A list of strings printed as Python prints it, e.g. `['a', 'b']`.
pub fn py_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}
