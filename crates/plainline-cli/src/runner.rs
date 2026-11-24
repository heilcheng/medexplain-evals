//! Generate, judge and score. Outputs are JSONL and the run is resumable.

use futures::stream::{self, StreamExt};
use plainline_client::{judge_explanation, ChatClient};
use plainline_core::metrics::{score_output, ScoredRow};
use plainline_core::prompts::{writer_messages, JUDGE_VERSION};
use plainline_core::pyjson;
use plainline_core::readers::Condition;
use plainline_core::schema::Item;
use sha1::{Digest, Sha1};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub n_samples: u32,
    pub per_fact: bool,
    pub workers: usize,
    pub max_tokens: u32,
    pub temperature: f64,
    pub mock: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions { n_samples: 1, per_fact: false, workers: 4, max_tokens: 1024, temperature: 0.0, mock: false }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Failed(String),
}

/// First 12 hex characters of the SHA-1 of `model|item_id|cond|sample`.
pub fn output_id(model: &str, item_id: &str, cond: &str, sample: u32) -> String {
    let digest = Sha1::digest(format!("{model}|{item_id}|{cond}|{sample}").as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect::<String>()[..12].to_string()
}

fn done_ids(path: &Path) -> Result<HashSet<String>, RunError> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let text = fs::read_to_string(path).map_err(|e| RunError::Io(format!("{}: {e}", path.display())))?;
    let mut ids = HashSet::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| RunError::Io(format!("{}: bad row: {e}", path.display())))?;
        let id = v
            .get("output_id")
            .and_then(|x| x.as_str())
            .ok_or_else(|| RunError::Io(format!("{}: row without output_id", path.display())))?;
        ids.insert(id.to_string());
    }
    Ok(ids)
}

struct Job<'a> {
    item: &'a Item,
    cond: &'a Condition,
    sample: u32,
}

async fn do_job(
    item: &Item,
    cond: &Condition,
    sample: u32,
    writer: &dyn ChatClient,
    judge: &dyn ChatClient,
    opts: &RunOptions,
) -> Result<ScoredRow, String> {
    let text = writer
        .complete(&writer_messages(&item.source.text, &cond.instruction), opts.temperature, opts.max_tokens)
        .await
        .map_err(|e| e.to_string())?;
    let jr = judge_explanation(judge, &text, &item.facts, opts.per_fact).await.map_err(|e| e.to_string())?;
    let mut row = score_output(&text, &item.facts, &jr, cond);
    row.output_id = output_id(writer.name(), &item.source.source_id, &cond.id, sample);
    row.item_id = item.source.source_id.clone();
    row.model = writer.name().to_string();
    row.sample = sample;
    row.judge = judge.name().to_string();
    row.judge_version = JUDGE_VERSION.to_string();
    row.mock = opts.mock;
    row.facts_status = item.facts.status.as_str().to_string();
    row.text = text;
    row.fact_verdicts = jr.fact_verdicts;
    row.mns_verdicts = jr.mns_verdicts;
    Ok(row)
}

/// Generate, judge and score every missing (item, condition, sample) and append the rows to
/// `scored.jsonl`. Rows that finished are kept when other jobs fail; a rerun does only the
/// missing ones. Returns the path of `scored.jsonl`.
pub async fn run(
    items: &[Item],
    conds: &[Condition],
    writer: &dyn ChatClient,
    judge: &dyn ChatClient,
    out_dir: impl AsRef<Path>,
    opts: &RunOptions,
) -> Result<PathBuf, RunError> {
    let out = out_dir.as_ref();
    fs::create_dir_all(out).map_err(|e| RunError::Io(format!("{}: {e}", out.display())))?;
    let scored_path = out.join("scored.jsonl");
    let done = done_ids(&scored_path)?;
    let mut jobs = Vec::new();
    for item in items {
        for cond in conds {
            for sample in 0..opts.n_samples {
                if !done.contains(&output_id(writer.name(), &item.source.source_id, &cond.id, sample)) {
                    jobs.push(Job { item, cond, sample });
                }
            }
        }
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&scored_path)
        .map_err(|e| RunError::Io(format!("{}: {e}", scored_path.display())))?;
    // `buffered` keeps the jobs' order, so rows land in the file in job order.
    let mut results = stream::iter(jobs.iter()).map(|j| do_job(j.item, j.cond, j.sample, writer, judge, opts)).buffered(opts.workers.max(1));
    let mut failures: Vec<(usize, String)> = Vec::new();
    let mut n = 0;
    while let Some(res) = results.next().await {
        match res {
            Ok(row) => {
                let line = pyjson::to_string(&row).map_err(|e| RunError::Io(e.to_string()))?;
                writeln!(file, "{line}").and_then(|_| file.flush()).map_err(|e| RunError::Io(e.to_string()))?;
            }
            Err(e) => failures.push((n, e)),
        }
        n += 1;
    }
    if let Some((first, err)) = failures.first() {
        let job = &jobs[*first];
        return Err(RunError::Failed(format!(
            "{} of {} jobs failed (first: item {}, condition {}: {err}). Completed rows were kept; rerun to resume.",
            failures.len(),
            jobs.len(),
            job.item.source.source_id,
            job.cond.id
        )));
    }
    Ok(scored_path)
}
