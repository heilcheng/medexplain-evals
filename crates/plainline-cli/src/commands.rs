//! The subcommands. Each returns its stdout and stderr text instead of printing, so the
//! same code serves the binary and the Python binding.

use crate::runner::{RunOptions, run};
use clap::{Args, Parser, Subcommand, ValueEnum};
use plainline_client::{ChatClient, MockClient, OpenAiCompatClient};
use plainline_core::judge::{Label, agreement};
use plainline_core::readers::conditions;
use plainline_core::report::write_report;
use plainline_core::schema::{DRAFT_STATUS, FactSet, Source, load_items, read_jsonl, validate_factset, write_jsonl};
use plainline_pipeline::build::{Endpoints, PILOT_DRUGS, PILOT_TOPICS, build_pilot, py_list};
use plainline_pipeline::draft::llm_draft;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// What a command produced.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Output {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
}

#[derive(Debug)]
pub struct CliError(pub String);

impl<E: std::fmt::Display> From<E> for CliError {
    fn from(e: E) -> Self {
        CliError(e.to_string())
    }
}

type CmdResult = Result<Output, CliError>;

#[derive(Parser, Debug)]
#[command(
    name = "plainline",
    version,
    about = "Fidelity under audience adaptation for medical explanations. Works with any OpenAI-compatible chat endpoint."
)]
pub struct Cli {
    #[arg(long, global = true, default_value = "data")]
    pub data_dir: PathBuf,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// fetch open sources and write the pilot set with draft facts
    BuildPilot,
    /// check the data files
    ValidateData,
    /// draft facts with a model (output is a draft, needs clinician review)
    Draft(DraftArgs),
    /// generate explanations, judge them and write a report
    Run(RunArgs),
    /// recompute the report from a run directory
    Report(ReportArgs),
    /// write a human labelling sheet from a run
    ExportLabels(ExportArgs),
    /// judge-versus-human agreement from a filled labelling sheet
    Agreement(AgreementArgs),
}

#[derive(Args, Debug, Clone, Default)]
pub struct Endpoint {
    /// OpenAI-compatible base URL, e.g. https://api.openai.com/v1
    #[arg(long)]
    pub base_url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    /// env var holding the API key
    #[arg(long, default_value = "OPENAI_API_KEY")]
    pub api_key_env: String,
    /// offline mock clients (measures nothing)
    #[arg(long)]
    pub mock: bool,
}

#[derive(Args, Debug, Clone)]
pub struct DraftArgs {
    #[command(flatten)]
    pub endpoint: Endpoint,
    #[arg(long, default_value = "data/facts/facts.llm_draft.jsonl")]
    pub out: PathBuf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    Adapt,
    Cue,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Adapt => "adapt",
            Mode::Cue => "cue",
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    #[command(flatten)]
    pub endpoint: Endpoint,
    #[arg(long)]
    pub judge_base_url: Option<String>,
    #[arg(long)]
    pub judge_model: Option<String>,
    #[arg(long)]
    pub judge_api_key_env: Option<String>,
    #[arg(long, value_enum, default_value = "adapt")]
    pub mode: Mode,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "dev")]
    pub split: String,
    #[arg(long)]
    pub limit: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub samples: u32,
    /// one judge call per fact
    #[arg(long)]
    pub per_fact: bool,
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..))]
    pub workers: u32,
    #[arg(long, default_value_t = 2000)]
    pub boot: usize,
}

#[derive(Args, Debug, Clone)]
pub struct ReportArgs {
    #[arg(long)]
    pub run: PathBuf,
    /// default: inferred from the run
    #[arg(long, value_enum)]
    pub mode: Option<Mode>,
    #[arg(long, default_value_t = 2000)]
    pub boot: usize,
}

#[derive(Args, Debug, Clone)]
pub struct ExportArgs {
    #[arg(long)]
    pub run: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct AgreementArgs {
    #[arg(long)]
    pub labels: PathBuf,
}

fn build_client(base: Option<&str>, model: Option<&str>, key_env: &str) -> Result<Box<dyn ChatClient>, CliError> {
    match (base, model) {
        (Some(b), Some(m)) if !b.is_empty() && !m.is_empty() => Ok(Box::new(OpenAiCompatClient::new(b, m, key_env))),
        _ => Err(CliError("--base-url and --model are required (or pass --mock)".into())),
    }
}

fn writer_client(e: &Endpoint) -> Result<Box<dyn ChatClient>, CliError> {
    if e.mock {
        return Ok(Box::new(MockClient));
    }
    build_client(e.base_url.as_deref(), e.model.as_deref(), &e.api_key_env)
}

fn judge_client(a: &RunArgs) -> Result<Box<dyn ChatClient>, CliError> {
    if a.endpoint.mock {
        return Ok(Box::new(MockClient));
    }
    let base = a
        .judge_base_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(a.endpoint.base_url.as_deref());
    let model = a
        .judge_model
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(a.endpoint.model.as_deref());
    let key = a
        .judge_api_key_env
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&a.endpoint.api_key_env);
    build_client(base, model, key)
}

/// Today's date (UTC) as YYYY-MM-DD.
pub fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn cmd_build_pilot(data_dir: &Path) -> CmdResult {
    let rep = block_on(build_pilot(
        data_dir,
        &PILOT_DRUGS,
        &PILOT_TOPICS,
        &Endpoints::default(),
        &today(),
    ))?;
    Ok(Output {
        stdout: format!("{}\n", serde_json::to_string_pretty(&rep)?),
        ..Default::default()
    })
}

fn cmd_draft(data_dir: &Path, a: &DraftArgs) -> CmdResult {
    let client = writer_client(&a.endpoint)?;
    let sources: Vec<Source> = read_jsonl(data_dir.join("sources").join("sources.jsonl"))?;
    let mut out = Output::default();
    let mut drafts: Vec<FactSet> = Vec::new();
    for s in &sources {
        let fs = block_on(llm_draft(client.as_ref(), s))?;
        let probs = validate_factset(&fs, true);
        if !probs.is_empty() {
            out.stderr
                .push_str(&format!("skip {}: {}\n", s.source_id, py_list(&probs)));
            continue;
        }
        drafts.push(fs);
    }
    write_jsonl(&a.out, &drafts)?;
    out.stdout = format!(
        "wrote {} draft fact sets to {} (status: {DRAFT_STATUS})\n",
        drafts.len(),
        a.out.display()
    );
    Ok(out)
}

fn cmd_validate(data_dir: &Path) -> CmdResult {
    let items = load_items(data_dir, None)?;
    let mut out = Output::default();
    let mut bad = 0;
    for it in &items {
        let probs = validate_factset(&it.facts, true);
        if !probs.is_empty() {
            bad += 1;
            out.stdout
                .push_str(&format!("{}: {}\n", it.source.source_id, py_list(&probs)));
        }
    }
    let n_facts: usize = items.iter().map(|i| i.facts.facts.len()).sum();
    let n_valid = items.iter().filter(|i| i.facts.is_validated()).count();
    out.stdout.push_str(&format!(
        "{} items, {n_facts} facts, {n_valid} clinician-validated fact sets, {bad} with problems\n",
        items.len()
    ));
    out.code = i32::from(bad > 0);
    Ok(out)
}

fn cmd_run(data_dir: &Path, a: &RunArgs) -> CmdResult {
    let mut items = load_items(data_dir, Some(&a.split))?;
    if let Some(limit) = a.limit.filter(|&l| l > 0) {
        items.truncate(limit);
    }
    let conds = conditions(a.mode.as_str(), None)?;
    let writer = writer_client(&a.endpoint)?;
    let judge = judge_client(a)?;
    let mut out = Output::default();
    if !a.endpoint.mock && judge.name() == writer.name() {
        out.stderr
            .push_str("warning: judge and writer are the same model; self-preference is not controlled\n");
    }
    let opts = RunOptions {
        n_samples: a.samples,
        per_fact: a.per_fact,
        workers: a.workers as usize,
        mock: a.endpoint.mock,
        ..RunOptions::default()
    };
    let path = block_on(run(&items, &conds, writer.as_ref(), judge.as_ref(), &a.out, &opts))?;
    let ids: Vec<String> = conds.iter().map(|c| c.id.clone()).collect();
    write_report(&path, &ids, &a.out, a.boot, 0)?;
    out.stdout = format!("{}\n", read_text(&a.out.join("results.md"))?);
    Ok(out)
}

fn read_text(p: &Path) -> Result<String, CliError> {
    std::fs::read_to_string(p).map_err(|e| CliError(format!("{}: {e}", p.display())))
}

fn cmd_report(a: &ReportArgs) -> CmdResult {
    let scored = a.run.join("scored.jsonl");
    let mode = match a.mode {
        Some(m) => m,
        None => {
            // Infer from the run so a cue run is not silently reported as adapt.
            let rows: Vec<Value> = read_jsonl(&scored)?;
            if rows
                .iter()
                .any(|r| r.get("kind").and_then(Value::as_str) == Some("cue"))
            {
                Mode::Cue
            } else {
                Mode::Adapt
            }
        }
    };
    let ids: Vec<String> = conditions(mode.as_str(), None)?.iter().map(|c| c.id.clone()).collect();
    write_report(&scored, &ids, &a.run, a.boot, 0)?;
    Ok(Output {
        stdout: format!("{}\n", read_text(&a.run.join("results.md"))?),
        ..Default::default()
    })
}

#[derive(Serialize)]
struct LabelRow {
    key: String,
    fact: String,
    tier: plainline_core::schema::Tier,
    explanation: String,
    judge_stated: bool,
    human_stated: Option<bool>,
}

/// Write a sheet for human labelling: one row per (output, fact).
fn cmd_export_labels(data_dir: &Path, a: &ExportArgs) -> CmdResult {
    let items: HashMap<String, _> = load_items(data_dir, None)?
        .into_iter()
        .map(|i| (i.source.source_id.clone(), i))
        .collect();
    let scored: Vec<plainline_core::metrics::ScoredRow> = read_jsonl(a.run.join("scored.jsonl"))?;
    let mut rows = Vec::new();
    for r in &scored {
        let item = items
            .get(&r.item_id)
            .ok_or_else(|| CliError(format!("item {} is not in the data", r.item_id)))?;
        for f in &item.facts.facts {
            let verdict = r
                .fact_verdicts
                .get(&f.fact_id)
                .ok_or_else(|| CliError(format!("output {} has no verdict for {}", r.output_id, f.fact_id)))?;
            rows.push(LabelRow {
                key: format!("{}|{}", r.output_id, f.fact_id),
                fact: f.text.clone(),
                tier: f.tier,
                explanation: r.text.clone(),
                judge_stated: verdict == "stated",
                human_stated: None,
            });
        }
    }
    write_jsonl(&a.out, &rows)?;
    Ok(Output {
        stdout: format!(
            "wrote {} rows to {}; fill human_stated with true or false\n",
            rows.len(),
            a.out.display()
        ),
        ..Default::default()
    })
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn cmd_agreement(a: &AgreementArgs) -> CmdResult {
    let sheet: Vec<Value> = read_jsonl(&a.labels)?;
    let (mut judged, mut human) = (Vec::new(), Vec::new());
    for r in sheet
        .iter()
        .filter(|r| r.get("human_stated").is_some_and(|v| !v.is_null()))
    {
        let key = r
            .get("key")
            .and_then(Value::as_str)
            .ok_or_else(|| CliError("a label row has no key".into()))?;
        judged.push(Label {
            key: key.into(),
            stated: truthy(&r["judge_stated"]),
        });
        human.push(Label {
            key: key.into(),
            stated: truthy(&r["human_stated"]),
        });
    }
    Ok(Output {
        stdout: format!("{}\n", serde_json::to_string_pretty(&agreement(&judged, &human))?),
        ..Default::default()
    })
}

/// Run a future to completion on a fresh multi-threaded runtime.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(f)
}

/// Run one parsed command.
pub fn execute(cli: &Cli) -> Output {
    let d = cli.data_dir.as_path();
    let res = match &cli.cmd {
        Cmd::BuildPilot => cmd_build_pilot(d),
        Cmd::ValidateData => cmd_validate(d),
        Cmd::Draft(a) => cmd_draft(d, a),
        Cmd::Run(a) => cmd_run(d, a),
        Cmd::Report(a) => cmd_report(a),
        Cmd::ExportLabels(a) => cmd_export_labels(d, a),
        Cmd::Agreement(a) => cmd_agreement(a),
    };
    res.unwrap_or_else(|e| Output {
        stderr: format!("error: {}\n", e.0),
        code: 1,
        ..Default::default()
    })
}

/// Parse `args` (program name first) and run. Returns the process exit code.
pub fn run_cli(args: &[String]) -> i32 {
    let out = run_cli_captured(args);
    print!("{}", out.stdout);
    eprint!("{}", out.stderr);
    out.code
}

/// Like [`run_cli`] but returns the text instead of printing it.
pub fn run_cli_captured(args: &[String]) -> Output {
    match Cli::try_parse_from(args) {
        Ok(cli) => execute(&cli),
        Err(e) => {
            let code = e.exit_code();
            let text = e.render().to_string();
            if code == 0 {
                Output {
                    stdout: text,
                    code,
                    ..Default::default()
                }
            } else {
                Output {
                    stderr: text,
                    code,
                    ..Default::default()
                }
            }
        }
    }
}
