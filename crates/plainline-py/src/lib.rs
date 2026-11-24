//! Python bindings: the Plainline Rust core exposed as `plainline._core`.
//!
//! Structured values cross the boundary as JSON, so the Python side sees plain dicts and
//! lists. Everything here is a thin shim; the logic lives in the `plainline-*` crates.

use plainline_cli::commands::{execute, run_cli_captured, Cli, Cmd, Endpoint, Mode, Output, RunArgs};
use plainline_core::judge::{self, Label};
use plainline_core::metrics::{self, ScoredRow};
use plainline_core::readability::{self, Band};
use plainline_core::readers::{self, Condition};
use plainline_core::report;
use plainline_core::rng::Generator;
use plainline_core::schema::{self, FactSet, Source};
use plainline_core::{judge::JudgeResult, prompts};
use plainline_pipeline::{draft, text};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::PathBuf;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Serialise `v` and load it back as Python objects.
fn to_py<'py, T: Serialize>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyAny>> {
    let s = serde_json::to_string(v).map_err(err)?;
    py.import("json")?.call_method1("loads", (s,))
}

/// Dump Python objects to JSON and parse them into `T`.
fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    let s: String = obj.py().import("json")?.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn condition_json(c: &Condition) -> Value {
    json!({"id": c.id, "kind": c.kind, "instruction": c.instruction,
           "band": c.band.map(|(a, b)| json!([a, b])), "group": c.group})
}

fn condition_from(v: &Value) -> PyResult<Condition> {
    let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let band: Option<Band> = match v.get("band") {
        Some(Value::Array(a)) if a.len() == 2 => Some((a[0].as_f64(), a[1].as_f64())),
        Some(Value::Null) | None => None,
        Some(_) => return Err(PyValueError::new_err("band must be null or [low, high]")),
    };
    Ok(Condition { id: s("id"), kind: s("kind"), instruction: s("instruction"), band, group: s("group") })
}

fn path(p: Option<String>, default: &str) -> PathBuf {
    PathBuf::from(p.unwrap_or_else(|| default.to_string()))
}

// ------------------------------------------------------------- readability

#[pyfunction]
fn fk_grade(text: &str) -> f64 {
    readability::fk_grade(text)
}

#[pyfunction]
fn smog_grade(text: &str) -> f64 {
    readability::smog_grade(text)
}

#[pyfunction]
fn word_count(text: &str) -> usize {
    readability::word_count(text)
}

#[pyfunction]
fn syllables(word: &str) -> usize {
    readability::syllables(word)
}

#[pyfunction]
#[pyo3(signature = (grade, band=None))]
fn in_band(grade: f64, band: Option<(Option<f64>, Option<f64>)>) -> Option<bool> {
    readability::in_band(grade, band)
}

// ------------------------------------------------------- readers, prompts

#[pyfunction]
#[pyo3(signature = (mode="adapt", path=None))]
fn conditions<'py>(py: Python<'py>, mode: &str, path: Option<String>) -> PyResult<Bound<'py, PyAny>> {
    let p = path.map(PathBuf::from);
    let conds = readers::conditions(mode, p.as_deref()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_py(py, &conds.iter().map(condition_json).collect::<Vec<_>>())
}

#[pyfunction]
fn writer_messages<'py>(py: Python<'py>, source_text: &str, reader_instruction: &str) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &prompts::writer_messages(source_text, reader_instruction))
}

#[pyfunction]
fn judge_messages<'py>(
    py: Python<'py>,
    explanation: &str,
    facts: Vec<(String, String)>,
    statements: Vec<(String, String)>,
) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &prompts::judge_messages(explanation, &facts, &statements))
}

#[pyfunction]
fn extract_messages<'py>(py: Python<'py>, source_text: &str) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &prompts::extract_messages(source_text))
}

// ------------------------------------------------------------------- data

#[pyfunction]
#[pyo3(signature = (data_dir="data", split=None))]
fn load_items<'py>(py: Python<'py>, data_dir: &str, split: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let items = schema::load_items(data_dir, split).map_err(err)?;
    to_py(py, &items)
}

#[pyfunction]
#[pyo3(signature = (factset, require_must=true))]
fn validate_factset(factset: &Bound<'_, PyAny>, require_must: bool) -> PyResult<Vec<String>> {
    let fs: FactSet = from_py(factset)?;
    Ok(schema::validate_factset(&fs, require_must))
}

// ----------------------------------------------------------------- judging

#[pyfunction]
fn parse_verdicts(
    py: Python<'_>,
    raw: &str,
    fact_ids: Vec<String>,
    stmt_ids: Vec<String>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, u32)> {
    let (f, s, e) = judge::parse_verdicts(raw, &fact_ids, &stmt_ids);
    Ok((to_py(py, &f)?.unbind(), to_py(py, &s)?.unbind(), e))
}

#[pyfunction]
fn agreement<'py>(py: Python<'py>, judged: &Bound<'py, PyAny>, human: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (j, h): (Vec<Label>, Vec<Label>) = (from_py(judged)?, from_py(human)?);
    to_py(py, &judge::agreement(&j, &h))
}

// ----------------------------------------------------------------- metrics

#[pyfunction]
fn score_output<'py>(
    py: Python<'py>,
    explanation: &str,
    factset: &Bound<'py, PyAny>,
    fact_verdicts: &Bound<'py, PyAny>,
    mns_verdicts: &Bound<'py, PyAny>,
    parse_errors: u32,
    condition: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let fs: FactSet = from_py(factset)?;
    let jr = JudgeResult {
        fact_verdicts: from_py(fact_verdicts)?,
        mns_verdicts: from_py(mns_verdicts)?,
        parse_errors,
        raw: vec![],
    };
    let cond = condition_from(&from_py::<Value>(condition)?)?;
    to_py(py, &metrics::score_output(explanation, &fs, &jr, &cond))
}

#[pyfunction]
#[pyo3(signature = (rows, cond_order, n_boot=2000, seed=0))]
fn summarize_model<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    cond_order: Vec<String>,
    n_boot: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let rows: Vec<ScoredRow> = from_py(rows)?;
    let s = py.detach(|| metrics::summarize_model(&rows, &cond_order, n_boot, seed));
    to_py(py, &s)
}

/// Point estimates of the headline metrics; an empty dict when they are undefined.
#[pyfunction]
fn headline_points<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>, cond_order: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
    let rows: Vec<ScoredRow> = from_py(rows)?;
    match metrics::headline_points(&rows, &cond_order) {
        None => Ok(PyDict::new(py).into_any()),
        Some(h) => {
            let d = PyDict::new(py);
            d.set_item("worst_reader_cfr", h.worst_reader_cfr)?;
            d.set_item("mean_adaptation_tax", h.mean_adaptation_tax)?;
            Ok(d.into_any())
        }
    }
}

#[pyfunction]
fn length_control<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let rows: Vec<ScoredRow> = from_py(rows)?;
    match metrics::length_control(&rows) {
        None => Ok(PyDict::new(py).into_any()),
        Some(lc) => to_py(py, &lc),
    }
}

/// Bootstrap resamples exactly as `numpy.random.default_rng(seed).integers(0, n, n)` draws them.
#[pyfunction]
fn bootstrap_draws(n_items: usize, n_boot: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Generator::new(seed);
    (0..n_boot)
        .map(|_| {
            let mut ix = vec![0usize; n_items];
            rng.fill_integers(n_items, &mut ix);
            ix
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (scored_path, cond_order, n_boot=2000, seed=0))]
fn build_summary<'py>(py: Python<'py>, scored_path: &str, cond_order: Vec<String>, n_boot: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
    let s = py.detach(|| report::build_summary(scored_path, &cond_order, n_boot, seed)).map_err(err)?;
    to_py(py, &s)
}

#[pyfunction]
#[pyo3(signature = (scored_path, cond_order, out_dir, n_boot=2000, seed=0))]
fn write_report<'py>(
    py: Python<'py>,
    scored_path: &str,
    cond_order: Vec<String>,
    out_dir: &str,
    n_boot: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let s = py.detach(|| report::write_report(scored_path, &cond_order, out_dir, n_boot, seed)).map_err(err)?;
    to_py(py, &s)
}

// ---------------------------------------------------------------- pipeline

#[pyfunction]
fn rule_draft<'py>(py: Python<'py>, source: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let src: Source = from_py(source)?;
    to_py(py, &draft::rule_draft(&src))
}

#[pyfunction]
fn clean(t: &str) -> String {
    text::clean(t)
}

#[pyfunction]
fn sentences(t: &str) -> Vec<String> {
    text::sentences(t)
}

#[pyfunction]
fn truncate_words(t: &str, max_words: usize) -> String {
    text::truncate_words(t, max_words)
}

#[pyfunction]
fn dedupe_sentences(t: &str) -> String {
    text::dedupe_sentences(t)
}

// ------------------------------------------------------------------ runner

/// Hand a command's text to Python's own stdout and stderr, so capture and redirection work.
fn emit(py: Python<'_>, out: &Output) -> PyResult<()> {
    let sys = py.import("sys")?;
    if !out.stdout.is_empty() {
        sys.getattr("stdout")?.call_method1("write", (&out.stdout,))?;
    }
    if !out.stderr.is_empty() {
        sys.getattr("stderr")?.call_method1("write", (&out.stderr,))?;
    }
    Ok(())
}

/// Run the command line (`argv[0]` is the program name). Returns the exit code.
#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<String>) -> PyResult<i32> {
    let out = py.detach(|| run_cli_captured(&argv));
    emit(py, &out)?;
    Ok(out.code)
}

/// Generate, judge, score and report one run; returns the markdown report.
#[pyfunction]
#[pyo3(signature = (out, data_dir="data", mode="adapt", split="dev", limit=None, samples=1, per_fact=false,
                    workers=4, boot=2000, mock=false, base_url=None, model=None,
                    api_key_env="OPENAI_API_KEY", judge_base_url=None, judge_model=None, judge_api_key_env=None))]
#[allow(clippy::too_many_arguments)]
fn run_eval(
    py: Python<'_>,
    out: String,
    data_dir: &str,
    mode: &str,
    split: &str,
    limit: Option<usize>,
    samples: u32,
    per_fact: bool,
    workers: u32,
    boot: usize,
    mock: bool,
    base_url: Option<String>,
    model: Option<String>,
    api_key_env: &str,
    judge_base_url: Option<String>,
    judge_model: Option<String>,
    judge_api_key_env: Option<String>,
) -> PyResult<String> {
    let mode = match mode {
        "adapt" => Mode::Adapt,
        "cue" => Mode::Cue,
        other => return Err(PyValueError::new_err(format!("unknown mode {other:?}; use 'adapt' or 'cue'"))),
    };
    let run = RunArgs {
        endpoint: Endpoint { base_url, model, api_key_env: api_key_env.to_string(), mock },
        judge_base_url,
        judge_model,
        judge_api_key_env,
        mode,
        out: path(Some(out), ""),
        split: split.to_string(),
        limit,
        samples,
        per_fact,
        workers: workers.max(1),
        boot,
    };
    let cli = Cli { data_dir: path(Some(data_dir.to_string()), "data"), cmd: Cmd::Run(run) };
    let res = py.detach(|| execute(&cli));
    emit(py, &Output { stdout: String::new(), ..res.clone() })?;
    if res.code != 0 {
        return Err(PyRuntimeError::new_err(res.stderr.trim().to_string()));
    }
    Ok(res.stdout)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("JUDGE_VERSION", prompts::JUDGE_VERSION)?;
    m.add("WRITER_SYSTEM", prompts::WRITER_SYSTEM)?;
    m.add("WRITER_TASK", prompts::WRITER_TASK)?;
    m.add("JUDGE_SYSTEM", prompts::JUDGE_SYSTEM)?;
    m.add("EXTRACT_SYSTEM", prompts::EXTRACT_SYSTEM)?;
    m.add("DRAFT_STATUS", schema::DRAFT_STATUS)?;
    m.add("CANARY", schema::CANARY)?;
    m.add_function(wrap_pyfunction!(fk_grade, m)?)?;
    m.add_function(wrap_pyfunction!(smog_grade, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_function(wrap_pyfunction!(syllables, m)?)?;
    m.add_function(wrap_pyfunction!(in_band, m)?)?;
    m.add_function(wrap_pyfunction!(conditions, m)?)?;
    m.add_function(wrap_pyfunction!(writer_messages, m)?)?;
    m.add_function(wrap_pyfunction!(judge_messages, m)?)?;
    m.add_function(wrap_pyfunction!(extract_messages, m)?)?;
    m.add_function(wrap_pyfunction!(load_items, m)?)?;
    m.add_function(wrap_pyfunction!(validate_factset, m)?)?;
    m.add_function(wrap_pyfunction!(parse_verdicts, m)?)?;
    m.add_function(wrap_pyfunction!(agreement, m)?)?;
    m.add_function(wrap_pyfunction!(score_output, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_model, m)?)?;
    m.add_function(wrap_pyfunction!(headline_points, m)?)?;
    m.add_function(wrap_pyfunction!(length_control, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_draws, m)?)?;
    m.add_function(wrap_pyfunction!(build_summary, m)?)?;
    m.add_function(wrap_pyfunction!(write_report, m)?)?;
    m.add_function(wrap_pyfunction!(rule_draft, m)?)?;
    m.add_function(wrap_pyfunction!(clean, m)?)?;
    m.add_function(wrap_pyfunction!(sentences, m)?)?;
    m.add_function(wrap_pyfunction!(truncate_words, m)?)?;
    m.add_function(wrap_pyfunction!(dedupe_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval, m)?)?;
    Ok(())
}
