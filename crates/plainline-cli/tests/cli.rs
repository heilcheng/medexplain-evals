use async_trait::async_trait;
use plainline_cli::run_cli_captured;
use plainline_cli::runner::{RunOptions, output_id, run};
use plainline_client::{ChatClient, ClientError, Message, MockClient, ScriptedClient};
use plainline_core::fixtures::{item, verdict_json};
use plainline_core::metrics::ScoredRow;
use plainline_core::readers::conditions;
use plainline_core::report::{build_summary, to_markdown};
use plainline_core::schema::read_jsonl;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("plainline-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn cli(args: &[&str]) -> plainline_cli::Output {
    let data = root().join("data");
    let mut argv: Vec<String> = vec!["plainline".into(), "--data-dir".into(), data.to_string_lossy().into()];
    argv.extend(args.iter().map(|s| s.to_string()));
    run_cli_captured(&argv)
}

fn ids(n: usize, mode: &str) -> Vec<String> {
    conditions(mode, None)
        .unwrap()
        .iter()
        .take(n)
        .map(|c| c.id.clone())
        .collect()
}

fn lines(p: &Path) -> Vec<String> {
    std::fs::read_to_string(p).unwrap().lines().map(String::from).collect()
}

#[tokio::test]
async fn run_with_mock_and_resume() {
    let dir = tmp("resume");
    let conds = conditions("adapt", None).unwrap();
    let opts = RunOptions {
        mock: true,
        ..RunOptions::default()
    };
    let p = run(&[item()], &conds, &MockClient, &MockClient, &dir, &opts)
        .await
        .unwrap();
    let rows: Vec<ScoredRow> = read_jsonl(&p).unwrap();
    assert_eq!(rows.len(), conds.len());
    assert!(rows.iter().all(|r| r.mock));
    run(&[item()], &conds, &MockClient, &MockClient, &dir, &opts)
        .await
        .unwrap(); // nothing new to do
    assert_eq!(lines(&p).len(), conds.len());
}

#[tokio::test]
async fn scripted_end_to_end() {
    let dir = tmp("scripted");
    let writer = ScriptedClient::new(&["Can cause liver damage. Take with food."], "w");
    let judge = ScriptedClient::new(
        &[&verdict_json(&[
            ("f1", "stated"),
            ("f2", "absent"),
            ("f3", "stated"),
            ("f4", "absent"),
            ("n1", "not_asserted"),
        ])],
        "j",
    );
    let conds: Vec<_> = conditions("adapt", None).unwrap().into_iter().take(2).collect();
    let opts = RunOptions {
        workers: 1,
        ..RunOptions::default()
    };
    let p = run(&[item()], &conds, &writer, &judge, &dir, &opts).await.unwrap();
    let rows: Vec<ScoredRow> = read_jsonl(&p).unwrap();
    assert_eq!(rows[0].cfr, Some(0.5));
    assert_eq!(rows[0].must_missing, ["f2"]);
    assert_eq!(rows[0].facts_status, "draft, needs clinician review");
    assert_eq!(rows[0].judge, "j");
    assert_eq!(rows[0].model, "w");
    let md = to_markdown(&build_summary(&p, &ids(2, "adapt"), 50, 0).unwrap());
    assert!(md.contains("NOT been reviewed by clinicians"));
}

#[tokio::test]
async fn report_banner_absent_only_when_all_validated() {
    let dir = tmp("banner");
    let mut it = item();
    it.facts.status = plainline_core::schema::Status::ClinicianReviewed;
    it.facts.reviewers = vec!["a".into(), "b".into()];
    let conds: Vec<_> = conditions("adapt", None).unwrap().into_iter().take(2).collect();
    let p = run(&[it], &conds, &MockClient, &MockClient, &dir, &RunOptions::default())
        .await
        .unwrap();
    assert!(!to_markdown(&build_summary(&p, &ids(2, "adapt"), 20, 0).unwrap()).contains("NOT been reviewed"));
}

#[test]
fn cli_mock_run_and_labels() {
    let dir = tmp("cli-run");
    let out = dir.join("r");
    let res = cli(&[
        "run",
        "--mock",
        "--limit",
        "3",
        "--out",
        out.to_str().unwrap(),
        "--boot",
        "50",
    ]);
    assert_eq!(res.code, 0, "{}", res.stderr);
    assert!(out.join("summary.json").exists() && out.join("results.md").exists());
    assert!(res.stdout.starts_with("# Plainline results"));
    let sheet = dir.join("labels.jsonl");
    let res = cli(&[
        "export-labels",
        "--run",
        out.to_str().unwrap(),
        "--out",
        sheet.to_str().unwrap(),
    ]);
    assert_eq!(res.code, 0, "{}", res.stderr);
    let rows: Vec<serde_json::Value> = lines(&sheet).iter().map(|l| serde_json::from_str(l).unwrap()).collect();
    let filled: Vec<String> = rows
        .iter()
        .map(|r| {
            let mut r = r.clone();
            r["human_stated"] = r["judge_stated"].clone();
            r.to_string()
        })
        .collect();
    std::fs::write(&sheet, filled.join("\n")).unwrap();
    let res = cli(&["agreement", "--labels", sheet.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&res.stdout).unwrap();
    assert_eq!(v["kappa"], 1.0);
    assert_eq!(v["n"], rows.len());
}

#[test]
fn cli_cue_mode_and_report_infers_it() {
    let dir = tmp("cue");
    let out = dir.join("c");
    let res = cli(&[
        "run",
        "--mock",
        "--mode",
        "cue",
        "--limit",
        "2",
        "--out",
        out.to_str().unwrap(),
        "--boot",
        "30",
    ]);
    assert_eq!(res.code, 0, "{}", res.stderr);
    let s: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("summary.json")).unwrap()).unwrap();
    assert!(s["models"]["mock"]["conditions"]["low_literacy"].is_object());
    // `report` without --mode must treat a cue run as a cue run.
    let res = cli(&["report", "--run", out.to_str().unwrap(), "--boot", "30"]);
    assert_eq!(res.code, 0, "{}", res.stderr);
    assert!(res.stdout.contains("low_literacy"));
}

#[test]
fn cli_requires_endpoint() {
    let dir = tmp("noendpoint");
    let res = cli(&["run", "--out", dir.to_str().unwrap()]);
    assert_eq!(res.code, 1);
    assert!(
        res.stderr.contains("--base-url and --model are required"),
        "{}",
        res.stderr
    );
}

#[test]
fn usage_errors_exit_2_and_help_exits_0() {
    let res = cli(&["run"]);
    assert_eq!(res.code, 2);
    assert!(res.stderr.contains("--out"));
    let res = cli(&["--help"]);
    assert_eq!(res.code, 0);
    for sub in [
        "build-pilot",
        "draft",
        "validate-data",
        "run",
        "report",
        "export-labels",
        "agreement",
    ] {
        assert!(res.stdout.contains(sub), "{sub}");
    }
}

#[test]
fn validate_data_cli() {
    let res = cli(&["validate-data"]);
    assert_eq!(res.code, 0);
    assert!(res.stdout.contains("0 with problems"), "{}", res.stdout);
    assert!(
        res.stdout
            .starts_with("30 items, 339 facts, 0 clinician-validated fact sets")
    );
}

struct FlakyWriter;

#[async_trait]
impl ChatClient for FlakyWriter {
    fn name(&self) -> &str {
        "flaky"
    }
    async fn complete(&self, messages: &[Message], _t: f64, _m: u32) -> Result<String, ClientError> {
        let u = messages.last().unwrap().content.to_lowercase();
        if u.contains("grade") || u.contains("sixth") {
            return Err(ClientError::Other("boom".into()));
        }
        Ok("Can cause liver damage.".into())
    }
}

#[tokio::test]
async fn run_keeps_completed_rows_when_one_job_fails() {
    let dir = tmp("flaky");
    let conds = conditions("adapt", None).unwrap();
    let judge = ScriptedClient::new(
        &[&verdict_json(&[
            ("f1", "stated"),
            ("f2", "stated"),
            ("f3", "stated"),
            ("f4", "stated"),
            ("n1", "not_asserted"),
        ])],
        "j",
    );
    let opts = RunOptions {
        workers: 1,
        ..RunOptions::default()
    };
    let err = run(&[item()], &conds, &FlakyWriter, &judge, &dir, &opts)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("jobs failed"), "{err}");
    assert!(err.to_string().contains("Completed rows were kept; rerun to resume."));
    let n = lines(&dir.join("scored.jsonl")).len();
    assert!(0 < n && n < conds.len());
    // a rerun with a healthy writer only does the missing jobs
    let healthy = ScriptedClient::new(&["ok"], "flaky");
    run(&[item()], &conds, &healthy, &judge, &dir, &opts).await.unwrap();
    assert_eq!(lines(&dir.join("scored.jsonl")).len(), conds.len());
    assert_eq!(healthy.calls().len(), conds.len() - n);
}

#[tokio::test]
async fn rows_land_in_job_order_even_with_many_workers() {
    let dir = tmp("order");
    let conds = conditions("adapt", None).unwrap();
    let opts = RunOptions {
        workers: 8,
        mock: true,
        ..RunOptions::default()
    };
    let p = run(&[item()], &conds, &MockClient, &MockClient, &dir, &opts)
        .await
        .unwrap();
    let rows: Vec<ScoredRow> = read_jsonl(&p).unwrap();
    let got: Vec<&str> = rows.iter().map(|r| r.condition.as_str()).collect();
    let want: Vec<&str> = conds.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(got, want);
}

#[test]
fn output_ids_match_python() {
    // hashlib.sha1(b"mock|dm-warfarin|neutral|0").hexdigest()[:12]
    assert_eq!(output_id("mock", "dm-warfarin", "neutral", 0), "30ce9133d528");
    assert_ne!(
        output_id("mock", "a", "neutral", 0),
        output_id("mock", "a", "neutral", 1)
    );
}

#[test]
fn mock_runs_match_python() {
    // tests/golden/ was recorded from the Python implementation on the pilot data.
    let g: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("tests/golden/mock_runs.json")).unwrap()).unwrap();
    for mode in ["adapt", "cue"] {
        let dir = tmp(&format!("golden-{mode}"));
        let res = cli(&[
            "run",
            "--mock",
            "--mode",
            mode,
            "--out",
            dir.to_str().unwrap(),
            "--boot",
            "500",
        ]);
        assert_eq!(res.code, 0, "{}", res.stderr);
        let scored = std::fs::read(dir.join("scored.jsonl")).unwrap();
        let hex: String = Sha256::digest(&scored).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            g[mode]["scored_sha256"].as_str().unwrap(),
            "scored.jsonl differs for {mode}"
        );
        assert_eq!(
            lines(&dir.join("scored.jsonl")).len() as u64,
            g[mode]["rows"].as_u64().unwrap()
        );
        let want_md = std::fs::read_to_string(root().join(format!("tests/golden/mock_{mode}_results.md"))).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("results.md")).unwrap(), want_md);
        let want: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root().join(format!("tests/golden/mock_{mode}_summary.json"))).unwrap(),
        )
        .unwrap();
        let got: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
        assert_json_close(&want, &got, mode);
    }
}

fn assert_json_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
    use serde_json::Value::*;
    match (a, b) {
        (Object(x), Object(y)) => {
            assert_eq!(x.keys().collect::<Vec<_>>(), y.keys().collect::<Vec<_>>(), "{path}");
            x.iter()
                .for_each(|(k, v)| assert_json_close(v, &y[k], &format!("{path}.{k}")));
        }
        (Array(x), Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}");
            x.iter()
                .zip(y)
                .enumerate()
                .for_each(|(i, (p, q))| assert_json_close(p, q, &format!("{path}[{i}]")));
        }
        (Number(x), Number(y)) => assert!((x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-9, "{path}"),
        _ => assert_eq!(a, b, "{path}"),
    }
}
