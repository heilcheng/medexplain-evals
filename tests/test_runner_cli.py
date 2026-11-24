import json

import httpx
import pytest

from conftest import ScriptedClient, verdict_json
from plainline import cli
from plainline.client import MockClient, OpenAICompatClient
from plainline.readers import conditions
from plainline.report import build_summary, to_markdown
from plainline.runner import run


def test_run_with_mock_and_resume(item, tmp_path):
    conds = conditions("adapt")
    p = run([item], conds, MockClient(), MockClient(), tmp_path, mock=True)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(rows) == len(conds) and all(r["mock"] for r in rows)
    run([item], conds, MockClient(), MockClient(), tmp_path, mock=True)  # nothing new to do
    assert len(p.read_text().splitlines()) == len(conds)


def test_scripted_end_to_end(item, tmp_path):
    writer = ScriptedClient(["Can cause liver damage. Take with food."], name="w")
    judge = ScriptedClient([verdict_json(f1="stated", f2="absent", f3="stated", f4="absent", n1="not_asserted")],
                           name="j")
    conds = conditions("adapt")[:2]
    p = run([item], conds, writer, judge, tmp_path, workers=1)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[0]["cfr"] == 0.5 and rows[0]["must_missing"] == ["f2"]
    assert rows[0]["facts_status"] == "draft, needs clinician review"
    md = to_markdown(build_summary(p, [c.id for c in conds], n_boot=50))
    assert "NOT been reviewed by clinicians" in md


def test_report_banner_absent_only_when_all_validated(item, tmp_path):
    item.facts.status = "clinician_reviewed"
    item.facts.reviewers = ["a", "b"]
    conds = conditions("adapt")[:2]
    p = run([item], conds, MockClient(), MockClient(), tmp_path)
    assert "NOT been reviewed" not in to_markdown(build_summary(p, [c.id for c in conds], n_boot=20))


def test_cli_mock_run_and_labels(tmp_path, capsys):
    out = tmp_path / "r"
    cli.main(["run", "--mock", "--limit", "3", "--out", str(out), "--boot", "50"])
    assert (out / "summary.json").exists() and "results.md" in {f.name for f in out.iterdir()}
    sheet = tmp_path / "labels.jsonl"
    cli.main(["export-labels", "--run", str(out), "--out", str(sheet)])
    rows = [json.loads(l) for l in sheet.read_text().splitlines()]
    for r in rows:
        r["human_stated"] = r["judge_stated"]
    sheet.write_text("\n".join(json.dumps(r) for r in rows))
    capsys.readouterr()
    cli.main(["agreement", "--labels", str(sheet)])
    res = json.loads(capsys.readouterr().out)
    assert res["kappa"] == 1.0 and res["n"] == len(rows)


def test_cli_cue_mode(tmp_path):
    out = tmp_path / "c"
    cli.main(["run", "--mock", "--mode", "cue", "--limit", "2", "--out", str(out), "--boot", "30"])
    s = json.loads((out / "summary.json").read_text())
    assert "low_literacy" in s["models"]["mock"]["conditions"]


def test_cli_requires_endpoint(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "--out", str(tmp_path)])


def test_validate_data_cli():
    with pytest.raises(SystemExit) as e:
        cli.main(["validate-data"])
    assert e.value.code == 0


def test_openai_client_request_shape(monkeypatch):
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("MY_KEY", "k")
    c = OpenAICompatClient("http://localhost:8000/v1/", "m", "MY_KEY")
    assert c.complete([{"role": "user", "content": "x"}]) == "hi"
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k" and seen["json"]["model"] == "m"


def test_openai_client_retries(monkeypatch):
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(1)
        code = 503 if len(calls) < 2 else 200
        return httpx.Response(code, json={"choices": [{"message": {"content": "ok"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert OpenAICompatClient("http://x", "m").complete([]) == "ok" and len(calls) == 2
