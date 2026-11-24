"""Parity check and golden generator. Stage 1 of the Rust port only.

Runs the original Python implementation (plainline.* pure-Python modules) and the Rust core
(plainline._core) side by side, asserts they agree (floats within 1e-9, NaN equal to null),
and records the Python outputs under tests/golden/ so the same checks keep running after the
Python logic is removed. Run from the repository root with both implementations importable.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from plainline import _core as rs
from plainline import judge as pj
from plainline import metrics as pm
from plainline import prompts as pp
from plainline import readability as pr
from plainline import readers as prd
from plainline.pipeline import dailymed, draft, medlineplus, text as ptext
from plainline.schema import FactSet, Source, load_items

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "tests" / "golden"
TOL = 1e-9
checks = {"n": 0}


def norm(x):
    """JSON-safe copy: NaN becomes None, tuples become lists."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    if isinstance(x, (np.floating,)):
        return norm(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def close(a, b, path="$"):
    a, b = norm(a), norm(b)
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None, f"{path}: {a!r} vs {b!r}"
        assert abs(a - b) <= TOL * max(1.0, abs(a)), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"
    checks["n"] += 1


def dump(name, obj):
    (GOLD / name).write_text(json.dumps(norm(obj), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ prompts / readers / data
def check_static():
    for n in ("JUDGE_VERSION", "WRITER_SYSTEM", "WRITER_TASK", "JUDGE_SYSTEM", "EXTRACT_SYSTEM"):
        assert getattr(pp, n) == getattr(rs, n), n
    from plainline.schema import CANARY, DRAFT_STATUS
    assert rs.CANARY == CANARY and rs.DRAFT_STATUS == DRAFT_STATUS
    src = "Take 1 tablet.\nDo not exceed 4 g."
    assert pp.writer_messages(src, "Reader: x") == rs.writer_messages(src, "Reader: x")
    assert pp.extract_messages(src) == rs.extract_messages(src)
    for stm in ([], [("n1", "bad")]):
        assert pp.judge_messages("E", [("f1", "a"), ("f2", "b")], stm) == rs.judge_messages("E", [("f1", "a"), ("f2", "b")], stm)
    for mode in ("adapt", "cue"):
        py = [{"id": c.id, "kind": c.kind, "instruction": c.instruction,
               "band": list(c.band) if c.band else None, "group": c.group} for c in prd.conditions(mode)]
        close(py, rs.conditions(mode))
    items = load_items(ROOT / "data")
    close([i.model_dump() for i in items], rs.load_items(str(ROOT / "data")))
    close([i.model_dump() for i in load_items(ROOT / "data", "test")], rs.load_items(str(ROOT / "data"), "test"))
    for it in items:
        d = it.facts.model_dump()
        assert pp and draft.validate_factset(it.facts) == rs.validate_factset(d)
    # byte round trip of the data files
    from plainline.schema import read_jsonl, write_jsonl
    return items


# --------------------------------------------------------------------------- readability
VOCAB = ("take one pill each day do not more call your doctor if you bleed anticoagulant concomitant "
         "administration necessitates international normalized ratio surveillance medicine liver damage "
         "rested closed shaped boxes passes cases edged raised noted tablet tablets "
         "doesn't rock-solid o'clock e.g. fast-acting").split()


def fuzz_texts(n=300, seed=3):
    rnd = random.Random(seed)
    out = ["", "   ", "...", "Hello", "Hello.", "Hello.\n", "Hello.\n\nWorld", "A.B.C. d? e! f", "no punctuation at all"]
    for _ in range(n):
        k = rnd.randint(1, 60)
        parts = []
        for _ in range(k):
            w = rnd.choice(VOCAB)
            parts.append(w + rnd.choice(["", "", "", ".", ".", "!", "?", ",", "\n\n", ". ", "..."]))
        out.append(rnd.choice([" ", " ", "\n"]).join(parts))
    return out


def check_readability(items):
    ids = {i.source.text: i.source.source_id for i in items}
    texts = [i.source.text for i in items] + fuzz_texts()
    gold = []
    for t in texts:
        py = {("source_id" if t in ids else "text"): ids.get(t, t), "words": pr.word_count(t),
              "fk": pr.fk_grade(t), "smog": pr.smog_grade(t)}
        assert rs.word_count(t) == py["words"]
        for k, f in (("fk", rs.fk_grade), ("smog", rs.smog_grade)):
            assert abs(f(t) - py[k]) <= TOL, (k, t[:50])
        gold.append(py)
    words = sorted(set(" ".join(texts).replace("\n", " ").split()))
    syl = {w: pr.syllables(w) for w in words}
    for w, n in syl.items():
        assert rs.syllables(w) == n, w
    for g in (-3.0, 5.0, 8.0, 8.0000001, 12.5):
        for band in (None, (None, 8), (10, None), (8, 16), (None, None)):
            assert pr.in_band(g, band) == rs.in_band(g, band), (g, band)
    checks["n"] += len(gold) * 3 + len(syl)
    dump("readability.json", {"texts": gold, "syllables": syl})


# ---------------------------------------------------------------------------- verdict parsing
def check_verdicts():
    from conftest_like import verdict_json  # noqa
    fids, sids = ["f1", "f2", "f3"], ["n1", "n2"]
    raws = [
        verdict_json(f1="stated", f2="absent", f3="contradicted", n1="asserted", n2="not_asserted"),
        "```json\n" + verdict_json(f1="stated", n1="asserted") + "\n```",
        "```" + verdict_json(f1="stated") + "```",
        "Here you go: " + verdict_json(f1="Stated ", f2=" ABSENT") + " thanks",
        verdict_json(f1="maybe"), "I cannot do that", "", "{}", "[]", "null", "123", '"x"',
        '{"verdicts": null}', '{"verdicts": {}}', '{"verdicts": "abc"}', '{"verdicts": [1, 2]}',
        '{"verdicts": [{"id": "f1", "verdict": "stated"}, 7, {"id": "f2", "verdict": "stated"}]}',
        '{"verdicts": [{"id": 1, "verdict": "stated"}, {"id": null, "verdict": null}, {"verdict": "absent"}]}',
        '{"verdicts": [{"id": "f1", "verdict": "stated"}, {"id": "f1", "verdict": "contradicted"}]}',
        '{"verdicts": [{"id": "f1", "verdict": ["stated"]}]}',
        '{"verdicts": [{"id": "n1", "verdict": "stated"}, {"id": "f1", "verdict": "asserted"}]}',
        'prefix {"verdicts": [{"id": "f1", "verdict": "stated"}]} and {"other": 1}',
        '{"verdicts": [{"id": "f1", "verdict": "stated"}',
        "{'verdicts': []}",
        '{"verdicts": [{"id": true, "verdict": "stated"}, {"id": 1.5, "verdict": "absent"}]}',
        "é {\"verdicts\": [{\"id\": \"f1\", \"verdict\": \"stated\"}]}",
    ]
    gold = []
    for raw in raws:
        py = pj.parse_verdicts(raw, fids, sids)
        got = rs.parse_verdicts(raw, fids, sids)
        close(list(py), list(got), f"verdicts[{raw[:30]!r}]")
        gold.append({"raw": raw, "fact_ids": fids, "stmt_ids": sids, "facts": py[0], "stmts": py[1], "errors": py[2]})
    dump("verdicts.json", gold)
    # agreement
    cases = []
    rnd = random.Random(5)
    for n in (0, 1, 6, 40):
        for _ in range(3):
            j = [{"key": str(i), "stated": rnd.random() < 0.6} for i in range(n)]
            h = [{"key": str(i), "stated": rnd.random() < 0.5} for i in range(n)]
            h = h[: max(0, n - 2)]
            cases.append({"judged": j, "human": h, "expected": pj.agreement(j, h)})
    cases.append({"judged": [{"key": "a", "stated": True}], "human": [{"key": "a", "stated": True}],
                  "expected": pj.agreement([{"key": "a", "stated": True}], [{"key": "a", "stated": True}])})
    for c in cases:
        close(c["expected"], rs.agreement(c["judged"], c["human"]))
    dump("agreement.json", cases)


# ------------------------------------------------------------------------------- metrics
def synth_rows(rnd, n_items, conds, samples=1, p_missing=0.0, p_none=0.0, words_fn=None):
    rows = []
    for i in range(n_items):
        base = rnd.uniform(0.3, 1.0)
        for c in conds:
            for s in range(samples):
                if rnd.random() < p_missing:
                    continue
                cfr = None if rnd.random() < p_none else round(min(1.0, max(0.0, base - (0.25 if c != "neutral" else 0) + rnd.uniform(-0.2, 0.2))), 3)
                rows.append({
                    "item_id": f"i{i:02d}", "condition": c, "cfr": cfr,
                    "sfr": None if rnd.random() < p_none else rnd.choice([0.0, 0.5, 1.0, 0.25]),
                    "distortion": rnd.choice([0.0, 0.0, 0.1, None]), "mns": rnd.choice([0.0, 1.0, None]),
                    "band_ok": None if c == "neutral" else rnd.random() < 0.5,
                    "words": (words_fn(rnd, c) if words_fn else rnd.randint(40, 300)),
                    "fk_grade": round(rnd.uniform(3, 14), 2), "parse_errors": rnd.choice([0, 0, 0, 2]),
                })
    return rows


def check_metrics():
    rnd = random.Random(11)
    conds = ["neutral", "clinician", "nurse", "grade6_adult"]
    cases = []

    def case(name, rows, order, n_boot=200, seed=0):
        py = pm.summarize_model(rows, order, n_boot=n_boot, seed=seed)
        # Est dataclasses are already dicts in the summary; headline too
        got = rs.summarize_model(rows, order, n_boot, seed)
        close(py, got, name)
        hp_py = pm.headline_points(rows, order)
        close(hp_py, rs.headline_points(rows, order), name + ".hp")
        lc_py = pm.length_control(rows)
        close(lc_py, rs.length_control(rows), name + ".lc")
        cases.append({"name": name, "rows": rows, "cond_order": order, "n_boot": n_boot, "seed": seed,
                      "summary": py, "headline_points": hp_py, "length_control": lc_py})

    case("basic", synth_rows(rnd, 20, conds), conds)
    case("seed7", synth_rows(rnd, 12, conds), conds, 300, 7)
    case("samples", synth_rows(rnd, 15, conds, samples=3), conds, 150, 1)
    case("missing", synth_rows(rnd, 18, conds, p_missing=0.25, p_none=0.15), conds, 150, 2)
    case("many_none", synth_rows(rnd, 10, conds, p_none=0.7), conds, 100, 3)
    case("single_item", synth_rows(rnd, 1, conds), conds, 50)
    case("two_items", synth_rows(rnd, 2, conds), conds, 50, 5)
    case("no_neutral", synth_rows(rnd, 12, conds[1:]), conds, 80)
    case("only_neutral", synth_rows(rnd, 12, conds[:1]), conds, 80)
    case("empty_cond_filter", synth_rows(rnd, 12, conds[:2]), conds, 80)
    case("order_changed", synth_rows(rnd, 12, conds), ["neutral", "grade6_adult", "clinician", "nurse"], 80)
    case("ties_words", synth_rows(rnd, 14, conds, words_fn=lambda r, c: r.choice([100, 120, 150])), conds, 90)
    case("matched_len_ties", synth_rows(rnd, 16, conds, words_fn=lambda r, c: 100 if c == "neutral" else r.choice([75, 133, 134, 74, 100])), conds, 90)
    case("zero_words", synth_rows(rnd, 12, conds, words_fn=lambda r, c: r.choice([0, 0, 50, 90])), conds, 60)
    case("constant_cfr", [dict(r, cfr=1.0) for r in synth_rows(rnd, 12, conds)], conds, 60)
    case("empty_rows", [], conds, 20)
    # the fixtures of the original unit tests
    rows = []
    r0 = np.random.default_rng(0)
    for i in range(20):
        base = r0.uniform(0.6, 1.0)
        for cond, drop, words in [("neutral", 0.0, 200), ("grade6_adult", 0.2, 120), ("clinician", 0.0, 210)]:
            rows.append({"item_id": f"i{i}", "condition": cond, "cfr": max(0, base - drop), "sfr": 0.5,
                         "distortion": 0.0, "mns": 0.0,
                         "band_ok": None if cond == "neutral" else (cond != "grade6_adult" or i % 2 == 0),
                         "words": (120 if i < 10 else 190) + i if cond == "grade6_adult" else words + i,
                         "fk_grade": 7.0, "parse_errors": 0})
    case("unit_fixture", rows, ["neutral", "clinician", "grade6_adult"], 300)
    case("unit_fixture_matched", rows, ["neutral", "grade6_adult"], 100)
    dump("metrics_cases.json", cases)

    # bootstrap draws: the Rust generator against numpy for many (n, seed)
    for n in (2, 3, 5, 20, 37, 200):
        for seed in (0, 1, 9, 123456789):
            rng = np.random.default_rng(seed)
            want = [rng.integers(0, n, n).tolist() for _ in range(40)]
            assert rs.bootstrap_draws(n, 40, seed) == want, (n, seed)
            checks["n"] += 1
    # fixed-index fixtures: bootstrap of a mean with explicit resamples
    for seed in (0, 4):
        x = np.array([0.1, 0.9, 0.5, 0.7, 1.0, 0.0, 0.3, 0.8])
        py = pm.bootstrap(lambda ix: float(x[ix].mean()), len(x), 500, seed=seed)
        draws = rs.bootstrap_draws(len(x), 500, seed)
        vals = [float(x[ix].mean()) for ix in draws]
        lo, hi = np.nanquantile(vals, [0.025, 0.975])
        assert abs(lo - py.lo) < TOL and abs(hi - py.hi) < TOL
        checks["n"] += 1


def check_score_output(items):
    rnd = random.Random(21)
    conds = {c.id: c for c in prd.conditions("adapt")}
    gold = []
    texts = fuzz_texts(20)
    for it in items[:12]:
        fs = it.facts
        for cid in ("neutral", "grade6_adult", "clinician"):
            cond = conds[cid]
            fv = {f.fact_id: rnd.choice(["stated", "stated", "absent", "contradicted"]) for f in fs.facts}
            sv = {f"n{i + 1}": rnd.choice(["asserted", "not_asserted"]) for i in range(len(fs.must_not_say))}
            text = rnd.choice(texts)
            jr = pj.JudgeResult(fv, sv, parse_errors=rnd.choice([0, 1]))
            py = pm.score_output(text, fs, jr, cond)
            got = rs.score_output(text, fs.model_dump(), fv, sv, jr.parse_errors,
                                  {"id": cond.id, "kind": cond.kind, "instruction": cond.instruction,
                                   "band": list(cond.band) if cond.band else None})
            close(py, {k: got[k] for k in py}, f"score_output[{it.source.source_id}/{cid}]")
            gold.append({"text": text, "factset": fs.model_dump(), "fact_verdicts": fv, "mns_verdicts": sv,
                         "parse_errors": jr.parse_errors, "condition": {"id": cond.id, "kind": cond.kind, "band": list(cond.band) if cond.band else None},
                         "expected": py})
    dump("score_output.json", gold)


# -------------------------------------------------------------------------------- pipeline
ENTITIES = ["&amp;", "&lt;p&gt;x&lt;/p&gt;", "&nbsp;", "&eacute;", "&rarr;", "&#8217;", "&#x2022;", "&mdash;", "&deg;C",
            "&micro;g", "&alpha;", "&amp", "&copy", "&notanentity;", "&#0;", "&#x80;", "&hellip;"]


def check_pipeline(items):
    corpus = [i.source.text for i in items]
    sids = {i.source.text: i.source.source_id for i in items}
    for i, it in enumerate(items):
        src = it.source
        py = draft.rule_draft(src)
        got = rs.rule_draft(src.model_dump())
        close(py.model_dump(), got, f"rule_draft[{src.source_id}]")
        # and it must equal the committed pilot facts (they were produced by rule_draft)
        close(it.facts.model_dump(), got, f"pilot_facts[{src.source_id}]")
    strings = corpus + ["A. B. 12 Dose.  Next sentence (see 5.1). \"Quoted\" end? [Bracket] x", "", "  ",
                        "One two. Three four five. Six seven.", "Bleeding can be fatal. Check INR. Bleeding can be fatal."]
    strings += [f"Text {e} more {e}." for e in ENTITIES] + ["<b>bold</b> text<br/>here <3 2 > 1 <p class='x'>y</p>"]
    gold = []
    for s in strings:
        row = {("source_id" if s in sids else "text"): sids.get(s, s), "clean": ptext.clean(s), "sentences": ptext.sentences(s),
               "truncate_30": ptext.truncate_words(s, 30), "truncate_4": ptext.truncate_words(s, 4),
               "dedupe": ptext.dedupe_sentences(s)}
        got = {"clean": rs.clean(s), "sentences": rs.sentences(s), "truncate_30": rs.truncate_words(s, 30),
               "truncate_4": rs.truncate_words(s, 4), "dedupe": rs.dedupe_sentences(s)}
        close({k: row[k] for k in got}, got, f"text[{s[:30]!r}]")
        gold.append(row)
    dump("pipeline_text.json", gold)


# ------------------------------------------------------------------------------ mock runs
def check_mock_runs():
    tmp = Path(tempfile.mkdtemp())
    gold = {}
    for mode in ("adapt", "cue"):
        py_out, rs_out = tmp / f"py_{mode}", tmp / f"rs_{mode}"
        py_cmd = [sys.executable, "-m", "plainline.cli", "--data-dir", str(ROOT / "data"), "run", "--mock",
                  "--mode", mode, "--out", str(py_out), "--boot", "500"]
        subprocess.run(py_cmd, check=True, capture_output=True, cwd=ROOT)
        txt = rs.run_eval(str(rs_out), data_dir=str(ROOT / "data"), mode=mode, mock=True, boot=500)
        assert (py_out / "scored.jsonl").read_bytes() == (rs_out / "scored.jsonl").read_bytes(), f"scored.jsonl {mode}"
        assert (py_out / "results.md").read_bytes() == (rs_out / "results.md").read_bytes(), f"results.md {mode}"
        assert txt == (rs_out / "results.md").read_text() + "\n"
        a = json.loads((py_out / "summary.json").read_text())
        b = json.loads((rs_out / "summary.json").read_text())
        close(a, b, f"summary[{mode}]")
        # resume does nothing, and a report recomputed from the rows agrees
        rs.run_eval(str(rs_out), data_dir=str(ROOT / "data"), mode=mode, mock=True, boot=500)
        assert (py_out / "scored.jsonl").read_bytes() == (rs_out / "scored.jsonl").read_bytes()
        ids = [c.id for c in prd.conditions(mode)]
        close(norm(json.loads(json.dumps(__import__("plainline.report", fromlist=["x"]).build_summary(py_out / "scored.jsonl", ids, 500)))),
              rs.build_summary(str(py_out / "scored.jsonl"), ids, 500), f"build_summary[{mode}]")
        gold[mode] = {"scored_sha256": hashlib.sha256((py_out / "scored.jsonl").read_bytes()).hexdigest(),
                      "rows": len((py_out / "scored.jsonl").read_text().splitlines())}
        shutil.copy(py_out / "results.md", GOLD / f"mock_{mode}_results.md")
        (GOLD / f"mock_{mode}_summary.json").write_text(json.dumps(norm(a), indent=1) + "\n")
        # export-labels and agreement CLIs
        py_sheet, rs_sheet = tmp / f"py_{mode}.labels.jsonl", tmp / f"rs_{mode}.labels.jsonl"
        subprocess.run([sys.executable, "-m", "plainline.cli", "--data-dir", str(ROOT / "data"), "export-labels",
                        "--run", str(py_out), "--out", str(py_sheet)], check=True, capture_output=True, cwd=ROOT)
        rs.cli_main(["plainline", "--data-dir", str(ROOT / "data"), "export-labels", "--run", str(rs_out), "--out", str(rs_sheet)])
        assert py_sheet.read_bytes() == rs_sheet.read_bytes(), "labels sheet"
    (GOLD / "mock_runs.json").write_text(json.dumps(gold, indent=1) + "\n")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "tests"))
    items = check_static()
    check_readability(items)
    check_verdicts()
    check_metrics()
    check_score_output(items)
    check_pipeline(items)
    check_mock_runs()
    print(f"parity OK: {checks['n']} comparisons, tolerance {TOL}")
