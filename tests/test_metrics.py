import numpy as np

from plainline.judge import JudgeResult
from plainline.metrics import bootstrap, headline_points, length_control, score_output, summarize_model
from plainline.readers import Condition


def _jr(**v):
    return JudgeResult(fact_verdicts=v, mns_verdicts={"n1": "not_asserted"})


def test_score_output(item):
    cond = Condition("grade6_adult", "reader", "x", (None, 8))
    r = score_output("Take it with food.", item.facts,
                     _jr(f1="absent", f2="stated", f3="stated", f4="contradicted"), cond)
    assert r["cfr"] == 0.5 and r["sfr"] == 1.0
    assert r["distortion"] == 0.25 and r["mns"] == 0.0
    assert r["must_missing"] == ["f1"] and r["band_ok"] is True


def test_score_output_mns_violation(item):
    jr = JudgeResult({"f1": "stated", "f2": "stated", "f3": "stated", "f4": "stated"}, {"n1": "asserted"})
    r = score_output("x", item.facts, jr, Condition("neutral", "neutral", "x"))
    assert r["mns"] == 1.0 and r["band_ok"] is None


def test_bootstrap_deterministic_and_covers():
    x = np.array([0.0, 1.0] * 20)
    a = bootstrap(lambda ix: x[ix].mean(), len(x), 500, seed=1)
    b = bootstrap(lambda ix: x[ix].mean(), len(x), 500, seed=1)
    assert a.to_dict() == b.to_dict()
    assert a.lo < 0.5 < a.hi


def _rows(n=20):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        base = rng.uniform(0.6, 1.0)
        for cond, drop, words in [("neutral", 0.0, 200), ("grade6_adult", 0.2, 120), ("clinician", 0.0, 210)]:
            rows.append({"item_id": f"i{i}", "condition": cond, "cfr": max(0, base - drop), "sfr": 0.5,
                         "distortion": 0.0, "mns": 0.0, "band_ok": None if cond == "neutral" else (cond != "grade6_adult" or i % 2 == 0),
                         "words": (120 if i < 10 else 190) + i if cond == "grade6_adult" else words + i, "fk_grade": 7.0, "parse_errors": 0})
    return rows


def test_adaptation_tax_is_paired_and_positive():
    s = summarize_model(_rows(), ["neutral", "clinician", "grade6_adult"], n_boot=300)
    tax = s["conditions"]["grade6_adult"]["adaptation_tax"]
    assert abs(tax["point"] - 0.2) < 1e-9
    assert tax["lo"] <= 0.2 <= tax["hi"]
    assert abs(s["conditions"]["clinician"]["adaptation_tax"]["point"]) < 1e-9
    assert "adaptation_tax" not in s["conditions"]["neutral"]


def test_worst_reader_and_band():
    s = summarize_model(_rows(), ["neutral", "clinician", "grade6_adult"], n_boot=300)
    cfr = {c: d["cfr"]["point"] for c, d in s["conditions"].items()}
    assert s["headline"]["worst_reader_cfr"]["point"] == min(cfr["clinician"], cfr["grade6_adult"])
    assert s["conditions"]["grade6_adult"]["band_pass"] == 0.5
    assert s["conditions"]["neutral"]["band_pass"] is None


def test_matched_length_excludes_far_outputs():
    s = summarize_model(_rows(), ["neutral", "grade6_adult"], n_boot=100)
    # The first ten items are far shorter than neutral (ratio about 0.6) and drop out.
    m = s["conditions"]["grade6_adult"]["matched_length_tax"]
    assert m is not None and m["n_items"] == 10


def test_headline_points_handles_missing_neutral():
    assert headline_points([], ["neutral", "x"]) == {}
    h = headline_points(_rows(), ["neutral", "clinician", "grade6_adult"])
    assert abs(h["mean_adaptation_tax"] - 0.1) < 1e-9


def test_length_control_detects_length_confound():
    rows = [{"cfr": w / 100, "words": w} for w in range(10, 70, 5)]
    lc = length_control(rows)
    assert lc["spearman_cfr_vs_words"] > 0.99
    assert lc["by_length_tercile"]["long"]["cfr"] > lc["by_length_tercile"]["short"]["cfr"]


def test_worst_reader_cfr_ignores_nan_condition():
    rows = _rows()
    for r in rows:
        if r["condition"] == "clinician":
            r["cfr"] = None
    s = summarize_model(rows, ["neutral", "clinician", "grade6_adult"], n_boot=100)
    w = s["headline"]["worst_reader_cfr"]["point"]
    assert w == w and abs(w - s["conditions"]["grade6_adult"]["cfr"]["point"]) < 1e-9
    h = headline_points(rows, ["neutral", "clinician", "grade6_adult"])
    assert h["worst_reader_cfr"] == h["worst_reader_cfr"]
