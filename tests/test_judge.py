from conftest import ScriptedClient, verdict_json
from plainline.judge import agreement, judge_explanation, parse_verdicts


def test_parse_plain_and_fenced():
    raw = "```json\n" + verdict_json(f1="stated", n1="asserted") + "\n```"
    f, s, err = parse_verdicts(raw, ["f1"], ["n1"])
    assert f == {"f1": "stated"} and s == {"n1": "asserted"} and err == 0


def test_missing_and_invalid_fall_back_conservatively():
    f, s, err = parse_verdicts(verdict_json(f1="maybe"), ["f1", "f2"], ["n1"])
    assert f == {"f1": "absent", "f2": "absent"}
    assert s == {"n1": "not_asserted"}
    assert err == 3


def test_garbage_output():
    f, s, err = parse_verdicts("I cannot do that", ["f1"], [])
    assert f == {"f1": "absent"} and err == 1


def test_batched_is_one_call(item):
    c = ScriptedClient([verdict_json(f1="stated", f2="absent", f3="stated", f4="contradicted", n1="not_asserted")])
    r = judge_explanation(c, "some text", item.facts)
    assert len(c.calls) == 1
    assert r.fact_verdicts["f4"] == "contradicted"
    assert r.parse_errors == 0


def test_per_fact_calls_each_item(item):
    c = ScriptedClient([verdict_json(f1="stated")])
    r = judge_explanation(c, "text", item.facts, per_fact=True)
    assert len(c.calls) == len(item.facts.facts) + 1  # 4 facts and 1 statement
    assert r.parse_errors > 0  # the scripted reply never contains the other ids


def test_judge_prompt_hides_tiers_and_source(item):
    c = ScriptedClient([verdict_json(f1="stated")])
    judge_explanation(c, "EXPL", item.facts)
    user = c.calls[0][1]["content"]
    assert "must_keep" not in user and "should_keep" not in user
    assert "EXPL" in user and "Can cause liver damage" in user


def test_agreement_kappa():
    j = [{"key": str(i), "stated": s} for i, s in enumerate([1, 1, 0, 0, 1, 0])]
    h = [{"key": str(i), "stated": s} for i, s in enumerate([1, 1, 0, 0, 1, 0])]
    assert agreement(j, h)["kappa"] == 1.0
    h2 = [{"key": str(i), "stated": s} for i, s in enumerate([0, 0, 0, 0, 1, 0])]
    r = agreement(j, h2)
    assert 0 < r["agreement"] < 1 and r["kappa"] < 1
    assert agreement(j, [])["n"] == 0
