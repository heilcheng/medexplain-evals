import json

import pytest

from conftest import ScriptedClient
from plainline.pipeline import dailymed, medlineplus
from plainline.pipeline.draft import llm_draft, rule_draft, validate_factset
from plainline.pipeline.text import sentences, truncate_words
from plainline.schema import DRAFT_STATUS, Fact, FactSet, Source

SPL = b"""<?xml version="1.0"?>
<document xmlns="urn:hl7-org:v3"><component><structuredBody>
<component><section><code code="34066-1"/><title>Boxed</title><text>WARNING: BLEEDING RISK Testdrug can cause major or fatal bleeding [see Warnings and Precautions (5.1)]. Monitor INR regularly in all patients.</text></section></component>
<component><section><code code="34070-3"/><text>Do not use in pregnancy. Contraindicated with severe liver disease.</text></section></component>
<component><section><code code="34068-7"/><text>Do not exceed 10 mg daily. Start at 2 mg once daily.</text></section></component>
<component><section><code code="99999-9"/><text>ignored</text></section></component>
</structuredBody></component></document>"""

TOPICS = b"""<?xml version="1.0"?><health-topics>
<health-topic title="Sepsis" language="English" url="https://medlineplus.gov/sepsis.html">
<full-summary>&lt;p&gt;Sepsis is a medical emergency. Call 911 if you have a fast heartbeat and confusion.&lt;/p&gt;</full-summary></health-topic>
<health-topic title="Sepsis" language="Spanish" url="x"><full-summary>hola</full-summary></health-topic>
</health-topics>"""


def test_parse_spl_selects_sections_and_strips_refs():
    s = dailymed.parse_spl(SPL)
    assert set(s) == {"34066-1", "34070-3", "34068-7"}
    assert "[see" not in s["34066-1"] and "(5.1)" not in s["34066-1"]
    assert s["34066-1"].startswith("Testdrug can cause")


def test_build_source_marks_provenance():
    src = dailymed.build_source("testdrug", {"title": "TESTDRUG TABLET", "setid": "abc", "spl_version": 2},
                                dailymed.parse_spl(SPL))
    assert "setid=abc" in src.version and "BOXED WARNING" in src.text
    assert src.origin_url.endswith("setid=abc")


def test_parse_topics_english_only():
    t = medlineplus.parse_topics(TOPICS)
    assert list(t) == ["sepsis"] and "<p>" not in t["sepsis"]["summary"]


def test_sentences_and_truncate():
    assert len(sentences("One two. Three four five. Six.")) == 3
    assert truncate_words("One two. Three four five. Six seven.", 4) == "One two."
    assert truncate_words("A very long single sentence here.", 2).startswith("A very")


def _src():
    s = dailymed.parse_spl(SPL)
    return dailymed.build_source("testdrug", {"title": "TESTDRUG TABLET", "setid": "abc", "spl_version": 2}, s)


def test_rule_draft_is_marked_draft_and_tiers():
    fs = rule_draft(_src())
    assert fs.status == DRAFT_STATUS and fs.generator == "rule-v0" and not fs.is_validated
    musts = " ".join(f.text for f in fs.tier("must_keep"))
    assert "fatal bleeding" in musts and "Do not use in pregnancy" in musts and "Do not exceed 10 mg" in musts
    assert fs.must_not_say and validate_factset(fs) == []


def test_rule_draft_topic():
    src = medlineplus.build_source(medlineplus.parse_topics(TOPICS)["sepsis"], "x.xml")
    fs = rule_draft(src)
    assert any("Call 911" in f.text for f in fs.tier("must_keep"))


def test_llm_draft_parses_and_stays_draft():
    reply = json.dumps({"facts": [{"text": "Do not exceed 10 mg", "tier": "must_keep", "section": "dose"},
                                  {"text": "bogus", "tier": "critical"},
                                  {"text": "Used for X", "tier": "optional"}],
                        "must_not_say": ["Take extra"]})
    fs = llm_draft(ScriptedClient([reply], name="m1"), _src())
    assert [f.tier for f in fs.facts] == ["must_keep", "optional"]
    assert fs.status == DRAFT_STATUS and fs.generator == "llm:m1"


def test_validate_rejects_fake_review():
    fs = FactSet(source_id="a", generator="g", status="clinician_reviewed",
                 facts=[Fact(fact_id="f1", text="t", tier="must_keep")], reviewers=["only one"])
    assert "fewer than 2 reviewers" in " ".join(validate_factset(fs))
    fs.reviewers.append("second")
    assert validate_factset(fs) == [] and fs.is_validated
    assert "no must_keep" in " ".join(validate_factset(FactSet(source_id="a", generator="g", facts=[])))


def test_source_requires_fields():
    with pytest.raises(Exception):
        Source(source_id="x")


def test_dedupe_sentences_and_aphena_filtered():
    from plainline.pipeline.dailymed import REPACKAGERS
    from plainline.pipeline.text import dedupe_sentences
    assert dedupe_sentences("Bleeding can be fatal. Check INR. Bleeding can be fatal.") == "Bleeding can be fatal. Check INR."
    assert "aphena" in REPACKAGERS
