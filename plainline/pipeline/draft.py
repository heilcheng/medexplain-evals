"""Draft atomic facts. Every output is marked draft and needs clinician review.

Two generators:
  llm_draft   a model call with EXTRACT_SYSTEM (the intended generator).
  rule_draft  a deterministic section-and-keyword heuristic, used for the pilot when no
              model budget was spent. Its facts are whole source sentences, not truly
              atomic, and its must_not_say list is a short generic template.
Neither is evidence of clinical validity.
"""

from __future__ import annotations

import json
import re

from ..client import ChatClient
from ..judge import _extract_json
from ..prompts import extract_messages
from ..schema import DRAFT_STATUS, Fact, FactSet, Source
from .text import sentences

RULE_VERSION = "rule-v0"

_HARD = re.compile(r"\b(not exceed|maximum|do not|must not|should not|never|contraindicated|avoid|"
                   r"discontinue|stop|fatal|death|life-threatening|seek|call|emergency|911|immediately)\b", re.I)
_SERIOUS = re.compile(r"\b(risk|serious|monitor|bleeding|toxicity|overdose|suicid|pregnan|reaction|"
                      r"symptoms?|warning)\b", re.I)


def _split_sections(text: str) -> dict[str, str]:
    """Split the source text on the upper-case headings written by the fetchers."""
    parts = re.split(r"\n\n([A-Z][A-Z ,/&-]+)\n", "\n\n" + text)
    secs: dict[str, str] = {"": parts[0]}
    for i in range(1, len(parts) - 1, 2):
        secs[parts[i].strip().lower()] = parts[i + 1]
    return secs


_HEAD = re.compile(r"^(WARNING:\s+[A-Z0-9 ,/&;-]+\s*[\u2022-]?\s*|\d+(\.\d+)*\s+[A-Z][A-Z &,/-]{3,}\s+|\d+(\.\d+)*\s+(?=[A-Z][a-z]+ ))(?=[A-Z][a-z])")


def _norm(sent: str) -> str:
    """Drop leftover section headings glued to the start of a sentence."""
    return _HEAD.sub("", sent.strip()).strip()


def _usable(s: str) -> bool:
    n = len(s.split())
    return 5 <= n <= 55 and not s.rstrip().endswith('?')


def rule_draft(src: Source) -> FactSet:
    secs = _split_sections(src.text)
    facts: list[Fact] = []
    seen: set[str] = set()

    def add(sent: str, tier: str, section: str) -> None:
        sent = _norm(sent)
        key = re.sub(r"\W+", "", sent.lower())[:80]
        if key in seen or not _usable(sent):
            return
        seen.add(key)
        facts.append(Fact(fact_id=f"f{len(facts) + 1}", text=sent, tier=tier, section=section))

    counts = {"must_keep": 0, "should_keep": 0, "optional": 0}
    caps = {"must_keep": 8, "should_keep": 7, "optional": 3}

    def add_cap(sent: str, tier: str, section: str) -> None:
        if counts[tier] < caps[tier]:
            before = len(facts)
            add(sent, tier, section)
            counts[tier] += len(facts) - before

    if src.kind == "dailymed_label":
        for sent in sentences(secs.get("boxed warning", "")):
            add_cap(sent, "must_keep", "boxed warning")
        for sent in sentences(secs.get("contraindications", "")):
            add_cap(sent, "must_keep", "contraindications")
        for sent in sentences(secs.get("dosage and administration", "")):
            if _HARD.search(sent):
                add_cap(sent, "must_keep", "dosage and administration")
            elif re.search(r"\b\d+\s?(mg|mcg|g|mL|units?)\b", sent):
                add_cap(sent, "should_keep", "dosage and administration")
        for key in ("warnings and precautions", "warnings"):
            for sent in sentences(secs.get(key, "")):
                if _HARD.search(sent):
                    add_cap(sent, "should_keep", key)
                elif _SERIOUS.search(sent):
                    add_cap(sent, "optional", key)
        for sent in sentences(secs.get("indications and usage", ""))[:2]:
            add_cap(sent, "should_keep", "indications and usage")
    else:  # medlineplus_topic and other plain-prose sources
        body = src.text.split("\n\n", 1)[-1]
        for sent in sentences(body):
            if re.search(r"\b(call|seek|emergency|911|immediately|right away|go to)\b", sent, re.I):
                add_cap(sent, "must_keep", "summary")
            elif re.search(r"\b(symptoms?|cause[sd]?|treat|treatment|risk|prevent|diagnos)", sent, re.I):
                add_cap(sent, "should_keep", "summary")
            else:
                add_cap(sent, "optional", "summary")
        # A short topic may have no urgent-care sentence; promote nothing in that case.

    mns: list[str] = []
    if src.kind == "dailymed_label":
        if "boxed warning" in secs:
            mns.append("This medicine has no serious or life-threatening risks.")
        if "dosage and administration" in secs:
            mns.append("If you miss a dose, take a double dose next time to catch up.")
            mns.append("It is fine to take more than the labelled maximum dose if symptoms do not improve.")
    return FactSet(source_id=src.source_id, facts=facts, must_not_say=mns,
                   status=DRAFT_STATUS, generator=RULE_VERSION)


def llm_draft(client: ChatClient, src: Source) -> FactSet:
    raw = client.complete(extract_messages(src.text), temperature=0.0, max_tokens=2000)
    data = _extract_json(raw)
    facts = []
    for f in data.get("facts", []):
        tier = f.get("tier")
        if tier in ("must_keep", "should_keep", "optional") and f.get("text"):
            facts.append(Fact(fact_id=f"f{len(facts) + 1}", text=f["text"].strip(), tier=tier,
                              section=str(f.get("section", ""))))
    return FactSet(source_id=src.source_id, facts=facts,
                   must_not_say=[str(x) for x in data.get("must_not_say", [])][:4],
                   status=DRAFT_STATUS, generator=f"llm:{client.name}")


def validate_factset(fs: FactSet, *, require_must: bool = True) -> list[str]:
    """Return a list of problems; empty means structurally fine."""
    problems = []
    ids = [f.fact_id for f in fs.facts]
    if len(set(ids)) != len(ids):
        problems.append("duplicate fact ids")
    if require_must and not fs.tier("must_keep"):
        problems.append("no must_keep facts")
    if fs.status != DRAFT_STATUS and not fs.is_validated:
        problems.append("status claims review but fewer than 2 reviewers are recorded")
    return problems
