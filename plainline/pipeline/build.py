"""Build the pilot dataset: fetch sources, draft facts, write JSONL."""

from __future__ import annotations

from pathlib import Path

import httpx

from ..schema import FactSet, Source, write_jsonl
from . import dailymed, medlineplus
from .draft import rule_draft, validate_factset

PILOT_DRUGS = [
    "warfarin", "methotrexate", "hydrocodone bitartrate and acetaminophen", "metformin", "lisinopril",
    "amiodarone", "digoxin", "levothyroxine", "isotretinoin", "lithium carbonate", "ciprofloxacin",
    "ibuprofen", "apixaban", "clozapine", "sertraline", "tramadol", "prednisone", "gabapentin",
    "zolpidem", "colchicine",
]
PILOT_TOPICS = [
    "Anaphylaxis", "Appendicitis", "Heart Attack", "Hypothermia", "Meningitis",
    "Heat Illness", "Stroke", "Sepsis", "Seizures", "Concussion",
]
UA = {"User-Agent": "plainline-research-pipeline/0.1 (open benchmark; contact via GitHub repo)"}


def build_pilot(out_dir: str | Path = "data", drugs=None, topics=None) -> dict:
    drugs = drugs or PILOT_DRUGS
    topics = topics or PILOT_TOPICS
    sources: list[Source] = []
    skipped: list[str] = []
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as c:
        for d in drugs:
            try:
                s = dailymed.fetch_label(c, d)
            except httpx.HTTPError as e:
                s = None
                skipped.append(f"{d}: {e}")
            if s:
                sources.append(s)
            else:
                skipped.append(d)
        got, missing = medlineplus.fetch_topics(c, topics)
        sources += got
        skipped += missing
    facts: list[FactSet] = []
    kept: list[Source] = []
    for s in sources:
        fs = rule_draft(s)
        if validate_factset(fs):
            skipped.append(f"{s.source_id}: {validate_factset(fs)}")
            continue
        kept.append(s)
        facts.append(fs)
    out = Path(out_dir)
    write_jsonl(out / "sources" / "sources.jsonl", [s.model_dump() for s in kept])
    write_jsonl(out / "facts" / "facts.jsonl", [f.model_dump() for f in facts])
    return {"sources": len(kept), "facts": sum(len(f.facts) for f in facts), "skipped": skipped}
