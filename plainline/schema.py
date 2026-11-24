"""Data schema. Every machine-drafted fact set carries status 'draft'."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DRAFT_STATUS = "draft, needs clinician review"
CANARY = "PLAINLINE-CANARY-7c1e5a52-do-not-train-on-this-file"

Tier = Literal["must_keep", "should_keep", "optional"]
Status = Literal["draft, needs clinician review", "clinician_reviewed"]
Split = Literal["dev", "test"]


class Source(BaseModel):
    """One source document chunk given to the model under test."""

    source_id: str
    kind: Literal["dailymed_label", "medlineplus_topic", "cdc_page", "vignette"]
    title: str
    text: str
    origin_url: str
    license_note: str
    fetched: str  # ISO date
    version: str = ""
    split: Split = "dev"
    canary: str = CANARY


class Fact(BaseModel):
    fact_id: str
    text: str
    tier: Tier
    # Section of the source the fact was drawn from, when known.
    section: str = ""


class FactSet(BaseModel):
    source_id: str
    facts: list[Fact]
    # Statements a wrong simplification might assert. Judged for violations.
    must_not_say: list[str] = Field(default_factory=list)
    status: Status = DRAFT_STATUS
    generator: str  # how the draft was produced, e.g. "rule-v0" or a model id
    reviewers: list[str] = Field(default_factory=list)

    def tier(self, tier: Tier) -> list[Fact]:
        return [f for f in self.facts if f.tier == tier]

    @property
    def is_validated(self) -> bool:
        return self.status == "clinician_reviewed" and len(self.reviewers) >= 2


class Item(BaseModel):
    """A source with its fact set, ready to evaluate."""

    source: Source
    facts: FactSet


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_items(data_dir: str | Path = "data", split: str | None = None) -> list[Item]:
    """Load sources and fact sets from `data_dir`; both files are JSONL."""
    d = Path(data_dir)
    sources = {r["source_id"]: Source(**r) for r in read_jsonl(d / "sources" / "sources.jsonl")}
    facts = {r["source_id"]: FactSet(**r) for r in read_jsonl(d / "facts" / "facts.jsonl")}
    items = [Item(source=s, facts=facts[sid]) for sid, s in sources.items() if sid in facts]
    if split:
        items = [i for i in items if i.source.split == split]
    return items
