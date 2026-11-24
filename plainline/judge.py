"""Judge-light fact checking: one entailment-style verdict per atomic fact.

The judge never rates quality. It answers a narrow question per fact. Verdicts can
be compared with human labels through `agreement` (see docs/judge_validation.md).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .client import ChatClient
from .prompts import JUDGE_VERSION, judge_messages
from .schema import FactSet

FACT_VERDICTS = {"stated", "contradicted", "absent"}
STMT_VERDICTS = {"asserted", "not_asserted"}


@dataclass
class JudgeResult:
    fact_verdicts: dict[str, str]  # fact_id -> stated|contradicted|absent
    mns_verdicts: dict[str, str]  # n1.. -> asserted|not_asserted
    parse_errors: int = 0
    judge_version: str = JUDGE_VERSION
    raw: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def parse_verdicts(raw: str, fact_ids: list[str], stmt_ids: list[str]) -> tuple[dict, dict, int]:
    """Return (fact verdicts, statement verdicts, number of ids that were missing or invalid).

    Missing or invalid ids fall back to the conservative reading: a fact counts as
    absent, a statement as not asserted. The error count is reported so a high rate is visible.
    """
    errors = 0
    got: dict[str, str] = {}
    try:
        for v in _extract_json(raw).get("verdicts", []):
            got[str(v.get("id"))] = str(v.get("verdict", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    facts: dict[str, str] = {}
    for i in fact_ids:
        v = got.get(i)
        if v not in FACT_VERDICTS:
            errors += 1
            v = "absent"
        facts[i] = v
    stmts: dict[str, str] = {}
    for i in stmt_ids:
        v = got.get(i)
        if v not in STMT_VERDICTS:
            errors += 1
            v = "not_asserted"
        stmts[i] = v
    return facts, stmts, errors


def judge_explanation(client: ChatClient, explanation: str, fs: FactSet,
                      per_fact: bool = False) -> JudgeResult:
    """Check every fact and must-not-say statement against one explanation.

    per_fact=False sends one call listing all items (cheap). per_fact=True sends one
    call per item, which avoids cross-item interference at higher cost.
    """
    facts = [(f.fact_id, f.text) for f in fs.facts]
    stmts = [(f"n{i + 1}", t) for i, t in enumerate(fs.must_not_say)]
    res = JudgeResult({}, {})
    groups = ([([f], []) for f in facts] + [([], [s]) for s in stmts]) if per_fact else [(facts, stmts)]
    for fgrp, sgrp in groups:
        raw = client.complete(judge_messages(explanation, fgrp, sgrp), temperature=0.0, max_tokens=800)
        fv, sv, err = parse_verdicts(raw, [i for i, _ in fgrp], [i for i, _ in sgrp])
        res.fact_verdicts.update(fv)
        res.mns_verdicts.update(sv)
        res.parse_errors += err
        res.raw.append(raw)
    return res


def agreement(judged: list[dict], human: list[dict]) -> dict:
    """Compare judge verdicts with human labels.

    Both inputs are lists of {"key": str, "stated": bool}. A key is any stable id such
    as "<output_id>|<fact_id>". Returns raw agreement and Cohen's kappa on the overlap.
    """
    h = {r["key"]: bool(r["stated"]) for r in human}
    pairs = [(bool(r["stated"]), h[r["key"]]) for r in judged if r["key"] in h]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    a = sum(x == y for x, y in pairs) / n
    p_j = sum(x for x, _ in pairs) / n
    p_h = sum(y for _, y in pairs) / n
    pe = p_j * p_h + (1 - p_j) * (1 - p_h)
    kappa = 1.0 if pe == 1 else (a - pe) / (1 - pe)
    return {"n": n, "agreement": a, "kappa": kappa, "judge_stated_rate": p_j, "human_stated_rate": p_h}
