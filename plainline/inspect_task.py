"""Inspect AI task. Run with, for example:

    inspect eval plainline/inspect_task.py --model openai/gpt-4o-mini \
        -T judge_model=openai/gpt-4o -T mode=adapt

The judge should be a different model from the one under test. If judge_model is
omitted the evaluated model judges itself, which is not controlled for self-preference.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, metric, scorer
from inspect_ai.solver import TaskState, generate, system_message

from .judge import parse_verdicts
from .metrics import headline_points
from .prompts import JUDGE_VERSION, WRITER_SYSTEM, judge_messages, writer_messages
from .readability import fk_grade, in_band, word_count
from .readers import conditions
from .schema import FactSet, load_items


def build_samples(mode: str, split: str, data_dir: str | None, limit_items: int | None) -> list[Sample]:
    items = load_items(data_dir or _default_data_dir(), split)
    if limit_items:
        items = items[:limit_items]
    samples = []
    for it in items:
        for c in conditions(mode):
            user = writer_messages(it.source.text, c.instruction)[1]["content"]
            samples.append(Sample(
                id=f"{it.source.source_id}|{c.id}", input=user,
                metadata={"item_id": it.source.source_id, "condition": c.id, "kind": c.kind,
                          "band": list(c.band) if c.band else None,
                          "facts": it.facts.model_dump(), "mode": mode},
            ))
    return samples


def _default_data_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / "data")


def _unwrap(s):
    """Metrics receive SampleScore or Score depending on the value layout."""
    return getattr(s, "score", s)


def _rows(scores) -> list[dict]:
    rows = []
    for s in scores:
        sc = _unwrap(s)
        v = sc.value
        if not isinstance(v, dict):
            continue
        md = sc.metadata or {}
        rows.append({"item_id": md.get("item_id"), "condition": md.get("condition"),
                     "cfr": v.get("cfr"), "words": v.get("words", 0)})
    return rows


def _order(scores) -> list[str]:
    seen = []
    for s in scores:
        c = (_unwrap(s).metadata or {}).get("condition")
        if c and c not in seen:
            seen.append(c)
    return ["neutral"] + [c for c in seen if c != "neutral"]


@metric
def worst_reader_cfr():
    def m(scores):
        rows = [r for r in _rows(scores) if r["cfr"] is not None]
        return headline_points(rows, _order(scores)).get("worst_reader_cfr", float("nan"))
    return m


@metric
def adaptation_tax():
    def m(scores):
        rows = [r for r in _rows(scores) if r["cfr"] is not None]
        return headline_points(rows, _order(scores)).get("mean_adaptation_tax", float("nan"))
    return m


def _mean_metric(key: str):
    @metric(name=f"{key}_mean")
    def factory():
        def m(scores):
            vals = [_unwrap(s).value[key] for s in scores
                    if isinstance(_unwrap(s).value, dict) and _unwrap(s).value.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        return m
    return factory


cfr_mean = _mean_metric("cfr")
sfr_mean = _mean_metric("sfr")
distortion_mean = _mean_metric("distortion")
mns_mean = _mean_metric("mns")
band_pass_mean = _mean_metric("band_ok")


@scorer(metrics=[worst_reader_cfr(), adaptation_tax(), cfr_mean(), sfr_mean(),
                 distortion_mean(), mns_mean(), band_pass_mean()])
def fact_retention(judge_model: str | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        fs = FactSet(**state.metadata["facts"])
        text = state.output.completion
        judge = get_model(judge_model) if judge_model else get_model()
        facts = [(f.fact_id, f.text) for f in fs.facts]
        stmts = [(f"n{i + 1}", t) for i, t in enumerate(fs.must_not_say)]
        msgs = judge_messages(text, facts, stmts)
        out = await judge.generate([ChatMessageSystem(content=msgs[0]["content"]),
                                    ChatMessageUser(content=msgs[1]["content"])])
        fv, sv, err = parse_verdicts(out.completion, [i for i, _ in facts], [i for i, _ in stmts])
        must = [f.fact_id for f in fs.tier("must_keep")]
        should = [f.fact_id for f in fs.tier("should_keep")]

        def frac(ids):
            return sum(fv[i] == "stated" for i in ids) / len(ids) if ids else None

        band = state.metadata.get("band")
        ok = in_band(fk_grade(text), tuple(band) if band else None)
        value = {"cfr": frac(must), "sfr": frac(should),
                 "distortion": sum(v == "contradicted" for v in fv.values()) / max(1, len(fv)),
                 "mns": (sum(v == "asserted" for v in sv.values()) / len(sv)) if sv else None,
                 "band_ok": None if ok is None else float(ok), "words": word_count(text)}
        value = {k: v for k, v in value.items() if v is not None}
        return Score(value=value, answer=text[:200],
                     metadata={"item_id": state.metadata["item_id"], "condition": state.metadata["condition"],
                               "fact_verdicts": fv, "mns_verdicts": sv, "parse_errors": err,
                               "judge_version": JUDGE_VERSION, "facts_status": fs.status})
    return score


@task
def plainline(mode: str = "adapt", split: str = "dev", data_dir: str | None = None,
              judge_model: str | None = None, limit_items: int | None = None):
    """Fidelity under audience adaptation. mode is 'adapt' (explicit reader) or 'cue'."""
    return Task(
        dataset=MemoryDataset(build_samples(mode, split, data_dir, limit_items), name=f"plainline-{mode}"),
        solver=[system_message(WRITER_SYSTEM), generate()],
        scorer=fact_retention(judge_model),
        metadata={"facts_status": "draft, needs clinician review"},
    )
