"""Generate, judge and score. Outputs are JSONL and the run is resumable."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import ChatClient
from .judge import judge_explanation
from .metrics import score_output
from .prompts import JUDGE_VERSION, writer_messages
from .readers import Condition
from .schema import Item


def output_id(model: str, item_id: str, cond: str, sample: int) -> str:
    return hashlib.sha1(f"{model}|{item_id}|{cond}|{sample}".encode()).hexdigest()[:12]


def _done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["output_id"]: r for r in map(json.loads, filter(None, path.read_text().splitlines()))}


def run(items: list[Item], conds: list[Condition], writer: ChatClient, judge: ChatClient,
        out_dir: str | Path, *, n_samples: int = 1, per_fact: bool = False, workers: int = 4,
        max_tokens: int = 1024, temperature: float = 0.0, mock: bool = False) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored_path = out / "scored.jsonl"
    done = _done(scored_path)
    jobs = [(it, c, s) for it in items for c in conds for s in range(n_samples)
            if output_id(writer.name, it.source.source_id, c.id, s) not in done]

    def work(job):
        it, c, s = job
        text = writer.complete(writer_messages(it.source.text, c.instruction),
                               temperature=temperature, max_tokens=max_tokens)
        jr = judge_explanation(judge, text, it.facts, per_fact=per_fact)
        row = score_output(text, it.facts, jr, c)
        row.update({
            "output_id": output_id(writer.name, it.source.source_id, c.id, s),
            "item_id": it.source.source_id, "model": writer.name, "sample": s,
            "judge": judge.name, "judge_version": JUDGE_VERSION, "mock": mock,
            "facts_status": it.facts.status, "text": text,
            "fact_verdicts": jr.fact_verdicts, "mns_verdicts": jr.mns_verdicts,
        })
        return row

    with open(scored_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        for row in ex.map(work, jobs):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    return scored_path
