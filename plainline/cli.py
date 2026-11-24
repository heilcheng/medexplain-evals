"""Command line interface. Works with any OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import ChatClient, MockClient, OpenAICompatClient
from .judge import agreement
from .readers import conditions
from .report import write_report
from .runner import run
from .schema import DRAFT_STATUS, FactSet, load_items, read_jsonl, write_jsonl


def _client(args, prefix: str = "") -> ChatClient:
    if args.mock:
        return MockClient()
    base = getattr(args, f"{prefix}base_url") or (args.base_url if prefix else None)
    model = getattr(args, f"{prefix}model") or (args.model if prefix else None)
    key = getattr(args, f"{prefix}api_key_env") or (args.api_key_env if prefix else "OPENAI_API_KEY")
    if not base or not model:
        sys.exit("error: --base-url and --model are required (or pass --mock)")
    return OpenAICompatClient(base, model, key)


def cmd_build_pilot(args) -> None:
    from .pipeline.build import build_pilot

    print(json.dumps(build_pilot(args.data_dir), indent=2))


def cmd_draft(args) -> None:
    from .pipeline.draft import llm_draft, validate_factset
    from .schema import Source

    client = _client(args)
    sources = [Source(**r) for r in read_jsonl(Path(args.data_dir) / "sources" / "sources.jsonl")]
    out = []
    for s in sources:
        fs = llm_draft(client, s)
        probs = validate_factset(fs)
        if probs:
            print(f"skip {s.source_id}: {probs}", file=sys.stderr)
            continue
        out.append(fs.model_dump())
    dest = Path(args.out)
    write_jsonl(dest, out)
    print(f"wrote {len(out)} draft fact sets to {dest} (status: {DRAFT_STATUS})")


def cmd_validate(args) -> None:
    from .pipeline.draft import validate_factset

    items = load_items(args.data_dir)
    bad = 0
    for it in items:
        probs = validate_factset(it.facts)
        if probs:
            bad += 1
            print(f"{it.source.source_id}: {probs}")
    n_facts = sum(len(i.facts.facts) for i in items)
    n_valid = sum(i.facts.is_validated for i in items)
    print(f"{len(items)} items, {n_facts} facts, {n_valid} clinician-validated fact sets, {bad} with problems")
    sys.exit(1 if bad else 0)


def cmd_run(args) -> None:
    items = load_items(args.data_dir, args.split)
    if args.limit:
        items = items[: args.limit]
    conds = conditions(args.mode)
    writer = _client(args)
    judge = _client(args, "judge_") if not args.mock else MockClient()
    if not args.mock and judge.name == writer.name:
        print("warning: judge and writer are the same model; self-preference is not controlled", file=sys.stderr)
    out = Path(args.out)
    path = run(items, conds, writer, judge, out, n_samples=args.samples, per_fact=args.per_fact,
               workers=args.workers, mock=args.mock)
    write_report(path, [c.id for c in conds], out, n_boot=args.boot)
    print((out / "results.md").read_text())


def cmd_report(args) -> None:
    conds = conditions(args.mode)
    write_report(Path(args.run) / "scored.jsonl", [c.id for c in conds], args.run, n_boot=args.boot)
    print((Path(args.run) / "results.md").read_text())


def cmd_export_labels(args) -> None:
    """Write a sheet for human labelling: one row per (output, fact)."""
    items = {i.source.source_id: i for i in load_items(args.data_dir)}
    rows = []
    for r in read_jsonl(Path(args.run) / "scored.jsonl"):
        fs: FactSet = items[r["item_id"]].facts
        for f in fs.facts:
            rows.append({"key": f"{r['output_id']}|{f.fact_id}", "fact": f.text, "tier": f.tier,
                         "explanation": r["text"], "judge_stated": r["fact_verdicts"][f.fact_id] == "stated",
                         "human_stated": None})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}; fill human_stated with true or false")


def cmd_agreement(args) -> None:
    sheet = read_jsonl(args.labels)
    judged = [{"key": r["key"], "stated": r["judge_stated"]} for r in sheet if r.get("human_stated") is not None]
    human = [{"key": r["key"], "stated": r["human_stated"]} for r in sheet if r.get("human_stated") is not None]
    print(json.dumps(agreement(judged, human), indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="plainline", description=__doc__)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-pilot", help="fetch open sources and write the pilot set with draft facts").set_defaults(fn=cmd_build_pilot)
    sub.add_parser("validate-data", help="check the data files").set_defaults(fn=cmd_validate)

    def endpoint(sp, judge: bool = True):
        sp.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
        sp.add_argument("--model")
        sp.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key")
        sp.add_argument("--mock", action="store_true", help="offline mock clients (measures nothing)")
        if judge:
            sp.add_argument("--judge-base-url")
            sp.add_argument("--judge-model")
            sp.add_argument("--judge-api-key-env")

    d = sub.add_parser("draft", help="draft facts with a model (output is a draft, needs clinician review)")
    endpoint(d, judge=False)
    d.add_argument("--out", default="data/facts/facts.llm_draft.jsonl")
    d.set_defaults(fn=cmd_draft)

    r = sub.add_parser("run", help="generate explanations, judge them and write a report")
    endpoint(r)
    r.add_argument("--mode", choices=["adapt", "cue"], default="adapt")
    r.add_argument("--out", required=True)
    r.add_argument("--split", default="dev")
    r.add_argument("--limit", type=int)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--per-fact", action="store_true", help="one judge call per fact")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--boot", type=int, default=2000)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="recompute the report from a run directory")
    rp.add_argument("--run", required=True)
    rp.add_argument("--mode", choices=["adapt", "cue"], default="adapt")
    rp.add_argument("--boot", type=int, default=2000)
    rp.set_defaults(fn=cmd_report)

    e = sub.add_parser("export-labels", help="write a human labelling sheet from a run")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export_labels)

    a = sub.add_parser("agreement", help="judge-versus-human agreement from a filled labelling sheet")
    a.add_argument("--labels", required=True)
    a.set_defaults(fn=cmd_agreement)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
