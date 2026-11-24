"""Turn scored.jsonl into summary.json and a markdown table."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import summarize_model

DRAFT_BANNER = ("Fact lists are machine-drafted and have NOT been reviewed by clinicians. "
                "Numbers below are pilot values for pipeline checking, not validated results.")


def _fmt(d: dict | None, pct: bool = True) -> str:
    if not d:
        return "n/a"
    k = 100 if pct else 1
    if d["lo"] != d["lo"]:
        return f"{d['point'] * k:.1f}"
    return f"{d['point'] * k:.1f} [{d['lo'] * k:.1f}, {d['hi'] * k:.1f}]"


def build_summary(scored_path: str | Path, cond_order: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    rows = [json.loads(l) for l in Path(scored_path).read_text().splitlines() if l.strip()]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    flags = {
        "mock": any(r.get("mock") for r in rows),
        "all_facts_validated": bool(rows) and all(r.get("facts_status") == "clinician_reviewed" for r in rows),
        "judge_versions": sorted({r.get("judge_version", "") for r in rows}),
        "judges": sorted({r.get("judge", "") for r in rows}),
    }
    return {"flags": flags,
            "models": {m: summarize_model(rs, cond_order, n_boot, seed) for m, rs in by_model.items()}}


def to_markdown(summary: dict) -> str:
    f = summary["flags"]
    lines = ["# Plainline results", ""]
    if not f["all_facts_validated"]:
        lines += [f"> {DRAFT_BANNER}", ""]
    if f["mock"]:
        lines += ["> These outputs came from the offline mock client. They measure nothing about any model.", ""]
    lines += [f"Judge: {', '.join(f['judges'])} ({', '.join(f['judge_versions'])}). Intervals are 95% bootstrap CIs over items.", ""]
    for model, s in summary["models"].items():
        h = s["headline"]
        lines += [f"## {model}", "", f"Items: {s['n_items']}. Judge parse-error rate: {s['parse_error_rate']:.1%}.",
                  f"Worst-reader CFR: {_fmt(h.get('worst_reader_cfr'))}. "
                  f"Mean adaptation tax (points): {_fmt(h.get('mean_adaptation_tax'))}. "
                  f"CFR spread across conditions (points): {_fmt(h.get('cfr_spread'))}.", "",
                  "| Condition | CFR % | Tax (pts) | Tax at matched length (pts) | SFR % | Distortion % | Must-not-say % | In band % | Words | FK grade |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c, d in s["conditions"].items():
            band = "n/a" if d["band_pass"] is None else f"{d['band_pass'] * 100:.0f}"
            ml = d.get("matched_length_tax")
            lines.append(
                f"| {c} | {_fmt(d['cfr'])} | {_fmt(d.get('adaptation_tax'))} | {_fmt(ml)} | {_fmt(d['sfr'])} | "
                f"{_fmt(d['distortion'])} | {_fmt(d['mns'])} | {band} | {d['mean_words']:.0f} | {d['mean_fk_grade']:.1f} |")
        lc = s.get("length_control", {})
        if lc:
            lines += ["", "CFR by output-length tercile: " + "; ".join(
                f"{k} ({v['mean_words']:.0f} words, n={v['n']}): {v['cfr'] * 100:.1f}%"
                for k, v in lc["by_length_tercile"].items())]
            rho = lc.get("spearman_cfr_vs_words")
            if rho is not None:
                lines.append(f"Spearman correlation of CFR with output length: {rho:.2f}.")
        lines.append("")
    return "\n".join(lines)


def write_report(scored_path: str | Path, cond_order: list[str], out_dir: str | Path,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    summary = build_summary(scored_path, cond_order, n_boot, seed)
    out = Path(out_dir)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "results.md").write_text(to_markdown(summary))
    return summary
