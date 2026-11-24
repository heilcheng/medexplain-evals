"""Reader specs and cue conditions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Condition:
    """One way of asking: neutral, an explicit reader, or an implicit cue."""

    id: str
    kind: str  # neutral | reader | cue
    instruction: str
    band: tuple[float | None, float | None] | None = None
    group: str = ""


def _load(path: str | Path | None = None) -> dict:
    if path:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return yaml.safe_load(resources.files("plainline.data").joinpath("readers.yaml").read_text(encoding="utf-8"))


def _band(b) -> tuple[float | None, float | None] | None:
    return None if b is None else (b[0], b[1])


def conditions(mode: str = "adapt", path: str | Path | None = None) -> list[Condition]:
    """Neutral first, then the readers (mode adapt) or the cues (mode cue)."""
    cfg = _load(path)
    n = cfg["neutral"]
    out = [Condition("neutral", "neutral", n["instruction"], _band(n.get("band")))]
    if mode == "adapt":
        for r in cfg["readers"]:
            out.append(Condition(r["id"], "reader", r["instruction"], _band(r.get("band"))))
    elif mode == "cue":
        for c in cfg["cues"]:
            out.append(Condition(c["id"], "cue", f"{c['text']} {cfg['cue_task']}", None, c.get("group", "")))
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'adapt' or 'cue'")
    return out
