"""Small text helpers shared by the pipeline."""

from __future__ import annotations

import html
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])")


def clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(clean(text)) if s.strip()]


def truncate_words(text: str, max_words: int) -> str:
    """Cut at a sentence boundary at or under max_words (at least one sentence)."""
    out: list[str] = []
    n = 0
    for s in sentences(text):
        w = len(s.split())
        if out and n + w > max_words:
            break
        out.append(s)
        n += w
    return " ".join(out)
