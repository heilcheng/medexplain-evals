"""Readability: Flesch-Kincaid grade and SMOG, with a pass/fail band.

Syllables are counted with a vowel-group heuristic, so values differ slightly from
dictionary-based tools. The band is a gate, not a quality score.
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENT = re.compile(r"[.!?]+(?:\s|$)|\n{2,}")


def syllables(word: str) -> int:
    w = word.lower().strip("'-")
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    if w.endswith("es") or w.endswith("ed"):
        if n > 1 and not re.search(r"(t|d)ed$|(s|x|z|ch|sh|ce|ge)s$", w):
            n -= 1
    return max(1, n)


def _counts(text: str) -> tuple[int, int, int, int]:
    words = _WORD.findall(text)
    sents = max(1, len([s for s in _SENT.split(text) if s.strip()]))
    syl = [syllables(w) for w in words]
    poly = sum(1 for s in syl if s >= 3)
    return len(words), sents, sum(syl), poly


def fk_grade(text: str) -> float:
    w, s, syl, _ = _counts(text)
    if w == 0:
        return 0.0
    return 0.39 * (w / s) + 11.8 * (syl / w) - 15.59


def smog_grade(text: str) -> float:
    w, s, _, poly = _counts(text)
    if w == 0:
        return 0.0
    return 1.043 * math.sqrt(poly * (30 / s)) + 3.1291


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def in_band(grade: float, band: tuple[float | None, float | None] | None) -> bool | None:
    """None when the condition has no band."""
    if band is None:
        return None
    lo, hi = band
    return (lo is None or grade >= lo) and (hi is None or grade <= hi)
