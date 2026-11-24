"""Metrics. No weighted composite: the output is a vector plus the worst-reader CFR.

Definitions (per model, per condition = reader or cue):
  CFR   fraction of must_keep facts judged "stated", averaged over items.
  SFR   same for should_keep facts.
  tax   Adaptation Tax, paired per item: CFR(neutral) - CFR(condition).
  distortion   fraction of facts (all tiers) judged "contradicted".
  mns   fraction of must_not_say statements judged "asserted".
  band  fraction of outputs inside the condition's reading-grade band.
All intervals are percentile bootstrap CIs over items (items are the sampling unit,
so the pairing across conditions is preserved).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .judge import JudgeResult
from .readability import fk_grade, in_band, smog_grade, word_count
from .readers import Condition
from .schema import FactSet


def score_output(explanation: str, fs: FactSet, jr: JudgeResult, cond: Condition) -> dict:
    must = [f.fact_id for f in fs.tier("must_keep")]
    should = [f.fact_id for f in fs.tier("should_keep")]
    v = jr.fact_verdicts

    def frac(ids: list[str], want: str) -> float | None:
        return None if not ids else sum(v[i] == want for i in ids) / len(ids)

    n_all = len(fs.facts)
    mns = list(jr.mns_verdicts.values())
    fk = fk_grade(explanation)
    return {
        "condition": cond.id,
        "kind": cond.kind,
        "words": word_count(explanation),
        "fk_grade": round(fk, 2),
        "smog": round(smog_grade(explanation), 2),
        "band_ok": in_band(fk, cond.band),
        "cfr": frac(must, "stated"),
        "sfr": frac(should, "stated"),
        "distortion": (sum(x == "contradicted" for x in v.values()) / n_all) if n_all else None,
        "mns": (sum(x == "asserted" for x in mns) / len(mns)) if mns else None,
        "must_missing": [i for i in must if v[i] != "stated"],
        "parse_errors": jr.parse_errors,
    }


# ---------------------------------------------------------------- bootstrap

@dataclass
class Est:
    point: float
    lo: float
    hi: float

    def fmt(self, pct: bool = True) -> str:
        k = 100 if pct else 1
        return f"{self.point * k:.1f} [{self.lo * k:.1f}, {self.hi * k:.1f}]"

    def to_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def bootstrap(stat: Callable[[np.ndarray], float], n_items: int, n_boot: int = 2000,
              seed: int = 0, level: float = 0.95) -> Est:
    """Percentile bootstrap over item indices. `stat` takes an index array."""
    full = float(stat(np.arange(n_items)))
    if n_items < 2 or math.isnan(full):
        return Est(full, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat(rng.integers(0, n_items, n_items))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(draws, [a, 1 - a])
    return Est(full, float(lo), float(hi))


# -------------------------------------------------------------- aggregation

def _matrix(rows: list[dict], key: str, items: list[str], conds: list[str]) -> np.ndarray:
    """items x conditions matrix of metric `key`, NaN when missing; repeats are averaged."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        val = r.get(key)
        if val is not None:
            acc[(r["item_id"], r["condition"])].append(float(val))
    m = np.full((len(items), len(conds)), np.nan)
    for i, it in enumerate(items):
        for j, c in enumerate(conds):
            if acc.get((it, c)):
                m[i, j] = np.mean(acc[(it, c)])
    return m


def _nm(x: np.ndarray, axis=None) -> float:
    with np.errstate(all="ignore"):
        return float(np.nanmean(x, axis=axis)) if np.isfinite(x).any() else float("nan")


def _row_nanmean(m: np.ndarray) -> np.ndarray:
    ok = np.isfinite(m)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, m, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def summarize_model(rows: list[dict], cond_order: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """Aggregate one model's scored rows. `cond_order` starts with 'neutral'."""
    items = sorted({r["item_id"] for r in rows})
    conds = [c for c in cond_order if any(r["condition"] == c for r in rows)]
    n = len(items)
    cfr = _matrix(rows, "cfr", items, conds)
    out: dict = {"n_items": n, "conditions": {}, "headline": {}}
    ni = conds.index("neutral") if "neutral" in conds else None

    for j, c in enumerate(conds):
        d: dict = {"cfr": bootstrap(lambda ix, j=j: _nm(cfr[ix, j]), n, n_boot, seed).to_dict()}
        for key in ("sfr", "distortion", "mns"):
            m = _matrix(rows, key, items, conds)
            d[key] = bootstrap(lambda ix, m=m, j=j: _nm(m[ix, j]), n, n_boot, seed).to_dict()
        band = [r["band_ok"] for r in rows if r["condition"] == c and r["band_ok"] is not None]
        d["band_pass"] = (sum(band) / len(band)) if band else None
        words = [r["words"] for r in rows if r["condition"] == c]
        d["mean_words"] = float(np.mean(words)) if words else None
        d["mean_fk_grade"] = float(np.mean([r["fk_grade"] for r in rows if r["condition"] == c]))
        if ni is not None and j != ni:
            d["adaptation_tax"] = bootstrap(
                lambda ix, j=j: _nm(cfr[ix, ni] - cfr[ix, j]), n, n_boot, seed).to_dict()
            d["matched_length_tax"] = _matched_length_tax(rows, items, c, cfr[:, ni], cfr[:, j],
                                                          n_boot, seed)
        out["conditions"][c] = d

    non_neutral = [j for j, c in enumerate(conds) if c != "neutral"]
    if non_neutral:
        out["headline"]["worst_reader_cfr"] = bootstrap(
            lambda ix: min(_nm(cfr[ix, j]) for j in non_neutral), n, n_boot, seed).to_dict()
        out["headline"]["mean_adaptation_tax"] = bootstrap(
            lambda ix: _nm(cfr[ix, ni] - _row_nanmean(cfr[ix][:, non_neutral]))
            if ni is not None else float("nan"), n, n_boot, seed).to_dict()
    out["headline"]["cfr_spread"] = bootstrap(
        lambda ix: max(_nm(cfr[ix, j]) for j in range(len(conds))) - min(
            _nm(cfr[ix, j]) for j in range(len(conds))), n, n_boot, seed).to_dict()
    out["length_control"] = length_control(rows)
    out["parse_error_rate"] = float(np.mean([r["parse_errors"] > 0 for r in rows])) if rows else 0.0
    return out


def _matched_length_tax(rows: list[dict], items: list[str], cond: str, neutral_cfr: np.ndarray,
                        cond_cfr: np.ndarray, n_boot: int, seed: int,
                        ratio: float = 1.33) -> dict | None:
    """Tax restricted to items where the reader output is within `ratio` of the neutral length."""
    w: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        w[(r["item_id"], r["condition"])].append(r["words"])
    keep = np.zeros(len(items), dtype=bool)
    for i, it in enumerate(items):
        a, b = w.get((it, "neutral")), w.get((it, cond))
        if a and b:
            x, y = np.mean(a), np.mean(b)
            keep[i] = x > 0 and y > 0 and 1 / ratio <= y / x <= ratio
    if keep.sum() < 2:
        return None
    d = (neutral_cfr - cond_cfr)[keep]
    est = bootstrap(lambda ix: _nm(d[ix]), int(keep.sum()), n_boot, seed).to_dict()
    est["n_items"] = int(keep.sum())
    return est


def length_control(rows: list[dict]) -> dict:
    """CFR by length tercile over all outputs, and rank correlation of CFR with length."""
    ok = [r for r in rows if r["cfr"] is not None and r["words"] > 0]
    if len(ok) < 6:
        return {}
    words = np.array([r["words"] for r in ok], float)
    cfr = np.array([r["cfr"] for r in ok], float)
    q = np.quantile(words, [1 / 3, 2 / 3])
    bucket = np.digitize(words, q)
    names = ["short", "medium", "long"]
    by = {names[b]: {"n": int((bucket == b).sum()),
                     "mean_words": float(words[bucket == b].mean()),
                     "cfr": float(cfr[bucket == b].mean())} for b in range(3) if (bucket == b).any()}
    rw = np.argsort(np.argsort(words)).astype(float)
    rc = np.argsort(np.argsort(cfr)).astype(float)
    rho = float(np.corrcoef(rw, rc)[0, 1]) if rw.std() > 0 and rc.std() > 0 else None
    return {"by_length_tercile": by, "spearman_cfr_vs_words": rho}
