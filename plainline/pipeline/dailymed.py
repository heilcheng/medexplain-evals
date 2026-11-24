"""Fetch FDA drug labels (SPL) from the DailyMed web services (NLM).

Only selected clinically important sections are kept, each truncated at a
sentence boundary, so the chunk is a short excerpt and not a full label copy.
"""

from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET

import httpx

from ..schema import Source
from .text import clean, dedupe_sentences, truncate_words

API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
NS = {"h": "urn:hl7-org:v3"}
TITLE = "{urn:hl7-org:v3}title"

# LOINC section codes used in SPL labels: key -> (heading, word cap)
SECTIONS = {
    "34066-1": ("Boxed warning", 220),
    "34067-9": ("Indications and usage", 110),
    "34068-7": ("Dosage and administration", 220),
    "34070-3": ("Contraindications", 130),
    "43685-7": ("Warnings and precautions", 260),
    "34071-1": ("Warnings", 260),
}
ORDER = ["34066-1", "34067-9", "34068-7", "34070-3", "43685-7", "34071-1"]
REPACKAGERS = ("repack", "bryant", "a-s medication", "proficient rx", "nucare", "preferred pharm",
               "quality care", "direct_rx", "direct rx", "henry schein", "medsource", "rebel dist",
               "northwind", "asclemed", "enovachem", "lake erie", "denton", "unit dose", "pd-rx",
               "aphena")


def find_setid(client: httpx.Client, drug_name: str) -> dict | None:
    """First non-repackager Rx label whose title starts with the drug name."""
    r = client.get(f"{API}/spls.json", params={"drug_name": drug_name, "pagesize": 50})
    r.raise_for_status()
    rows = r.json().get("data", [])
    first = drug_name.split()[0].lower()
    for row in rows:
        title = row["title"].lower()
        if title.startswith(first) and not any(p in title for p in REPACKAGERS):
            return row
    return None


def _tidy(text: str) -> str:
    """Remove cross-reference markers and highlight boilerplate from SPL text."""
    text = re.sub(r"\[\s*see[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\(\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*\)", "", text)
    text = re.sub(r"See full prescribing information for complete boxed warning\.?", "", text, flags=re.I)
    text = re.sub(r"^WARNING:\s+[A-Z0-9 ,/&;()-]{4,}?\s+(?=[A-Z][a-z])", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_spl(xml_bytes: bytes) -> dict[str, str]:
    """Return {loinc_code: text} for the sections of interest."""
    root = ET.fromstring(xml_bytes)
    found: dict[str, str] = {}
    for sec in root.iter("{urn:hl7-org:v3}section"):
        code = sec.find("h:code", NS)
        if code is None:
            continue
        c = code.get("code")
        if c in SECTIONS:
            txt = _tidy(clean(" ".join(" ".join(c.itertext()) for c in sec if c.tag != TITLE)))
            # Highlights sections reuse the same codes as the full text; keep the longest.
            if len(txt) > len(found.get(c, "")):
                found[c] = txt
    return found


def build_source(drug_name: str, row: dict, sections: dict[str, str]) -> Source:
    name = re.sub(r"\s*\[.*?\]", "", row["title"]).title()
    parts = [f"{name} (prescribing information excerpt)"]
    used = {}
    for code in ORDER:
        if code == "34071-1" and "43685-7" in sections:
            continue  # prefer the PLR-format section when both exist
        if code in sections:
            heading, cap = SECTIONS[code]
            body = truncate_words(dedupe_sentences(sections[code]), cap)
            used[heading] = body
            parts.append(f"{heading.upper()}\n{body}")
    sid = "dm-" + drug_name.lower().replace(" ", "-").replace("/", "-")
    return Source(
        source_id=sid,
        kind="dailymed_label",
        title=row["title"],
        text="\n\n".join(parts),
        origin_url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={row['setid']}",
        license_note="DailyMed label text (manufacturer-authored FDA labeling). Short excerpt; see DATA_LICENSES.md.",
        fetched=dt.date.today().isoformat(),
        version=f"spl_version={row.get('spl_version')};setid={row['setid']}",
    )


def fetch_label(client: httpx.Client, drug_name: str) -> Source | None:
    row = find_setid(client, drug_name)
    if row is None:
        return None
    r = client.get(f"{API}/spls/{row['setid']}.xml")
    r.raise_for_status()
    sections = parse_spl(r.content)
    if not sections:
        return None
    return build_source(drug_name, row, sections)
