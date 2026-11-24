"""Fetch MedlinePlus health topic summaries (NLM, public domain per NLM policy).

Only the health topic summary text is used. Drug monographs (ASHP), the A.D.A.M.
encyclopedia and supplement monographs are copyrighted and are never fetched.
"""

from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET

import httpx

from ..schema import Source
from .text import clean, truncate_words

INDEX = "https://medlineplus.gov/xml.html"


def latest_topics_url(client: httpx.Client) -> str:
    page = client.get(INDEX).text
    names = sorted(set(re.findall(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml", page)))
    if not names:
        raise RuntimeError("could not find a MedlinePlus topics XML file")
    return f"https://medlineplus.gov/xml/{names[-1]}"


def parse_topics(xml_bytes: bytes) -> dict[str, dict]:
    """Map lowercase English title to {title, url, summary}."""
    root = ET.fromstring(xml_bytes)
    out: dict[str, dict] = {}
    for ht in root.iter("health-topic"):
        if ht.get("language") != "English":
            continue
        fs = ht.find("full-summary")
        if fs is None or not (fs.text or "").strip():
            continue
        title = ht.get("title", "")
        out[title.lower()] = {"title": title, "url": ht.get("url", ""), "summary": clean(fs.text)}
    return out


def build_source(topic: dict, xml_name: str) -> Source:
    slug = re.sub(r"[^a-z0-9]+", "-", topic["title"].lower()).strip("-")
    return Source(
        source_id=f"mp-{slug}",
        kind="medlineplus_topic",
        title=topic["title"],
        text=f"{topic['title']} (MedlinePlus health topic summary)\n\n{truncate_words(topic['summary'], 350)}",
        origin_url=topic["url"],
        license_note="NLM health topic summary, public domain (NLM copyright policy). Courtesy of the U.S. National Library of Medicine.",
        fetched=dt.date.today().isoformat(),
        version=xml_name,
    )


def fetch_topics(client: httpx.Client, titles: list[str]) -> tuple[list[Source], list[str]]:
    url = latest_topics_url(client)
    r = client.get(url, timeout=120)
    r.raise_for_status()
    topics = parse_topics(r.content)
    name = url.rsplit("/", 1)[-1]
    got, missing = [], []
    for t in titles:
        if t.lower() in topics:
            got.append(build_source(topics[t.lower()], name))
        else:
            missing.append(t)
    return got, missing
