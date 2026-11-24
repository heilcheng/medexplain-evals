import re
from pathlib import Path

from plainline.readers import conditions
from plainline.schema import DRAFT_STATUS, load_items

ROOT = Path(__file__).resolve().parent.parent


def test_pilot_data():
    items = load_items(ROOT / "data")
    assert 28 <= len(items) <= 40
    assert all(i.facts.status == DRAFT_STATUS for i in items), "no pilot fact set may claim review"
    assert all(i.facts.tier("must_keep") for i in items)
    assert all("PLAINLINE-CANARY" in i.source.canary for i in items)
    kinds = {i.source.kind for i in items}
    assert kinds == {"dailymed_label", "medlineplus_topic"}
    assert all(i.source.origin_url.startswith("https://") for i in items)


def test_readers():
    adapt = conditions("adapt")
    assert adapt[0].id == "neutral" and len(adapt) == 6
    cue = conditions("cue")
    assert cue[0].id == "neutral" and len(cue) >= 6
    assert all("physician" not in c.instruction for c in cue[1:])


def test_no_em_dashes_or_arrows():
    bad = re.compile("[" + "".join(map(chr, (0x2014, 0x2192, 0x2190, 0x21d2, 0x2194))) + "]")
    for p in list(ROOT.glob("*.md")) + list(ROOT.glob("docs/*.md")) + list((ROOT / "plainline").rglob("*")) \
            + list((ROOT / "tests").glob("*.py")) + list((ROOT / "data").rglob("*.jsonl")):
        if p.is_file() and p.suffix in {".md", ".py", ".yaml", ".jsonl"}:
            assert not bad.search(p.read_text(encoding="utf-8")), f"em dash or arrow in {p}"
