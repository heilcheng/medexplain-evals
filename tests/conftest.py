import json

import pytest

from plainline.schema import DRAFT_STATUS, Fact, FactSet, Item, Source


@pytest.fixture
def item():
    src = Source(source_id="t1", kind="dailymed_label", title="Testdrug", origin_url="http://x",
                 license_note="test", fetched="2026-01-01",
                 text="Testdrug can cause liver damage. Do not exceed 4 grams per day. "
                      "Take with food. It is used for pain.")
    fs = FactSet(source_id="t1", generator="test", status=DRAFT_STATUS,
                 facts=[Fact(fact_id="f1", text="Can cause liver damage", tier="must_keep"),
                        Fact(fact_id="f2", text="Do not exceed 4 grams per day", tier="must_keep"),
                        Fact(fact_id="f3", text="Take with food", tier="should_keep"),
                        Fact(fact_id="f4", text="Used for pain", tier="optional")],
                 must_not_say=["It is fine to take more than 4 grams per day."])
    return Item(source=src, facts=fs)


class ScriptedClient:
    """Returns queued replies, records calls."""

    def __init__(self, replies, name="scripted"):
        self.replies = list(replies)
        self.calls = []
        self.name = name

    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def verdict_json(**kw):
    return json.dumps({"verdicts": [{"id": k, "verdict": v} for k, v in kw.items()]})
