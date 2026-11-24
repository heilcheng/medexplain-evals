import itertools
import json

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from plainline.inspect_task import build_samples, plainline  # noqa: E402


def test_samples_cover_items_and_conditions():
    s = build_samples("adapt", "dev", None, 2)
    assert len(s) == 2 * 6
    assert {x.metadata["condition"] for x in s} >= {"neutral", "clinician", "grade6_adult"}
    assert "<source>" in s[0].input and s[0].metadata["facts"]["status"] == "draft, needs clinician review"
    assert len(build_samples("cue", "dev", None, 1)) == 1 * 8


def test_task_runs_with_mock_models(tmp_path):
    ids = [{"id": f"f{i}", "verdict": "stated"} for i in range(1, 13)]
    reply = ModelOutput.from_content("mockllm/model", json.dumps({"verdicts": ids}))
    judge = get_model("mockllm/model", custom_outputs=itertools.repeat(reply))
    task = plainline(limit_items=2)
    # Use the same mock for writer; the scorer fetches its judge through get_model().
    from plainline import inspect_task

    orig = inspect_task.get_model
    inspect_task.get_model = lambda *a, **k: judge
    try:
        logs = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    finally:
        inspect_task.get_model = orig
    log = logs[0]
    assert log.status == "success"
    names = list(log.results.scores[0].metrics)
    assert {"worst_reader_cfr", "adaptation_tax", "cfr_mean"} <= set(names)
    assert log.results.scores[0].metrics["cfr_mean"].value > 0
