import json


def verdict_json(**kw):
    return json.dumps({"verdicts": [{"id": k, "verdict": v} for k, v in kw.items()]})
