"""Chat clients. Any OpenAI-compatible endpoint works; a mock is provided for offline runs."""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

import httpx

Messages = list[dict[str, str]]


class ChatClient(Protocol):
    name: str

    def complete(self, messages: Messages, *, temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


class OpenAICompatClient:
    """POST {base_url}/chat/completions. The API key is read from an environment variable."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 120.0, retries: int = 3, extra_body: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.retries = retries
        self.extra_body = extra_body or {}

    def complete(self, messages: Messages, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **self.extra_body}
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                               timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                if 400 <= r.status_code < 500:
                    # Auth or request errors will not fix themselves; fail at once.
                    raise RuntimeError(f"model call failed with HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as e:
                last = e
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"model call failed after {self.retries} attempts: {last}")


_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


class MockClient:
    """Deterministic offline stand-in. Not a measurement; used for tests and smoke runs.

    Writer role: echoes a share of the source sentences (fewer for lay readers).
    Judge role: marks a fact stated when most of its content words occur in the explanation.
    """

    name = "mock"

    def complete(self, messages: Messages, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        user = messages[-1]["content"]
        if "PLAINLINE_JUDGE" in messages[0]["content"]:
            return self._judge(user)
        return self._write(user)

    @staticmethod
    def _write(user: str) -> str:
        m = re.search(r"<source>\n(.*?)\n</source>", user, re.S)
        src = m.group(1) if m else user
        sents = [s for s in _SENT.split(src) if s.strip()]
        keep = max(2, len(sents) // (3 if "grade" in user.lower() else 2))
        return " ".join(sents[:keep])

    @staticmethod
    def _judge(user: str) -> str:
        import json

        m = re.search(r"<explanation>\n(.*?)\n</explanation>", user, re.S)
        expl = set(_WORD.findall((m.group(1) if m else "").lower()))
        out = []
        for fid, text in re.findall(r"^\[(\S+)\] (.*)$", user.split("</explanation>")[-1], re.M):
            if fid.startswith("n"):
                out.append({"id": fid, "verdict": "not_asserted"})
                continue
            words = [w for w in _WORD.findall(text.lower()) if len(w) > 3]
            hit = sum(w in expl for w in words) / max(1, len(words))
            out.append({"id": fid, "verdict": "stated" if hit >= 0.6 else "absent"})
        return json.dumps({"verdicts": out})
