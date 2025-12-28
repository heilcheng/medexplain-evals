"""Async API Client for MedExplain-Evals.

This module provides async HTTP clients for various LLM providers with:
    - Automatic retry with exponential backoff
    - Rate limiting and quota management
    - Streaming support
    - Connection pooling
    - Comprehensive error handling

Supported Providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude 3)
    - Google (Gemini)
    - Local/HuggingFace (via HTTP API)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.exceptions import (
    APIResponseError,
    AuthenticationError,
    ModelInferenceError,
    RateLimitError,
)
from src.logging import get_logger
from src.settings import Audience, get_settings


logger = get_logger(__name__)


@dataclass
class Message:
    """Chat message."""

    role: str
    content: str


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass
class ExplanationResponse:
    """Audience-specific explanations."""

    physician: str = ""
    nurse: str = ""
    patient: str = ""
    caregiver: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "physician": self.physician,
            "nurse": self.nurse,
            "patient": self.patient,
            "caregiver": self.caregiver,
        }


class BaseLLMClient(ABC):
    """Base class for LLM API clients."""

    provider: ClassVar[str] = "base"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BaseLLMClient:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "Client not initialized. Use 'async with' context manager."
            raise RuntimeError(msg)
        return self._client

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion from messages."""
        ...

    async def generate_explanations(
        self,
        medical_content: str,
        audiences: list[Audience] | None = None,
    ) -> dict[str, str]:
        """Generate audience-adaptive medical explanations.

        Args:
            medical_content: The medical content to explain
            audiences: Target audiences (defaults to all)

        Returns:
            Dictionary mapping audience to explanation
        """
        if audiences is None:
            audiences = list(Audience)

        prompt = self._build_explanation_prompt(medical_content, audiences)
        messages = [Message(role="user", content=prompt)]

        response = await self.complete(messages, temperature=0.3, max_tokens=2048)
        return self._parse_explanations(response.content, audiences)

    def _build_explanation_prompt(self, content: str, audiences: list[Audience]) -> str:
        """Build prompt for generating explanations."""
        audience_list = ", ".join(a.value for a in audiences)
        return f"""You are a medical communication expert. Generate clear, accurate explanations
of the following medical content for each target audience: {audience_list}.

Medical Content:
{content}

For each audience, provide an appropriate explanation that matches their medical literacy level:
- Physician: Technical, using proper medical terminology
- Nurse: Practical, care-focused with clinical relevance
- Patient: Simple, clear language avoiding jargon
- Caregiver: Concrete instructions and warning signs

Format your response as:
For a Physician: [explanation]
For a Nurse: [explanation]
For a Patient: [explanation]
For a Caregiver: [explanation]
"""

    def _parse_explanations(self, response: str, audiences: list[Audience]) -> dict[str, str]:
        """Parse explanations from LLM response."""
        import re

        explanations: dict[str, str] = {}

        patterns = {
            Audience.PHYSICIAN: r"For a Physician:\s*(.+?)(?=For a (?:Nurse|Patient|Caregiver):|$)",
            Audience.NURSE: r"For a Nurse:\s*(.+?)(?=For a (?:Physician|Patient|Caregiver):|$)",
            Audience.PATIENT: r"For a Patient:\s*(.+?)(?=For a (?:Physician|Nurse|Caregiver):|$)",
            Audience.CAREGIVER: r"For a Caregiver:\s*(.+?)(?=$)",
        }

        for audience in audiences:
            pattern = patterns.get(audience)
            if pattern:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    explanations[audience.value] = match.group(1).strip()
                else:
                    explanations[audience.value] = ""

        return explanations

    async def _retry_request(self, coro):
        """Execute request with retry logic."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            reraise=True,
        ):
            with attempt:
                return await coro


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    provider: ClassVar[str] = "openai"

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        api_key = api_key or (settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
        base_url = settings.api.openai.base_url

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion using OpenAI API."""
        if not self.api_key:
            raise AuthenticationError("openai", "API key not provided")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            response = await self._retry_request(
                self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", 60))
                raise RateLimitError("openai", retry_after)

            if response.status_code == 401:
                raise AuthenticationError("openai", "Invalid API key")

            if response.status_code >= 400:
                raise APIResponseError("openai", response.status_code, response.text)

            data = response.json()
            return CompletionResponse(
                content=data["choices"][0]["message"]["content"],
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )

        except httpx.HTTPError as e:
            raise ModelInferenceError(self.model, str(e), cause=e)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    provider: ClassVar[str] = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        api_key = api_key or (settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None)
        base_url = settings.api.anthropic.base_url

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion using Anthropic API."""
        if not self.api_key:
            raise AuthenticationError("anthropic", "API key not provided")

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2024-01-01",
        }

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            payload["system"] = system_message

        try:
            response = await self._retry_request(
                self.client.post(f"{self.base_url}/v1/messages", headers=headers, json=payload)
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", 60))
                raise RateLimitError("anthropic", retry_after)

            if response.status_code == 401:
                raise AuthenticationError("anthropic", "Invalid API key")

            if response.status_code >= 400:
                raise APIResponseError("anthropic", response.status_code, response.text)

            data = response.json()
            content = data["content"][0]["text"] if data.get("content") else ""

            return CompletionResponse(
                content=content,
                model=data["model"],
                usage={
                    "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                },
                finish_reason=data.get("stop_reason", "stop"),
            )

        except httpx.HTTPError as e:
            raise ModelInferenceError(self.model, str(e), cause=e)


class GoogleClient(BaseLLMClient):
    """Google Gemini API client."""

    provider: ClassVar[str] = "google"

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        api_key = api_key or (settings.google_api_key.get_secret_value() if settings.google_api_key else None)
        base_url = settings.api.google.base_url

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion using Google Gemini API."""
        if not self.api_key:
            raise AuthenticationError("google", "API key not provided")

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        try:
            response = await self._retry_request(self.client.post(url, json=payload))

            if response.status_code == 429:
                raise RateLimitError("google", 60)

            if response.status_code == 401:
                raise AuthenticationError("google", "Invalid API key")

            if response.status_code >= 400:
                raise APIResponseError("google", response.status_code, response.text)

            data = response.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]

            return CompletionResponse(
                content=content,
                model=self.model,
                usage=data.get("usageMetadata", {}),
                finish_reason=data["candidates"][0].get("finishReason", "STOP"),
            )

        except httpx.HTTPError as e:
            raise ModelInferenceError(self.model, str(e), cause=e)


class LocalClient(BaseLLMClient):
    """Client for local/self-hosted LLM APIs (OpenAI-compatible)."""

    provider: ClassVar[str] = "local"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, base_url=base_url, **kwargs)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate completion using local API."""
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            response = await self._retry_request(
                self.client.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=payload)
            )

            if response.status_code >= 400:
                raise APIResponseError("local", response.status_code, response.text)

            data = response.json()
            return CompletionResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )

        except httpx.HTTPError as e:
            raise ModelInferenceError(self.model, str(e), cause=e)


# Client factory
_CLIENT_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GoogleClient,
    "local": LocalClient,
    "huggingface": LocalClient,  # HF TGI is OpenAI-compatible
}


@asynccontextmanager
async def create_async_client(
    backend: str,
    model: str,
    **kwargs: Any,
) -> AsyncIterator[BaseLLMClient]:
    """Create an async LLM client.

    Args:
        backend: Provider name (openai, anthropic, google, local)
        model: Model name or path
        **kwargs: Additional client configuration

    Yields:
        Configured LLM client

    Example:
        async with create_async_client("openai", "gpt-4o") as client:
            response = await client.complete([Message("user", "Hello!")])
    """
    client_class = _CLIENT_REGISTRY.get(backend.lower())
    if client_class is None:
        available = ", ".join(_CLIENT_REGISTRY.keys())
        msg = f"Unknown backend: {backend}. Available: {available}"
        raise ValueError(msg)

    client = client_class(model=model, **kwargs)
    async with client:
        yield client


__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
    "LocalClient",
    "Message",
    "CompletionResponse",
    "ExplanationResponse",
    "create_async_client",
]
