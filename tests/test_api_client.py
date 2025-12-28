"""Tests for the async API client module.

Tests cover:
    - Client initialization
    - Message formatting
    - Completion responses
    - Explanation parsing
    - Error handling
    - Provider-specific behavior
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test message creation."""
        from src.api_client import Message

        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test completion response creation."""
        from src.api_client import CompletionResponse

        response = CompletionResponse(
            content="Test response",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["prompt_tokens"] == 10
        assert response.finish_reason == "stop"


class TestExplanationResponse:
    """Tests for ExplanationResponse dataclass."""

    def test_to_dict(self) -> None:
        """Test converting explanation response to dictionary."""
        from src.api_client import ExplanationResponse

        response = ExplanationResponse(
            physician="Technical explanation",
            nurse="Care explanation",
            patient="Simple explanation",
            caregiver="Practical explanation",
        )

        result = response.to_dict()

        assert result["physician"] == "Technical explanation"
        assert result["nurse"] == "Care explanation"
        assert len(result) == 4


class TestBaseLLMClient:
    """Tests for base LLM client functionality."""

    def test_explanation_prompt_building(self) -> None:
        """Test explanation prompt generation."""
        from src.api_client import LocalClient
        from src.settings import Audience

        client = LocalClient(model="test-model", base_url="http://localhost:8000")

        prompt = client._build_explanation_prompt(
            "High blood pressure condition",
            [Audience.PHYSICIAN, Audience.PATIENT],
        )

        assert "physician" in prompt.lower()
        assert "patient" in prompt.lower()
        assert "High blood pressure" in prompt

    def test_explanation_parsing(self) -> None:
        """Test parsing explanations from LLM response."""
        from src.api_client import LocalClient
        from src.settings import Audience

        client = LocalClient(model="test-model", base_url="http://localhost:8000")

        response = """
For a Physician: Technical medical explanation here.

For a Nurse: Nursing care explanation here.

For a Patient: Simple patient explanation here.

For a Caregiver: Caregiver instructions here.
"""

        result = client._parse_explanations(response, list(Audience))

        assert "Technical medical explanation" in result["physician"]
        assert "Nursing care explanation" in result["nurse"]
        assert "Simple patient explanation" in result["patient"]
        assert "Caregiver instructions" in result["caregiver"]

    def test_explanation_parsing_partial(self) -> None:
        """Test parsing when some explanations are missing."""
        from src.api_client import LocalClient
        from src.settings import Audience

        client = LocalClient(model="test-model", base_url="http://localhost:8000")

        response = """
For a Physician: Only physician explanation.
"""

        result = client._parse_explanations(response, list(Audience))

        assert "Only physician explanation" in result["physician"]
        # Other audiences should have empty strings
        assert result["nurse"] == ""
        assert result["patient"] == ""


class TestOpenAIClient:
    """Tests for OpenAI client."""

    @pytest.mark.asyncio
    async def test_client_initialization(self) -> None:
        """Test OpenAI client initialization."""
        from src.api_client import OpenAIClient

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = MagicMock()
            mock_settings.return_value.openai_api_key.get_secret_value.return_value = "sk-test"
            mock_settings.return_value.api.openai.base_url = "https://api.openai.com/v1"

            client = OpenAIClient(model="gpt-4")

            assert client.model == "gpt-4"
            assert client.api_key == "sk-test"

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_httpx_client: AsyncMock) -> None:
        """Test successful completion."""
        from src.api_client import Message, OpenAIClient

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = MagicMock()
            mock_settings.return_value.openai_api_key.get_secret_value.return_value = "sk-test"
            mock_settings.return_value.api.openai.base_url = "https://api.openai.com/v1"

            client = OpenAIClient(model="gpt-4")
            client._client = mock_httpx_client

            messages = [Message(role="user", content="Hello")]
            response = await client.complete(messages)

            assert response.content == "Test response"
            mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self) -> None:
        """Test rate limit handling."""
        from src.api_client import OpenAIClient
        from src.exceptions import RateLimitError

        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = MagicMock()
            mock_settings.return_value.openai_api_key.get_secret_value.return_value = "sk-test"
            mock_settings.return_value.api.openai.base_url = "https://api.openai.com/v1"

            client = OpenAIClient(model="gpt-4", max_retries=1)
            client._client = mock_client

            from src.api_client import Message

            with pytest.raises(RateLimitError) as exc_info:
                await client.complete([Message(role="user", content="test")])

            assert exc_info.value.retry_after == 30.0


class TestAnthropicClient:
    """Tests for Anthropic client."""

    @pytest.mark.asyncio
    async def test_client_initialization(self) -> None:
        """Test Anthropic client initialization."""
        from src.api_client import AnthropicClient

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.api.anthropic.base_url = "https://api.anthropic.com"

            client = AnthropicClient(model="claude-3-opus")

            assert client.model == "claude-3-opus"
            assert client.api_key == "test-key"


class TestGoogleClient:
    """Tests for Google Gemini client."""

    @pytest.mark.asyncio
    async def test_client_initialization(self) -> None:
        """Test Google client initialization."""
        from src.api_client import GoogleClient

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.google_api_key = MagicMock()
            mock_settings.return_value.google_api_key.get_secret_value.return_value = "google-key"
            mock_settings.return_value.api.google.base_url = "https://generativelanguage.googleapis.com"

            client = GoogleClient(model="gemini-pro")

            assert client.model == "gemini-pro"
            assert client.api_key == "google-key"


class TestLocalClient:
    """Tests for local/self-hosted client."""

    def test_client_initialization(self) -> None:
        """Test local client initialization."""
        from src.api_client import LocalClient

        client = LocalClient(model="llama-3", base_url="http://localhost:8080")

        assert client.model == "llama-3"
        assert client.base_url == "http://localhost:8080"


class TestCreateAsyncClient:
    """Tests for client factory function."""

    @pytest.mark.asyncio
    async def test_create_openai_client(self) -> None:
        """Test creating OpenAI client."""
        from src.api_client import OpenAIClient, create_async_client

        with patch("src.api_client.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = MagicMock()
            mock_settings.return_value.openai_api_key.get_secret_value.return_value = "sk-test"
            mock_settings.return_value.api.openai.base_url = "https://api.openai.com/v1"

            async with create_async_client("openai", "gpt-4") as client:
                assert isinstance(client, OpenAIClient)

    @pytest.mark.asyncio
    async def test_create_unknown_backend(self) -> None:
        """Test creating client with unknown backend raises error."""
        from src.api_client import create_async_client

        with pytest.raises(ValueError, match="Unknown backend"):
            async with create_async_client("unknown", "model"):
                pass

    @pytest.mark.asyncio
    async def test_create_local_client(self) -> None:
        """Test creating local client."""
        from src.api_client import LocalClient, create_async_client

        async with create_async_client("local", "test-model", base_url="http://localhost:8000") as client:
            assert isinstance(client, LocalClient)
