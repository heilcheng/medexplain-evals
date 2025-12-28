"""Pytest configuration and fixtures for MedExplain-Evals tests.

This module provides comprehensive test fixtures and configuration using
modern pytest features including:
    - Typed fixtures with proper annotations
    - Async test support with pytest-asyncio
    - Factory fixtures for flexible test data generation
    - Shared fixtures with appropriate scopes
    - Environment isolation for tests
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set mock environment variables before importing modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key-for-testing"
os.environ["GOOGLE_API_KEY"] = "test-google-key-for-testing"

if TYPE_CHECKING:
    from src.api_client import BaseLLMClient
    from src.benchmark import MedExplain, MedExplainItem
    from src.evaluator import MedExplainEvaluator
    from src.settings import Settings


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests (>5s)")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_api: Tests requiring real API keys")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in test_integration.py as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests with 'slow' in their name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def sample_medical_terms() -> list[str]:
    """Common medical terms for testing."""
    return [
        "hypertension",
        "diabetes",
        "myocardial",
        "infarction",
        "pneumonia",
        "cardiac",
        "pulmonary",
        "renal",
    ]


# =============================================================================
# Module-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Settings instance for testing."""
    from src.settings import Settings

    return Settings(
        openai_api_key="sk-test-key",  # type: ignore[arg-type]
        anthropic_api_key="test-key",  # type: ignore[arg-type]
    )


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================


@pytest.fixture
def sample_medical_content() -> str:
    """Sample medical content for testing."""
    return (
        "Hypertension is a chronic medical condition characterized by persistently "
        "elevated blood pressure in the arteries. It is defined as systolic blood "
        "pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg. Risk factors include "
        "obesity, sedentary lifestyle, high sodium intake, and genetic predisposition."
    )


@pytest.fixture
def sample_benchmark_item() -> MedExplainItem:
    """Sample benchmark item for testing."""
    from src.benchmark import MedExplainItem

    return MedExplainItem(
        id="test_item_001",
        medical_content=(
            "Type 2 diabetes mellitus is a metabolic disorder characterized by "
            "insulin resistance and relative insulin deficiency, leading to hyperglycemia."
        ),
        complexity_level="intermediate",
        source_dataset="test_dataset",
    )


@pytest.fixture
def sample_explanations() -> dict[str, str]:
    """Sample audience-adaptive explanations for testing."""
    return {
        "physician": (
            "Essential hypertension with systolic BP >140 mmHg or diastolic >90 mmHg, "
            "requiring antihypertensive therapy and cardiovascular risk stratification. "
            "Consider ACE inhibitors or ARBs as first-line treatment."
        ),
        "nurse": (
            "Patient has elevated blood pressure requiring regular monitoring. "
            "Ensure medication compliance, educate on lifestyle modifications, "
            "and watch for side effects like dizziness or cough."
        ),
        "patient": (
            "Your blood pressure is higher than normal, which means your heart has to "
            "work harder. We'll give you medicine to help lower it, and you can help by "
            "eating less salt and getting regular exercise."
        ),
        "caregiver": (
            "Their blood pressure is too high and needs daily monitoring. Make sure they "
            "take their medication at the same time each day. Watch for headaches, "
            "dizziness, or nosebleeds, and call the doctor if these occur."
        ),
    }


@pytest.fixture
def benchmark_instance() -> MedExplain:
    """MedExplain instance for testing."""
    from src.benchmark import MedExplain

    return MedExplain()


@pytest.fixture
def evaluator_instance() -> MedExplainEvaluator:
    """MedExplainEvaluator instance for testing."""
    from src.evaluator import MedExplainEvaluator

    return MedExplainEvaluator()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_model_function() -> MagicMock:
    """Mock model function that returns structured explanations."""

    def model_func(prompt: str) -> str:
        return """
For a Physician: The patient presents with essential hypertension (ICD-10: I10) with
systolic BP consistently >140 mmHg. Recommend initiating ACE inhibitor therapy
with regular monitoring of renal function and electrolytes.

For a Nurse: Monitor BP every 4 hours. Document readings and report systolic >160
or diastolic >100. Assess for medication compliance and educate on low-sodium diet.

For a Patient: Your blood pressure is a bit high. We'll give you medicine to help
bring it down. Try to eat less salty food and walk for 30 minutes each day.

For a Caregiver: Help them remember to take their blood pressure pill every morning.
Check their blood pressure at home and write it down. Call the doctor if it's very high.
"""

    mock = MagicMock(side_effect=model_func)
    return mock


@pytest.fixture
def mock_async_llm_client() -> AsyncMock:
    """Mock async LLM client for testing."""
    from src.api_client import CompletionResponse

    client = AsyncMock()
    client.complete.return_value = CompletionResponse(
        content="Test response content",
        model="test-model",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    client.generate_explanations.return_value = {
        "physician": "Technical medical explanation",
        "nurse": "Care-focused explanation",
        "patient": "Simple explanation",
        "caregiver": "Practical instructions",
    }
    return client


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx async client."""
    import httpx

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    client = AsyncMock()
    client.post.return_value = mock_response
    return client


# =============================================================================
# Factory Fixtures
# =============================================================================


@pytest.fixture
def benchmark_item_factory():
    """Factory for creating benchmark items with custom attributes."""
    from src.benchmark import MedExplainItem

    def _create_item(
        id: str = "test_001",
        medical_content: str = "Sample medical content",
        complexity_level: str = "basic",
        source_dataset: str = "test",
        **kwargs: Any,
    ) -> MedExplainItem:
        return MedExplainItem(
            id=id,
            medical_content=medical_content,
            complexity_level=complexity_level,
            source_dataset=source_dataset,
            **kwargs,
        )

    return _create_item


@pytest.fixture
def explanation_factory():
    """Factory for creating explanation dictionaries."""

    def _create_explanations(
        physician: str = "Technical explanation",
        nurse: str = "Nursing explanation",
        patient: str = "Patient explanation",
        caregiver: str = "Caregiver explanation",
    ) -> dict[str, str]:
        return {
            "physician": physician,
            "nurse": nurse,
            "patient": patient,
            "caregiver": caregiver,
        }

    return _create_explanations


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
async def async_temp_file(tmp_path: Path) -> AsyncIterator[Path]:
    """Async fixture for temporary file operations."""
    temp_file = tmp_path / "async_test_file.json"
    yield temp_file
    if temp_file.exists():
        temp_file.unlink()


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def clean_environment() -> Iterator[None]:
    """Fixture that provides a clean environment for tests."""
    # Store original environment
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["MEQ_BENCH_APP__LOG_LEVEL"] = "DEBUG"
    os.environ["MEQ_BENCH_PERFORMANCE__BATCH_SIZE"] = "5"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def no_api_keys() -> Iterator[None]:
    """Fixture that removes API keys from environment."""
    keys_to_remove = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    original_values = {k: os.environ.pop(k, None) for k in keys_to_remove}

    yield

    # Restore keys
    for key, value in original_values.items():
        if value is not None:
            os.environ[key] = value


# =============================================================================
# Logging Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def configure_test_logging() -> None:
    """Configure logging for tests (disabled by default)."""
    import logging

    logging.disable(logging.CRITICAL)


@pytest.fixture
def enable_logging() -> Iterator[None]:
    """Enable logging for specific tests."""
    import logging

    logging.disable(logging.NOTSET)
    yield
    logging.disable(logging.CRITICAL)
