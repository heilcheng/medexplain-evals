"""Tests for the Pydantic v2 settings module.

Tests cover:
    - Settings validation
    - Environment variable loading
    - YAML configuration loading
    - Default values
    - API key management
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings are valid."""
        from src.settings import Settings

        settings = Settings()

        assert settings.app.name == "MedExplain-Evals"
        assert settings.app.version == "2.0.0"
        assert len(settings.audiences) == 4
        assert len(settings.complexity_levels) == 3

    def test_settings_from_env(self) -> None:
        """Test settings can be loaded from environment variables."""
        from src.settings import Settings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-from-env"}):
            settings = Settings()
            assert settings.openai_api_key is not None
            assert settings.openai_api_key.get_secret_value() == "sk-test-from-env"

    def test_settings_validation(self) -> None:
        """Test settings validation catches invalid values."""
        from src.settings import ScoringWeights

        # Weights must sum to 1.0
        with pytest.raises(ValidationError):
            ScoringWeights(
                readability=0.5,
                terminology=0.5,
                safety=0.5,
                coverage=0.5,
                quality=0.5,
            )

    def test_readability_target_validation(self) -> None:
        """Test readability target validation."""
        from src.settings import ReadabilityTarget

        # Valid target
        target = ReadabilityTarget(min_grade_level=6, max_grade_level=10)
        assert target.min_grade_level == 6

        # Invalid: min > max
        with pytest.raises(ValidationError):
            ReadabilityTarget(min_grade_level=15, max_grade_level=10)

    def test_terminology_density_validation(self) -> None:
        """Test terminology density validation."""
        from src.settings import TerminologyDensity

        # Valid density
        density = TerminologyDensity(target=0.15, tolerance=0.05)
        assert density.target == 0.15

        # Invalid: target > 1.0
        with pytest.raises(ValidationError):
            TerminologyDensity(target=1.5, tolerance=0.05)


class TestAudience:
    """Tests for Audience enum."""

    def test_audience_values(self) -> None:
        """Test audience enum values."""
        from src.settings import Audience

        assert Audience.PHYSICIAN.value == "physician"
        assert Audience.NURSE.value == "nurse"
        assert Audience.PATIENT.value == "patient"
        assert Audience.CAREGIVER.value == "caregiver"

    def test_audience_iteration(self) -> None:
        """Test iterating over audiences."""
        from src.settings import Audience

        audiences = list(Audience)
        assert len(audiences) == 4


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_complexity_level_values(self) -> None:
        """Test complexity level enum values."""
        from src.settings import ComplexityLevel

        assert ComplexityLevel.BASIC.value == "basic"
        assert ComplexityLevel.INTERMEDIATE.value == "intermediate"
        assert ComplexityLevel.ADVANCED.value == "advanced"


class TestLLMJudgeSettings:
    """Tests for LLM Judge settings."""

    def test_default_values(self) -> None:
        """Test default LLM judge settings."""
        from src.settings import LLMJudgeSettings

        settings = LLMJudgeSettings()

        assert settings.default_model == "gpt-4-turbo"
        assert settings.timeout == 30
        assert settings.max_retries == 3
        assert settings.temperature == 0.1

    def test_validation(self) -> None:
        """Test LLM judge settings validation."""
        from src.settings import LLMJudgeSettings

        # Invalid temperature
        with pytest.raises(ValidationError):
            LLMJudgeSettings(temperature=3.0)  # Max is 2.0

        # Invalid timeout
        with pytest.raises(ValidationError):
            LLMJudgeSettings(timeout=1000)  # Max is 300


class TestPerformanceSettings:
    """Tests for performance settings."""

    def test_default_values(self) -> None:
        """Test default performance settings."""
        from src.settings import PerformanceSettings

        settings = PerformanceSettings()

        assert settings.batch_size == 10
        assert settings.max_workers == 4
        assert settings.cache_enabled is True
        assert settings.cache_ttl == 3600

    def test_validation(self) -> None:
        """Test performance settings validation."""
        from src.settings import PerformanceSettings

        # Invalid batch size
        with pytest.raises(ValidationError):
            PerformanceSettings(batch_size=0)

        # Invalid max_workers
        with pytest.raises(ValidationError):
            PerformanceSettings(max_workers=100)


class TestAPIKeyManagement:
    """Tests for API key management."""

    def test_get_api_key_success(self) -> None:
        """Test successful API key retrieval."""
        from src.settings import Settings

        settings = Settings(openai_api_key="sk-test-key")  # type: ignore[arg-type]
        key = settings.get_api_key("openai")
        assert key == "sk-test-key"

    def test_get_api_key_missing(self) -> None:
        """Test missing API key raises error."""
        from src.exceptions import MissingAPIKeyError
        from src.settings import Settings

        settings = Settings()
        with pytest.raises(MissingAPIKeyError) as exc_info:
            settings.get_api_key("openai")

        assert exc_info.value.provider == "openai"
        assert exc_info.value.env_var == "OPENAI_API_KEY"

    def test_get_api_key_unknown_provider(self) -> None:
        """Test unknown provider raises error."""
        from src.settings import Settings

        settings = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            settings.get_api_key("unknown")


class TestGetSettings:
    """Tests for get_settings function."""

    def test_cached_settings(self) -> None:
        """Test settings are cached."""
        from src.settings import get_settings

        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same cached instance
        assert settings1 is settings2

    def test_settings_reload(self) -> None:
        """Test settings can be reloaded."""
        from src.settings import get_settings

        settings1 = get_settings()

        # Clear cache
        get_settings.cache_clear()

        settings2 = get_settings()

        # Should be different instances after cache clear
        # (though values should be the same)
        assert settings1 is not settings2
