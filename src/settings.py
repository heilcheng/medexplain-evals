"""Pydantic v2 Settings Management for MEQ-Bench.

This module provides type-safe, validated configuration using Pydantic v2
with support for environment variables, .env files, and YAML configuration.

Features:
    - Type-safe settings with full validation
    - Environment variable support with MEQ_BENCH_ prefix
    - YAML configuration file loading
    - Nested configuration models
    - Runtime validation and coercion
    - Secrets management for API keys
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Audience(StrEnum):
    """Target audience types for medical explanations."""

    PHYSICIAN = "physician"
    NURSE = "nurse"
    PATIENT = "patient"
    CAREGIVER = "caregiver"


class ComplexityLevel(StrEnum):
    """Complexity levels for medical content."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DatasetName(StrEnum):
    """Supported dataset names."""

    MEDQUAD = "medquad"
    MEDQA = "medqa"
    ICLINIQ = "icliniq"
    COCHRANE = "cochrane"
    HEALTHSEARCHQA = "healthsearchqa"


class AppSettings(BaseModel):
    """Application-level settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "MEQ-Bench"
    version: str = "2.0.0"
    log_level: LogLevel = LogLevel.INFO
    data_path: Path = Path("data/")
    output_path: Path = Path("results/")
    cache_path: Path = Path(".cache/")

    @field_validator("data_path", "output_path", "cache_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v) if isinstance(v, str) else v


class ReadabilityTarget(BaseModel):
    """Readability grade level targets for an audience."""

    model_config = ConfigDict(frozen=True)

    min_grade_level: Annotated[int, Field(ge=1, le=20)]
    max_grade_level: Annotated[int, Field(ge=1, le=20)]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.min_grade_level > self.max_grade_level:
            msg = "min_grade_level must be <= max_grade_level"
            raise ValueError(msg)
        return self


class TerminologyDensity(BaseModel):
    """Terminology density targets for an audience."""

    model_config = ConfigDict(frozen=True)

    target: Annotated[float, Field(ge=0.0, le=1.0)]
    tolerance: Annotated[float, Field(ge=0.0, le=0.5)]


class SafetySettings(BaseModel):
    """Safety evaluation settings."""

    model_config = ConfigDict(frozen=True)

    danger_words: list[str] = Field(default_factory=list)
    safety_words: list[str] = Field(default_factory=list)
    medical_terms: list[str] = Field(default_factory=list)


class EvaluationSettings(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(frozen=True)

    readability_targets: dict[Audience, ReadabilityTarget] = Field(default_factory=dict)
    terminology_density: dict[Audience, TerminologyDensity] = Field(default_factory=dict)
    safety: SafetySettings = Field(default_factory=SafetySettings)


class ScoringWeights(BaseModel):
    """Weights for different evaluation metrics."""

    model_config = ConfigDict(frozen=True)

    readability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    terminology: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    safety: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    quality: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        total = self.readability + self.terminology + self.safety + self.coverage + self.quality
        if abs(total - 1.0) > 0.001:
            msg = f"Scoring weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class ScoringParameters(BaseModel):
    """Scoring parameters."""

    model_config = ConfigDict(frozen=True)

    min_explanation_length: Annotated[int, Field(ge=10)] = 50
    max_explanation_length: dict[Audience, int] = Field(
        default_factory=lambda: {
            Audience.PHYSICIAN: 2000,
            Audience.NURSE: 1500,
            Audience.PATIENT: 1000,
            Audience.CAREGIVER: 1000,
        }
    )
    coverage_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    safety_multiplier: Annotated[float, Field(ge=1.0, le=5.0)] = 2.0


class ScoringSettings(BaseModel):
    """Scoring configuration."""

    model_config = ConfigDict(frozen=True)

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    parameters: ScoringParameters = Field(default_factory=ScoringParameters)


class LLMJudgeSettings(BaseModel):
    """LLM-as-a-judge configuration."""

    model_config = ConfigDict(frozen=True)

    default_model: str = "gpt-4-turbo"
    available_models: list[str] = Field(
        default_factory=lambda: ["gpt-4-turbo", "gpt-4o", "claude-3-sonnet", "claude-3-opus"]
    )
    timeout: Annotated[int, Field(ge=5, le=300)] = 30
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.1
    max_tokens: Annotated[int, Field(ge=100, le=8000)] = 1000


class APIProviderSettings(BaseModel):
    """API provider configuration."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    models: list[str] = Field(default_factory=list)


class APISettings(BaseModel):
    """API configuration for all providers."""

    model_config = ConfigDict(frozen=True)

    openai: APIProviderSettings = Field(
        default_factory=lambda: APIProviderSettings(
            base_url="https://api.openai.com/v1",
            models=["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
        )
    )
    anthropic: APIProviderSettings = Field(
        default_factory=lambda: APIProviderSettings(
            base_url="https://api.anthropic.com",
            models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        )
    )
    google: APIProviderSettings = Field(
        default_factory=lambda: APIProviderSettings(
            base_url="https://generativelanguage.googleapis.com",
            models=["gemini-pro", "gemini-1.5-pro"],
        )
    )


class PerformanceSettings(BaseModel):
    """Performance and caching configuration."""

    model_config = ConfigDict(frozen=True)

    batch_size: Annotated[int, Field(ge=1, le=100)] = 10
    max_workers: Annotated[int, Field(ge=1, le=32)] = 4
    cache_enabled: bool = True
    cache_ttl: Annotated[int, Field(ge=60, le=86400)] = 3600


class Settings(BaseSettings):
    """Main settings class with environment variable support.

    Environment variables are prefixed with MEQ_BENCH_ and use double underscore
    for nested settings (e.g., MEQ_BENCH_APP__LOG_LEVEL=DEBUG).
    """

    model_config = SettingsConfigDict(
        env_prefix="MEQ_BENCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # API Keys (from environment)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    # Configuration sections
    app: AppSettings = Field(default_factory=AppSettings)
    audiences: list[Audience] = Field(default_factory=lambda: list(Audience))
    complexity_levels: list[ComplexityLevel] = Field(default_factory=lambda: list(ComplexityLevel))
    llm_judge: LLMJudgeSettings = Field(default_factory=LLMJudgeSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    api: APISettings = Field(default_factory=APISettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings from YAML file, merged with environment variables."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        return cls(**yaml_config)

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, google)

        Returns:
            The API key as a string

        Raises:
            MissingAPIKeyError: If the API key is not set
        """
        from src.exceptions import MissingAPIKeyError

        key_mapping = {
            "openai": (self.openai_api_key, "OPENAI_API_KEY"),
            "anthropic": (self.anthropic_api_key, "ANTHROPIC_API_KEY"),
            "google": (self.google_api_key, "GOOGLE_API_KEY"),
        }

        provider_lower = provider.lower()
        if provider_lower not in key_mapping:
            msg = f"Unknown provider: {provider}"
            raise ValueError(msg)

        secret, env_var = key_mapping[provider_lower]
        if secret is None:
            raise MissingAPIKeyError(provider, env_var)

        return secret.get_secret_value()

    def model_dump_yaml(self) -> str:
        """Export settings to YAML format (excluding secrets)."""
        data = self.model_dump(exclude={"openai_api_key", "anthropic_api_key", "google_api_key"})
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Settings are loaded once and cached. To reload, call get_settings.cache_clear().
    """
    config_path = Path("config.yaml")
    if config_path.exists():
        return Settings.from_yaml(config_path)
    return Settings()


def _convert_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy config.yaml format to new settings format."""
    # Handle audience conversion
    if "audiences" in data and isinstance(data["audiences"], list):
        data["audiences"] = [Audience(a) if isinstance(a, str) else a for a in data["audiences"]]

    # Handle complexity levels
    if "complexity_levels" in data and isinstance(data["complexity_levels"], list):
        data["complexity_levels"] = [ComplexityLevel(c) if isinstance(c, str) else c for c in data["complexity_levels"]]

    return data


# Convenience alias
settings = get_settings()

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "AppSettings",
    "EvaluationSettings",
    "ScoringSettings",
    "ScoringWeights",
    "LLMJudgeSettings",
    "APISettings",
    "PerformanceSettings",
    "Audience",
    "ComplexityLevel",
    "LogLevel",
    "DatasetName",
]
