"""MEQ-Bench Exception Hierarchy.

This module defines a comprehensive exception hierarchy for the MEQ-Bench framework,
providing structured error handling with rich context and recovery suggestions.

Exception Hierarchy:
    MEQBenchError (base)
    ├── ConfigurationError
    │   ├── ConfigFileNotFoundError
    │   ├── ConfigValidationError
    │   └── MissingAPIKeyError
    ├── DataError
    │   ├── DataLoadError
    │   ├── DataValidationError
    │   └── DatasetNotFoundError
    ├── EvaluationError
    │   ├── MetricCalculationError
    │   ├── ModelInferenceError
    │   └── TimeoutError
    ├── APIError
    │   ├── RateLimitError
    │   ├── AuthenticationError
    │   └── APIResponseError
    └── LeaderboardError
        ├── RenderingError
        └── ExportError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Structured context for error details."""

    operation: str
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.component}] {self.operation}"]
        if self.details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class MEQBenchError(Exception):
    """Base exception for all MEQ-Bench errors.

    Provides structured error handling with context, cause chaining,
    and optional recovery suggestions.

    Attributes:
        message: Human-readable error description
        context: Structured error context with operation details
        cause: Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        *,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message
        self.context = context
        self.cause = cause
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.context:
            parts.append(str(self.context))
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)

    @property
    def suggestion(self) -> str | None:
        """Get recovery suggestion if available."""
        return self.context.suggestion if self.context else None


# Configuration Errors
class ConfigurationError(MEQBenchError):
    """Base class for configuration-related errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file cannot be found."""

    def __init__(self, path: str, searched_locations: list[str] | None = None) -> None:
        context = ErrorContext(
            operation="load_config",
            component="Config",
            details={"path": path, "searched": searched_locations or []},
            suggestion="Create config.yaml or set MEQ_BENCH_CONFIG_PATH environment variable",
        )
        super().__init__(f"Configuration file not found: {path}", context=context)


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[dict[str, Any]], config_path: str | None = None) -> None:
        context = ErrorContext(
            operation="validate_config",
            component="Config",
            details={"errors": errors, "config_path": config_path},
            suggestion="Check configuration schema and fix validation errors",
        )
        super().__init__(f"Configuration validation failed: {len(errors)} error(s)", context=context)
        self.validation_errors = errors


class MissingAPIKeyError(ConfigurationError):
    """Raised when required API key is not set."""

    def __init__(self, provider: str, env_var: str) -> None:
        context = ErrorContext(
            operation="get_api_key",
            component="Config",
            details={"provider": provider, "env_var": env_var},
            suggestion=f"Set {env_var} environment variable or add to .env file",
        )
        super().__init__(f"API key not found for {provider}", context=context)
        self.provider = provider
        self.env_var = env_var


# Data Errors
class DataError(MEQBenchError):
    """Base class for data-related errors."""

    pass


class DataLoadError(DataError):
    """Raised when data loading fails."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        context = ErrorContext(
            operation="load_data",
            component="DataLoader",
            details={"source": source, "reason": reason},
            suggestion="Check data source path and format",
        )
        super().__init__(f"Failed to load data from {source}: {reason}", context=context, cause=cause)
        self.source = source


class DataValidationError(DataError):
    """Raised when data validation fails."""

    def __init__(
        self,
        item_id: str | None,
        field: str,
        expected: str,
        actual: str,
    ) -> None:
        context = ErrorContext(
            operation="validate_data",
            component="DataLoader",
            details={"item_id": item_id, "field": field, "expected": expected, "actual": actual},
            suggestion="Ensure data conforms to expected schema",
        )
        super().__init__(f"Data validation failed for field '{field}'", context=context)


class DatasetNotFoundError(DataError):
    """Raised when requested dataset is not found."""

    def __init__(self, dataset_name: str, available: list[str]) -> None:
        context = ErrorContext(
            operation="get_dataset",
            component="DataLoader",
            details={"dataset": dataset_name, "available": available},
            suggestion=f"Use one of: {', '.join(available)}",
        )
        super().__init__(f"Dataset not found: {dataset_name}", context=context)


# Evaluation Errors
class EvaluationError(MEQBenchError):
    """Base class for evaluation-related errors."""

    pass


class MetricCalculationError(EvaluationError):
    """Raised when metric calculation fails."""

    def __init__(
        self,
        metric_name: str,
        reason: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        context = ErrorContext(
            operation="calculate_metric",
            component="Evaluator",
            details={"metric": metric_name, "reason": reason},
            suggestion="Check input data format and metric configuration",
        )
        super().__init__(f"Failed to calculate metric '{metric_name}'", context=context, cause=cause)
        self.metric_name = metric_name


class ModelInferenceError(EvaluationError):
    """Raised when model inference fails."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        context = ErrorContext(
            operation="model_inference",
            component="Model",
            details={"model": model_name, "reason": reason},
            suggestion="Check model configuration and API connectivity",
        )
        super().__init__(f"Model inference failed for '{model_name}'", context=context, cause=cause)
        self.model_name = model_name


class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation times out."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        context = ErrorContext(
            operation=operation,
            component="Evaluator",
            details={"timeout_seconds": timeout_seconds},
            suggestion="Increase timeout or reduce batch size",
        )
        super().__init__(f"Operation timed out after {timeout_seconds}s", context=context)
        self.timeout_seconds = timeout_seconds


# API Errors
class APIError(MEQBenchError):
    """Base class for API-related errors."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
    ) -> None:
        context = ErrorContext(
            operation="api_call",
            component=f"API/{provider}",
            details={"retry_after": retry_after},
            suggestion=f"Wait {retry_after}s before retrying" if retry_after else "Implement exponential backoff",
        )
        super().__init__(f"Rate limit exceeded for {provider}", context=context)
        self.provider = provider
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, provider: str, reason: str) -> None:
        context = ErrorContext(
            operation="authenticate",
            component=f"API/{provider}",
            details={"reason": reason},
            suggestion="Check API key validity and permissions",
        )
        super().__init__(f"Authentication failed for {provider}: {reason}", context=context)
        self.provider = provider


class APIResponseError(APIError):
    """Raised when API returns an unexpected response."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        response_body: str | None = None,
    ) -> None:
        context = ErrorContext(
            operation="api_call",
            component=f"API/{provider}",
            details={"status_code": status_code, "response": response_body[:200] if response_body else None},
            suggestion="Check API documentation for error codes",
        )
        super().__init__(f"API error from {provider}: HTTP {status_code}", context=context)
        self.status_code = status_code
        self.response_body = response_body


# Leaderboard Errors
class LeaderboardError(MEQBenchError):
    """Base class for leaderboard-related errors."""

    pass


class RenderingError(LeaderboardError):
    """Raised when leaderboard rendering fails."""

    def __init__(self, reason: str, *, cause: Exception | None = None) -> None:
        context = ErrorContext(
            operation="render_leaderboard",
            component="Leaderboard",
            details={"reason": reason},
            suggestion="Check template and data format",
        )
        super().__init__(f"Failed to render leaderboard: {reason}", context=context, cause=cause)


class ExportError(LeaderboardError):
    """Raised when leaderboard export fails."""

    def __init__(
        self,
        format_type: str,
        path: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        context = ErrorContext(
            operation="export_leaderboard",
            component="Leaderboard",
            details={"format": format_type, "path": path},
            suggestion="Check write permissions and disk space",
        )
        super().__init__(f"Failed to export leaderboard to {format_type}", context=context, cause=cause)


__all__ = [
    "MEQBenchError",
    "ErrorContext",
    # Configuration
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "MissingAPIKeyError",
    # Data
    "DataError",
    "DataLoadError",
    "DataValidationError",
    "DatasetNotFoundError",
    # Evaluation
    "EvaluationError",
    "MetricCalculationError",
    "ModelInferenceError",
    "EvaluationTimeoutError",
    # API
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "APIResponseError",
    # Leaderboard
    "LeaderboardError",
    "RenderingError",
    "ExportError",
]
