"""Tests for the exception hierarchy module.

Tests cover:
    - Exception instantiation
    - Error context formatting
    - Exception chaining
    - Recovery suggestions
"""

from __future__ import annotations

import pytest


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_basic_context(self) -> None:
        """Test basic error context creation."""
        from src.exceptions import ErrorContext

        ctx = ErrorContext(
            operation="test_op",
            component="TestComponent",
        )

        assert ctx.operation == "test_op"
        assert ctx.component == "TestComponent"
        assert ctx.details == {}
        assert ctx.suggestion is None

    def test_context_with_details(self) -> None:
        """Test error context with details."""
        from src.exceptions import ErrorContext

        ctx = ErrorContext(
            operation="load_data",
            component="DataLoader",
            details={"file": "test.json", "line": 42},
            suggestion="Check file format",
        )

        assert ctx.details["file"] == "test.json"
        assert ctx.suggestion == "Check file format"

    def test_context_string_representation(self) -> None:
        """Test error context string formatting."""
        from src.exceptions import ErrorContext

        ctx = ErrorContext(
            operation="validate",
            component="Config",
            details={"key": "value"},
            suggestion="Fix it",
        )

        str_repr = str(ctx)
        assert "[Config]" in str_repr
        assert "validate" in str_repr
        assert "Fix it" in str_repr


class TestMedExplainError:
    """Tests for base MedExplainError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        from src.exceptions import MedExplainError

        error = MedExplainError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.context is None
        assert error.cause is None

    def test_error_with_context(self) -> None:
        """Test error with context."""
        from src.exceptions import ErrorContext, MedExplainError

        ctx = ErrorContext(
            operation="test",
            component="Test",
            suggestion="Try again",
        )
        error = MedExplainError("Test error", context=ctx)

        assert error.context is not None
        assert error.suggestion == "Try again"

    def test_error_with_cause(self) -> None:
        """Test error with cause chaining."""
        from src.exceptions import MedExplainError

        cause = ValueError("Original error")
        error = MedExplainError("Wrapped error", cause=cause)

        assert error.cause is cause
        assert "ValueError" in str(error)
        assert "Original error" in str(error)


class TestConfigurationErrors:
    """Tests for configuration-related errors."""

    def test_config_file_not_found(self) -> None:
        """Test ConfigFileNotFoundError."""
        from src.exceptions import ConfigFileNotFoundError

        error = ConfigFileNotFoundError(
            path="/path/to/config.yaml",
            searched_locations=["/home", "/etc"],
        )

        assert "config.yaml" in str(error)
        assert error.context is not None
        assert "MEQ_BENCH_CONFIG_PATH" in error.context.suggestion

    def test_config_validation_error(self) -> None:
        """Test ConfigValidationError."""
        from src.exceptions import ConfigValidationError

        errors = [
            {"loc": ("app", "name"), "msg": "field required"},
            {"loc": ("llm_judge", "model"), "msg": "invalid value"},
        ]
        error = ConfigValidationError(errors, config_path="config.yaml")

        assert "2 error(s)" in str(error)
        assert error.validation_errors == errors

    def test_missing_api_key_error(self) -> None:
        """Test MissingAPIKeyError."""
        from src.exceptions import MissingAPIKeyError

        error = MissingAPIKeyError("openai", "OPENAI_API_KEY")

        assert error.provider == "openai"
        assert error.env_var == "OPENAI_API_KEY"
        assert "OPENAI_API_KEY" in error.context.suggestion


class TestDataErrors:
    """Tests for data-related errors."""

    def test_data_load_error(self) -> None:
        """Test DataLoadError."""
        from src.exceptions import DataLoadError

        cause = FileNotFoundError("File not found")
        error = DataLoadError("medquad.json", "File not found", cause=cause)

        assert error.source == "medquad.json"
        assert error.cause is cause
        assert "medquad.json" in str(error)

    def test_data_validation_error(self) -> None:
        """Test DataValidationError."""
        from src.exceptions import DataValidationError

        error = DataValidationError(
            item_id="item_001",
            field="medical_content",
            expected="non-empty string",
            actual="empty",
        )

        assert "medical_content" in str(error)

    def test_dataset_not_found_error(self) -> None:
        """Test DatasetNotFoundError."""
        from src.exceptions import DatasetNotFoundError

        error = DatasetNotFoundError("unknown", ["medquad", "medqa"])

        assert "unknown" in str(error)
        assert "medquad" in error.context.suggestion


class TestEvaluationErrors:
    """Tests for evaluation-related errors."""

    def test_metric_calculation_error(self) -> None:
        """Test MetricCalculationError."""
        from src.exceptions import MetricCalculationError

        error = MetricCalculationError("readability", "Division by zero")

        assert error.metric_name == "readability"
        assert "readability" in str(error)

    def test_model_inference_error(self) -> None:
        """Test ModelInferenceError."""
        from src.exceptions import ModelInferenceError

        cause = TimeoutError("Request timed out")
        error = ModelInferenceError("gpt-4", "Timeout", cause=cause)

        assert error.model_name == "gpt-4"
        assert error.cause is cause

    def test_evaluation_timeout_error(self) -> None:
        """Test EvaluationTimeoutError."""
        from src.exceptions import EvaluationTimeoutError

        error = EvaluationTimeoutError("batch_evaluation", 60.0)

        assert error.timeout_seconds == 60.0
        assert "60.0s" in str(error)


class TestAPIErrors:
    """Tests for API-related errors."""

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError."""
        from src.exceptions import RateLimitError

        error = RateLimitError("openai", retry_after=30.0)

        assert error.provider == "openai"
        assert error.retry_after == 30.0
        assert "30.0s" in error.context.suggestion

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        from src.exceptions import AuthenticationError

        error = AuthenticationError("anthropic", "Invalid key format")

        assert error.provider == "anthropic"
        assert "Invalid key format" in str(error)

    def test_api_response_error(self) -> None:
        """Test APIResponseError."""
        from src.exceptions import APIResponseError

        error = APIResponseError("openai", 500, "Internal server error")

        assert error.status_code == 500
        assert error.response_body == "Internal server error"
        assert "HTTP 500" in str(error)


class TestLeaderboardErrors:
    """Tests for leaderboard-related errors."""

    def test_rendering_error(self) -> None:
        """Test RenderingError."""
        from src.exceptions import RenderingError

        cause = KeyError("missing_key")
        error = RenderingError("Template variable not found", cause=cause)

        assert "Template variable not found" in str(error)
        assert error.cause is cause

    def test_export_error(self) -> None:
        """Test ExportError."""
        from src.exceptions import ExportError

        error = ExportError("html", "/path/to/output.html")

        assert "html" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_hierarchy_inheritance(self) -> None:
        """Test that exceptions inherit correctly."""
        from src.exceptions import (
            APIError,
            ConfigurationError,
            DataError,
            EvaluationError,
            LeaderboardError,
            MedExplainError,
            RateLimitError,
        )

        # All should inherit from MedExplainError
        assert issubclass(ConfigurationError, MedExplainError)
        assert issubclass(DataError, MedExplainError)
        assert issubclass(EvaluationError, MedExplainError)
        assert issubclass(APIError, MedExplainError)
        assert issubclass(LeaderboardError, MedExplainError)

        # Specific errors should inherit from their category
        assert issubclass(RateLimitError, APIError)

    def test_exception_catching(self) -> None:
        """Test that exceptions can be caught at different levels."""
        from src.exceptions import MedExplainError, RateLimitError

        error = RateLimitError("openai", 30.0)

        # Should be catchable as MedExplainError
        try:
            raise error
        except MedExplainError as e:
            assert isinstance(e, RateLimitError)
