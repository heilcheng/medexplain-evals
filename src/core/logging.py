"""Structured Logging for MedExplain-Evals.

This module provides structured logging using structlog with support for:
    - JSON output for production environments
    - Pretty console output for development
    - Automatic context binding (request IDs, correlation IDs)
    - Performance timing decorators
    - Exception logging with full context

Usage:
    from src.logging import get_logger, configure_logging

    # Configure logging at application startup
    configure_logging(level="DEBUG", json_output=False)

    # Get a logger for your module
    logger = get_logger(__name__)

    # Log with structured context
    logger.info("Processing item", item_id="123", status="started")

    # Bind context for subsequent logs
    log = logger.bind(request_id="abc123")
    log.info("First operation")
    log.info("Second operation")  # Both have request_id
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import structlog
from rich.console import Console
from rich.logging import RichHandler


if TYPE_CHECKING:
    from structlog.typing import FilteringBoundLogger

P = ParamSpec("P")
T = TypeVar("T")


def _add_log_level(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add log level to event dict for filtering."""
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO timestamp to event dict."""
    import datetime

    event_dict["timestamp"] = datetime.datetime.now(tz=datetime.UTC).isoformat()
    return event_dict


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output logs as JSON (for production)
        log_file: Optional file path for log output
    """
    # Shared processors for structlog
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        # Development: Pretty console output with Rich
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.RichTracebackFormatter(
                    show_locals=True,
                    max_frames=10,
                ),
            ),
        ]
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Add file handler if specified
    if log_file:
        from pathlib import Path

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))

        # Use JSON format for file output
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    for logger_name in ["httpx", "httpcore", "urllib3", "openai", "anthropic"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ of the module

    Returns:
        A structlog FilteringBoundLogger instance
    """
    return structlog.get_logger(name)


@contextmanager
def log_context(**kwargs: Any):
    """Context manager for temporary logging context.

    Usage:
        with log_context(request_id="abc123", user_id="user456"):
            logger.info("Processing request")  # Includes request_id and user_id
        logger.info("After context")  # No longer has context
    """
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*kwargs.keys())


def timed(
    logger: FilteringBoundLogger | None = None,
    level: str = "debug",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use (defaults to module logger)
        level: Log level for timing messages

    Usage:
        @timed()
        def slow_operation():
            time.sleep(1)

        @timed(logger=my_logger, level="info")
        async def async_operation():
            await asyncio.sleep(1)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        log = logger or get_logger(func.__module__)
        log_method = getattr(log, level)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log_method(
                    "Function completed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed * 1000, 2),
                    status="success",
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log_method(
                    "Function failed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed * 1000, 2),
                    status="error",
                    error=str(e),
                )
                raise

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)  # type: ignore[misc]
                elapsed = time.perf_counter() - start
                log_method(
                    "Async function completed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed * 1000, 2),
                    status="success",
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log_method(
                    "Async function failed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed * 1000, 2),
                    status="error",
                    error=str(e),
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper

    return decorator


def log_exception(
    logger: FilteringBoundLogger,
    exc: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    """Log an exception with full context.

    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context to include
    """
    from src.exceptions import MedExplainError

    error_info: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }

    if isinstance(exc, MedExplainError):
        if exc.context:
            error_info["error_context"] = {
                "operation": exc.context.operation,
                "component": exc.context.component,
                "details": exc.context.details,
                "suggestion": exc.context.suggestion,
            }
        if exc.cause:
            error_info["cause"] = {
                "type": type(exc.cause).__name__,
                "message": str(exc.cause),
            }

    if context:
        error_info.update(context)

    logger.exception("Exception occurred", **error_info)


__all__ = [
    "configure_logging",
    "get_logger",
    "log_context",
    "log_exception",
    "timed",
]
