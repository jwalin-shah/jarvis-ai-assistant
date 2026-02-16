"""Structured logging and timing utilities for JARVIS observability.

Provides:
- StructuredFormatter: JSON log formatter for structured log output
- timed_operation: Context manager that logs operation timing
- log_event: Helper for structured event logging with metrics

Uses stdlib logging only (no external dependencies).

Usage:
    from jarvis.observability.logging import timed_operation, log_event

    with timed_operation(logger, "generation.model_inference", model_id="lfm-1.2b"):
        result = model.generate(prompt)

    log_event(logger, "classifier.inference.complete",
              classifier="category", result="question", confidence=0.87, latency_ms=12.3)
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter.

    Outputs log records as single-line JSON with standard fields:
    timestamp, level, logger, message, plus any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra structured fields (set via log_event or extra={})
        if hasattr(record, "event_type"):
            entry["event"] = record.event_type  # type: ignore[attr-defined, unused-ignore]
        if hasattr(record, "metrics"):
            entry["metrics"] = record.metrics  # type: ignore[attr-defined, unused-ignore]
        if hasattr(record, "metadata"):
            entry["metadata"] = record.metadata  # type: ignore[attr-defined, unused-ignore]

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            entry["error"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(entry, default=str)


@contextmanager
def timed_operation(
    log: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **extra: Any,
) -> Any:
    """Context manager that logs operation start/complete with timing.

    Args:
        log: Logger instance.
        operation: Operation name (e.g., "generation.model_inference").
        level: Log level for completion message (start is always DEBUG).
        **extra: Additional key-value pairs included in the log.

    Yields:
        dict that can be updated with additional metrics during the operation.

    Example:
        with timed_operation(logger, "model.load", model_id="lfm-1.2b") as ctx:
            model.load()
            ctx["memory_mb"] = model.get_memory_usage_mb()
    """
    ctx: dict[str, Any] = {}
    start = time.perf_counter()
    log.debug(
        "%s started",
        operation,
        extra={"event_type": f"{operation}.start", "metadata": extra},
    )
    try:
        yield ctx
        elapsed_ms = (time.perf_counter() - start) * 1000
        merged = {**extra, **ctx}
        log.log(
            level,
            "%s completed in %.1fms",
            operation,
            elapsed_ms,
            extra={
                "event_type": f"{operation}.complete",
                "metrics": {
                    "latency_ms": round(elapsed_ms, 1),
                    **{k: v for k, v in merged.items() if isinstance(v, (int, float))},
                },
                "metadata": {k: v for k, v in merged.items() if not isinstance(v, (int, float))},
            },
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.error(
            "%s failed after %.1fms",
            operation,
            elapsed_ms,
            exc_info=True,
            extra={
                "event_type": f"{operation}.failed",
                "metrics": {"latency_ms": round(elapsed_ms, 1)},
                "metadata": extra,
            },
        )
        raise


def log_event(
    log: logging.Logger,
    event_type: str,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Log a structured event with typed fields.

    Args:
        log: Logger instance.
        event_type: Event type string (e.g., "classifier.inference.complete").
        level: Log level.
        message: Optional human-readable message. Defaults to event_type.
        **fields: Arbitrary key-value fields. Numeric values go to metrics,
                  others go to metadata.
    """
    metrics = {k: v for k, v in fields.items() if isinstance(v, (int, float))}
    metadata = {k: v for k, v in fields.items() if not isinstance(v, (int, float))}

    log.log(
        level,
        message or event_type,
        extra={
            "event_type": event_type,
            "metrics": metrics if metrics else None,
            "metadata": metadata if metadata else None,
        },
    )


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON output.

    Call this once at application startup (e.g., in socket_server.main()).

    Args:
        level: Root log level.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers to avoid duplicate output
    root.handlers = [handler]
