"""Unified exception hierarchy for JARVIS.

This module provides a consistent error handling system across CLI, API, and models.
All JARVIS-specific exceptions inherit from JarvisError, enabling consistent handling.

Exception Hierarchy:
    JarvisError (base)
    +-- ConfigurationError - Configuration and settings issues
    +-- ModelError - Model loading and generation failures
    |   +-- ModelLoadError - Failed to load model
    |   +-- ModelGenerationError - Generation failed
    +-- iMessageError - iMessage access and query issues
    |   +-- iMessageAccessError - Permission/access denied
    |   +-- iMessageQueryError - Database query failure
    +-- ValidationError - Input validation failures
    +-- ResourceError - System resource issues
    |   +-- MemoryResourceError - Insufficient memory
    |   +-- DiskResourceError - Disk space/access issues
    +-- TaskError - Task queue failures
    +-- CalendarError - Calendar operation failures
    +-- ExperimentError - A/B testing failures
    +-- FeedbackError - Feedback tracking failures
    +-- GraphError - Knowledge graph failures
    +-- DatabaseError - Database operation failures
    +-- ExportError - Export generation failures
    +-- EmbeddingError - Embedding operation failures

Usage:
    from jarvis.errors import ModelError, iMessageError

    try:
        result = model.generate(prompt)
    except ModelError as e:
        logger.error("Model error: %s (code: %s)", e.message, e.code)
"""

# --- base ---
from jarvis.errors.base import (
    ConfigurationError,
    ErrorCode,
    JarvisError,
)

# --- domain errors ---
from jarvis.errors.domain import (
    CalendarAccessError,
    CalendarCreateError,
    CalendarError,
    DatabaseError,
    DiskResourceError,
    EmbeddingError,
    EventParseError,
    ExperimentConfigError,
    ExperimentError,
    ExperimentNotFoundError,
    ExportError,
    FeedbackError,
    FeedbackInvalidActionError,
    FeedbackNotFoundError,
    GraphError,
    MemoryResourceError,
    ResourceError,
    TaskError,
    TaskExecutionError,
    TaskNotFoundError,
    ValidationError,
)

# --- convenience factories ---
from jarvis.errors.factories import (
    calendar_permission_denied,
    imessage_db_not_found,
    imessage_permission_denied,
    model_generation_timeout,
    model_not_found,
    model_out_of_memory,
    task_invalid_status,
    task_not_found,
    validation_required,
    validation_type_error,
)

# --- model & imessage ---
from jarvis.errors.model import (
    ModelError,
    ModelGenerationError,
    ModelLoadError,
    iMessageAccessError,
    iMessageError,
    iMessageQueryError,
)

__all__ = [
    # Error codes
    "ErrorCode",
    # Base exception
    "JarvisError",
    # Configuration errors
    "ConfigurationError",
    # Model errors
    "ModelError",
    "ModelLoadError",
    "ModelGenerationError",
    # iMessage errors
    "iMessageError",
    "iMessageAccessError",
    "iMessageQueryError",
    # Validation errors
    "ValidationError",
    # Resource errors
    "ResourceError",
    "MemoryResourceError",
    "DiskResourceError",
    # Task errors
    "TaskError",
    "TaskNotFoundError",
    "TaskExecutionError",
    # Calendar errors
    "CalendarError",
    "CalendarAccessError",
    "CalendarCreateError",
    "EventParseError",
    # Experiment errors
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentConfigError",
    # Feedback errors
    "FeedbackError",
    "FeedbackNotFoundError",
    "FeedbackInvalidActionError",
    # Graph errors
    "GraphError",
    # Database errors
    "DatabaseError",
    # Export errors
    "ExportError",
    # Embedding errors
    "EmbeddingError",
    # Convenience functions
    "model_not_found",
    "model_out_of_memory",
    "model_generation_timeout",
    "imessage_permission_denied",
    "imessage_db_not_found",
    "validation_required",
    "validation_type_error",
    "task_not_found",
    "task_invalid_status",
    "calendar_permission_denied",
]
