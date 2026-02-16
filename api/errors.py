"""FastAPI exception handlers for JARVIS errors.

This module provides exception handlers that map JarvisError subclasses
to appropriate HTTP status codes with a standardized response format.

Response Format:
    {
        "error": "ErrorClassName",
        "code": "ERROR_CODE",
        "detail": "Human-readable error message",
        "details": {...}  # Optional additional context
    }

Usage:
    from api.errors import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from jarvis.errors import (
    ErrorCode,
    JarvisError,
    ModelLoadError,
    ResourceError,
    ValidationError,
    iMessageAccessError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Map error codes to HTTP status codes.
# This is the single source of truth for error-code -> HTTP status mapping.
# The get_status_code_for_error() function also checks for a `status_code` class
# attribute on the exception (via getattr), enabling future migration of status
# codes onto the exception classes themselves.
ERROR_CODE_STATUS_CODES: dict[ErrorCode, int] = {
    # 400 Bad Request
    ErrorCode.VAL_INVALID_INPUT: 400,
    ErrorCode.VAL_MISSING_REQUIRED: 400,
    ErrorCode.VAL_TYPE_ERROR: 400,
    ErrorCode.MDL_INVALID_REQUEST: 400,
    ErrorCode.CAL_PARSE_FAILED: 400,
    ErrorCode.EXPORT_INVALID_FORMAT: 400,
    # 403 Forbidden
    ErrorCode.MSG_ACCESS_DENIED: 403,
    ErrorCode.CAL_ACCESS_DENIED: 403,
    # 404 Not Found
    ErrorCode.MDL_NOT_FOUND: 404,
    ErrorCode.MSG_DB_NOT_FOUND: 404,
    ErrorCode.MSG_NOT_FOUND: 404,
    ErrorCode.CFG_MISSING: 404,
    ErrorCode.GRF_CONTACT_NOT_FOUND: 404,
    # 408 Request Timeout
    ErrorCode.MDL_TIMEOUT: 408,
    # 500 Internal Server Error
    ErrorCode.CFG_INVALID: 500,
    ErrorCode.CFG_MIGRATION_FAILED: 500,
    ErrorCode.MDL_GENERATION_FAILED: 500,
    ErrorCode.MSG_QUERY_FAILED: 500,
    ErrorCode.MSG_SCHEMA_UNSUPPORTED: 500,
    ErrorCode.MSG_SEND_FAILED: 500,
    ErrorCode.GRF_BUILD_FAILED: 500,
    ErrorCode.DB_QUERY_FAILED: 500,
    ErrorCode.DB_INTEGRITY_ERROR: 500,
    ErrorCode.EXPORT_GENERATION_FAILED: 500,
    ErrorCode.EMB_ENCODING_FAILED: 500,
    # 500 Internal Server Error (Calendar)
    ErrorCode.CAL_CREATE_FAILED: 500,
    ErrorCode.CAL_NOT_AVAILABLE: 500,
    # 503 Service Unavailable
    ErrorCode.MDL_LOAD_FAILED: 503,
    ErrorCode.RES_MEMORY_LOW: 503,
    ErrorCode.RES_MEMORY_EXHAUSTED: 503,
    ErrorCode.RES_DISK_FULL: 503,
    ErrorCode.RES_DISK_ACCESS: 503,
    ErrorCode.DB_CONNECTION_FAILED: 503,
    ErrorCode.EMB_INDEX_NOT_READY: 503,
}


def get_status_code_for_error(error: JarvisError) -> int:
    """Determine the appropriate HTTP status code for an error.

    Priority order:
    1. Error-code-level override in ERROR_CODE_STATUS_CODES
    2. The exception class's `status_code` attribute (defined on each JarvisError subclass)
    3. Default 500

    Args:
        error: The JARVIS error instance.

    Returns:
        HTTP status code (400-599).
    """
    # Check error-code-level overrides first (e.g. MDL_NOT_FOUND -> 404)
    if error.code in ERROR_CODE_STATUS_CODES:
        return ERROR_CODE_STATUS_CODES[error.code]

    # Use the status_code defined on the exception class
    return getattr(error, "status_code", 500)


def build_error_response(error: JarvisError) -> dict[str, Any]:
    """Build a standardized error response dictionary.

    Args:
        error: The JARVIS error instance.

    Returns:
        Dictionary with error, code, detail, and optional details fields.
    """
    response: dict[str, Any] = {
        "error": error.__class__.__name__,
        "code": error.code.value,
        "detail": error.message,
    }

    # Include additional details if present
    if error.details:
        response["details"] = error.details

    return response


async def jarvis_error_handler(request: Request, exc: JarvisError) -> JSONResponse:
    """Handle JarvisError and subclasses.

    Converts JARVIS errors to appropriate HTTP responses with
    standardized error format.

    Args:
        request: The FastAPI request object.
        exc: The JARVIS error that was raised.

    Returns:
        JSONResponse with appropriate status code and error body.
    """
    status_code = get_status_code_for_error(exc)
    response_body = build_error_response(exc)

    # Log the error with appropriate level
    if status_code >= 500:
        logger.error(
            "Server error: %s (code=%s, status=%d)",
            exc.message,
            exc.code.value,
            status_code,
            exc_info=exc.cause if exc.cause else exc,
        )
    else:
        logger.warning(
            "Client error: %s (code=%s, status=%d)",
            exc.message,
            exc.code.value,
            status_code,
        )

    headers = {}
    if status_code == 503:
        headers["Retry-After"] = "30"

    return JSONResponse(
        status_code=status_code,
        content=response_body,
        headers=headers if headers else None,
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle ValidationError with additional field information.

    Provides more detailed error responses for validation failures.

    Args:
        request: The FastAPI request object.
        exc: The validation error that was raised.

    Returns:
        JSONResponse with 400 status and validation details.
    """
    response_body = build_error_response(exc)

    logger.debug(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=400,
        content=response_body,
    )


async def imessage_access_error_handler(request: Request, exc: iMessageAccessError) -> JSONResponse:
    """Handle iMessageAccessError with permission instructions.

    Provides helpful instructions for resolving permission issues.

    Args:
        request: The FastAPI request object.
        exc: The iMessage access error that was raised.

    Returns:
        JSONResponse with 403 status and permission instructions.
    """
    response_body = build_error_response(exc)

    logger.warning(
        "iMessage access denied for %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=403,
        content=response_body,
    )


async def model_load_error_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    """Handle ModelLoadError with service availability information.

    Args:
        request: The FastAPI request object.
        exc: The model load error that was raised.

    Returns:
        JSONResponse with 503 status.
    """
    response_body = build_error_response(exc)

    logger.error(
        "Model load failed for %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
        exc_info=exc.cause if exc.cause else exc,
    )

    return JSONResponse(
        status_code=503,
        content=response_body,
        headers={"Retry-After": "30"},  # Suggest retry after 30 seconds
    )


async def resource_error_handler(request: Request, exc: ResourceError) -> JSONResponse:
    """Handle ResourceError with service availability information.

    Args:
        request: The FastAPI request object.
        exc: The resource error that was raised.

    Returns:
        JSONResponse with 503 status.
    """
    response_body = build_error_response(exc)

    logger.error(
        "Resource error for %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=503,
        content=response_body,
        headers={"Retry-After": "60"},  # Suggest retry after 60 seconds
    )


async def timeout_error_handler(request: Request, exc: TimeoutError) -> JSONResponse:
    """Handle timeout errors with 408 Request Timeout.

    This handler catches asyncio.TimeoutError and similar timeout exceptions
    and returns an appropriate 408 response.

    Args:
        request: The FastAPI request object.
        exc: The timeout error that was raised.

    Returns:
        JSONResponse with 408 status and retry information.
    """
    logger.warning(
        "Request timeout for %s %s",
        request.method,
        request.url.path,
    )

    response_body = {
        "error": "RequestTimeout",
        "code": "REQUEST_TIMEOUT",
        "detail": "The request timed out. Please try again.",
    }

    return JSONResponse(
        status_code=408,
        content=response_body,
        headers={"Retry-After": "5"},  # Suggest retry after 5 seconds
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with a generic error response.

    This is a catch-all handler for exceptions that aren't JarvisError
    subclasses. It logs the full exception and returns a safe error message.

    Args:
        request: The FastAPI request object.
        exc: The unexpected exception.

    Returns:
        JSONResponse with 500 status and generic error message.
    """
    logger.exception(
        "Unexpected error handling %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
    )

    response_body = {
        "error": "InternalError",
        "code": "INTERNAL_ERROR",
        "detail": "An unexpected error occurred. Please try again later.",
    }

    return JSONResponse(
        status_code=500,
        content=response_body,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all JARVIS exception handlers with a FastAPI app.

    This function should be called once when setting up the FastAPI app
    to ensure all JARVIS errors are handled consistently.

    Args:
        app: The FastAPI application instance.

    Example:
        app = FastAPI()
        register_exception_handlers(app)
    """
    # Register specific handlers first (most specific to least specific)
    # Note: type: ignore is needed due to Starlette's overly strict typing
    app.add_exception_handler(ValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(iMessageAccessError, imessage_access_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ModelLoadError, model_load_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ResourceError, resource_error_handler)  # type: ignore[arg-type]

    # Register the base JarvisError handler (catches all JarvisError subclasses)
    app.add_exception_handler(JarvisError, jarvis_error_handler)  # type: ignore[arg-type]

    # Register timeout handler for asyncio.TimeoutError
    app.add_exception_handler(TimeoutError, timeout_error_handler)  # type: ignore[arg-type]

    # Register a generic exception handler for unexpected errors
    app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore[arg-type]

    logger.debug("Registered JARVIS exception handlers")


# Export all public symbols
__all__ = [
    "register_exception_handlers",
    "jarvis_error_handler",
    "validation_error_handler",
    "imessage_access_error_handler",
    "model_load_error_handler",
    "resource_error_handler",
    "timeout_error_handler",
    "generic_exception_handler",
    "get_status_code_for_error",
    "build_error_response",
    "ERROR_CODE_STATUS_CODES",
]
