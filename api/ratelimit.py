"""Rate limiting configuration for JARVIS API.

Uses slowapi for rate limiting with configurable limits per endpoint type.
Supports per-client tracking with proper 429 responses.

Usage:
    from api.ratelimit import limiter, get_remote_address

    @app.get("/endpoint")
    @limiter.limit("60/minute")
    async def endpoint(request: Request):
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address as _get_remote_address

from jarvis.config import get_config

if TYPE_CHECKING:
    pass  # Type imports only

logger = logging.getLogger(__name__)

# Type for decorator functions
F = TypeVar("F", bound=Callable[..., Any])

# Config cache with 60-second TTL to avoid reading on every call
_config_cache: dict[str, tuple[Any, float]] = {}
_CONFIG_CACHE_TTL = 60.0  # seconds


def _get_cached_config_value(key: str, getter: Callable[[], Any], default: Any) -> Any:
    """Get config value with caching.

    Args:
        key: Cache key
        getter: Function to get the value if cache miss
        default: Default value if getter fails

    Returns:
        Config value
    """
    now = time.time()
    if key in _config_cache:
        value, timestamp = _config_cache[key]
        if now - timestamp < _CONFIG_CACHE_TTL:
            return value

    try:
        value = getter()
        _config_cache[key] = (value, now)
        return value
    except Exception:
        logger.debug("Config lookup failed for %s, using default", key)
        return default


def get_remote_address(request: Request) -> str:
    """Get client identifier for rate limiting.

    For local-first apps, we use a combination of IP and user-agent
    to differentiate between different local clients.

    Args:
        request: The FastAPI request object.

    Returns:
        String identifier for the client.
    """
    # Use slowapi's default for IP
    ip = _get_remote_address(request) or "unknown"

    # For localhost, add user-agent to differentiate clients
    if ip in ("127.0.0.1", "localhost", "::1"):
        user_agent = request.headers.get("user-agent", "unknown")
        # Hash the user agent for privacy
        return f"{ip}:{hash(user_agent) % 10000}"

    return ip


# Create the limiter instance
# Key function extracts client identifier from request
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    enabled=True,  # Can be toggled via config at runtime
)


# Rate limit constants for different endpoint types
RATE_LIMIT_GENERATION = "10/minute"  # For CPU-intensive generation endpoints
RATE_LIMIT_READ = "60/minute"  # For read-only endpoints
RATE_LIMIT_WRITE = "30/minute"  # For write endpoints


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Handle rate limit exceeded errors with proper 429 response.

    Args:
        request: The FastAPI request object.
        exc: The rate limit exception.

    Returns:
        JSON response with 429 status and retry-after header.
    """
    from fastapi.responses import JSONResponse

    # Extract retry-after from the exception if available
    retry_after = 60  # Default to 60 seconds
    if hasattr(exc, "detail") and exc.detail:
        # Try to parse the limit from the detail message
        try:
            # Detail format: "Rate limit exceeded: X per Y minute"
            parts = str(exc.detail).split()
            for i, part in enumerate(parts):
                if part.isdigit() and i + 1 < len(parts):
                    if "minute" in parts[i + 1]:
                        retry_after = 60
                    elif "second" in parts[i + 1]:
                        retry_after = int(part)
                    break
        except (ValueError, IndexError):
            pass

    logger.warning(
        "Rate limit exceeded for %s %s from %s",
        request.method,
        request.url.path,
        get_remote_address(request),
    )

    response_body = {
        "error": "RateLimitExceeded",
        "code": "RATE_LIMIT_EXCEEDED",
        "detail": "Too many requests. Please slow down.",
        "retry_after_seconds": retry_after,
    }

    return JSONResponse(
        status_code=429,
        content=response_body,
        headers={"Retry-After": str(retry_after)},
    )


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout to async endpoints.

    Args:
        timeout_seconds: Maximum time in seconds before timeout.

    Returns:
        Decorator function.

    Usage:
        @with_timeout(30.0)
        async def my_endpoint():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=408,
                    detail=f"Request timed out after {timeout_seconds} seconds",
                ) from None

        return wrapper  # type: ignore[return-value]

    return decorator


def run_in_threadpool(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Decorator to run a synchronous function in a thread pool.

    Use this for CPU-bound operations like model generation to avoid
    blocking the event loop.

    Args:
        func: Synchronous function to wrap.

    Returns:
        Async function that runs the original in a thread pool.

    Usage:
        @run_in_threadpool
        def cpu_intensive_work():
            ...

        async def endpoint():
            result = await cpu_intensive_work()
    """
    from starlette.concurrency import run_in_threadpool as starlette_threadpool

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await starlette_threadpool(func, *args, **kwargs)

    return wrapper


def get_timeout_generation() -> float:
    """Get generation timeout from config.

    Uses caching to avoid reading config file on every call.

    Returns:
        Timeout in seconds for generation operations.
    """
    return float(
        _get_cached_config_value(
            "timeout_generation",
            lambda: get_config().rate_limit.generation_timeout_seconds,
            30.0,  # Default fallback
        )
    )


def get_timeout_read() -> float:
    """Get read timeout from config.

    Uses caching to avoid reading config file on every call.

    Returns:
        Timeout in seconds for read operations.
    """
    return float(
        _get_cached_config_value(
            "timeout_read",
            lambda: get_config().rate_limit.read_timeout_seconds,
            10.0,  # Default fallback
        )
    )


# TIMEOUT_GENERATION and TIMEOUT_READ were removed (they froze at import time).
# Use get_timeout_generation() and get_timeout_read() at call sites instead.


# Export all public symbols
__all__ = [
    "limiter",
    "get_remote_address",
    "rate_limit_exceeded_handler",
    "with_timeout",
    "run_in_threadpool",
    "get_timeout_generation",
    "get_timeout_read",
    "RATE_LIMIT_GENERATION",
    "RATE_LIMIT_READ",
    "RATE_LIMIT_WRITE",
]
