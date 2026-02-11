"""Resilient HTTP client with automatic retry, circuit breaker, and fallback.

Wraps requests with resilience patterns for reliable API communication.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.health.circuit import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategy for handling failures."""

    FAIL = "fail"  # Raise exception
    RETURN_NONE = "return_none"  # Return None
    RETURN_EMPTY = "return_empty"  # Return empty dict/list
    USE_CACHE = "use_cache"  # Return cached value
    USE_DEFAULT = "use_default"  # Return default value


@dataclass
class ResilientClientConfig:
    """Configuration for resilient client."""

    # Retry configuration
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Timeout configuration
    connect_timeout: float = 5.0
    read_timeout: float = 30.0

    # Circuit breaker configuration
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0

    # Cache configuration
    cache_enabled: bool = True
    cache_ttl_seconds: float = 60.0

    # Fallback configuration
    fallback_strategy: FallbackStrategy = FallbackStrategy.FAIL
    fallback_default: Any = None

    # Request configuration
    user_agent: str = "JARVIS-ResilientClient/1.0"


@dataclass
class RequestContext:
    """Context for a request with metadata."""

    method: str
    url: str
    attempts: int = 0
    start_time: float = field(default_factory=time.time)
    last_error: Exception | None = None


class ResilientClient:
    """HTTP client with resilience patterns.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker pattern
    - Request/response caching
    - Configurable fallback strategies
    - Timeout handling

    Example:
        >>> config = ResilientClientConfig(max_retries=3)
        >>> client = ResilientClient("http://localhost:8742", config)
        >>>
        >>> # Simple GET with automatic retry
        >>> response = client.get("/health")
        >>>
        >>> # POST with fallback
        >>> result = client.post("/generate", json={"prompt": "hello"},
        ...                      fallback_strategy=FallbackStrategy.RETURN_EMPTY)
    """

    def __init__(
        self,
        base_url: str,
        config: ResilientClientConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize resilient client.

        Args:
            base_url: Base URL for all requests
            config: Client configuration
            session: Optional requests session to use
        """
        self._base_url = base_url.rstrip("/")
        self._config = config or ResilientClientConfig()
        self._session = session or self._create_session()
        self._circuit = CircuitBreaker(
            name=f"client_{base_url}",
            config=CircuitBreakerConfig(
                failure_threshold=self._config.circuit_failure_threshold,
                recovery_timeout_seconds=self._config.circuit_recovery_timeout,
            ),
        )
        self._cache: dict[str, tuple[Any, float]] = {}  # url -> (data, timestamp)
        self._cache_lock = False  # Simple lock for cache access

    def _create_session(self) -> requests.Session:
        """Create configured requests session."""
        session = requests.Session()

        # Mount retry adapter
        retry_strategy = Retry(
            total=0,  # We handle retries manually for more control
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(
            {
                "User-Agent": self._config.user_agent,
                "Accept": "application/json",
            }
        )

        return session

    def _get_cache_key(self, method: str, url: str, **kwargs: Any) -> str:
        """Generate cache key for request."""
        return f"{method}:{url}"

    def _get_from_cache(self, key: str) -> Any | None:
        """Get cached response if valid."""
        if not self._config.cache_enabled:
            return None

        if key not in self._cache:
            return None

        data, timestamp = self._cache[key]
        if time.time() - timestamp > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None

        return data

    def _set_cache(self, key: str, data: Any) -> None:
        """Cache response data."""
        if not self._config.cache_enabled:
            return

        self._cache[key] = (data, time.time())

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self._config.base_delay * (self._config.exponential_base**attempt)
        delay = min(delay, self._config.max_delay)

        if self._config.jitter:
            # Add randomness to prevent thundering herd
            delay = delay * (0.5 + random.random())

        return delay

    def _execute_request(
        self,
        context: RequestContext,
        **request_kwargs: Any,
    ) -> requests.Response:
        """Execute single request with circuit breaker."""
        if not self._circuit.can_execute():
            raise CircuitOpenError(f"Circuit open for {self._base_url}")

        url = f"{self._base_url}{context.url}"

        try:
            response = self._session.request(
                method=context.method,
                url=url,
                timeout=(self._config.connect_timeout, self._config.read_timeout),
                **request_kwargs,
            )

            # Record success for 2xx/3xx responses
            if response.status_code < 400:
                self._circuit.record_success()
            else:
                # 4xx client errors don't trigger circuit breaker
                # 5xx server errors do
                if response.status_code >= 500:
                    self._circuit.record_failure()
                response.raise_for_status()

            return response

        except requests.exceptions.Timeout:
            self._circuit.record_failure()
            raise
        except requests.exceptions.ConnectionError:
            self._circuit.record_failure()
            raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                self._circuit.record_failure()
            raise
        except Exception:
            self._circuit.record_failure()
            raise

    def _request_with_retry(
        self,
        method: str,
        url: str,
        fallback_strategy: FallbackStrategy | None = None,
        use_cache: bool = False,
        **request_kwargs: Any,
    ) -> Any:
        """Execute request with retry logic.

        Args:
            method: HTTP method
            url: URL path (appended to base_url)
            fallback_strategy: Override default fallback strategy
            use_cache: Whether to use cache for this request
            **request_kwargs: Additional arguments for requests

        Returns:
            Response data (JSON parsed if applicable)
        """
        context = RequestContext(method=method, url=url)
        strategy = fallback_strategy or self._config.fallback_strategy

        # Check cache for GET requests
        cache_key = self._get_cache_key(method, url, **request_kwargs)
        if use_cache and method.upper() == "GET":
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {url}")
                return cached

        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            context.attempts = attempt

            try:
                response = self._execute_request(context, **request_kwargs)
                data = None

                # Parse response
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = response.json()
                else:
                    data = response.text

                # Cache successful GET requests
                if use_cache and method.upper() == "GET":
                    self._set_cache(cache_key, data)

                return data

            except CircuitOpenError as e:
                # Circuit open - don't retry, use fallback
                last_error = e
                logger.warning(f"Circuit open for {url}, skipping retries")
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                context.last_error = e

                if attempt < self._config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self._config.max_retries + 1} attempts: {e}"
                    )

            except requests.exceptions.HTTPError as e:
                last_error = e
                context.last_error = e

                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    logger.error(f"Client error {e.response.status_code}: {e}")
                    break

                # Retry server errors (5xx)
                if attempt < self._config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Server error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Server error after retries: {e}")

            except Exception as e:
                last_error = e
                context.last_error = e
                logger.exception(f"Unexpected error in request: {e}")
                break

        # All retries exhausted - use fallback
        return self._handle_fallback(strategy, url, last_error)

    def _handle_fallback(
        self,
        strategy: FallbackStrategy,
        url: str,
        error: Exception | None,
    ) -> Any:
        """Handle request failure with fallback strategy."""
        logger.warning(f"Using fallback {strategy.value} for {url}")

        if strategy == FallbackStrategy.FAIL:
            if error:
                raise error
            raise RuntimeError(f"Request failed: {url}")

        elif strategy == FallbackStrategy.RETURN_NONE:
            return None

        elif strategy == FallbackStrategy.RETURN_EMPTY:
            return {}

        elif strategy == FallbackStrategy.USE_CACHE:
            # Try to return stale cache data
            return None  # Already checked above

        elif strategy == FallbackStrategy.USE_DEFAULT:
            return self._config.fallback_default

        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")

    # Convenience methods

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        fallback_strategy: FallbackStrategy | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute GET request."""
        return self._request_with_retry(
            "GET",
            url,
            params=params,
            fallback_strategy=fallback_strategy,
            use_cache=use_cache,
            **kwargs,
        )

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any = None,
        fallback_strategy: FallbackStrategy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute POST request."""
        return self._request_with_retry(
            "POST",
            url,
            json=json,
            data=data,
            fallback_strategy=fallback_strategy,
            **kwargs,
        )

    def put(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        fallback_strategy: FallbackStrategy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute PUT request."""
        return self._request_with_retry(
            "PUT",
            url,
            json=json,
            fallback_strategy=fallback_strategy,
            **kwargs,
        )

    def delete(
        self,
        url: str,
        fallback_strategy: FallbackStrategy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute DELETE request."""
        return self._request_with_retry(
            "DELETE",
            url,
            fallback_strategy=fallback_strategy,
            **kwargs,
        )

    def get_circuit_state(self) -> dict[str, Any]:
        """Get circuit breaker state for monitoring."""
        return {
            "state": self._circuit.state.value,
            "stats": {
                "total_executions": self._circuit.stats.total_executions,
                "total_successes": self._circuit.stats.total_successes,
                "total_failures": self._circuit.stats.total_failures,
                "failure_count": self._circuit.stats.failure_count,
            },
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()

    def close(self) -> None:
        """Close client and release resources."""
        self._session.close()
