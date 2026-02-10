"""Resilient model server with retry and fallback logic.

Wraps the MLXGenerator with retry handling, health checks,
and graceful degradation for generation failures.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from contracts.models import GenerationRequest, GenerationResponse

if TYPE_CHECKING:
    from models.generator import MLXGenerator

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0  # seconds
FALLBACK_RESPONSE = "I'm having trouble generating a response right now. Please try again."


class ResilientModelServer:
    """Wraps MLXGenerator with retry logic and fallback responses.

    On generation failure:
    1. Retries up to max_retries times with delay
    2. Returns a graceful fallback response if all retries fail
    3. Tracks failure counts for health monitoring
    """

    def __init__(
        self,
        generator: MLXGenerator | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._generator = generator
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_requests = 0

    @property
    def generator(self) -> MLXGenerator:
        """Lazy-load the generator if not provided."""
        if self._generator is None:
            from models import get_generator

            self._generator = get_generator(skip_templates=True)
        return self._generator

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response with retry logic.

        Args:
            request: The generation request.

        Returns:
            GenerationResponse, either from the model or a fallback.
        """
        self._total_requests += 1
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                response = self.generator.generate(request)
                self._consecutive_failures = 0
                return response
            except Exception as e:
                last_error = e
                self._total_failures += 1
                self._consecutive_failures += 1
                logger.warning(
                    "Generation attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.error(
            "All %d generation attempts failed, returning fallback",
            1 + self._max_retries,
            exc_info=last_error,
        )
        return GenerationResponse(
            text=FALLBACK_RESPONSE,
            finish_reason="error",
            metadata={
                "error": str(last_error),
                "retries_exhausted": True,
                "consecutive_failures": self._consecutive_failures,
            },
        )

    def is_healthy(self) -> bool:
        """Check if the model server is in a healthy state.

        Returns False if there have been 5+ consecutive failures,
        indicating a persistent issue.
        """
        return self._consecutive_failures < 5

    def get_stats(self) -> dict[str, int]:
        """Get server statistics."""
        return {
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "consecutive_failures": self._consecutive_failures,
        }


_server: ResilientModelServer | None = None


def get_resilient_server() -> ResilientModelServer:
    """Get or create the singleton ResilientModelServer."""
    global _server
    if _server is None:
        _server = ResilientModelServer()
    return _server
