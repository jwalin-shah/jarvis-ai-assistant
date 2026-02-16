"""Base classes and handlers for prefetch operations.

Provides a structured way to implement prediction handlers with built-in
validation, error handling, and performance tracking.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

# Use the prediction schema from jarvis.schemas.prediction if it exists
# For now, we'll assume a generic interface

logger = logging.getLogger(__name__)

class PrefetchHandler(ABC):
    """Abstract base class for all prefetch handlers.

    Handles common boilerplate:
    - Parameter validation
    - Error handling (graceful failure)
    - Performance timing
    - Result formatting
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__.replace("Handler", "").lower()

    @property
    @abstractmethod
    def required_params(self) -> list[str]:
        """List of parameter keys required for this handler."""
        pass

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Actual logic implementation for the prefetch handler."""
        pass

    def __call__(self, prediction: Any) -> dict[str, Any] | None:
        """Entry point for the handler. Called with a Prediction object."""
        # 1. Validate parameters
        params = getattr(prediction, "params", {})
        for param in self.required_params:
            if param not in params:
                return None

        # 2. Execute with timing and error handling
        from jarvis.utils.latency_tracker import track_latency

        try:
            with track_latency(f"prefetch.{self.name}"):
                result = self.execute(params)

            if result is None:
                return None

            # Ensure result has timing info for the cache
            if "prefetch_time" not in result:
                result["prefetch_time"] = time.time()

            return result
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.debug(f"{self.name} prefetch failed: {e}")
            return None
