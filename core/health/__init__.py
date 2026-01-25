"""System health monitoring (Workstreams 6-7).

Provides graceful degradation and circuit breaker functionality.
"""

from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
)
from .degradation import (
    GracefulDegradationController,
    get_degradation_controller,
    reset_degradation_controller,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitOpenError",
    "CircuitState",
    "GracefulDegradationController",
    "get_degradation_controller",
    "reset_degradation_controller",
]
