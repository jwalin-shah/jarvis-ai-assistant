"""System health monitoring (Workstreams 6-7).

Provides graceful degradation, circuit breaker, permission monitoring,
and schema detection functionality.
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
from .permissions import (
    TCCPermissionMonitor,
    get_permission_monitor,
    reset_permission_monitor,
)
from .schema import (
    ChatDBSchemaDetector,
    get_schema_detector,
    reset_schema_detector,
)

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitOpenError",
    "CircuitState",
    # Degradation controller
    "GracefulDegradationController",
    "get_degradation_controller",
    "reset_degradation_controller",
    # Permission monitor
    "TCCPermissionMonitor",
    "get_permission_monitor",
    "reset_permission_monitor",
    # Schema detector
    "ChatDBSchemaDetector",
    "get_schema_detector",
    "reset_schema_detector",
]
