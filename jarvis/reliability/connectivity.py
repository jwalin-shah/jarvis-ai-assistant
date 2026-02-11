"""Connectivity monitoring for services and network.

Provides real-time monitoring of:
- Network connectivity
- API server health
- Socket server availability
- Model loading status
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ConnectivityState(Enum):
    """Overall connectivity state."""

    ONLINE = "online"  # All services available
    DEGRADED = "degraded"  # Some services unavailable
    OFFLINE = "offline"  # No connectivity


class ServiceStatus(Enum):
    """Status of an individual service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health information for a service."""

    name: str
    status: ServiceStatus
    latency_ms: float
    last_check: datetime
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY


@dataclass
class ConnectivitySnapshot:
    """Snapshot of overall connectivity."""

    state: ConnectivityState
    services: dict[str, ServiceHealth]
    timestamp: datetime
    overall_latency_ms: float

    def get_degraded_services(self) -> list[str]:
        """Get names of degraded services."""
        return [
            name
            for name, health in self.services.items()
            if health.status in (ServiceStatus.DEGRADED, ServiceStatus.UNAVAILABLE)
        ]


class ConnectivityMonitor:
    """Monitor connectivity to JARVIS services.

    Continuously checks health of:
    - HTTP API (localhost:8742)
    - WebSocket (localhost:8743)
    - Unix Socket (~/.jarvis/jarvis.sock)
    - Model availability

    Example:
        >>> monitor = ConnectivityMonitor()
        >>> monitor.on_state_change = lambda old, new: print(f"{old} -> {new}")
        >>> monitor.start()
        >>> # Later...
        >>> snapshot = monitor.get_snapshot()
        >>> print(f"State: {snapshot.state}")
    """

    DEFAULT_CHECK_INTERVAL = 30.0  # seconds
    DEFAULT_TIMEOUT = 5.0  # seconds

    def __init__(
        self,
        api_url: str = "http://localhost:8742",
        ws_url: str = "ws://localhost:8743",
        socket_path: str | None = None,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize connectivity monitor.

        Args:
            api_url: Base URL for HTTP API
            ws_url: URL for WebSocket
            socket_path: Path to Unix socket (None to skip)
            check_interval: Seconds between health checks
        """
        self._api_url = api_url.rstrip("/")
        self._ws_url = ws_url
        self._socket_path = socket_path or str(Path.home() / ".jarvis" / "jarvis.sock")
        self._check_interval = check_interval

        self._services: dict[str, ServiceHealth] = {}
        self._current_state = ConnectivityState.UNKNOWN
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: threading.Thread | None = None

        # Callbacks
        self.on_state_change: Callable[[ConnectivityState, ConnectivityState], None] | None = None
        self.on_service_change: Callable[[str, ServiceHealth, ServiceHealth], None] | None = None

        # History for trend analysis
        self._history: list[ConnectivitySnapshot] = []
        self._max_history = 100

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Connectivity monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self._check_interval + 1)
        logger.info("Connectivity monitor stopped")

    def get_snapshot(self) -> ConnectivitySnapshot:
        """Get current connectivity snapshot."""
        with self._lock:
            services = dict(self._services)
            state = self._current_state

        # Calculate overall latency
        latencies = [h.latency_ms for h in services.values() if h.latency_ms > 0]
        overall_latency = sum(latencies) / len(latencies) if latencies else 0

        return ConnectivitySnapshot(
            state=state,
            services=services,
            timestamp=datetime.utcnow(),
            overall_latency_ms=overall_latency,
        )

    def get_history(self, last_n: int = 10) -> list[ConnectivitySnapshot]:
        """Get recent connectivity history."""
        with self._lock:
            return self._history[-last_n:]

    def check_now(self) -> ConnectivitySnapshot:
        """Perform immediate connectivity check."""
        self._check_all_services()
        return self.get_snapshot()

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_all_services()
            except Exception as e:
                logger.error(f"Error in connectivity check: {e}")

            # Sleep with early exit
            for _ in range(int(self._check_interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _check_all_services(self) -> None:
        """Check all services and update state."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._check_api): "api",
                executor.submit(self._check_websocket): "websocket",
                executor.submit(self._check_socket): "socket",
                executor.submit(self._check_model): "model",
            }

            for future in concurrent.futures.as_completed(futures):
                service_name = futures[future]
                try:
                    health = future.result()
                    self._update_service(service_name, health)
                except Exception as e:
                    logger.error(f"Error checking {service_name}: {e}")

        self._update_overall_state()

    def _check_api(self) -> ServiceHealth:
        """Check HTTP API health."""
        start = time.time()
        try:
            response = requests.get(
                f"{self._api_url}/health",
                timeout=self.DEFAULT_TIMEOUT,
            )
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                status = ServiceStatus.HEALTHY
                if data.get("status") != "healthy":
                    status = ServiceStatus.DEGRADED

                return ServiceHealth(
                    name="api",
                    status=status,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    metadata={"response": data},
                )
            else:
                return ServiceHealth(
                    name="api",
                    status=ServiceStatus.UNAVAILABLE,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}",
                )
        except requests.exceptions.Timeout:
            return ServiceHealth(
                name="api",
                status=ServiceStatus.UNAVAILABLE,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error="Timeout",
            )
        except requests.exceptions.ConnectionError as e:
            return ServiceHealth(
                name="api",
                status=ServiceStatus.UNAVAILABLE,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error=f"Connection error: {e}",
            )
        except Exception as e:
            return ServiceHealth(
                name="api",
                status=ServiceStatus.UNKNOWN,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error=str(e),
            )

    def _check_websocket(self) -> ServiceHealth:
        """Check WebSocket availability."""
        start = time.time()
        try:
            import websocket

            ws = websocket.create_connection(
                self._ws_url.replace("ws://", "http://"),
                timeout=self.DEFAULT_TIMEOUT,
            )
            ws.close()
            latency = (time.time() - start) * 1000

            return ServiceHealth(
                name="websocket",
                status=ServiceStatus.HEALTHY,
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )
        except ImportError:
            return ServiceHealth(
                name="websocket",
                status=ServiceStatus.UNKNOWN,
                latency_ms=0,
                last_check=datetime.utcnow(),
                error="websocket-client not installed",
            )
        except Exception as e:
            return ServiceHealth(
                name="websocket",
                status=ServiceStatus.UNAVAILABLE,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error=str(e),
            )

    def _check_socket(self) -> ServiceHealth:
        """Check Unix socket availability."""
        start = time.time()
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.DEFAULT_TIMEOUT)
            sock.connect(self._socket_path)
            sock.close()
            latency = (time.time() - start) * 1000

            return ServiceHealth(
                name="socket",
                status=ServiceStatus.HEALTHY,
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )
        except FileNotFoundError:
            return ServiceHealth(
                name="socket",
                status=ServiceStatus.UNAVAILABLE,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error="Socket file not found",
            )
        except Exception as e:
            return ServiceHealth(
                name="socket",
                status=ServiceStatus.UNAVAILABLE,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error=str(e),
            )

    def _check_model(self) -> ServiceHealth:
        """Check model loading status."""
        start = time.time()
        try:
            response = requests.get(
                f"{self._api_url}/health",
                timeout=self.DEFAULT_TIMEOUT,
            )
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                model_loaded = data.get("model_loaded", False)

                return ServiceHealth(
                    name="model",
                    status=ServiceStatus.HEALTHY if model_loaded else ServiceStatus.DEGRADED,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    metadata={"loaded": model_loaded},
                )
            else:
                return ServiceHealth(
                    name="model",
                    status=ServiceStatus.UNKNOWN,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}",
                )
        except Exception as e:
            return ServiceHealth(
                name="model",
                status=ServiceStatus.UNKNOWN,
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow(),
                error=str(e),
            )

    def _update_service(self, name: str, new_health: ServiceHealth) -> None:
        """Update service health and trigger callbacks."""
        with self._lock:
            old_health = self._services.get(name)
            self._services[name] = new_health

        # Trigger callback if status changed
        if old_health and old_health.status != new_health.status:
            logger.info(f"Service {name}: {old_health.status.value} -> {new_health.status.value}")
            if self.on_service_change:
                self.on_service_change(name, old_health, new_health)

    def _update_overall_state(self) -> None:
        """Update overall connectivity state."""
        with self._lock:
            services = dict(self._services)

        # Determine state based on services
        healthy_count = sum(1 for h in services.values() if h.status == ServiceStatus.HEALTHY)
        unavailable_count = sum(
            1 for h in services.values() if h.status == ServiceStatus.UNAVAILABLE
        )
        total = len(services)

        if total == 0:
            new_state = ConnectivityState.UNKNOWN
        elif healthy_count == total:
            new_state = ConnectivityState.ONLINE
        elif unavailable_count == total:
            new_state = ConnectivityState.OFFLINE
        else:
            new_state = ConnectivityState.DEGRADED

        with self._lock:
            old_state = self._current_state
            self._current_state = new_state

            # Update history
            snapshot = self.get_snapshot()
            self._history.append(snapshot)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # Trigger callback if state changed
        if old_state != new_state:
            logger.info(f"Connectivity state: {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(old_state, new_state)


# Import Path for type hints
from pathlib import Path
