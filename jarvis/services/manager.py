"""JARVIS Service Manager.

Provides unified management for all JARVIS services with dependency resolution,
health monitoring, and lifecycle management.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import TracebackType

from .api import FastAPIService
from .base import Service, ServiceError
from .embedding import EmbeddingService
from .ner import NERService
from .socket import SocketService

logger = logging.getLogger(__name__)


class ServiceDependencyError(ServiceError):
    """Error when service dependencies are not satisfied."""

    pass


class ServiceManager:
    """Unified service manager for all JARVIS services.

    Handles service lifecycle, dependencies, health monitoring, and coordination.
    Services are started in dependency order and stopped in reverse order.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = project_root
        self.services: dict[str, Service] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._initialized = False

    def _initialize_services(self) -> None:
        """Initialize all service instances."""
        if self._initialized:
            return

        jarvis_home = Path.home() / ".jarvis"
        venvs_dir = jarvis_home / "venvs"

        # Main venv (FastAPI + Socket)
        main_venv = self.project_root / ".venv"

        # Service-specific venvs (with legacy fallbacks)
        embedding_venv = venvs_dir / "embedding"
        legacy_embedding_dir = jarvis_home / "mlx-embed-service"
        if not embedding_venv.exists() and legacy_embedding_dir.exists():
            embedding_venv = legacy_embedding_dir

        ner_venv = venvs_dir / "ner"
        legacy_ner_venv = jarvis_home / "ner_venv"
        if not ner_venv.exists() and legacy_ner_venv.exists():
            ner_venv = legacy_ner_venv

        self.services = {
            "embedding": EmbeddingService(
                venv_path=embedding_venv,
                service_dir=embedding_venv,
            ),
            "ner": NERService(
                venv_path=ner_venv,
                script_path=self.project_root / "scripts" / "ner_server.py",
            ),
            "socket": SocketService(
                venv_path=main_venv,
            ),
            "api": FastAPIService(
                venv_path=main_venv,
            ),
        }

        self._initialized = True
        logger.info("Service manager initialized")

    def start_all(self, timeout: float = 120.0) -> None:
        """Start all services in dependency order.

        Args:
            timeout: Maximum seconds to wait for each service to become healthy.
                     Must be between 1 and 600 seconds.

        Raises:
            ValueError: If timeout is not in valid range.
        """
        if timeout <= 0 or timeout > 600:
            raise ValueError(f"timeout must be between 1 and 600 seconds, got {timeout}")

        with self._lock:
            self._initialize_services()

            # Dependency order: embedding/ner can start in parallel, then socket, then api
            start_order = ["embedding", "ner", "socket", "api"]

            logger.info("Starting all services...")

            for service_name in start_order:
                if service_name not in self.services:
                    continue

                service = self.services[service_name]

                # Check dependencies
                for dep in service.config.dependencies:
                    if dep not in self.services or not self.services[dep].is_running:
                        raise ServiceDependencyError(
                            f"Service {service_name} depends on {dep}, but {dep} is not running"
                        )

                try:
                    service.start()

                    # Wait for service to be ready
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if service.health_check():
                            logger.info(f"Service {service_name} is healthy")
                            break
                        time.sleep(1.0)
                    else:
                        raise ServiceError(
                            f"Service {service_name} failed to become healthy within {timeout}s"
                        )

                except Exception as e:
                    if service.config.optional:
                        logger.warning(f"Optional service {service_name} failed to start: {e}")
                        continue

                    logger.error(f"Failed to start service {service_name}: {e}")
                    # Stop any services we started
                    self._stop_started_services(start_order[: start_order.index(service_name) + 1])
                    raise

            logger.info("All services started successfully")

    def stop_all(self) -> None:
        """Stop all services in reverse dependency order."""
        with self._lock:
            if not self._initialized:
                return

            # Reverse order: api, socket, then embedding/ner
            stop_order = ["api", "socket", "ner", "embedding"]

            logger.info("Stopping all services...")

            for service_name in stop_order:
                if service_name in self.services:
                    try:
                        self.services[service_name].stop()
                    except Exception as e:
                        logger.error(f"Error stopping service {service_name}: {e}")

            logger.info("All services stopped")

    def start_service(self, service_name: str) -> None:
        """Start a specific service."""
        with self._lock:
            self._initialize_services()

            if service_name not in self.services:
                raise ServiceError(f"Unknown service: {service_name}")

            service = self.services[service_name]

            # Check dependencies
            for dep in service.config.dependencies:
                if dep not in self.services or not self.services[dep].is_running:
                    raise ServiceDependencyError(
                        f"Service {service_name} depends on {dep}, but {dep} is not running"
                    )

            service.start()

    def stop_service(self, service_name: str) -> None:
        """Stop a specific service."""
        with self._lock:
            if not self._initialized:
                return

            if service_name not in self.services:
                raise ServiceError(f"Unknown service: {service_name}")

            self.services[service_name].stop()

    def restart_service(self, service_name: str) -> None:
        """Restart a specific service."""
        with self._lock:
            self.stop_service(service_name)
            self.start_service(service_name)

    def get_status(self) -> dict[str, dict[str, object]]:
        """Get status of all services."""
        with self._lock:
            self._initialize_services()

            return {name: service.get_info() for name, service in self.services.items()}

    def get_service_status(self, service_name: str) -> dict[str, object]:
        """Get status of a specific service."""
        with self._lock:
            self._initialize_services()

            if service_name not in self.services:
                raise ServiceError(f"Unknown service: {service_name}")

            return self.services[service_name].get_info()

    def health_check_all(self) -> dict[str, bool]:
        """Check health of all running services."""
        with self._lock:
            if not self._initialized:
                return {}

            return {
                name: service.health_check()
                for name, service in self.services.items()
                if service.is_running
            }

    def _stop_started_services(self, service_names: list[str]) -> None:
        """Stop a list of services (used for cleanup on failure)."""
        for name in reversed(service_names):
            if name in self.services:
                try:
                    self.services[name].stop()
                except Exception as e:
                    logger.error(f"Error stopping {name} during cleanup: {e}")

    def __enter__(self) -> ServiceManager:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - ensure services are stopped."""
        self.stop_all()
