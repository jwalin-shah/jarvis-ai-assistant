"""Base service classes for JARVIS services."""

from __future__ import annotations

import abc
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from jarvis.errors import ErrorCode, JarvisError

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceConfig:
    """Configuration for a service."""

    name: str
    venv_path: Path | None = None
    command: list[str] = field(default_factory=list)
    working_dir: Path | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    health_check_url: str | None = None
    health_check_socket: Path | None = None
    health_check_timeout: float = 5.0
    startup_timeout: float = 30.0
    stop_timeout: float = 10.0
    restart_on_failure: bool = True
    optional: bool = False
    dependencies: list[str] = field(default_factory=list)


class ServiceError(JarvisError):
    """Base error for service-related issues."""

    default_message = "Service error"
    default_code = ErrorCode.UNKNOWN


class ServiceStartError(ServiceError):
    """Error starting a service."""

    default_message = "Failed to start service"
    default_code = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str | None = None,
        *,
        service_name: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a service start error.

        Args:
            message: Human-readable error message.
            service_name: Name of the service that failed to start.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        super().__init__(message, code=code, details=details, cause=cause)


class ServiceStopError(ServiceError):
    """Error stopping a service."""

    default_message = "Failed to stop service"
    default_code = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str | None = None,
        *,
        service_name: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a service stop error.

        Args:
            message: Human-readable error message.
            service_name: Name of the service that failed to stop.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        super().__init__(message, code=code, details=details, cause=cause)


class ServiceHealthError(ServiceError):
    """Error with service health check."""

    default_message = "Service health check failed"
    default_code = ErrorCode.UNKNOWN


class Service(abc.ABC):
    """Abstract base class for all JARVIS services.

    Services are long-running processes that can be started, stopped,
    and monitored for health. Each service runs in its own environment
    and can have dependencies on other services.
    """

    def __init__(self, config: ServiceConfig) -> None:
        """Initialize the service.

        Args:
            config: Service configuration.
        """
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._status = ServiceStatus.STOPPED
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._health_check_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Service name."""
        return self.config.name

    @property
    def status(self) -> ServiceStatus:
        """Current service status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if service is currently running."""
        return self._status in (ServiceStatus.RUNNING, ServiceStatus.HEALTHY)

    @property
    def pid(self) -> int | None:
        """Get process ID if running."""
        return self._process.pid if self._process else None

    def start(self) -> None:
        """Start the service.

        Raises:
            ServiceStartError: If the service fails to start.
        """
        with self._lock:
            if self._status != ServiceStatus.STOPPED:
                logger.warning(
                    "Service %s is not stopped (status: %s)",
                    self.name,
                    self._status.value,
                )
                return

            self._status = ServiceStatus.STARTING
            logger.info("Starting service %s", self.name)

            try:
                self._start_process()
                self._start_time = time.time()
                self._status = ServiceStatus.RUNNING
                self._start_health_monitor()
                logger.info("Service %s started (PID: %s)", self.name, self.pid)
            except ServiceStartError:
                # Re-raise service errors as-is
                raise
            except Exception as e:
                self._status = ServiceStatus.FAILED
                logger.error("Failed to start service %s: %s", self.name, e, exc_info=True)
                raise ServiceStartError(
                    f"Failed to start {self.name}",
                    service_name=self.name,
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the service.

        Raises:
            ServiceStopError: If the service fails to stop.
        """
        with self._lock:
            if self._status == ServiceStatus.STOPPED:
                return

            self._status = ServiceStatus.STOPPING
            logger.info("Stopping service %s", self.name)

            try:
                self._shutdown_event.set()
                self._stop_process()
                self._stop_time = time.time()
                self._status = ServiceStatus.STOPPED
                logger.info("Service %s stopped", self.name)
            except ServiceStopError:
                # Re-raise service errors as-is
                raise
            except Exception as e:
                logger.error("Error stopping service %s: %s", self.name, e, exc_info=True)
                raise ServiceStopError(
                    f"Failed to stop {self.name}",
                    service_name=self.name,
                    cause=e,
                ) from e

    def restart(self) -> None:
        """Restart the service."""
        logger.info(f"Restarting service {self.name}")
        self.stop()
        self.start()

    def health_check(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise.

        Note:
            Errors during health check are logged but do not raise exceptions
            to allow graceful degradation.
        """
        try:
            return self._perform_health_check()
        except Exception as e:
            logger.debug("Health check failed for %s: %s", self.name, e, exc_info=True)
            return False

    def get_info(self) -> dict[str, Any]:
        """Get service information.

        Returns:
            Dictionary with service status, PID, uptime, and health information.
        """
        # For services that may already be running (like embedding service),
        # check health even if we don't have a process reference
        can_run_without_process = getattr(self, "_can_run_without_process", False)
        healthy = self.health_check() if self._process or can_run_without_process else False
        status = self._status
        if healthy and status == ServiceStatus.STOPPED:
            status = ServiceStatus.HEALTHY

        uptime: float | None = None
        if self._start_time:
            uptime = time.time() - self._start_time

        return {
            "name": self.name,
            "status": status.value,
            "pid": self.pid,
            "uptime": uptime,
            "healthy": healthy,
            "dependencies": self.config.dependencies,
        }

    def _start_process(self) -> None:
        """Start the actual process. Override in subclasses."""
        if not self.config.command:
            raise ServiceStartError(f"No command configured for service {self.name}")

        # Set up environment
        env = os.environ.copy()
        env.update(self.config.env_vars)

        # Add venv to PATH if specified
        if self.config.venv_path:
            venv_bin = self.config.venv_path / "bin"
            env["PATH"] = f"{venv_bin}:{env['PATH']}"

        # Start process with platform-specific settings
        # Use process_group (Python 3.11+) instead of deprecated preexec_fn
        popen_kwargs: dict[str, Any] = {
            "cwd": self.config.working_dir,
            "env": env,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if hasattr(os, "setsid"):  # Unix/Linux/macOS
            popen_kwargs["process_group"] = 0

        self._process = subprocess.Popen(self.config.command, **popen_kwargs)

    def _stop_process(self) -> None:
        """Stop the process gracefully.

        Uses platform-specific process group termination on Unix/Linux/macOS
        for cleaner shutdown. Falls back to direct process termination on Windows.
        """
        if not self._process:
            return

        try:
            # Try graceful shutdown first
            # On Unix: kill entire process group (includes child processes)
            # On Windows: terminate single process
            if hasattr(os, "killpg") and hasattr(os, "getpgid") and hasattr(signal, "SIGTERM"):
                # Unix/Linux/macOS: send SIGTERM to process group
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already exited
            else:
                # Windows or fallback: terminate process directly
                self._process.terminate()

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Force killing service %s", self.name)
                if hasattr(os, "killpg") and hasattr(os, "getpgid") and hasattr(signal, "SIGKILL"):
                    # Unix/Linux/macOS: send SIGKILL to process group
                    try:
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Process already exited
                else:
                    # Windows or fallback: kill process directly
                    self._process.kill()
                self._process.wait()

        finally:
            self._process = None

    def _perform_health_check(self) -> bool:
        """Perform the actual health check. Override in subclasses."""
        # Default: check if process is still running
        if not self._process:
            return False

        return self._process.poll() is None

    def _start_health_monitor(self) -> None:
        """Start background health monitoring."""
        if not self.config.health_check_url and not self.config.health_check_socket:
            return

        self._health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            name=f"{self.name}-health",
            daemon=True,
        )
        self._health_check_thread.start()

    def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set() and self.is_running:
            try:
                healthy = self.health_check()
                new_status = ServiceStatus.HEALTHY if healthy else ServiceStatus.UNHEALTHY

                with self._lock:
                    if self._status in (
                        ServiceStatus.RUNNING,
                        ServiceStatus.HEALTHY,
                        ServiceStatus.UNHEALTHY,
                    ):
                        if self._status != new_status:
                            self._status = new_status
                            logger.info(
                                "Service %s health changed to %s",
                                self.name,
                                new_status.value,
                            )

            except Exception as e:
                logger.error("Health monitor error for %s: %s", self.name, e, exc_info=True)

            # Wait before next check
            self._shutdown_event.wait(10.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, status={self._status.value})"
