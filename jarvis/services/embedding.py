"""MLX embedding service implementation."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
from pathlib import Path

from .base import Service, ServiceConfig, ServiceStatus

logger = logging.getLogger(__name__)


class EmbeddingService(Service):
    """MLX embedding service for GPU-accelerated embeddings."""

    def __init__(
        self,
        port: int = 8766,
        socket_path: Path | None = None,
        venv_path: Path | None = None,
        service_dir: Path | None = None,
    ) -> None:
        if socket_path is None:
            socket_path = Path("/tmp/jarvis-embed.sock")
        if service_dir is None:
            service_dir = Path.home() / ".jarvis" / "venvs" / "embedding"
        if venv_path is None:
            venv_path = service_dir

        # Use venv-based approach with legacy fallback
        config = ServiceConfig(
            name="embedding",
            venv_path=venv_path,
            command=["python", "server.py"],
            working_dir=service_dir,
            health_check_socket=socket_path,
            startup_timeout=60.0,  # MLX models take time to load
            optional=True,
            dependencies=[],  # No dependencies
        )
        super().__init__(config)
        self.port = port
        self.socket_path = socket_path
        self._can_run_without_process = True  # Can be healthy without process reference
        self.service_dir = service_dir
        self._log_handles: list = []  # Track open log file handles for cleanup

    def _kill_orphan_processes(self) -> None:
        """Kill any orphaned embedding server processes."""
        import subprocess
        import time

        # Also check for processes using the socket file
        try:
            lsof_result = subprocess.run(
                ["lsof", "/tmp/jarvis-embed.sock"],
                capture_output=True,
                text=True,
            )
            if lsof_result.returncode == 0:
                # Parse lsof output to get PIDs (skip header line)
                for line in lsof_result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            logger.info("Killing process %d using embed socket", pid)
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(0.3)
                            try:
                                os.kill(pid, 0)
                                os.kill(pid, signal.SIGKILL)
                                logger.info("Force killed PID %d", pid)
                            except ProcessLookupError:
                                pass
                        except (ValueError, ProcessLookupError, PermissionError):
                            pass
        except Exception as e:
            logger.debug("Error checking socket users: %s", e)

        # Find orphaned embedding server processes by checking the service directory.
        # Use the specific service_dir path to avoid killing unrelated server.py processes.
        try:
            service_dir_str = str(self.service_dir)
            legacy_dir_str = str(Path.home() / ".jarvis" / "mlx-embed-service")

            result = subprocess.run(
                ["pgrep", "-f", "server.py"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return  # No matching processes

            pids = [int(pid.strip()) for pid in result.stdout.strip().split("\n") if pid.strip()]

            # Skip our own PID and parent PID to avoid self-kill
            my_pid = os.getpid()
            my_ppid = os.getppid()

            for pid in pids:
                if pid in (my_pid, my_ppid):
                    continue
                try:
                    # macOS: use ps to get full command line for PID validation
                    ps_result = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "command="],
                        capture_output=True,
                        text=True,
                    )
                    cmdline = ps_result.stdout.strip()

                    # Strict validation: only kill if the command line contains
                    # our specific service directory path (not just "embedding")
                    is_our_service = service_dir_str in cmdline or legacy_dir_str in cmdline
                    if not is_our_service:
                        continue

                    logger.info(
                        "Killing orphaned embedding server PID %d (cmd: %s)",
                        pid,
                        cmdline[:100],
                    )
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.3)
                    # Check if still alive and force kill
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        os.kill(pid, signal.SIGKILL)
                        logger.info("Force killed PID %d", pid)
                    except ProcessLookupError:
                        pass  # Already dead
                except (ProcessLookupError, PermissionError, ValueError):
                    pass  # Process already gone or not ours

        except Exception as e:
            logger.debug("Error checking for orphan processes: %s", e)

    def _cleanup_sockets(self) -> None:
        """Remove stale socket files."""
        socket_files = [
            Path("/tmp/jarvis-embed.sock"),
            Path("/tmp/jarvis-embed-minimal.sock"),
        ]
        for sock_file in socket_files:
            if sock_file.exists():
                try:
                    sock_file.unlink()
                    logger.debug("Removed stale socket %s", sock_file)
                except OSError as e:
                    logger.debug("Could not remove socket %s: %s", sock_file, e)

    def _close_log_handles(self) -> None:
        """Close any open log file handles."""
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._log_handles.clear()

    def _start_process(self) -> None:
        """Start the embedding service."""
        # First check if service is already running and healthy
        if self.health_check():
            # Service is already running, don't start another instance
            self._status = ServiceStatus.HEALTHY
            return

        # Clean up orphan processes and stale sockets before starting
        self._kill_orphan_processes()
        self._cleanup_sockets()

        # Close any leftover log handles from a previous start
        self._close_log_handles()

        # The embedding service uses its own venv structure
        # It needs to be run from the service directory with its python
        import subprocess

        venv_python = self.service_dir / "bin" / "python"
        server_script = self.service_dir / "server.py"

        # Redirect output to log file to avoid pipe buffer blocking
        log_file = self.service_dir / "server.log"

        if venv_python.exists() and server_script.exists():
            log_handle = open(log_file, "a")
            try:
                self._process = subprocess.Popen(
                    [str(venv_python), str(server_script)],
                    cwd=self.service_dir,
                    stdout=log_handle,
                    stderr=log_handle,
                    preexec_fn=None,
                )
            except Exception:
                log_handle.close()
                raise
            self._log_handles.append(log_handle)
            return

        # Fallback to legacy uv-run layout
        legacy_dir = Path.home() / ".jarvis" / "mlx-embed-service"
        legacy_server = legacy_dir / "server.py"
        if legacy_server.exists():
            legacy_log = legacy_dir / "server.log"
            legacy_log_handle = open(legacy_log, "a")
            try:
                self._process = subprocess.Popen(
                    ["uv", "run", "python", str(legacy_server)],
                    cwd=legacy_dir,
                    stdout=legacy_log_handle,
                    stderr=legacy_log_handle,
                    preexec_fn=None,
                )
            except Exception:
                legacy_log_handle.close()
                raise
            self._log_handles.append(legacy_log_handle)
            return

        raise RuntimeError(
            "Embedding service not found. Run scripts/migrate_mlx_embedding.py "
            "or install the MLX embedding service."
        )

    def _stop_process(self) -> None:
        """Stop the process and clean up log file handles."""
        try:
            super()._stop_process()
        finally:
            self._close_log_handles()

    def _perform_health_check(self) -> bool:
        """Check if embedding service is responding."""
        # First try socket health check
        if self.config.health_check_socket:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.settimeout(self.config.health_check_timeout)
                    sock.connect(str(self.config.health_check_socket))

                    # Send health check request (newline-delimited like the server expects)
                    health_request = {"jsonrpc": "2.0", "method": "health", "id": 1}
                    request_data = json.dumps(health_request).encode() + b"\n"
                    sock.sendall(request_data)

                    # Read response (newline-delimited)
                    response_data = sock.recv(4096)

                    if response_data:
                        response = json.loads(response_data.decode().strip())
                        result = response.get("result", {})
                        if isinstance(result, dict):
                            return result.get("status") == "healthy"
                        return False

                    return False
                finally:
                    sock.close()
            except Exception:
                # If socket check fails, fall back to process check
                pass

        # Fall back to checking if process is running
        return super()._perform_health_check()
