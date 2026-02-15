"""NER Client - Connect to the NER service via Unix socket.

This client communicates with the spaCy NER server running in a separate
Python environment (to avoid dependency conflicts with MLX).

Usage:
    from jarvis.nlp.ner_client import get_entities, is_service_running

    if is_service_running():
        entities = get_entities("My mom Sarah works at Apple in San Francisco")
        # [{"text": "Sarah", "start": 7, "end": 12, "label": "PERSON"}, ...]
    else:
        # Fall back to regex-based entity detection
        pass
"""

import json
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.config import get_config

logger = logging.getLogger(__name__)

# Default socket path
SOCKET_PATH = Path(os.getenv("JARVIS_NER_SOCKET", str(Path.home() / ".jarvis" / "jarvis-ner.sock")))
PID_FILE = Path.home() / ".jarvis" / "ner_server.pid"


# Connection timeouts (from config, with fallback defaults)
def _get_connect_timeout() -> float:
    return get_config().ner.connect_timeout


def _get_read_timeout() -> float:
    return get_config().ner.read_timeout


@dataclass
class Entity:
    """A named entity extracted from text."""

    text: str
    start: int
    end: int
    label: str

    def is_person(self) -> bool:
        """Check if this is a person entity."""
        return self.label == "PERSON"

    def is_location(self) -> bool:
        """Check if this is a location entity."""
        return self.label in ("GPE", "LOC")

    def is_organization(self) -> bool:
        """Check if this is an organization entity."""
        return self.label == "ORG"


_auto_start_attempted = False

# Cached socket connection for reuse (Issue 1: avoid connect/disconnect per request)
_cached_socket: socket.socket | None = None
_socket_last_used: float = 0.0
_SOCKET_REUSE_TIMEOUT = 30.0  # seconds before closing idle socket

# Cached service-running state (Issue 2: avoid redundant check before each request)
_service_running_cache: bool | None = None
_service_running_checked_at: float = 0.0
_SERVICE_RUNNING_TTL = 5.0  # seconds to cache the "running" state


def _check_socket_responsive() -> bool:
    """Check if the NER socket exists and accepts connections."""
    if not SOCKET_PATH.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(_get_connect_timeout())
        sock.connect(str(SOCKET_PATH))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ensure_service_running() -> bool:
    """Auto-start the NER service if not running.

    Checks once per process. If the service is down and the NER venv exists,
    starts scripts/ner_server.py in background and waits up to 5s.

    Returns:
        True if service is running after this call.
    """
    global _auto_start_attempted

    if _check_socket_responsive():
        return True

    if _auto_start_attempted:
        return False
    _auto_start_attempted = True

    ner_venv = Path.home() / ".jarvis" / "ner_venv"
    python_path = ner_venv / "bin" / "python"
    if not python_path.exists():
        logger.info("NER venv not found at %s, skipping auto-start", python_path)
        return False

    server_script = Path(__file__).parent.parent.parent / "scripts" / "ner_server.py"
    if not server_script.exists():
        logger.warning("NER server script not found: %s", server_script)
        return False

    logger.info("Auto-starting NER service...")
    try:
        log_path = Path.home() / ".jarvis" / "ner_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")  # noqa: SIM115
        subprocess.Popen(
            [str(python_path), str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            start_new_session=True,
        )
    except Exception as e:
        logger.warning("Failed to start NER service: %s", e)
        return False

    # Wait up to 5s for socket to appear and become responsive
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        time.sleep(0.3)
        if _check_socket_responsive():
            logger.info("NER service started successfully")
            return True

    logger.warning("NER service did not start within 5s")
    return False


def is_service_running() -> bool:
    """Check if the NER service is running, auto-starting if needed.

    Caches the result for a short TTL to avoid redundant socket checks
    before every request.

    Returns:
        True if the service is running and accepting connections.
    """
    global _service_running_cache, _service_running_checked_at

    now = time.monotonic()
    if (
        _service_running_cache is not None
        and (now - _service_running_checked_at) < _SERVICE_RUNNING_TTL
    ):
        return _service_running_cache

    result = _ensure_service_running()
    _service_running_cache = result
    _service_running_checked_at = now
    return result


def get_pid() -> int | None:
    """Get the PID of the running NER server.

    Returns:
        PID if server is running, None otherwise.
    """
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _get_cached_socket() -> socket.socket:
    """Get or create a cached socket connection.

    Reuses an existing connection if it was used within the timeout window.
    Creates a new connection otherwise.

    Returns:
        Connected socket ready for use.

    Raises:
        ConnectionError: If connection fails.
    """
    global _cached_socket, _socket_last_used

    now = time.monotonic()

    # Reuse existing socket if still within timeout window
    if _cached_socket is not None and (now - _socket_last_used) < _SOCKET_REUSE_TIMEOUT:
        return _cached_socket

    # Close stale socket if any
    _close_cached_socket()

    # Create fresh connection
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(_get_connect_timeout())
    sock.connect(str(SOCKET_PATH))
    sock.settimeout(_get_read_timeout())

    _cached_socket = sock
    _socket_last_used = now
    return sock


def _close_cached_socket() -> None:
    """Close the cached socket if open."""
    global _cached_socket, _socket_last_used

    if _cached_socket is not None:
        try:
            _cached_socket.close()
        except OSError:
            pass
        _cached_socket = None
        _socket_last_used = 0.0


def _send_on_socket(sock: socket.socket, request: dict[str, Any]) -> dict[str, Any]:
    """Send a request and receive response on the given socket.

    Args:
        sock: Connected socket.
        request: Request dictionary to send.

    Returns:
        Response dictionary from server.
    """
    global _socket_last_used

    # Send length-prefixed JSON
    data = json.dumps(request).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "big") + data)

    # Receive response (length-prefixed)
    # Loop until all 4 bytes are received (sock.recv may return fewer)
    length_bytes = b""
    while len(length_bytes) < 4:
        chunk = sock.recv(4 - len(length_bytes))
        if not chunk:
            raise ConnectionError("Connection closed by server")
        length_bytes += chunk
    length = int.from_bytes(length_bytes, "big")

    response_data = b""
    while len(response_data) < length:
        chunk = sock.recv(min(4096, length - len(response_data)))
        if not chunk:
            break
        response_data += chunk

    _socket_last_used = time.monotonic()
    return json.loads(response_data.decode("utf-8"))


def _send_request(request: dict[str, Any]) -> dict[str, Any]:
    """Send a request to the NER server.

    Reuses cached socket connections. On broken pipe or connection reset,
    retries once with a fresh socket.

    Args:
        request: Request dictionary to send.

    Returns:
        Response dictionary from server.

    Raises:
        ConnectionError: If connection fails after retry.
        TimeoutError: If request times out.
    """
    try:
        sock = _get_cached_socket()
        return _send_on_socket(sock, request)
    except (BrokenPipeError, ConnectionResetError, ConnectionError, OSError):
        # Socket went stale - close and retry with a fresh one
        _close_cached_socket()
        sock = _get_cached_socket()
        return _send_on_socket(sock, request)


def get_entities(text: str) -> list[Entity]:
    """Extract named entities from text using the NER service.

    Falls back gracefully if service is unavailable.

    Args:
        text: Text to extract entities from.

    Returns:
        List of Entity objects. Empty list if service unavailable or error.
    """
    if not text or not is_service_running():
        return []

    try:
        response = _send_request({"text": text})

        if "error" in response:
            logger.warning("NER service error: %s", response["error"])
            return []

        entities = response.get("entities", [])
        return [
            Entity(
                text=e["text"],
                start=e["start"],
                end=e["end"],
                label=e["label"],
            )
            for e in entities
        ]

    except (ConnectionError, TimeoutError) as e:
        logger.warning("NER service connection failed: %s", e)
        return []
    except Exception as e:
        logger.warning("NER client error: %s", e)
        return []


def get_entities_batch(texts: list[str]) -> list[list[Entity]]:
    """Extract named entities from multiple texts.

    More efficient than calling get_entities() in a loop.

    Args:
        texts: List of texts to extract entities from.

    Returns:
        List of entity lists, one per input text.
    """
    if not texts or not is_service_running():
        return [[] for _ in texts]

    try:
        response = _send_request({"texts": texts})

        if "error" in response:
            logger.warning("NER service error: %s", response["error"])
            return [[] for _ in texts]

        results = response.get("results", [])
        return [
            [
                Entity(
                    text=e["text"],
                    start=e["start"],
                    end=e["end"],
                    label=e["label"],
                )
                for e in entities
            ]
            for entities in results
        ]

    except (ConnectionError, TimeoutError) as e:
        logger.warning("NER service connection failed: %s", e)
        return [[] for _ in texts]
    except Exception as e:
        logger.warning("NER client error: %s", e)
        return [[] for _ in texts]


def extract_person_names(text: str) -> list[str]:
    """Extract person names from text.

    Convenience function for relationship detection.

    Args:
        text: Text to extract names from.

    Returns:
        List of person name strings.
    """
    entities = get_entities(text)
    return [e.text for e in entities if e.is_person()]


def extract_locations(text: str) -> list[str]:
    """Extract location names from text.

    Args:
        text: Text to extract locations from.

    Returns:
        List of location name strings.
    """
    entities = get_entities(text)
    return [e.text for e in entities if e.is_location()]


def extract_organizations(text: str) -> list[str]:
    """Extract organization names from text.

    Args:
        text: Text to extract organizations from.

    Returns:
        List of organization name strings.
    """
    entities = get_entities(text)
    return [e.text for e in entities if e.is_organization()]


def get_syntactic_features(text: str) -> list[float]:
    """Extract syntactic features for dialogue act classification.

    Returns 14 features:
    - Directive indicators (5): imperative, you+modal, request verbs,
      starts_modal, directive_question
    - Commissive indicators (4): i_will, promise_verb,
      first_person_count, agreement
    - General syntactic (5): modal_count, verb_count,
      second_person_count, has_negation, is_interrogative

    Args:
        text: Text to extract features from.

    Returns:
        List of 14 float features. Returns zeros if service unavailable.
    """
    if not text or not is_service_running():
        return [0.0] * 14

    try:
        response = _send_request({"type": "syntactic", "text": text})

        if "error" in response:
            logger.warning("NER service error: %s", response["error"])
            return [0.0] * 14

        return response.get("features", [0.0] * 14)

    except (ConnectionError, TimeoutError) as e:
        logger.warning("NER service connection failed: %s", e)
        return [0.0] * 14
    except Exception as e:
        logger.warning("NER client error: %s", e)
        return [0.0] * 14


def get_syntactic_features_batch(texts: list[str]) -> list[list[float]]:
    """Extract syntactic features from multiple texts.

    More efficient than calling get_syntactic_features() in a loop.

    Args:
        texts: List of texts to extract features from.

    Returns:
        List of 14-element feature lists, one per input text.
    """
    if not texts or not is_service_running():
        return [[0.0] * 14 for _ in texts]

    try:
        response = _send_request({"type": "syntactic_batch", "texts": texts})

        if "error" in response:
            logger.warning("NER service error: %s", response["error"])
            return [[0.0] * 14 for _ in texts]

        return response.get("results", [[0.0] * 14 for _ in texts])

    except (ConnectionError, TimeoutError) as e:
        logger.warning("NER service connection failed: %s", e)
        return [[0.0] * 14 for _ in texts]
    except Exception as e:
        logger.warning("NER client error: %s", e)
        return [[0.0] * 14 for _ in texts]


__all__ = [
    "Entity",
    "is_service_running",
    "get_pid",
    "get_entities",
    "get_entities_batch",
    "extract_person_names",
    "extract_locations",
    "extract_organizations",
    "get_syntactic_features",
    "get_syntactic_features_batch",
]
