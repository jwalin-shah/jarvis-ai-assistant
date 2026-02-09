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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default socket path
SOCKET_PATH = Path(os.getenv("JARVIS_NER_SOCKET", str(Path.home() / ".jarvis" / "jarvis-ner.sock")))
PID_FILE = Path.home() / ".jarvis" / "ner_server.pid"

# Connection timeout
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 10.0


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


def is_service_running() -> bool:
    """Check if the NER service is running.

    Returns:
        True if the service is running and accepting connections.
    """
    if not SOCKET_PATH.exists():
        return False

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect(str(SOCKET_PATH))
        return True
    except OSError:
        return False
    finally:
        sock.close()


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


def _send_request(request: dict[str, Any]) -> dict[str, Any]:
    """Send a request to the NER server.

    Args:
        request: Request dictionary to send.

    Returns:
        Response dictionary from server.

    Raises:
        ConnectionError: If connection fails.
        TimeoutError: If request times out.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(CONNECT_TIMEOUT)

    try:
        sock.connect(str(SOCKET_PATH))
        sock.settimeout(READ_TIMEOUT)

        # Send length-prefixed JSON
        data = json.dumps(request).encode("utf-8")
        sock.sendall(len(data).to_bytes(4, "big") + data)

        # Receive response (length-prefixed)
        length_bytes = sock.recv(4)
        if not length_bytes:
            raise ConnectionError("Connection closed by server")
        length = int.from_bytes(length_bytes, "big")

        response_data = b""
        while len(response_data) < length:
            chunk = sock.recv(min(4096, length - len(response_data)))
            if not chunk:
                break
            response_data += chunk

        return json.loads(response_data.decode("utf-8"))

    finally:
        sock.close()


def get_entities(text: str) -> list[Entity]:
    """Extract named entities from text using the NER service.

    Falls back gracefully if service is unavailable.

    Args:
        text: Text to extract entities from.

    Returns:
        List of Entity objects. Empty list if service unavailable or error.
    """
    if not text or not SOCKET_PATH.exists():
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
    if not texts or not SOCKET_PATH.exists():
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
    - Directive indicators (5): imperative, you+modal, request verbs, starts_modal, directive_question
    - Commissive indicators (4): i_will, promise_verb, first_person_count, agreement
    - General syntactic (5): modal_count, verb_count, second_person_count, has_negation, is_interrogative

    Args:
        text: Text to extract features from.

    Returns:
        List of 14 float features. Returns zeros if service unavailable.
    """
    if not text or not SOCKET_PATH.exists():
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
    if not texts or not SOCKET_PATH.exists():
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
