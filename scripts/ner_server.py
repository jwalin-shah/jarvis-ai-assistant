#!/usr/bin/env python3
"""spaCy NER server - Unix socket server for named entity recognition.

Communicates via length-prefixed JSON over a Unix domain socket.
Designed to run inside the NER venv (~/.jarvis/ner_venv/) which has spaCy
installed separately from the MLX environment to avoid dependency conflicts.

Protocol (matches jarvis/nlp/ner_client.py):
    Request:  4-byte big-endian length + JSON payload
    Response: 4-byte big-endian length + JSON payload

Request types:
    {"text": "..."}                    -> {"entities": [...]}
    {"texts": ["...", "..."]}          -> {"results": [[...], [...]]}
    {"type": "syntactic", "text": "."} -> {"features": [14 floats]}
    {"type": "syntactic_batch", "texts": [...]} -> {"results": [[14 floats], ...]}

Usage:
    ~/.jarvis/ner_venv/bin/python scripts/ner_server.py
"""

import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
from pathlib import Path

SOCKET_PATH = Path(os.getenv("JARVIS_NER_SOCKET", str(Path.home() / ".jarvis" / "jarvis-ner.sock")))
PID_FILE = Path.home() / ".jarvis" / "ner_server.pid"
LOG_FILE = Path.home() / ".jarvis" / "ner_server.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Loaded once at startup
nlp = None

# Syntactic feature extraction helpers
MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
REQUEST_VERBS = {"please", "help", "send", "tell", "give", "show", "let", "bring", "take", "get"}
PROMISE_VERBS = {"promise", "swear", "guarantee", "commit", "pledge", "vow"}
NEGATION_WORDS = {"not", "n't", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere"}


def load_model():
    """Load spaCy model once at startup."""
    global nlp
    import spacy

    logger.info("Loading spaCy model en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Model loaded successfully")


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text."""
    doc = nlp(text)
    return [
        {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        for ent in doc.ents
    ]


def extract_syntactic_features(text: str) -> list[float]:
    """Extract 14 syntactic features for dialogue act classification.

    Features (14 total):
    - Directive indicators (5): imperative, you+modal, request verbs,
      starts_modal, directive_question
    - Commissive indicators (4): i_will, promise_verb,
      first_person_count, agreement
    - General syntactic (5): modal_count, verb_count,
      second_person_count, has_negation, is_interrogative
    """
    doc = nlp(text)
    tokens = list(doc)
    lower_tokens = [t.text.lower() for t in tokens]

    # Directive indicators
    imperative = 1.0 if tokens and tokens[0].pos_ == "VERB" and tokens[0].morph.get("Mood") == ["Imp"] else 0.0
    if not imperative and tokens and tokens[0].pos_ == "VERB" and tokens[0].tag_ == "VB":
        imperative = 0.5  # Base form verb at start is soft imperative signal

    you_modal = 0.0
    for i, t in enumerate(lower_tokens):
        if t in ("you", "u") and i + 1 < len(lower_tokens) and lower_tokens[i + 1] in MODAL_VERBS:
            you_modal = 1.0
            break

    request_verb = 1.0 if any(t in REQUEST_VERBS for t in lower_tokens) else 0.0

    starts_modal = 1.0 if lower_tokens and lower_tokens[0] in MODAL_VERBS else 0.0

    directive_question = 0.0
    if text.strip().endswith("?") and any(t in MODAL_VERBS for t in lower_tokens):
        if any(t in ("you", "u") for t in lower_tokens):
            directive_question = 1.0

    # Commissive indicators
    i_will = 0.0
    for i, t in enumerate(lower_tokens):
        if t == "i" and i + 1 < len(lower_tokens) and lower_tokens[i + 1] in ("will", "'ll", "ll"):
            i_will = 1.0
            break

    promise_verb = 1.0 if any(t in PROMISE_VERBS for t in lower_tokens) else 0.0

    first_person_count = sum(1 for t in lower_tokens if t in ("i", "me", "my", "mine", "myself")) / max(len(lower_tokens), 1)

    agreement_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "agreed", "definitely", "absolutely"}
    agreement = 1.0 if any(t in agreement_words for t in lower_tokens) else 0.0

    # General syntactic
    modal_count = sum(1 for t in lower_tokens if t in MODAL_VERBS) / max(len(lower_tokens), 1)
    verb_count = sum(1 for t in tokens if t.pos_ == "VERB") / max(len(tokens), 1)
    second_person_count = sum(1 for t in lower_tokens if t in ("you", "your", "yours", "yourself")) / max(len(lower_tokens), 1)
    has_negation = 1.0 if any(t in NEGATION_WORDS for t in lower_tokens) else 0.0
    is_interrogative = 1.0 if text.strip().endswith("?") else 0.0

    return [
        imperative, you_modal, request_verb, starts_modal, directive_question,
        i_will, promise_verb, first_person_count, agreement,
        modal_count, verb_count, second_person_count, has_negation, is_interrogative,
    ]


def handle_request(data: dict) -> dict:
    """Route request to appropriate handler."""
    req_type = data.get("type", "ner")

    if req_type == "syntactic":
        text = data.get("text", "")
        if not text:
            return {"features": [0.0] * 14}
        features = extract_syntactic_features(text)
        return {"features": features}

    if req_type == "syntactic_batch":
        texts = data.get("texts", [])
        results = [extract_syntactic_features(t) if t else [0.0] * 14 for t in texts]
        return {"results": results}

    # NER: single text
    if "text" in data and "texts" not in data:
        text = data["text"]
        if not text:
            return {"entities": []}
        entities = extract_entities(text)
        return {"entities": entities}

    # NER: batch
    if "texts" in data:
        texts = data["texts"]
        results = [extract_entities(t) if t else [] for t in texts]
        return {"results": results}

    return {"error": f"Unknown request format: {list(data.keys())}"}


def recv_exactly(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf


def handle_client(conn: socket.socket, addr):
    """Handle a single client connection."""
    try:
        # Read length prefix (4 bytes, big-endian)
        length_bytes = recv_exactly(conn, 4)
        length = struct.unpack(">I", length_bytes)[0]

        if length > 10_000_000:  # 10MB safety limit
            response = {"error": "Request too large"}
        else:
            request_data = recv_exactly(conn, length)
            request = json.loads(request_data.decode("utf-8"))
            response = handle_request(request)

        # Send response
        response_bytes = json.dumps(response).encode("utf-8")
        conn.sendall(struct.pack(">I", len(response_bytes)) + response_bytes)

    except ConnectionError:
        pass
    except Exception as e:
        logger.error("Error handling client: %s", e)
        try:
            error_response = json.dumps({"error": str(e)}).encode("utf-8")
            conn.sendall(struct.pack(">I", len(error_response)) + error_response)
        except Exception:
            pass
    finally:
        conn.close()


def cleanup(signum=None, frame=None):
    """Clean up socket and PID file on shutdown."""
    logger.info("Shutting down NER server...")
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    if PID_FILE.exists():
        PID_FILE.unlink()
    sys.exit(0)


def main():
    # Register signal handlers
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Clean up stale socket
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Load model
    load_model()

    # Start Unix socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCKET_PATH))
    server.listen(16)
    # Restrict socket permissions to owner only
    os.chmod(str(SOCKET_PATH), 0o600)

    logger.info("NER server listening on %s (PID: %d)", SOCKET_PATH, os.getpid())

    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()
    except Exception:
        pass
    finally:
        server.close()
        cleanup()


if __name__ == "__main__":
    main()
