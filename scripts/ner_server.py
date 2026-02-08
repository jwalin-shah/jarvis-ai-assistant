#!/usr/bin/env python3
"""NER Server - Unix socket server for spaCy NER.

This server runs in a separate Python environment (ner_venv) to avoid
dependency conflicts with MLX and other JARVIS components.

Communication is via Unix socket for low latency (~5ms for typical messages).

Protocol:
- Request: JSON with "text" field (or list for batch)
- Response: JSON with "entities" list of (text, start, end, label) tuples

Usage:
    # Start server (from ner_venv)
    ~/.jarvis/ner_venv/bin/python scripts/ner_server.py

    # Or via CLI
    jarvis ner start
"""

import json
import logging
import os
import signal
import socket
import sys
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ner_server")

# Default paths (must match ner_client.py)
SOCKET_PATH = Path(os.getenv("JARVIS_NER_SOCKET", str(Path.home() / ".jarvis" / "jarvis-ner.sock")))
PID_FILE = Path.home() / ".jarvis" / "ner_server.pid"
SPACY_MODEL = os.getenv("JARVIS_SPACY_MODEL", "en_core_web_sm")

# Global state
nlp = None
server_socket = None
shutdown_flag = threading.Event()


def load_model():
    """Load the spaCy model."""
    global nlp
    try:
        import spacy

        logger.info("Loading spaCy model: %s", SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load spaCy model: %s", e)
        return False


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text.

    Returns:
        List of entity dicts with keys: text, start, end, label
    """
    if not nlp or not text:
        return []

    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
        }
        for ent in doc.ents
    ]


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract entities from multiple texts using pipe for efficiency."""
    if not nlp or not texts:
        return [[] for _ in texts]

    results = []
    for doc in nlp.pipe(texts, batch_size=32):
        entities = [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
        ]
        results.append(entities)
    return results


def extract_syntactic_features(text: str) -> list[float]:
    """Extract 14 syntactic features for dialogue act classification.

    Features:
    - Directive indicators (5): imperative, you+modal, request verbs, starts_modal, directive_question
    - Commissive indicators (4): i_will, promise_verb, first_person_count, agreement
    - General syntactic (5): modal_count, verb_count, second_person_count, has_negation, is_interrogative

    Returns:
        14-element list of float features
    """
    if not nlp or not text:
        return [0.0] * 14

    import re

    doc = nlp(text)
    features = []

    # Directive indicators (5)
    # 1. Imperative verb (VB tag at sentence start)
    features.append(1.0 if doc and doc[0].tag_ == "VB" else 0.0)

    # 2. "you/u" + modal pattern
    features.append(1.0 if re.search(r"\b(you|u)\s+(can|could|should|will|would)\b", text, re.I) else 0.0)

    # 3. Request verbs
    request_verbs = {"ask", "tell", "send", "call", "let", "give", "show", "help"}
    features.append(1.0 if any(tok.lemma_ in request_verbs for tok in doc) else 0.0)

    # 4. Starts with modal (polite request)
    modal_verbs = {"can", "could", "will", "would", "should", "must", "may", "might"}
    features.append(1.0 if doc and doc[0].lemma_ in modal_verbs else 0.0)

    # 5. Directive question (ends with ? and has you/your)
    features.append(1.0 if text.endswith("?") and any(t.text.lower() in {"you", "your"} for t in doc) else 0.0)

    # Commissive indicators (4)
    # 6. First person future (I'll, I will, we'll, we will)
    features.append(1.0 if re.search(r"\b(i'?ll|i will|we'?ll|we will)\b", text, re.I) else 0.0)

    # 7. Promise verbs
    promise_verbs = {"promise", "commit", "agree", "guarantee", "swear"}
    features.append(1.0 if any(tok.lemma_ in promise_verbs for tok in doc) else 0.0)

    # 8. First person pronoun count
    first_person = sum(1 for t in doc if t.text.lower() in {"i", "me", "my", "we", "us", "our"})
    features.append(float(first_person))

    # 9. Agreement words
    agreement_words = {"ok", "okay", "sure", "yeah", "yes", "yep", "yup", "alright"}
    features.append(1.0 if any(t.text.lower() in agreement_words for t in doc) else 0.0)

    # General syntactic (5)
    # 10. Modal verb count
    modal_count = sum(1 for t in doc if t.lemma_ in modal_verbs)
    features.append(float(modal_count))

    # 11. Verb count
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    features.append(float(verb_count))

    # 12. Second person count
    second_person = sum(1 for t in doc if t.text.lower() in {"you", "your", "yours"})
    features.append(float(second_person))

    # 13. Has negation
    features.append(1.0 if any(t.dep_ == "neg" for t in doc) else 0.0)

    # 14. Is interrogative (wh-word or ends with ?)
    wh_tags = {"WDT", "WP", "WP$", "WRB"}
    features.append(1.0 if (doc and doc[0].tag_ in wh_tags) or text.endswith("?") else 0.0)

    return features


def extract_syntactic_features_batch(texts: list[str]) -> list[list[float]]:
    """Extract syntactic features from multiple texts using pipe for efficiency."""
    if not nlp or not texts:
        return [[0.0] * 14 for _ in texts]

    import re

    logger.info(f"Processing batch of {len(texts)} texts for syntactic features")

    results = []
    modal_verbs = {"can", "could", "will", "would", "should", "must", "may", "might"}
    request_verbs = {"ask", "tell", "send", "call", "let", "give", "show", "help"}
    promise_verbs = {"promise", "commit", "agree", "guarantee", "swear"}
    agreement_words = {"ok", "okay", "sure", "yeah", "yes", "yep", "yup", "alright"}
    wh_tags = {"WDT", "WP", "WP$", "WRB"}

    # Use larger batch size for better throughput (256 is optimal for most SpaCy models)
    for doc, text in zip(nlp.pipe(texts, batch_size=256), texts):
        features = []

        # Directive indicators (5)
        features.append(1.0 if doc and doc[0].tag_ == "VB" else 0.0)
        features.append(1.0 if re.search(r"\b(you|u)\s+(can|could|should|will|would)\b", text, re.I) else 0.0)
        features.append(1.0 if any(tok.lemma_ in request_verbs for tok in doc) else 0.0)
        features.append(1.0 if doc and doc[0].lemma_ in modal_verbs else 0.0)
        features.append(1.0 if text.endswith("?") and any(t.text.lower() in {"you", "your"} for t in doc) else 0.0)

        # Commissive indicators (4)
        features.append(1.0 if re.search(r"\b(i'?ll|i will|we'?ll|we will)\b", text, re.I) else 0.0)
        features.append(1.0 if any(tok.lemma_ in promise_verbs for tok in doc) else 0.0)
        first_person = sum(1 for t in doc if t.text.lower() in {"i", "me", "my", "we", "us", "our"})
        features.append(float(first_person))
        features.append(1.0 if any(t.text.lower() in agreement_words for t in doc) else 0.0)

        # General syntactic (5)
        modal_count = sum(1 for t in doc if t.lemma_ in modal_verbs)
        features.append(float(modal_count))
        verb_count = sum(1 for t in doc if t.pos_ == "VERB")
        features.append(float(verb_count))
        second_person = sum(1 for t in doc if t.text.lower() in {"you", "your", "yours"})
        features.append(float(second_person))
        features.append(1.0 if any(t.dep_ == "neg" for t in doc) else 0.0)
        features.append(1.0 if (doc and doc[0].tag_ in wh_tags) or text.endswith("?") else 0.0)

        results.append(features)

    return results


def handle_request(data: bytes) -> bytes:
    """Handle a single request.

    Args:
        data: JSON bytes with request

    Returns:
        JSON bytes with response
    """
    try:
        request = json.loads(data.decode("utf-8"))
        request_type = request.get("type", "entities")

        # Handle syntactic features batch request
        if request_type == "syntactic_batch":
            texts = request.get("texts", [])
            if not isinstance(texts, list):
                return json.dumps({"error": "texts must be a list"}).encode()
            features = extract_syntactic_features_batch(texts)
            return json.dumps({"results": features}).encode()

        # Handle syntactic features single request
        if request_type == "syntactic":
            text = request.get("text", "")
            if not text:
                return json.dumps({"features": [0.0] * 14}).encode()
            features = extract_syntactic_features(text)
            return json.dumps({"features": features}).encode()

        # Handle batch entity request
        if "texts" in request:
            texts = request["texts"]
            if not isinstance(texts, list):
                return json.dumps({"error": "texts must be a list"}).encode()
            entities = extract_entities_batch(texts)
            return json.dumps({"results": entities}).encode()

        # Handle single entity request
        text = request.get("text", "")
        if not text:
            return json.dumps({"entities": []}).encode()

        entities = extract_entities(text)
        return json.dumps({"entities": entities}).encode()

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}).encode()
    except Exception as e:
        logger.error("Error handling request: %s", e)
        return json.dumps({"error": str(e)}).encode()


def handle_client(conn: socket.socket, addr):
    """Handle a client connection."""
    try:
        # Read request (length-prefixed)
        length_bytes = conn.recv(4)
        if not length_bytes:
            return
        length = int.from_bytes(length_bytes, "big")

        # Read data
        data = b""
        while len(data) < length:
            chunk = conn.recv(min(4096, length - len(data)))
            if not chunk:
                break
            data += chunk

        # Process and respond
        response = handle_request(data)
        conn.sendall(len(response).to_bytes(4, "big") + response)

    except Exception as e:
        logger.error("Error handling client: %s", e)
    finally:
        conn.close()


def cleanup():
    """Clean up resources."""
    global server_socket
    if server_socket:
        server_socket.close()
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    if PID_FILE.exists():
        PID_FILE.unlink()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received signal %d, shutting down...", signum)
    shutdown_flag.set()
    cleanup()
    sys.exit(0)


def run_server():
    """Run the NER server."""
    global server_socket

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load model
    if not load_model():
        sys.exit(1)

    # Remove stale socket
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    # Create socket
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(str(SOCKET_PATH))
    server_socket.listen(5)
    server_socket.settimeout(1.0)  # For shutdown check

    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    logger.info("NER server listening on %s (PID: %d)", SOCKET_PATH, os.getpid())

    try:
        while not shutdown_flag.is_set():
            try:
                conn, addr = server_socket.accept()
                # Handle in thread for concurrent requests
                thread = threading.Thread(target=handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except TimeoutError:
                continue
    except Exception as e:
        logger.error("Server error: %s", e)
    finally:
        cleanup()


if __name__ == "__main__":
    run_server()
