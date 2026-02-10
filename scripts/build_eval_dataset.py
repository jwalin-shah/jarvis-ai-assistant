#!/usr/bin/env python3
"""Build evaluation dataset from archive data and real iMessages.

Sources:
- Archive: trigger_auto_labeled.jsonl, trigger_commitment_corrected.jsonl,
  trigger_needs_review.jsonl, trigger_new_batch_3000.jsonl
- Real iMessages: paired incoming/reply messages from chat.db

Output: evals/data/pipeline_eval.jsonl

Usage:
    uv run python scripts/build_eval_dataset.py
    uv run python scripts/build_eval_dataset.py --skip-imessage  # archive only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

def _setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("build_eval_dataset.log")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


logger = _setup_logging()

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "archive" / "data"
OUTPUT_PATH = ROOT / "evals" / "data" / "pipeline_eval.jsonl"

# Old 5-category → new 6-category + mobilization mapping
LABEL_MAP: dict[str, dict] = {
    "question": {
        "category": "question",
        "mobilization": "HIGH",
        "confidence": "auto",
    },
    "commitment": {
        "category": "request",
        "mobilization": "HIGH",
        "confidence": "auto",
    },
}

# Labels that need text-based heuristics
HEURISTIC_LABELS = {"reaction", "social", "statement"}

# iMessage reaction prefixes (tapbacks)
REACTION_PREFIXES = (
    "Liked",
    "Loved",
    "Disliked",
    "Laughed at",
    "Emphasized",
    "Questioned",
)


def map_label(text: str, old_label: str) -> dict:
    """Map old 5-category label to new 6-category + mobilization.

    Returns dict with category, mobilization, label_confidence.
    """
    # Direct mappings
    if old_label in LABEL_MAP:
        m = LABEL_MAP[old_label]
        return {
            "category": m["category"],
            "mobilization": m["mobilization"],
            "label_confidence": m["confidence"],
        }

    stripped = text.strip()

    # Reaction: check if iMessage tapback
    if old_label == "reaction":
        if any(stripped.startswith(p) for p in REACTION_PREFIXES):
            return {
                "category": "acknowledge",
                "mobilization": "NONE",
                "label_confidence": "auto",
            }
        # Emotional reaction
        return {
            "category": "emotion",
            "mobilization": "MEDIUM",
            "label_confidence": "needs_review",
        }

    # Social: ambiguous between statement and emotion
    if old_label == "social":
        # Short social messages are acknowledgments
        if len(stripped.split()) <= 3:
            return {
                "category": "acknowledge",
                "mobilization": "NONE",
                "label_confidence": "needs_review",
            }
        return {
            "category": "statement",
            "mobilization": "LOW",
            "label_confidence": "needs_review",
        }

    # Statement
    if old_label == "statement":
        if len(stripped.split()) < 5:
            return {
                "category": "acknowledge",
                "mobilization": "NONE",
                "label_confidence": "auto",
            }
        return {
            "category": "statement",
            "mobilization": "LOW",
            "label_confidence": "auto",
        }

    # Fallback
    return {
        "category": "statement",
        "mobilization": "LOW",
        "label_confidence": "needs_review",
    }


def load_archive_data() -> list[dict]:
    """Load and deduplicate archive data from all trigger files."""
    examples: list[dict] = []
    seen_texts: set[str] = set()

    files = [
        ("trigger_auto_labeled.jsonl", "label"),
        ("trigger_commitment_corrected.jsonl", "label"),
        ("trigger_needs_review.jsonl", "auto_label"),
        ("trigger_new_batch_3000.jsonl", "auto_label"),
    ]

    for filename, label_key in files:
        path = ARCHIVE_DIR / filename
        if not path.exists():
            logger.warning("Missing archive file: %s", path)
            continue

        count = 0
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry.get("text", "").strip()
            old_label = entry.get(label_key, "")

            if not text or not old_label:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)

            mapping = map_label(text, old_label)
            examples.append({
                "text": text,
                "old_label": old_label,
                **mapping,
                "source": "archive",
                "source_file": filename,
            })
            count += 1

        logger.info("Loaded %d examples from %s", count, filename)

    return examples


def load_imessage_data(limit: int = 200) -> list[dict]:
    """Load real iMessage pairs (incoming → your reply within 1 hour)."""
    try:
        from integrations.imessage.reader import ChatDBReader
    except ImportError:
        logger.error("Cannot import ChatDBReader - skipping iMessage data")
        return []

    reader = ChatDBReader()
    if not reader.check_access():
        logger.warning("No iMessage access - skipping iMessage data")
        return []

    examples: list[dict] = []
    conversations = reader.get_conversations(limit=50)
    logger.info("Scanning %d conversations for message pairs...", len(conversations))

    for conv_idx, conv in enumerate(conversations):
        if len(examples) >= limit:
            break

        messages = reader.get_messages(conv.chat_id, limit=500)
        if not messages:
            continue

        # Find incoming messages followed by your reply within 1 hour
        for i, msg in enumerate(messages):
            if len(examples) >= limit:
                break
            if msg.is_from_me or not msg.text or not msg.text.strip():
                continue

            # Look for your reply in subsequent messages
            reply_text = None
            for j in range(i + 1, min(i + 10, len(messages))):
                next_msg = messages[j]
                if not next_msg.is_from_me:
                    continue
                if not next_msg.text or not next_msg.text.strip():
                    continue
                # Check within 1 hour
                if next_msg.date - msg.date > timedelta(hours=1):
                    break
                reply_text = next_msg.text.strip()
                break

            # Build thread context (up to 5 preceding messages)
            thread: list[str] = []
            for k in range(max(0, i - 5), i):
                prev = messages[k]
                if prev.text and prev.text.strip():
                    prefix = "me: " if prev.is_from_me else f"{prev.sender_name or prev.sender}: "
                    thread.append(prefix + prev.text.strip())

            text = msg.text.strip()
            examples.append({
                "text": text,
                "old_label": None,
                "category": None,  # needs manual labeling
                "mobilization": None,
                "label_confidence": "needs_review",
                "source": "imessage",
                "should_reply": reply_text is not None,
                "actual_reply": reply_text,
                "contact_name": msg.sender_name or msg.sender,
                "thread": thread,
                "chat_id": conv.chat_id,
            })

        if (conv_idx + 1) % 10 == 0:
            logger.info(
                "  Scanned %d/%d conversations, %d examples so far",
                conv_idx + 1, len(conversations), len(examples),
            )

    logger.info("Loaded %d iMessage examples", len(examples))
    return examples


def build_dataset(skip_imessage: bool = False) -> None:
    """Build the full evaluation dataset."""
    logger.info("Building evaluation dataset...")

    # Load archive
    archive = load_archive_data()
    logger.info("Archive: %d total examples", len(archive))

    # Count by confidence
    auto_count = sum(1 for e in archive if e.get("label_confidence") == "auto")
    review_count = sum(1 for e in archive if e.get("label_confidence") == "needs_review")
    logger.info("  auto-mapped: %d, needs_review: %d", auto_count, review_count)

    # Load iMessages
    imessage: list[dict] = []
    if not skip_imessage:
        imessage = load_imessage_data()
    else:
        logger.info("Skipping iMessage data (--skip-imessage)")

    # Combine and assign IDs
    all_examples: list[dict] = []
    for i, ex in enumerate(archive):
        ex["id"] = f"archive_{i}"
        ex.setdefault("should_reply", None)
        ex.setdefault("actual_reply", None)
        ex.setdefault("thread", [])
        all_examples.append(ex)

    for i, ex in enumerate(imessage):
        ex["id"] = f"imessage_{i}"
        all_examples.append(ex)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Wrote %d examples to %s", len(all_examples), OUTPUT_PATH)

    # Summary
    categories = {}
    for ex in all_examples:
        cat = ex.get("category") or "unlabeled"
        categories[cat] = categories.get(cat, 0) + 1
    logger.info("Category distribution: %s", json.dumps(categories, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pipeline evaluation dataset")
    parser.add_argument(
        "--skip-imessage", action="store_true",
        help="Skip iMessage data (archive only)",
    )
    args = parser.parse_args()
    build_dataset(skip_imessage=args.skip_imessage)


if __name__ == "__main__":
    main()
