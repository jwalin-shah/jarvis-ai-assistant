#!/usr/bin/env python3
"""
Prepare training data for GLiNER-based fact extraction pipeline.

This script:
1. Loads real iMessage conversations
2. Runs GLiNER candidate extraction (zero-shot)
3. Creates labeled dataset for classifier training
4. Exports in GLiNER fine-tuning format (optional)

Usage:
    # Generate candidates from your messages
    python scripts/prepare_gliner_training.py --source imessage --output training_data.jsonl

    # Generate from synthetic test cases
    python scripts/prepare_gliner_training.py --source synthetic --output test_candidates.jsonl

Output format for classifier training:
    {
        "text": "I love Thai food",
        "candidate": "Thai food",
        "entity_type": "food_preference",
        "gliner_score": 0.85,
        "label": 1,  # 1 = valid fact, 0 = false positive
        "category": "preference"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("prepare_gliner_training.log")
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


@dataclass
class TrainingExample:
    """A training example for the fact classifier."""

    text: str  # Full message text
    candidate: str  # The extracted entity span
    entity_type: str  # GLiNER label
    gliner_score: float  # GLiNER confidence
    category: str  # Mapped category
    label: int = 0  # 1 = valid fact, 0 = false positive
    context: str = ""  # Surrounding messages
    reasoning: str = ""  # Why this label was assigned


# Synthetic test cases with labels for validation
SYNTHETIC_TEST_CASES: list[dict[str, Any]] = [
    # === VALID FACTS (label=1) ===

    # Preferences - food
    {"text": "I love Thai food, especially pad thai", "candidate": "Thai food", "entity_type": "food_preference", "label": 1, "category": "preference", "reasoning": "Clear preference with specific food"},
    {"text": "I can't stand cilantro, it tastes like soap", "candidate": "cilantro", "entity_type": "disliked_food", "label": 1, "category": "preference", "reasoning": "Clear dislike with specific reason"},
    {"text": "obsessed with this new ramen place", "candidate": "ramen place", "entity_type": "food_preference", "label": 1, "category": "preference", "reasoning": "Slang 'obsessed with' indicates strong preference"},
    {"text": "I'm allergic to peanuts so be careful", "candidate": "peanuts", "entity_type": "allergy", "label": 1, "category": "health", "reasoning": "Medical fact - allergy"},

    # Locations
    {"text": "I live in San Francisco now", "candidate": "San Francisco", "entity_type": "current_location", "label": 1, "category": "location", "reasoning": "Clear current location statement"},
    {"text": "moving to Austin next month", "candidate": "Austin", "entity_type": "future_location", "label": 1, "category": "location", "reasoning": "Future location with temporal marker"},
    {"text": "grew up in Ohio but left for college", "candidate": "Ohio", "entity_type": "past_location", "label": 1, "category": "location", "reasoning": "Past location with context"},

    # Work
    {"text": "I work at Google as a software engineer", "candidate": "Google", "entity_type": "employer", "label": 1, "category": "work", "reasoning": "Clear employer statement"},
    {"text": "just got a job at a startup downtown", "candidate": "startup", "entity_type": "employer", "label": 1, "category": "work", "reasoning": "Employment context clear"},

    # Relationships
    {"text": "My sister Sarah is visiting this weekend", "candidate": "Sarah", "entity_type": "family_member", "label": 1, "category": "relationship", "reasoning": "Family relationship with name"},
    {"text": "my mom's birthday is tomorrow", "candidate": "mom", "entity_type": "family_member", "label": 1, "category": "relationship", "reasoning": "Family relationship mentioned"},

    # === FALSE POSITIVES (label=0) ===

    # Vague/subjectless
    {"text": "yeah same, I love it", "candidate": "it", "entity_type": "thing_preference", "label": 0, "category": "preference", "reasoning": "Vague pronoun 'it' - unknown referent"},
    {"text": "me too! love that", "candidate": "that", "entity_type": "thing_preference", "label": 0, "category": "preference", "reasoning": "Vague pronoun 'that' - context dependent"},
    {"text": "it's great", "candidate": "it", "entity_type": "thing_preference", "label": 0, "category": "preference", "reasoning": "Vague pronoun, no specific referent"},
    {"text": "that thing is amazing", "candidate": "that thing", "entity_type": "thing_preference", "label": 0, "category": "preference", "reasoning": "Vague noun phrase"},

    # Too short/insufficient context
    {"text": "I like sf", "candidate": "sf", "entity_type": "current_location", "label": 0, "category": "location", "reasoning": "Abbreviation without clear context"},
    {"text": "love this", "candidate": "this", "entity_type": "thing_preference", "label": 0, "category": "preference", "reasoning": "Vague demonstrative"},

    # Bot/spam patterns
    {"text": "Your CVS Pharmacy prescription is ready", "candidate": "CVS Pharmacy", "entity_type": "employer", "label": 0, "category": "work", "reasoning": "Bot message - automated"},
    {"text": "Check out this job at Amazon", "candidate": "Amazon", "entity_type": "employer", "label": 0, "category": "work", "reasoning": "Spam/LinkedIn message"},

    # Misclassified
    {"text": "Dear John, I hope this finds you well", "candidate": "John", "entity_type": "friend_name", "label": 0, "category": "relationship", "reasoning": "Professional email, not personal fact"},

    # === EDGE CASES (ambiguous) ===

    {"text": "moving next week", "candidate": "next week", "entity_type": "date_reference", "label": 0, "category": "temporal", "reasoning": "Temporal marker but missing location (dropped subject)"},
    {"text": "same lol", "candidate": "", "entity_type": "", "label": 0, "category": "", "reasoning": "No extractable content"},
    {"text": "my friend is great", "candidate": "friend", "entity_type": "friend_name", "label": 0, "category": "relationship", "reasoning": "Generic, no specific name mentioned"},
]


def generate_synthetic_dataset(logger: logging.Logger) -> list[TrainingExample]:
    """Generate training examples from synthetic test cases."""
    from tqdm import tqdm
    examples = []
    for case in tqdm(SYNTHETIC_TEST_CASES, desc="Generating synthetic", unit="case"):
        # Skip empty cases
        if not case.get("candidate"):
            continue

        ex = TrainingExample(
            text=case["text"],
            candidate=case["candidate"],
            entity_type=case.get("entity_type", ""),
            gliner_score=case.get("gliner_score", 0.7),
            category=case["category"],
            label=case["label"],
            reasoning=case.get("reasoning", ""),
        )
        examples.append(ex)

    return examples


def load_imessage_samples(db_path: Path | None = None, limit: int = 100, logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    """Load sample messages from iMessage database.

    Returns messages that might contain personal facts.
    """
    if db_path is None:
        db_path = Path.home() / "Library" / "Messages" / "chat.db"

    if not db_path.exists():
        if logger:
            logger.warning(f"iMessage DB not found at {db_path}")
        return []

    messages = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Query for messages with potential fact content
        # Look for first-person statements with keywords
        cursor.execute(
            """
            SELECT m.text, m.date, c.display_name
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL
              AND LENGTH(m.text) > 10
              AND (
                LOWER(m.text) LIKE '%i love%'
                OR LOWER(m.text) LIKE '%i like%'
                OR LOWER(m.text) LIKE '%i hate%'
                OR LOWER(m.text) LIKE '%i work%'
                OR LOWER(m.text) LIKE '%my sister%'
                OR LOWER(m.text) LIKE '%my mom%'
                OR LOWER(m.text) LIKE '%live in%'
                OR LOWER(m.text) LIKE '%moving to%'
                OR LOWER(m.text) LIKE '%allergic%'
                OR LOWER(m.text) LIKE '%obsessed with%'
              )
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        )

        for row in cursor.fetchall():
            text, date, chat_name = row
            messages.append({
                "text": text,
                "date": date,
                "chat": chat_name,
                "source": "imessage",
            })

        conn.close()
        if logger:
            logger.info(f"Loaded {len(messages)} messages from iMessage DB")

    except Exception as e:
        if logger:
            logger.error(f"Error loading iMessage DB: {e}")

    return messages


def extract_candidates_with_gliner(messages: list[dict[str, Any]], logger: logging.Logger | None = None) -> list[TrainingExample]:
    """Run GLiNER on messages via CandidateExtractor and create training examples.

    Uses the canonical CandidateExtractor for consistent label sets and thresholds.
    Results need manual labeling (or heuristics for initial labels).
    """
    try:
        from jarvis.contacts.candidate_extractor import CandidateExtractor
    except ImportError:
        if logger:
            logger.error("CandidateExtractor not available")
        return []

    extractor = CandidateExtractor()

    examples = []
    from tqdm import tqdm
    for msg in tqdm(messages, desc="Extracting candidates", unit="msg"):
        text = msg.get("text", "")
        if not text:
            continue

        candidates = extractor.extract_candidates(
            text, message_id=0,  # no real ROWID for training prep
        )

        for c in candidates:
            ex = TrainingExample(
                text=text,
                candidate=c.span_text,
                entity_type=c.span_label,
                gliner_score=c.gliner_score,
                category=_fact_type_to_category(c.fact_type),
                label=0,  # Needs manual labeling
                context=f"chat: {msg.get('chat', 'unknown')}",
                reasoning="Auto-extracted, needs review",
            )
            examples.append(ex)

    return examples


def _fact_type_to_category(fact_type: str) -> str:
    """Map fact_type (e.g. 'location.current') to broad category."""
    if "." in fact_type:
        return fact_type.split(".")[0]
    return "unknown"


def apply_heuristic_labels(examples: list[TrainingExample], logger: logging.Logger | None = None) -> list[TrainingExample]:
    """Apply heuristic rules to suggest labels (for initial training)."""
    import re
    from tqdm import tqdm

    for ex in tqdm(examples, desc="Applying heuristics", unit="ex"):
        candidate_lower = ex.candidate.lower()
        text_lower = ex.text.lower()

        # Mark as likely false positive
        vague_words = {"it", "that", "this", "them", "there", "thing", "stuff"}
        if candidate_lower in vague_words:
            ex.label = 0
            ex.reasoning = "Vague pronoun"
            continue

        # Too short
        if len(candidate_lower.split()) < 2 and ex.category == "preference":
            ex.label = 0
            ex.reasoning = "Too short for category"
            continue

        # Bot patterns
        bot_patterns = [
            r"CVS Pharmacy",
            r"Rx Ready",
            r"Check out this job",
        ]
        if any(re.search(p, ex.text) for p in bot_patterns):
            ex.label = 0
            ex.reasoning = "Likely bot message"
            continue

        # Clear positive indicators
        if ex.gliner_score > 0.8 and len(candidate_lower.split()) >= 2:
            if candidate_lower not in vague_words:
                ex.label = 1
                ex.reasoning = "High confidence + specific subject"

    return examples


def export_for_training(examples: list[TrainingExample], output_path: Path, logger: logging.Logger) -> None:
    """Export training examples in JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), indent=None) + "\n")

    logger.info(f"Exported {len(examples)} examples to {output_path}")

    # Print statistics
    valid = sum(1 for e in examples if e.label == 1)
    invalid = sum(1 for e in examples if e.label == 0)
    logger.info(f"  Valid facts: {valid}")
    logger.info(f"  False positives: {invalid}")


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(
        description="Prepare training data for GLiNER-based fact extraction"
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "imessage", "both"],
        default="synthetic",
        help="Data source for training examples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/fact_candidates.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max messages to load from iMessage",
    )
    parser.add_argument(
        "--apply-heuristics",
        action="store_true",
        help="Apply heuristic rules to suggest labels",
    )

    args = parser.parse_args()

    all_examples = []

    if args.source in ("synthetic", "both"):
        logger.info("Generating synthetic training examples...")
        synthetic = generate_synthetic_dataset(logger)
        all_examples.extend(synthetic)
        logger.info(f"  Generated {len(synthetic)} synthetic examples")

    if args.source in ("imessage", "both"):
        logger.info("Loading messages from iMessage...")
        messages = load_imessage_samples(limit=args.limit, logger=logger)

        if messages:
            logger.info("Extracting candidates with GLiNER...")
            candidates = extract_candidates_with_gliner(messages, logger)
            all_examples.extend(candidates)
            logger.info(f"  Extracted {len(candidates)} candidates")

    if args.apply_heuristics:
        logger.info("Applying heuristic labels...")
        all_examples = apply_heuristic_labels(all_examples, logger)

    if all_examples:
        export_for_training(all_examples, args.output, logger)

        # Also print sample
        print("\n" + "=" * 70, flush=True)
        print("Sample training examples:", flush=True)
        print("=" * 70, flush=True)
        for ex in all_examples[:5]:
            print(f"\nText: {ex.text}", flush=True)
            print(f"Candidate: '{ex.candidate}' ({ex.entity_type})", flush=True)
            print(f"Label: {'VALID' if ex.label == 1 else 'INVALID'}", flush=True)
            print(f"Reasoning: {ex.reasoning}", flush=True)
    else:
        logger.warning("No examples generated!")


if __name__ == "__main__":
    main()
