#!/usr/bin/env python3
"""Extract facts from iMessage DB and validate precision.

Phase 5: Manual validation of fact extraction quality improvements.
- Extracts facts from real conversation data
- Samples random facts for manual evaluation
- Compares before/after quality
"""
from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis.utils.logging import setup_script_logging

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of recent messages to load from iMessage DB (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of extracted facts to sample for review (default: %(default)s).",
    )
    parser.add_argument(
        "--report-path",
        default="fact_extraction_review.md",
        help="Output path for markdown review report (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-json-path",
        default="fact_extraction_sample.json",
        help="Output path for sampled facts JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Fallback progress print interval when tqdm is unavailable (default: %(default)s).",
    )
    parser.add_argument(
        "--rule-based",
        action="store_true",
        default=False,
        help="Use rule-based FactExtractor instead of GLiNER+NLI (for comparison).",
    )
    return parser.parse_args(argv)


def iter_with_progress(
    items: list,
    label: str,
    progress_every: int = 100,
) -> Iterator:
    """Yield items with visible progress for larger loops."""
    total = len(items)
    if total <= 10:
        yield from items
        return

    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=label, unit="item")
        return
    except ImportError:
        pass

    print(f"{label}: processing {total} items...", flush=True)
    for idx, item in enumerate(items, 1):
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"{label}: {idx}/{total}", flush=True)
        yield item


def get_recent_messages(limit: int = 5000) -> list[dict]:
    """Get recent messages from iMessage chat.db for fact extraction."""
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return []

    try:
        # Open iMessage database in read-only mode
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT m.text, m.ROWID, c.chat_identifier
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0
            ORDER BY m.ROWID DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in iter_with_progress(rows, "Filtering messages"):
            text = row["text"]
            if text and len(text.strip()) > 5:  # Skip very short messages
                messages.append(
                    {
                        "text": text,
                        "id": row["ROWID"],
                        "chat_id": row["chat_identifier"],
                    }
                )
        logger.info(f"Loaded {len(messages)} messages from iMessage database")
        return messages
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_facts_with_filters(messages: list[dict], progress_every: int = 100) -> list[Fact]:
    """Extract facts using new quality filter pipeline."""
    from jarvis.contacts.fact_extractor import FactExtractor

    # Keep threshold at 0.5 - professional message + coherence filters remove obvious bad facts
    extractor = FactExtractor(confidence_threshold=0.5)
    logger.info("Extracting facts with quality filters...")
    start = time.perf_counter()

    # Group by chat_id for better extraction
    chats: dict[str, list[dict]] = {}
    for msg in messages:
        chat_id = msg.get("chat_id", "unknown")
        if chat_id not in chats:
            chats[chat_id] = []
        chats[chat_id].append(msg)

    all_facts = []
    chat_items = list(chats.items())
    for chat_id, chat_messages in iter_with_progress(chat_items, "Extracting per-chat facts", progress_every):
        facts = extractor.extract_facts(chat_messages, contact_id=chat_id)
        all_facts.extend(facts)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"Extracted {len(all_facts)} facts from {len(messages)} messages in {elapsed:.1f}ms"
    )
    return all_facts


def extract_facts_with_gliner(messages: list[dict], progress_every: int = 100) -> list[Fact]:
    """Extract facts using GLiNER + NLI entailment verification pipeline."""
    from jarvis.contacts.candidate_extractor import CandidateExtractor
    from jarvis.contacts.contact_profile import Fact

    extractor = CandidateExtractor(label_profile="balanced", use_entailment=True)
    logger.info("Extracting facts with GLiNER + NLI entailment...")
    start = time.perf_counter()

    # Group by chat_id for batch extraction
    chats: dict[str, list[dict]] = {}
    for msg in messages:
        chat_id = msg.get("chat_id", "unknown")
        chats.setdefault(chat_id, []).append(msg)

    from jarvis.text_normalizer import normalize_text

    all_facts: list[Fact] = []
    chat_items = list(chats.items())
    for chat_id, chat_msgs in iter_with_progress(
        chat_items, "GLiNER+NLI extraction", progress_every
    ):
        batch_msgs = []
        for m in chat_msgs:
            raw = m.get("text")
            if not raw:
                continue
            normalized = normalize_text(raw)
            if not normalized or not normalized.strip():
                continue
            batch_msgs.append(
                {
                    "text": normalized,
                    "message_id": m.get("id", 0),
                    "is_from_me": m.get("is_from_me", False),
                    "chat_id": chat_id,
                }
            )
        candidates = extractor.extract_batch(batch_msgs)
        for c in candidates:
            all_facts.append(
                Fact(
                    category=c.fact_type or c.span_label,
                    subject=c.span_text,
                    predicate=c.span_label,
                    source_text=c.source_text[:200],
                    confidence=c.gliner_score,
                    contact_id=str(chat_id),
                    extracted_at="",
                )
            )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"GLiNER+NLI extracted {len(all_facts)} facts from {len(messages)} messages "
        f"in {elapsed:.1f}ms"
    )
    return all_facts


def sample_facts(facts: list[Fact], sample_size: int = 50) -> list[Fact]:
    """Randomly sample facts for manual evaluation."""
    if len(facts) <= sample_size:
        return facts
    return random.sample(facts, sample_size)


def format_fact_for_review(fact: Fact, index: int) -> str:
    """Format a fact for human review."""
    return f"""
{index}. [{fact.category.upper()}] {fact.subject} ({fact.predicate})
   Confidence: {fact.confidence:.2f}
   Source: {fact.source_text[:100]}...
   Contact: {fact.contact_id}
"""


def generate_review_report(facts: list[Fact]) -> str:
    """Generate a report for manual fact evaluation."""
    report = """
# Fact Extraction Quality Review Report

## Instructions
For each fact below, rate it as:
- ✓ GOOD: Actionable, specific, clear context
- ✗ BAD: Vague, missing context, questionable accuracy

After reviewing all facts, calculate:
- Precision = (GOOD facts) / (total facts)
- Target: ≥80% (40+ GOOD out of 50)

---

## Facts for Review

"""
    for i, fact in enumerate(facts, 1):
        report += format_fact_for_review(fact, i)

    report += f"""

---

## Summary

Total facts sampled: {len(facts)}
Categories:
"""
    by_category = {}
    for fact in facts:
        by_category[fact.category] = by_category.get(fact.category, 0) + 1

    for category, count in sorted(by_category.items()):
        report += f"  - {category}: {count}\n"

    report += f"""
Confidence range: {min(f.confidence for f in facts):.2f} - {max(f.confidence for f in facts):.2f}
Average confidence: {sum(f.confidence for f in facts) / len(facts):.2f}

---

## Evaluation Template

Copy this template and fill in your assessment:

```
Review Results:
- GOOD facts: ___ / {len(facts)}
- BAD facts: ___ / {len(facts)}
- Precision: ___% (target: ≥80%)

Notes:
[Your observations and feedback here]
```
"""
    return report


def save_report(facts: list[Fact], output_path: str = "fact_extraction_review.md") -> None:
    """Save fact review report to file."""
    report = generate_review_report(facts)
    try:
        with open(output_path, "w") as f:
            f.write(report)
    except OSError as exc:
        logger.error("Failed to write review report %s: %s", output_path, exc)
        raise SystemExit(1) from exc
    logger.info(f"Review report saved to {output_path}")


def get_extraction_stats(facts: list[Fact]) -> dict:
    """Calculate extraction statistics."""
    stats = {
        "total_facts": len(facts),
        "by_category": {},
        "confidence_stats": {
            "min": min(f.confidence for f in facts) if facts else 0,
            "max": max(f.confidence for f in facts) if facts else 0,
            "avg": sum(f.confidence for f in facts) / len(facts) if facts else 0,
        },
        "filtered_out_estimate": {
            "vague_subjects": sum(
                1 for f in facts if f.subject.lower() in {"me", "you", "that", "this", "it", "them", "he", "she"}
            ),
            "short_phrases": sum(
                1 for f in facts if f.category == "preference" and len(f.subject.split()) < 3
            ),
        },
    }

    for fact in facts:
        if fact.category not in stats["by_category"]:
            stats["by_category"][fact.category] = 0
        stats["by_category"][fact.category] += 1

    return stats


def main(argv: Sequence[str] | None = None) -> None:
    """Run fact extraction and generate review report."""
    setup_script_logging("extract_and_validate_facts")
    args = parse_args(argv)
    logger.info("Starting fact extraction validation...")
    print("\n" + "=" * 70, flush=True)
    print("FACT EXTRACTION QUALITY VALIDATION", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Extract from recent messages
    logger.info("Loading messages from database...")
    messages = get_recent_messages(limit=args.limit)

    if not messages:
        logger.error("No messages found - database may be empty")
        return

    logger.info(f"Loaded {len(messages)} messages")

    # Extract facts with chosen pipeline
    if args.rule_based:
        logger.info("Using rule-based FactExtractor pipeline")
        facts = extract_facts_with_filters(messages, progress_every=args.progress_every)
    else:
        logger.info("Using GLiNER + NLI entailment pipeline")
        facts = extract_facts_with_gliner(messages, progress_every=args.progress_every)

    if not facts:
        logger.error("No facts extracted - review extraction logic")
        return

    # Show statistics
    stats = get_extraction_stats(facts)
    logger.info(f"Extraction statistics:")
    logger.info(f"  Total facts: {stats['total_facts']}")
    logger.info(f"  By category: {stats['by_category']}")
    logger.info(f"  Confidence range: {stats['confidence_stats']['min']:.2f} - {stats['confidence_stats']['max']:.2f}")
    logger.info(f"  Average confidence: {stats['confidence_stats']['avg']:.2f}")

    # Sample for manual review
    sample = sample_facts(facts, sample_size=args.sample_size)
    logger.info(f"Sampled {len(sample)} facts for manual review")

    # Generate review report
    save_report(sample, output_path=args.report_path)

    # Print review report
    print(generate_review_report(sample), flush=True)

    # Save facts to JSON for reference
    facts_json = [
        {
            "category": f.category,
            "subject": f.subject,
            "predicate": f.predicate,
            "confidence": f.confidence,
            "source_text": f.source_text[:100],
        }
        for f in sample
    ]
    try:
        with open(args.sample_json_path, "w") as f:
            json.dump(facts_json, f, indent=2)
    except OSError as exc:
        logger.error("Failed to write sample JSON %s: %s", args.sample_json_path, exc)
        raise SystemExit(1) from exc

    logger.info("Extraction complete. Review '%s' for evaluation.", args.report_path)


if __name__ == "__main__":
    main()
