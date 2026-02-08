#!/usr/bin/env python3
"""
Backfill context for validation sets from chat.db.

Validation sets were sampled with broken SQL that left ~97% with empty context.
This script looks up each message in chat.db and retrieves up to 5 prior messages
in the same thread to populate the context field.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional


def get_chat_db_path() -> Path:
    """Get path to iMessage chat.db."""
    path = Path.home() / "Library" / "Messages" / "chat.db"
    if not path.exists():
        raise FileNotFoundError(f"chat.db not found at {path}")
    return path


def find_message(cursor: sqlite3.Cursor, text: str) -> Optional[tuple[int, int, str]]:
    """
    Find a message by exact text match.

    Returns: (ROWID, date, thread_id) or None if not found
    """
    cursor.execute(
        """
        SELECT m.ROWID, m.date, m.cache_roomnames
        FROM message m
        WHERE m.text = ?
          AND m.text IS NOT NULL
        LIMIT 1
        """,
        (text,),
    )
    result = cursor.fetchone()
    return result if result else None


def get_prior_messages(
    cursor: sqlite3.Cursor, thread_id: str, date: int, limit: int = 5
) -> list[str]:
    """
    Get prior messages in the same thread, before the given date.

    Returns: list of message texts in chronological order (oldest first)
    """
    cursor.execute(
        """
        SELECT m.text
        FROM message m
        WHERE m.cache_roomnames = ?
          AND m.date < ?
          AND m.text IS NOT NULL
          AND m.text != ''
        ORDER BY m.date DESC
        LIMIT ?
        """,
        (thread_id, date, limit),
    )
    # Query returns newest-first, reverse to get chronological order
    messages = [row[0] for row in cursor.fetchall()]
    return list(reversed(messages))


def backfill_validation_set(validation_file: Path, db_path: Path) -> dict:
    """
    Backfill context for a single validation set.

    Returns: dict with stats (total, matched, with_context)
    """
    # Read validation set
    examples = []
    with open(validation_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Connect to chat.db (read-only)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    stats = {"total": len(examples), "matched": 0, "with_context": 0, "updated": 0}

    # Process each example
    for i, example in enumerate(examples, 1):
        text = example["text"]

        # Print progress
        if i % 10 == 0 or i == len(examples):
            print(
                f"  Processing {i}/{len(examples)} "
                f"(matched: {stats['matched']}, with context: {stats['with_context']})",
                flush=True,
            )

        # Find message in chat.db
        result = find_message(cursor, text)
        if not result:
            continue

        stats["matched"] += 1
        rowid, date, thread_id = result

        # Get prior messages
        prior_messages = get_prior_messages(cursor, thread_id, date, limit=5)
        if prior_messages:
            # Only update if we got non-empty context
            example["context"] = prior_messages
            stats["with_context"] += 1
            # Check if this actually changed the context
            if not example.get("context") or example["context"] != prior_messages:
                stats["updated"] += 1

    conn.close()

    # Write updated validation set
    with open(validation_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    return stats


def main():
    """Backfill context for all validation sets."""
    db_path = get_chat_db_path()
    print(f"Using chat.db: {db_path}\n", flush=True)

    # Validation sets to process
    validation_files = [
        Path("validation_sets_multilabel.jsonl"),  # Combined set - this one has empty context
        Path("validation_set_2_labeled.jsonl"),
        Path("validation_set_3_labeled.jsonl"),
        Path("validation_set_4_labeled.jsonl"),
        Path("validation_set_5_labeled.jsonl"),
    ]

    all_stats = {}

    for val_file in validation_files:
        if not val_file.exists():
            print(f"Skipping {val_file.name} (not found)", flush=True)
            continue

        print(f"Processing {val_file.name}...", flush=True)
        stats = backfill_validation_set(val_file, db_path)
        all_stats[val_file.name] = stats

        print(f"  ✓ Total: {stats['total']}", flush=True)
        print(f"  ✓ Matched in chat.db: {stats['matched']}", flush=True)
        print(f"  ✓ With context: {stats['with_context']}", flush=True)
        print(f"  ✓ Actually updated: {stats['updated']}", flush=True)
        print(
            f"  ✓ Context coverage: "
            f"{stats['with_context']/stats['total']*100:.1f}%\n",
            flush=True,
        )

    # Summary
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total_examples = sum(s["total"] for s in all_stats.values())
    total_matched = sum(s["matched"] for s in all_stats.values())
    total_with_context = sum(s["with_context"] for s in all_stats.values())
    total_updated = sum(s["updated"] for s in all_stats.values())

    print(f"Total examples: {total_examples}", flush=True)
    print(f"Matched in chat.db: {total_matched} ({total_matched/total_examples*100:.1f}%)", flush=True)
    print(
        f"With context: {total_with_context} ({total_with_context/total_examples*100:.1f}%)",
        flush=True,
    )
    print(
        f"Actually updated: {total_updated} ({total_updated/total_examples*100:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
