#!/usr/bin/env python3
"""
Backfill context for validation sets using topic segmentation.

Instead of fixed 5-message window, uses topic boundaries to include
only messages from the same topic segment as context.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from contracts.imessage import Message
from jarvis.topics.topic_segmenter import segment_conversation


def get_chat_db_path() -> Path:
    """Get path to iMessage chat.db."""
    path = Path.home() / "Library" / "Messages" / "chat.db"
    if not path.exists():
        raise FileNotFoundError(f"chat.db not found at {path}")
    return path


def find_message_with_thread(cursor: sqlite3.Cursor, text: str) -> Optional[tuple[int, int, str]]:
    """
    Find a message by exact text match and get its thread.

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


def get_thread_messages(
    cursor: sqlite3.Cursor, thread_id: str, before_date: int, limit: int = 100
) -> list[tuple[str, int, bool]]:
    """
    Get messages from the same thread before the target message.

    Returns: list of (text, date, is_from_me) tuples
    """
    cursor.execute(
        """
        SELECT m.text, m.date, m.is_from_me
        FROM message m
        WHERE m.cache_roomnames = ?
          AND m.date <= ?
          AND m.text IS NOT NULL
          AND m.text != ''
        ORDER BY m.date DESC
        LIMIT ?
        """,
        (thread_id, before_date, limit),
    )
    # Reverse to get chronological order
    return list(reversed(cursor.fetchall()))


def extract_topic_context(
    target_text: str, thread_messages: list[tuple[str, int, bool]]
) -> list[str]:
    """
    Use topic segmentation to find messages in the same topic as target.

    Args:
        target_text: The message we're finding context for
        thread_messages: All messages in thread (text, date, is_from_me)

    Returns:
        List of context messages (only those before target in same topic)
    """
    # Convert to Message objects for topic segmenter
    messages = []
    target_index = None

    for i, (text, date, is_from_me) in enumerate(thread_messages):
        # Convert Core Data timestamp to datetime
        # Core Data epoch: Jan 1, 2001 (date is in nanoseconds, divide by 10^9)
        dt = datetime(2001, 1, 1) + timedelta(seconds=date / 1_000_000_000)

        msg = Message(
            id=i,
            chat_id="temp_chat",
            sender="temp_sender" if not is_from_me else "me",
            sender_name=None,
            text=text,
            date=dt,
            is_from_me=bool(is_from_me),
        )
        messages.append(msg)

        if text == target_text:
            target_index = i

    if target_index is None:
        return []

    # Segment the conversation
    try:
        segments = segment_conversation(messages)
    except Exception as e:
        print(f"    Warning: topic segmentation failed: {e}", flush=True)
        # Fallback to last 5 messages
        return [msg.text for msg in messages[max(0, target_index - 5):target_index]]

    # Find which segment contains the target message
    target_segment = None
    for segment in segments:
        for seg_msg in segment.messages:
            # Match by timestamp since text might be duplicated
            target_msg = messages[target_index]
            if seg_msg.timestamp == target_msg.date:
                target_segment = segment
                break
        if target_segment:
            break

    if not target_segment:
        # Fallback to last 5
        return [msg.text for msg in messages[max(0, target_index - 5):target_index]]

    # Extract messages from segment that come BEFORE target
    context = []
    for seg_msg in target_segment.messages:
        # Find corresponding original message
        for i, msg in enumerate(messages):
            if msg.date == seg_msg.timestamp:
                if i < target_index:
                    context.append(msg.text)
                break

    return context


def backfill_validation_set(validation_file: Path, db_path: Path) -> dict:
    """
    Backfill context using topic segmentation.

    Returns: dict with stats (total, matched, with_context, topic_avg_length)
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

    stats = {
        "total": len(examples),
        "matched": 0,
        "with_context": 0,
        "updated": 0,
        "context_lengths": [],
    }

    # Process each example
    for i, example in enumerate(examples, 1):
        text = example["text"]

        # Print progress
        if i % 10 == 0 or i == len(examples):
            avg_len = (
                sum(stats["context_lengths"]) / len(stats["context_lengths"])
                if stats["context_lengths"]
                else 0
            )
            print(
                f"  Processing {i}/{len(examples)} "
                f"(matched: {stats['matched']}, with context: {stats['with_context']}, "
                f"avg context: {avg_len:.1f} msgs)",
                flush=True,
            )

        # Find message in chat.db
        result = find_message_with_thread(cursor, text)
        if not result:
            continue

        stats["matched"] += 1
        rowid, date, thread_id = result

        # Get thread messages
        thread_messages = get_thread_messages(cursor, thread_id, date, limit=100)

        if not thread_messages:
            continue

        # Extract topic-aware context
        context = extract_topic_context(text, thread_messages)

        if context:
            old_context = example.get("context", [])
            example["context"] = context
            stats["with_context"] += 1
            stats["context_lengths"].append(len(context))

            # Check if actually different from before
            if old_context != context:
                stats["updated"] += 1

    conn.close()

    # Write updated validation set
    with open(validation_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    # Compute average context length
    if stats["context_lengths"]:
        stats["avg_context_length"] = sum(stats["context_lengths"]) / len(
            stats["context_lengths"]
        )
    else:
        stats["avg_context_length"] = 0.0

    return stats


def main():
    """Backfill context for all validation sets using topic segmentation."""
    db_path = get_chat_db_path()
    print(f"Using chat.db: {db_path}", flush=True)
    print("Using TOPIC SEGMENTATION for context extraction\n", flush=True)

    # Validation sets to process
    validation_files = [
        Path("validation_sets_multilabel.jsonl"),
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
        print(f"  ✓ Updated: {stats['updated']}", flush=True)
        print(
            f"  ✓ Context coverage: {stats['with_context']/stats['total']*100:.1f}%",
            flush=True,
        )
        print(f"  ✓ Avg context length: {stats['avg_context_length']:.1f} messages\n", flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total_examples = sum(s["total"] for s in all_stats.values())
    total_matched = sum(s["matched"] for s in all_stats.values())
    total_with_context = sum(s["with_context"] for s in all_stats.values())
    total_updated = sum(s["updated"] for s in all_stats.values())

    all_lengths = []
    for s in all_stats.values():
        all_lengths.extend(s.get("context_lengths", []))
    avg_context = sum(all_lengths) / len(all_lengths) if all_lengths else 0

    print(f"Total examples: {total_examples}", flush=True)
    print(
        f"Matched in chat.db: {total_matched} ({total_matched/total_examples*100:.1f}%)",
        flush=True,
    )
    print(
        f"With context: {total_with_context} ({total_with_context/total_examples*100:.1f}%)",
        flush=True,
    )
    print(
        f"Updated: {total_updated} ({total_updated/total_examples*100:.1f}%)",
        flush=True,
    )
    print(f"Avg context length: {avg_context:.1f} messages", flush=True)
    print(
        f"\nCompare to fixed 5-message window: "
        f"topic-aware context is {'longer' if avg_context > 5 else 'shorter'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
