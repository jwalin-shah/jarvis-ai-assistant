#!/usr/bin/env python3
"""
Backfill context using semantic similarity filtering.

Instead of expensive topic segmentation, this uses a simpler approach:
1. Get last 10 messages before target
2. Embed all messages + target
3. Keep only messages with high cosine similarity to target (same topic)

This is much faster than full topic segmentation and still filters out off-topic noise.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


def get_chat_db_path() -> Path:
    """Get path to iMessage chat.db."""
    path = Path.home() / "Library" / "Messages" / "chat.db"
    if not path.exists():
        raise FileNotFoundError(f"chat.db not found at {path}")
    return path


def find_message_with_thread(cursor: sqlite3.Cursor, text: str) -> Optional[tuple[int, int, str]]:
    """Find a message by exact text match and get its thread."""
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
    cursor: sqlite3.Cursor, thread_id: str, before_date: int, limit: int = 10
) -> list[str]:
    """Get prior messages in the same thread."""
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
        (thread_id, before_date, limit),
    )
    # Reverse to get chronological order
    return [row[0] for row in reversed(cursor.fetchall())]


def filter_by_semantic_similarity(
    target_text: str,
    prior_messages: list[str],
    embedder,
    similarity_threshold: float = 0.6,
) -> list[str]:
    """
    Filter prior messages by semantic similarity to target.

    Only keeps messages that are topically related to the target message.

    Args:
        target_text: The message we're finding context for
        prior_messages: Candidate context messages
        embedder: BERT embedder instance
        similarity_threshold: Cosine similarity threshold (0-1)

    Returns:
        Filtered list of semantically relevant messages
    """
    if not prior_messages:
        return []

    # Embed all messages at once
    all_texts = [target_text] + prior_messages
    try:
        embeddings = embedder.encode(all_texts, normalize=True)
    except Exception as e:
        print(f"    Warning: embedding failed: {e}", flush=True)
        # Fallback: return all messages
        return prior_messages

    target_emb = embeddings[0]
    prior_embs = embeddings[1:]

    # Compute cosine similarities
    similarities = np.dot(prior_embs, target_emb)

    # Filter by threshold
    filtered = []
    for i, sim in enumerate(similarities):
        if sim >= similarity_threshold:
            filtered.append(prior_messages[i])

    return filtered


def backfill_validation_set(validation_file: Path, db_path: Path, embedder) -> dict:
    """
    Backfill context using semantic filtering.

    Returns: dict with stats
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
        "avg_similarity": [],
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
                f"avg: {avg_len:.1f} msgs)",
                flush=True,
            )

        # Find message in chat.db
        result = find_message_with_thread(cursor, text)
        if not result:
            continue

        stats["matched"] += 1
        rowid, date, thread_id = result

        # Get last 10 prior messages
        prior_messages = get_prior_messages(cursor, thread_id, date, limit=10)

        if not prior_messages:
            continue

        # Filter by semantic similarity
        filtered_context = filter_by_semantic_similarity(
            text,
            prior_messages,
            embedder,
            similarity_threshold=0.6,
        )

        if filtered_context:
            old_context = example.get("context", [])
            example["context"] = filtered_context
            stats["with_context"] += 1
            stats["context_lengths"].append(len(filtered_context))

            if old_context != filtered_context:
                stats["updated"] += 1

    conn.close()

    # Write updated validation set
    with open(validation_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    # Compute average
    if stats["context_lengths"]:
        stats["avg_context_length"] = sum(stats["context_lengths"]) / len(
            stats["context_lengths"]
        )
    else:
        stats["avg_context_length"] = 0.0

    return stats


def main():
    """Backfill context for validation sets using semantic filtering."""
    from jarvis.embedding_adapter import get_embedder

    db_path = get_chat_db_path()
    print(f"Using chat.db: {db_path}", flush=True)
    print("Using SEMANTIC FILTERING for context extraction", flush=True)
    print("(Keeps last 10 messages filtered by cosine similarity > 0.6)\n", flush=True)

    # Load embedder once
    print("Loading BERT embedder...", flush=True)
    embedder = get_embedder()

    # Validation sets to process
    validation_files = [
        Path("validation_sets_multilabel.jsonl"),
    ]

    all_stats = {}

    for val_file in validation_files:
        if not val_file.exists():
            print(f"Skipping {val_file.name} (not found)", flush=True)
            continue

        print(f"\nProcessing {val_file.name}...", flush=True)
        stats = backfill_validation_set(val_file, db_path, embedder)
        all_stats[val_file.name] = stats

        print(f"  ✓ Total: {stats['total']}", flush=True)
        print(f"  ✓ Matched in chat.db: {stats['matched']}", flush=True)
        print(f"  ✓ With context: {stats['with_context']}", flush=True)
        print(f"  ✓ Updated: {stats['updated']}", flush=True)
        print(
            f"  ✓ Context coverage: {stats['with_context']/stats['total']*100:.1f}%",
            flush=True,
        )
        print(f"  ✓ Avg context length: {stats['avg_context_length']:.1f} messages", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total_examples = sum(s["total"] for s in all_stats.values())
    total_with_context = sum(s["with_context"] for s in all_stats.values())

    all_lengths = []
    for s in all_stats.values():
        all_lengths.extend(s.get("context_lengths", []))
    avg_context = sum(all_lengths) / len(all_lengths) if all_lengths else 0

    print(
        f"With context: {total_with_context}/{total_examples} "
        f"({total_with_context/total_examples*100:.1f}%)",
        flush=True,
    )
    print(f"Avg context length: {avg_context:.1f} messages", flush=True)
    print(
        f"\nCompare to fixed 5-message window: "
        f"semantic filter gives {avg_context:.1f} msgs on average",
        flush=True,
    )


if __name__ == "__main__":
    main()
