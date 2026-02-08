#!/usr/bin/env python3
"""Analyze production message distribution vs training data.

Classifies all messages from chat.db and compares the distribution
to the training data distribution.
"""

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from jarvis.classifiers.category_classifier import classify_category


def get_all_messages(db_path: Path) -> list[tuple[int, str]]:
    """Extract all message texts from chat.db.

    Returns:
        List of (rowid, text) tuples for messages that are from me.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all my outgoing messages with text
    query = """
        SELECT ROWID, text
        FROM message
        WHERE is_from_me = 1
        AND text IS NOT NULL
        AND text != ''
        ORDER BY ROWID
    """

    cursor.execute(query)
    messages = cursor.fetchall()
    conn.close()

    return messages


def classify_all(messages: list[tuple[int, str]]) -> dict[str, int]:
    """Classify all messages and return distribution.

    Args:
        messages: List of (rowid, text) tuples

    Returns:
        Counter of category -> count
    """
    distribution = Counter()

    print(f"Classifying {len(messages):,} messages...")
    start = time.time()

    # Process in batches for progress updates
    batch_size = 1000
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]

        for rowid, text in batch:
            try:
                result = classify_category(text, context=[])
                distribution[result.category] += 1
            except Exception as e:
                print(f"Error classifying message {rowid}: {e}")
                continue

        if (i + batch_size) % 10000 == 0 or i + batch_size >= len(messages):
            elapsed = time.time() - start
            rate = (i + batch_size) / elapsed
            print(f"  {i + batch_size:,}/{len(messages):,} ({rate:.0f} msg/s)")

    elapsed = time.time() - start
    print(f"\nClassified {len(messages):,} messages in {elapsed:.1f}s")
    print(f"Average: {len(messages) / elapsed:.0f} messages/second")

    return dict(distribution)


def load_training_distribution(data_dir: Path) -> dict[str, int]:
    """Load training data distribution from metadata."""
    meta_path = data_dir / "category_training" / "metadata.json"
    if not meta_path.exists():
        return {}

    metadata = json.loads(meta_path.read_text())
    return metadata.get("label_distribution_balanced", {})


def print_comparison(prod_dist: dict[str, int], train_dist: dict[str, int]) -> None:
    """Print side-by-side comparison of distributions."""
    total_prod = sum(prod_dist.values())
    total_train = sum(train_dist.values())

    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"{'Category':<12} {'Production':<25} {'Training Data':<25}")
    print("-" * 70)

    all_categories = sorted(set(prod_dist.keys()) | set(train_dist.keys()))

    for cat in all_categories:
        prod_count = prod_dist.get(cat, 0)
        train_count = train_dist.get(cat, 0)

        prod_pct = 100 * prod_count / total_prod if total_prod > 0 else 0
        train_pct = 100 * train_count / total_train if total_train > 0 else 0

        prod_str = f"{prod_count:>7,} ({prod_pct:>5.1f}%)"
        train_str = f"{train_count:>7,} ({train_pct:>5.1f}%)"

        print(f"{cat:<12} {prod_str:<25} {train_str:<25}")

    print("-" * 70)
    print(f"{'TOTAL':<12} {total_prod:>7,} {'':<18} {total_train:>7,}")
    print("=" * 70)

    # Calculate divergence
    print("\nKEY INSIGHTS:")
    for cat in all_categories:
        prod_pct = 100 * prod_dist.get(cat, 0) / total_prod if total_prod > 0 else 0
        train_pct = 100 * train_dist.get(cat, 0) / total_train if total_train > 0 else 0
        diff = prod_pct - train_pct

        if abs(diff) > 5:
            direction = "over" if diff > 0 else "under"
            print(f"  - {cat}: {direction}-represented by {abs(diff):.1f}% in training")


def main() -> None:
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    data_dir = Path(__file__).parent.parent / "data"

    if not db_path.exists():
        print(f"Error: chat.db not found at {db_path}")
        return

    # Get all messages
    messages = get_all_messages(db_path)
    print(f"Found {len(messages):,} outgoing messages\n")

    # Classify all
    prod_dist = classify_all(messages)

    # Load training distribution
    train_dist = load_training_distribution(data_dir)

    # Compare
    print_comparison(prod_dist, train_dist)


if __name__ == "__main__":
    main()
