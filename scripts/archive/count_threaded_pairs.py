#!/usr/bin/env python3
"""
Count gold pairs from iMessage's explicit thread_originator_guid linking.

Since iOS 14, when someone uses the "reply" feature in iMessage, Apple creates
a database linkage via thread_originator_guid. These are GOLD pairs - no
heuristics needed, high confidence (85-95%) that reply is responding to original.

Note: Not 100% confidence because:
- Sarcastic replies ("Sure..." meaning NO)
- Context from phone calls (text alone may not make sense)
- Accidental wrong reply clicks
- Multi-message context missing

Usage:
    uv run python -m scripts.count_threaded_pairs
    uv run python -m scripts.count_threaded_pairs --sample 10  # Show examples
    uv run python -m scripts.count_threaded_pairs --export gold_pairs.jsonl
    uv run python -m scripts.count_threaded_pairs --export sample.jsonl --limit 100 --randomize
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ThreadedPair:
    """A pair extracted from explicit iMessage threading."""

    trigger_text: str
    response_text: str
    trigger_is_from_me: bool
    response_is_from_me: bool
    trigger_date: datetime
    response_date: datetime
    chat_id: int | None = None
    is_group: bool = False
    confidence: float = 0.90  # High but not 100% - sarcasm, context, accidental clicks


def get_chat_db_path() -> Path:
    """Get the iMessage database path."""
    return Path.home() / "Library" / "Messages" / "chat.db"


def count_threaded_pairs(db_path: Path) -> dict:
    """
    Query chat.db for threaded reply pairs.

    Returns statistics about the threaded pairs found.
    """
    # Read-only connection
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # First, check if thread_originator_guid column exists
        cursor.execute("PRAGMA table_info(message)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "thread_originator_guid" not in columns:
            return {
                "error": "thread_originator_guid column not found. Requires macOS 14+/iOS 14+",
                "total_threaded_pairs": 0,
            }

        # Count total messages with threading
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM message
            WHERE thread_originator_guid IS NOT NULL
              AND thread_originator_guid != ''
        """)
        total_replies = cursor.fetchone()["cnt"]

        # Main query: Get threaded pairs with full metadata
        # Join reply message with its original message via GUID
        query = """
            SELECT
                reply.ROWID as reply_id,
                reply.text AS response_text,
                reply.attributedBody AS response_attributed,
                reply.is_from_me AS response_is_from_me,
                reply.date AS response_date,
                original.ROWID as original_id,
                original.text AS trigger_text,
                original.attributedBody AS trigger_attributed,
                original.is_from_me AS trigger_is_from_me,
                original.date AS trigger_date,
                cmj.chat_id
            FROM message reply
            JOIN message original ON reply.thread_originator_guid = original.guid
            LEFT JOIN chat_message_join cmj ON reply.ROWID = cmj.message_id
            WHERE (reply.text IS NOT NULL AND reply.text != '')
               OR reply.attributedBody IS NOT NULL
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        # Process and categorize pairs
        stats = {
            "total_replies_with_threading": total_replies,
            "total_matched_pairs": len(rows),
            "them_to_me": 0,  # They send, I reply (most useful for suggestions)
            "me_to_them": 0,  # I send, they reply
            "me_to_me": 0,    # Self-replies
            "them_to_them": 0,  # Their conversation
            "with_text_both": 0,
            "trigger_empty": 0,
            "response_empty": 0,
            "group_chat_pairs": 0,
            "direct_chat_pairs": 0,
        }

        pairs: list[ThreadedPair] = []
        chat_ids = set()

        for row in rows:
            trigger_text = row["trigger_text"] or ""
            response_text = row["response_text"] or ""
            trigger_from_me = bool(row["trigger_is_from_me"])
            response_from_me = bool(row["response_is_from_me"])
            chat_id = row["chat_id"]

            # Skip if both texts are empty
            if not trigger_text.strip() and not response_text.strip():
                continue

            if not trigger_text.strip():
                stats["trigger_empty"] += 1
                continue

            if not response_text.strip():
                stats["response_empty"] += 1
                continue

            stats["with_text_both"] += 1

            # Categorize by direction
            if not trigger_from_me and response_from_me:
                stats["them_to_me"] += 1
            elif trigger_from_me and not response_from_me:
                stats["me_to_them"] += 1
            elif trigger_from_me and response_from_me:
                stats["me_to_me"] += 1
            else:
                stats["them_to_them"] += 1

            if chat_id:
                chat_ids.add(chat_id)

            # Convert Apple's cocoa timestamp to datetime
            # Apple uses seconds since 2001-01-01, stored as nanoseconds since ~macOS 10.13
            def apple_to_datetime(ts: int | None) -> datetime:
                if ts is None:
                    return datetime.now()
                # If timestamp is in nanoseconds (very large), convert
                if ts > 1e15:
                    ts = ts / 1e9
                # Add offset from Unix epoch to Apple epoch (978307200 seconds)
                return datetime.fromtimestamp(ts + 978307200)

            pair = ThreadedPair(
                trigger_text=trigger_text,
                response_text=response_text,
                trigger_is_from_me=trigger_from_me,
                response_is_from_me=response_from_me,
                trigger_date=apple_to_datetime(row["trigger_date"]),
                response_date=apple_to_datetime(row["response_date"]),
                chat_id=chat_id,
            )
            pairs.append(pair)

        # Check how many are group chats
        if chat_ids:
            chat_id_list = ",".join(str(cid) for cid in chat_ids)
            cursor.execute(f"""
                SELECT c.ROWID as chat_id, COUNT(DISTINCT h.ROWID) as participant_count
                FROM chat c
                LEFT JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
                LEFT JOIN handle h ON chj.handle_id = h.ROWID
                WHERE c.ROWID IN ({chat_id_list})
                GROUP BY c.ROWID
            """)
            group_chats = {
                row["chat_id"] for row in cursor.fetchall() if row["participant_count"] > 1
            }

            for pair in pairs:
                if pair.chat_id in group_chats:
                    pair.is_group = True
                    stats["group_chat_pairs"] += 1
                else:
                    stats["direct_chat_pairs"] += 1

        stats["unique_chats"] = len(chat_ids)

        return {
            "stats": stats,
            "pairs": pairs,
        }

    finally:
        conn.close()


def print_sample_pairs(
    pairs: list[ThreadedPair], n: int = 10, direction: str = "them_to_me"
) -> None:
    """Print sample pairs for review."""
    if direction == "them_to_me":
        filtered = [p for p in pairs if not p.trigger_is_from_me and p.response_is_from_me]
    elif direction == "me_to_them":
        filtered = [p for p in pairs if p.trigger_is_from_me and not p.response_is_from_me]
    else:
        filtered = pairs

    print(f"\n{'='*60}")
    print(f"Sample {direction} pairs (showing {min(n, len(filtered))} of {len(filtered)}):")
    print('='*60)

    for i, pair in enumerate(filtered[:n]):
        print(f"\n--- Pair {i+1} ---")
        who_t = "(me)" if pair.trigger_is_from_me else "(them)"
        who_r = "(me)" if pair.response_is_from_me else "(them)"
        print(f"[TRIGGER] {who_t}: {pair.trigger_text[:200]}")
        print(f"[RESPONSE] {who_r}: {pair.response_text[:200]}")
        print(f"Time: {pair.trigger_date} -> {pair.response_date}")
        print(f"Group: {pair.is_group}")


def export_pairs(
    pairs: list[ThreadedPair],
    output_path: Path,
    direction: str | None = None,
    limit: int | None = None,
    randomize: bool = False,
) -> int:
    """Export pairs to JSONL file.

    Args:
        pairs: List of pairs to export
        output_path: Path to output JSONL file
        direction: Filter by direction (them_to_me, me_to_them, or None for all)
        limit: Maximum number of pairs to export (None for all)
        randomize: If True, shuffle pairs before limiting (for representative sample)
    """
    import random

    if direction == "them_to_me":
        filtered = [p for p in pairs if not p.trigger_is_from_me and p.response_is_from_me]
    elif direction == "me_to_them":
        filtered = [p for p in pairs if p.trigger_is_from_me and not p.response_is_from_me]
    else:
        filtered = pairs

    # Shuffle for representative sample if requested
    if randomize:
        filtered = filtered.copy()
        random.shuffle(filtered)

    # Apply limit
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    with open(output_path, "w") as f:
        for pair in filtered:
            record = {
                "trigger": pair.trigger_text,
                "response": pair.response_text,
                "trigger_is_from_me": pair.trigger_is_from_me,
                "response_is_from_me": pair.response_is_from_me,
                "trigger_date": pair.trigger_date.isoformat(),
                "response_date": pair.response_date.isoformat(),
                "is_group": pair.is_group,
                "confidence": pair.confidence,
                "source": "thread_originator_guid",
            }
            f.write(json.dumps(record) + "\n")

    return len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count gold pairs from iMessage threading")
    parser.add_argument("--sample", type=int, default=0, help="Show N sample pairs")
    parser.add_argument("--export", type=str, help="Export pairs to JSONL file")
    parser.add_argument("--limit", type=int, help="Limit export to N pairs (use with --export)")
    parser.add_argument("--randomize", action="store_true",
                       help="Randomize pairs before limiting (for representative sample)")
    parser.add_argument("--direction", choices=["them_to_me", "me_to_them", "all"],
                       default="them_to_me", help="Filter by direction")
    parser.add_argument("--db-path", type=str, help="Custom chat.db path")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_chat_db_path()

    if not db_path.exists():
        print(f"Error: chat.db not found at {db_path}")
        print("Make sure you have Full Disk Access enabled for your terminal.")
        return

    print(f"Analyzing threaded pairs in {db_path}...")
    print("-" * 60)

    result = count_threaded_pairs(db_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    stats = result["stats"]
    pairs = result["pairs"]

    print("\n📊 THREADED PAIRS STATISTICS")
    print("=" * 60)
    print(f"Total messages with thread_originator_guid: {stats['total_replies_with_threading']:,}")
    print(f"Successfully matched to original message:   {stats['total_matched_pairs']:,}")
    print(f"With valid text in both trigger & response: {stats['with_text_both']:,}")
    print()
    print("By Direction:")
    print(f"  • them → me (GOLD for suggestions):  {stats['them_to_me']:,}")
    print(f"  • me → them (shows my style):        {stats['me_to_them']:,}")
    print(f"  • me → me (self-replies):            {stats['me_to_me']:,}")
    print(f"  • them → them (their conversation):  {stats['them_to_them']:,}")
    print()
    print("By Chat Type:")
    print(f"  • Direct messages: {stats['direct_chat_pairs']:,}")
    print(f"  • Group chats:     {stats['group_chat_pairs']:,}")
    print(f"  • Unique chats:    {stats['unique_chats']:,}")
    print()
    print("Skipped:")
    print(f"  • Empty trigger: {stats['trigger_empty']:,}")
    print(f"  • Empty response: {stats['response_empty']:,}")

    # Key insight
    print()
    print("=" * 60)
    print("🎯 KEY INSIGHT")
    print("=" * 60)
    gold_count = stats["them_to_me"]
    print(f"You have {gold_count:,} GOLD pairs (them→me with explicit threading).")
    print("These have ~85-95% confidence - Apple proves the reply linkage")
    print("but sarcasm, phone call context, or accidental clicks may reduce quality.")
    print()
    print("RECOMMENDED: Export a random sample and manually review 100 pairs")
    print("to measure actual quality before building on this data.")
    print()
    print("  uv run python -m scripts.count_threaded_pairs \\")
    print("    --export sample.jsonl --limit 100 --randomize")

    if args.sample > 0:
        print_sample_pairs(pairs, args.sample, args.direction)

    if args.export:
        export_path = Path(args.export)
        direction_filter = args.direction if args.direction != "all" else None
        count = export_pairs(
            pairs,
            export_path,
            direction=direction_filter,
            limit=args.limit,
            randomize=args.randomize,
        )
        total_available = len([
            p for p in pairs
            if direction_filter is None
            or (direction_filter == "them_to_me"
                and not p.trigger_is_from_me and p.response_is_from_me)
            or (direction_filter == "me_to_them"
                and p.trigger_is_from_me and not p.response_is_from_me)
        ])
        if args.limit and count < total_available:
            print(f"\n✅ Exported {count:,} pairs (of {total_available:,}) to {export_path}")
            if args.randomize:
                print("   (randomized for representative sample)")
        else:
            print(f"\n✅ Exported {count:,} pairs to {export_path}")


if __name__ == "__main__":
    main()
