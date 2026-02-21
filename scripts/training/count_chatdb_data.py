#!/usr/bin/env python3
"""Quick analysis of chat.db to estimate usable fine-tuning data.

Applies filters and shows counts without full extraction.
Use this to decide what slice to train on.

Example:
    uv run python scripts/training/count_chatdb_data.py --min-date 2023-01-01
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Bot/verification patterns
BOT_PATTERNS = [
    r"^\d{4,8}$",
    r"verification.*code",
    r"auth.*code",
    r"2fa.*code",
    r"security.*code",
    r"one-time.*code",
    r"otp",
    r"password.*is",
    r"your code is",
    r"login.*code",
]
BOT_RE = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)

REACTION_PATTERNS = [r"^liked\s+", r"^loved\s+", r"^laughed\s+at\s+", r"^emphasized\s+", r"^questioned\s+"]
REACTION_RE = re.compile("|".join(REACTION_PATTERNS), re.IGNORECASE)


def parse_apple_date(apple_timestamp: int) -> datetime:
    """Convert Apple timestamp to datetime.
    
    Apple timestamps are in nanoseconds since Jan 1, 2001.
    We convert to seconds to avoid overflow.
    """
    apple_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    # Convert nanoseconds to seconds
    seconds = apple_timestamp / 1_000_000_000
    return apple_epoch + timedelta(seconds=seconds)


def analyze_chat_db(
    db_path: Path,
    min_date: datetime | None = None,
    gap_minutes: int = 60,
    min_reply_words: int = 6,
    max_context_turns: int = 8,
) -> dict:
    """Analyze chat.db and return statistics."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    stats = {
        "total_messages": 0,
        "total_chats": 0,
        "one_on_one_chats": 0,
        "group_chats": 0,
        "by_year": defaultdict(int),
        "by_contact": defaultdict(int),
        "skipped_group_chats": 0,
        "skipped_reactions": 0,
        "skipped_bots": 0,
        "skipped_short": 0,
        "skipped_old": 0,
        "skipped_no_context": 0,
        "training_examples": 0,
        "context_distribution": defaultdict(int),
    }

    # Count total chats
    cursor.execute("SELECT COUNT(*) FROM chat")
    stats["total_chats"] = cursor.fetchone()[0]

    # Count 1:1 vs group
    cursor.execute("""
        SELECT 
            c.ROWID,
            COUNT(chj.handle_id) as participant_count
        FROM chat c
        JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        GROUP BY c.ROWID
    """)
    for row in cursor.fetchall():
        if row["participant_count"] == 1:
            stats["one_on_one_chats"] += 1
        else:
            stats["group_chats"] += 1

    # Analyze each 1:1 chat
    cursor.execute("""
        SELECT 
            c.guid as chat_guid,
            h.id as handle_id,
            h.ROWID as handle_rowid
        FROM chat c
        JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        JOIN handle h ON chj.handle_id = h.ROWID
        WHERE c.ROWID IN (
            SELECT chat_id 
            FROM chat_handle_join 
            GROUP BY chat_id 
            HAVING COUNT(*) = 1
        )
        ORDER BY c.ROWID
    """)

    one_on_one_chats = cursor.fetchall()
    print(f"Analyzing {len(one_on_one_chats)} 1:1 conversations...")

    for chat_row in one_on_one_chats:
        chat_guid = chat_row["chat_guid"]
        handle_id = chat_row["handle_id"]

        # Get messages for this chat
        cursor.execute("""
            SELECT 
                m.ROWID,
                m.text,
                m.is_from_me,
                m.date
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id = (SELECT ROWID FROM chat WHERE guid = ?)
            AND m.text IS NOT NULL
            AND m.text != ''
            ORDER BY m.date ASC
        """, (chat_guid,))

        messages = cursor.fetchall()
        stats["total_messages"] += len(messages)

        # Track by year (from timestamps)
        for msg_row in messages:
            date = parse_apple_date(msg_row["date"])
            stats["by_year"][date.year] += 1

        # Simulate extraction logic
        gap_threshold = timedelta(minutes=gap_minutes)
        my_replies = [m for m in messages if m["is_from_me"]]

        for reply in my_replies:
            reply_date = parse_apple_date(reply["date"])

            # Check date filter
            if min_date and reply_date < min_date:
                stats["skipped_old"] += 1
                continue

            # Check reaction
            if REACTION_RE.match(reply["text"]):
                stats["skipped_reactions"] += 1
                continue

            # Check bot
            if BOT_RE.search(reply["text"]):
                stats["skipped_bots"] += 1
                continue

            # Check length
            if len(reply["text"].split()) < min_reply_words:
                stats["skipped_short"] += 1
                continue

            # Find context
            reply_idx = next(
                (i for i, m in enumerate(messages) if m["ROWID"] == reply["ROWID"]),
                None,
            )
            if reply_idx is None or reply_idx == 0:
                stats["skipped_no_context"] += 1
                continue

            # Count context turns
            context_count = 0
            prev_date = reply_date
            for i in range(reply_idx - 1, -1, -1):
                msg = messages[i]
                msg_date = parse_apple_date(msg["date"])

                if prev_date - msg_date > gap_threshold:
                    break

                context_count += 1
                prev_date = msg_date

                if context_count >= max_context_turns:
                    break

            if context_count == 0:
                stats["skipped_no_context"] += 1
                continue

            stats["training_examples"] += 1
            stats["by_contact"][handle_id] += 1
            stats["context_distribution"][context_count] += 1

    conn.close()
    return stats


def print_stats(stats: dict, min_date: datetime | None = None) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("CHAT.DB ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Conversations:")
    print(f"  Total chats:        {stats['total_chats']:,}")
    print(f"  1:1 conversations:  {stats['one_on_one_chats']:,}")
    print(f"  Group chats:        {stats['group_chats']:,} (excluded)")

    print(f"\n📝 Messages:")
    print(f"  Total messages:     {stats['total_messages']:,}")

    print(f"\n📅 By year:")
    for year in sorted(stats["by_year"].keys(), reverse=True):
        count = stats["by_year"][year]
        print(f"  {year}: {count:,}")

    print("\n" + "-" * 70)
    print("FILTERING BREAKDOWN")
    print("-" * 70)

    print(f"  Group chats (excluded):     {stats['skipped_group_chats']:,}")
    print(f"  Reactions/tapbacks:         {stats['skipped_reactions']:,}")
    print(f"  Bot/verification codes:     {stats['skipped_bots']:,}")
    print(f"  Short replies (<6 words):   {stats['skipped_short']:,}")
    if min_date:
        print(f"  Before {min_date.date()}:          {stats['skipped_old']:,}")
    print(f"  No context available:       {stats['skipped_no_context']:,}")

    print("-" * 70)
    print(f"✅ USABLE TRAINING EXAMPLES:  {stats['training_examples']:,}")
    print("=" * 70)

    if stats["training_examples"] > 0:
        print("\n📏 Context length distribution:")
        for turns in sorted(stats["context_distribution"].keys()):
            count = stats["context_distribution"][turns]
            pct = count / stats["training_examples"] * 100
            bar = "█" * int(pct / 3)
            print(f"  {turns:2d} turns: {count:5,} ({pct:5.1f}%) {bar}")

        print("\n👤 Top contacts (usable pairs):")
        sorted_contacts = sorted(
            stats["by_contact"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        for contact, count in sorted_contacts:
            pct = count / stats["training_examples"] * 100
            print(f"  {contact[:35]:35s} {count:4d} ({pct:4.1f}%)")

        # Warn about distribution
        if sorted_contacts:
            top_pct = sorted_contacts[0][1] / stats["training_examples"] * 100
            if top_pct > 40:
                print(f"\n⚠️  WARNING: Top contact is {top_pct:.1f}% of data.")
                print("   Consider using --max-per-contact to balance.")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / "Library" / "Messages" / "chat.db",
        help="Path to chat.db",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Only count messages after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=60,
        help="Minutes of silence that defines new conversation (default: 60)",
    )
    parser.add_argument(
        "--min-reply-words",
        type=int,
        default=6,
        help="Minimum words in reply to count (default: 6)",
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"❌ chat.db not found at {args.db}")
        return

    min_date = None
    if args.min_date:
        min_date = datetime.strptime(args.min_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    print(f"Analyzing {args.db}...")
    stats = analyze_chat_db(
        args.db,
        min_date=min_date,
        gap_minutes=args.gap_minutes,
        min_reply_words=args.min_reply_words,
    )

    print_stats(stats, min_date)


if __name__ == "__main__":
    main()
