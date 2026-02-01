#!/usr/bin/env python3
"""
Extract full conversation context from iMessage threads.

Key insight: When someone uses iMessage's "reply" feature, it creates a THREAD.
A thread can have many messages, not just a single (trigger, response) pair.

This script extracts (context, my_response) pairs where:
- context = all previous messages in the thread
- my_response = my reply within that thread

This gives us FULL CONVERSATION CONTEXT, not isolated pairs.

Usage:
    uv run python -m scripts.extract_threaded_conversations
    uv run python -m scripts.extract_threaded_conversations --sample 5
    uv run python -m scripts.extract_threaded_conversations --export conversations.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ThreadMessage:
    """A single message within a thread."""
    text: str
    is_from_me: bool
    date: datetime
    rowid: int


@dataclass
class ConversationContext:
    """A (context, my_response) pair extracted from a thread."""
    # The conversation context (all messages before my response)
    context_messages: list[ThreadMessage]
    # My response
    my_response: ThreadMessage
    # Thread metadata
    thread_guid: str
    thread_size: int  # Total messages in this thread
    position_in_thread: int  # Where my response falls (1-indexed)
    is_group: bool = False
    chat_id: int | None = None

    @property
    def context_text(self) -> str:
        """Format context as a conversation string."""
        lines = []
        for msg in self.context_messages:
            who = "Me" if msg.is_from_me else "Them"
            lines.append(f"{who}: {msg.text}")
        return "\n".join(lines)

    @property
    def trigger_text(self) -> str:
        """Get the immediate trigger (last message before my response)."""
        if self.context_messages:
            return self.context_messages[-1].text
        return ""

    def to_dict(self) -> dict:
        return {
            "context": [
                {"text": m.text, "is_from_me": m.is_from_me, "date": m.date.isoformat()}
                for m in self.context_messages
            ],
            "context_formatted": self.context_text,
            "immediate_trigger": self.trigger_text,
            "my_response": self.my_response.text,
            "my_response_date": self.my_response.date.isoformat(),
            "thread_guid": self.thread_guid,
            "thread_size": self.thread_size,
            "position_in_thread": self.position_in_thread,
            "context_length": len(self.context_messages),
            "is_group": self.is_group,
            "source": "thread_conversation",
        }


def get_chat_db_path() -> Path:
    """Get the iMessage database path."""
    return Path.home() / "Library" / "Messages" / "chat.db"


def apple_to_datetime(ts: int | None) -> datetime:
    """Convert Apple timestamp to datetime."""
    if ts is None:
        return datetime.now()
    # If timestamp is in nanoseconds (very large), convert
    if ts > 1e15:
        ts = ts / 1e9
    # Add offset from Unix epoch to Apple epoch (978307200 seconds)
    return datetime.fromtimestamp(ts + 978307200)


def extract_thread_conversations(db_path: Path) -> list[ConversationContext]:
    """
    Extract (context, my_response) pairs from all iMessage threads.

    Returns list of ConversationContext objects with full thread context.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Check if thread_originator_guid column exists
        cursor.execute("PRAGMA table_info(message)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "thread_originator_guid" not in columns:
            print("Error: thread_originator_guid column not found. Requires macOS 14+/iOS 14+")
            return []

        # Get all unique thread originators
        cursor.execute("""
            SELECT DISTINCT thread_originator_guid
            FROM message
            WHERE thread_originator_guid IS NOT NULL
              AND thread_originator_guid != ''
        """)
        thread_guids = [row[0] for row in cursor.fetchall()]

        print(f"Found {len(thread_guids):,} unique threads")

        # Get group chat info
        cursor.execute("""
            SELECT c.ROWID as chat_id, COUNT(DISTINCT h.ROWID) as participant_count
            FROM chat c
            LEFT JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
            LEFT JOIN handle h ON chj.handle_id = h.ROWID
            GROUP BY c.ROWID
        """)
        group_chats = {row["chat_id"] for row in cursor.fetchall() if row["participant_count"] > 1}

        all_contexts: list[ConversationContext] = []

        for i, thread_guid in enumerate(thread_guids):
            if (i + 1) % 1000 == 0:
                print(f"Processing thread {i + 1:,}/{len(thread_guids):,}...")

            # Get the original message that started the thread
            cursor.execute("""
                SELECT ROWID, text, is_from_me, date
                FROM message
                WHERE guid = ?
            """, (thread_guid,))
            original_row = cursor.fetchone()

            if not original_row or not original_row["text"]:
                continue

            original = ThreadMessage(
                text=original_row["text"],
                is_from_me=bool(original_row["is_from_me"]),
                date=apple_to_datetime(original_row["date"]),
                rowid=original_row["ROWID"],
            )

            # Get all replies in this thread, ordered by date
            cursor.execute("""
                SELECT m.ROWID, m.text, m.is_from_me, m.date, cmj.chat_id
                FROM message m
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE m.thread_originator_guid = ?
                  AND m.text IS NOT NULL
                  AND m.text != ''
                ORDER BY m.date
            """, (thread_guid,))
            reply_rows = cursor.fetchall()

            # Build the full thread: original + all replies
            thread_messages = [original]
            chat_id = None

            for row in reply_rows:
                thread_messages.append(ThreadMessage(
                    text=row["text"],
                    is_from_me=bool(row["is_from_me"]),
                    date=apple_to_datetime(row["date"]),
                    rowid=row["ROWID"],
                ))
                if row["chat_id"]:
                    chat_id = row["chat_id"]

            is_group = chat_id in group_chats if chat_id else False
            thread_size = len(thread_messages)

            # Extract (context, my_response) for each of MY messages in the thread
            for pos, msg in enumerate(thread_messages):
                if msg.is_from_me and pos > 0:  # Skip if I started the thread (no context)
                    # Context = all messages before this one
                    context = thread_messages[:pos]

                    # Only include if there's at least one message from them in context
                    has_their_message = any(not m.is_from_me for m in context)
                    if not has_their_message:
                        continue

                    all_contexts.append(ConversationContext(
                        context_messages=context,
                        my_response=msg,
                        thread_guid=thread_guid,
                        thread_size=thread_size,
                        position_in_thread=pos + 1,
                        is_group=is_group,
                        chat_id=chat_id,
                    ))

        return all_contexts

    finally:
        conn.close()


def print_sample_conversations(contexts: list[ConversationContext], n: int = 5) -> None:
    """Print sample conversations for review."""
    import random

    samples = random.sample(contexts, min(n, len(contexts)))

    print(f"\n{'='*70}")
    print(f"SAMPLE CONVERSATIONS (showing {len(samples)} of {len(contexts):,})")
    print('='*70)

    for i, ctx in enumerate(samples, 1):
        print(f"\n{'─'*70}")
        print(f"CONVERSATION {i} (thread: {ctx.thread_size}, pos: {ctx.position_in_thread})")
        print(f"{'─'*70}")
        print("\n[CONTEXT]:")
        for msg in ctx.context_messages[-5:]:  # Show last 5 context messages
            who = "  Me" if msg.is_from_me else "Them"
            print(f"  {who}: {msg.text[:100]}")
        if len(ctx.context_messages) > 5:
            print(f"  ... ({len(ctx.context_messages) - 5} earlier messages)")
        print(f"\n[MY RESPONSE]: {ctx.my_response.text[:200]}")
        print(f"\nGroup chat: {ctx.is_group}")


def export_conversations(
    contexts: list[ConversationContext],
    output_path: Path,
    limit: int | None = None,
    min_context: int = 1,
    max_context: int | None = None,
) -> int:
    """Export conversations to JSONL file."""
    import random

    # Filter by context length
    filtered = [c for c in contexts if len(c.context_messages) >= min_context]
    if max_context:
        filtered = [c for c in filtered if len(c.context_messages) <= max_context]

    # Randomize for representative sample
    random.shuffle(filtered)

    if limit:
        filtered = filtered[:limit]

    with open(output_path, "w") as f:
        for ctx in filtered:
            f.write(json.dumps(ctx.to_dict()) + "\n")

    return len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full conversation context from iMessage threads"
    )
    parser.add_argument("--sample", type=int, default=0,
                       help="Show N sample conversations")
    parser.add_argument("--export", type=str,
                       help="Export conversations to JSONL file")
    parser.add_argument("--limit", type=int,
                       help="Limit export to N conversations")
    parser.add_argument("--min-context", type=int, default=1,
                       help="Minimum context messages required (default: 1)")
    parser.add_argument("--max-context", type=int,
                       help="Maximum context messages (default: no limit)")
    parser.add_argument("--db-path", type=str,
                       help="Custom chat.db path")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_chat_db_path()

    if not db_path.exists():
        print(f"Error: chat.db not found at {db_path}")
        return

    print(f"Extracting threaded conversations from {db_path}...")
    print("-" * 60)

    contexts = extract_thread_conversations(db_path)

    # Statistics
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"Total (context, my_response) pairs: {len(contexts):,}")

    # Context length distribution
    context_lengths = [len(c.context_messages) for c in contexts]
    print("\nContext length distribution:")
    print(f"  Min: {min(context_lengths)}")
    print(f"  Max: {max(context_lengths)}")
    print(f"  Avg: {sum(context_lengths) / len(context_lengths):.1f}")

    # By context length buckets
    buckets = {
        "1 message context": len([c for c in contexts if len(c.context_messages) == 1]),
        "2-5 messages": len([c for c in contexts if 2 <= len(c.context_messages) <= 5]),
        "6-20 messages": len([c for c in contexts if 6 <= len(c.context_messages) <= 20]),
        "21+ messages": len([c for c in contexts if len(c.context_messages) > 20]),
    }
    print("\nBy context length:")
    for bucket, count in buckets.items():
        pct = (count / len(contexts)) * 100
        print(f"  {bucket:20} {count:6,} ({pct:5.1f}%)")

    # Group vs direct
    group_count = len([c for c in contexts if c.is_group])
    direct_count = len(contexts) - group_count
    print("\nBy chat type:")
    print(f"  Direct messages: {direct_count:,}")
    print(f"  Group chats:     {group_count:,}")

    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHT")
    print("=" * 60)
    print(f"You have {len(contexts):,} conversation pairs WITH FULL CONTEXT!")
    print("Each pair includes all previous messages in the thread.")
    print("This is much better than isolated (trigger, response) pairs.")

    if args.sample > 0:
        print_sample_conversations(contexts, args.sample)

    if args.export:
        export_path = Path(args.export)
        count = export_conversations(
            contexts,
            export_path,
            limit=args.limit,
            min_context=args.min_context,
            max_context=args.max_context,
        )
        print(f"\n✅ Exported {count:,} conversations to {export_path}")


if __name__ == "__main__":
    main()
