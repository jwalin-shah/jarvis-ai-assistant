#!/usr/bin/env python3
"""Export topic segments for fact extraction evaluation.

Reads persisted segments from JARVIS DB, joins with iMessage chat.db to get
message text, resolves contact names, and formats as 'Name: "text"'.

Pre-processing:
- Bot messages filtered (CVS, short codes, spam)
- Slang expanded for model consumption
- Unresolved contacts (raw phone numbers) flagged

Stratified sampling: ~5 segments per chat, 3+ messages each, target 150-200
total segments from 20+ chats.

Usage:
    uv run python scripts/export_eval_segments.py
    uv run python scripts/export_eval_segments.py --per-chat 5 --min-messages 3
"""

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

APPLE_EPOCH_UNIX = 978307200  # 2001-01-01 00:00:00 UTC
NANOSECONDS_PER_SECOND = 1_000_000_000
OUTPUT_PATH = Path("training_data/segment_eval/segments_exported.json")

# Matches raw phone numbers, chat_id handles, or email-only identifiers
_PHONE_RE = re.compile(r"^\+?\d{10,15}$")
_HANDLE_RE = re.compile(r"^(?:iMessage|SMS|RCS);")


def _is_unresolved_contact(name: str) -> bool:
    """Check if a contact name is unresolved (phone number, raw handle, or email)."""
    name = name.strip()
    if _PHONE_RE.match(name):
        return True
    if _HANDLE_RE.match(name):
        return True
    # Email-only (no real name resolved)
    if "@" in name and " " not in name:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export segments for eval")
    parser.add_argument("--per-chat", type=int, default=5, help="Segments per chat")
    parser.add_argument("--min-messages", type=int, default=3, help="Min messages per segment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    from integrations.imessage import CHAT_DB_PATH, ChatDBReader
    from jarvis.contacts.junk_filters import is_bot_message
    from jarvis.db import get_db
    from jarvis.nlp.slang import expand_slang
    from jarvis.topics.segment_storage import get_segments_for_chat

    db = get_db()

    # Step 1: Get all chats that have segments
    print("Querying chats with segments...", flush=True)
    with db.connection() as conn:
        chat_rows = conn.execute(
            """
            SELECT DISTINCT chat_id, contact_id, COUNT(*) as seg_count
            FROM conversation_segments
            GROUP BY chat_id
            HAVING seg_count >= 1
            ORDER BY seg_count DESC
            """
        ).fetchall()

    print(f"  Found {len(chat_rows)} chats with segments", flush=True)

    # Step 2: Open iMessage chat.db + ChatDBReader for name resolution
    imessage_uri = f"file:{CHAT_DB_PATH}?mode=ro"
    imessage_conn = sqlite3.connect(imessage_uri, uri=True)
    imessage_conn.row_factory = sqlite3.Row
    reader = ChatDBReader()
    reader.__enter__()  # open connection for contact resolution

    exported = []
    stats = {"total_segments": 0, "bot_filtered": 0, "unresolved_filtered": 0}
    t0 = time.time()

    for chat_row in chat_rows:
        chat_id = chat_row["chat_id"]
        contact_id = chat_row["contact_id"]

        # Get segments for this chat
        with db.connection() as conn:
            segments = get_segments_for_chat(conn, chat_id, limit=100)

        # Filter: min message count
        eligible = [s for s in segments if s["message_count"] >= args.min_messages]
        if not eligible:
            continue

        # Sample up to per_chat segments
        sampled = random.sample(eligible, min(len(eligible), args.per_chat))

        for seg in sampled:
            stats["total_segments"] += 1

            # Get message rowids
            msg_rowids = [m["message_rowid"] for m in seg["messages"]]
            if not msg_rowids:
                continue

            # Fetch message text + metadata from iMessage
            placeholders = ",".join("?" * len(msg_rowids))
            rows = imessage_conn.execute(
                f"""
                SELECT m.ROWID, m.text, m.is_from_me, m.date,
                       COALESCE(h.id, 'me') AS sender_handle
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID IN ({placeholders})
                ORDER BY m.date ASC, m.ROWID ASC
                """,  # noqa: S608
                msg_rowids,
            ).fetchall()

            if not rows:
                continue

            # Format as 'Name: "text"' with bot filtering + slang expansion
            lines = []
            contact_name = None
            all_bot = True

            for row in rows:
                text = row["text"]
                if not text or not text.strip():
                    continue

                # Bot filtering: skip bot messages
                if is_bot_message(text, chat_id):
                    continue

                all_bot = False

                if row["is_from_me"]:
                    speaker = "Jwalin"
                else:
                    handle = row["sender_handle"]
                    resolved = reader._resolve_contact_name(handle)
                    speaker = resolved or handle
                    if contact_name is None:
                        contact_name = speaker

                # Apply slang expansion for model consumption
                cleaned_text = expand_slang(text.strip())

                lines.append(f'{speaker}: "{cleaned_text}"')

            if not lines:
                if all_bot:
                    stats["bot_filtered"] += 1
                continue

            resolved_name = contact_name or contact_id or chat_id

            # Flag unresolved contacts (raw phone numbers)
            if _is_unresolved_contact(resolved_name):
                stats["unresolved_filtered"] += 1
                continue

            exported.append(
                {
                    "segment_id": seg["segment_id"],
                    "chat_id": chat_id,
                    "contact_name": resolved_name,
                    "topic_label": seg["topic_label"],
                    "message_count": len(lines),
                    "formatted_text": "\n".join(lines),
                }
            )

        print(
            f"  {chat_id[:30]}: {len(sampled)} segments sampled",
            flush=True,
        )

    reader.__exit__(None, None, None)
    imessage_conn.close()

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(exported, f, indent=2)

    elapsed = time.time() - t0

    # Stats
    unique_contacts = len({e["contact_name"] for e in exported})
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"  Exported: {len(exported)} segments", flush=True)
    print(f"  Unique contacts: {unique_contacts}", flush=True)
    print(f"  Bot-filtered segments: {stats['bot_filtered']}", flush=True)
    print(f"  Unresolved-contact segments: {stats['unresolved_filtered']}", flush=True)
    print(f"  Output: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
