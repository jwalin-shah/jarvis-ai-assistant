#!/usr/bin/env python3
"""Pull real conversation segments from iMessage DB for model testing.

Resolves phone numbers to contact names via AddressBook.
Outputs segments in the format the 350M model will see.

Usage:
    uv run python scripts/pull_segments.py
    uv run python scripts/pull_segments.py --chat-id "iMessage;-;+14081234567"
    uv run python scripts/pull_segments.py --limit 5 --msgs 30
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, ".")

# AddressBook path for contact name resolution
ADDRESSBOOK_PATH = Path.home() / "Library/Application Support/AddressBook/Sources"
CHAT_DB = Path.home() / "Library/Messages/chat.db"


def load_contacts() -> dict[str, str]:
    """Load phone->name mapping from macOS AddressBook."""
    contacts: dict[str, str] = {}
    if not ADDRESSBOOK_PATH.exists():
        print("Warning: AddressBook not found, using phone numbers", flush=True)
        return contacts

    for source_dir in ADDRESSBOOK_PATH.iterdir():
        if not source_dir.is_dir():
            continue
        ab_db = source_dir / "AddressBook-v22.abcddb"
        if not ab_db.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{ab_db}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT
                    ZABCDPHONENUMBER.ZFULLNUMBER as phone,
                    ZABCDRECORD.ZFIRSTNAME as first,
                    ZABCDRECORD.ZLASTNAME as last
                FROM ZABCDPHONENUMBER
                JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK
                WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL
                  AND ZABCDRECORD.ZFIRSTNAME IS NOT NULL
            """)
            for row in cur.fetchall():
                phone = row["phone"]
                first = row["first"] or ""
                last = row["last"] or ""
                name = f"{first} {last}".strip()
                if phone and name:
                    # Normalize: strip spaces, dashes, parens
                    normalized = "".join(c for c in phone if c.isdigit() or c == "+")
                    if not normalized.startswith("+") and len(normalized) == 10:
                        normalized = "+1" + normalized
                    contacts[normalized] = name
            conn.close()
        except Exception as e:
            print(f"Warning: Could not read {ab_db}: {e}", flush=True)

    print(f"Loaded {len(contacts)} contacts from AddressBook", flush=True)
    return contacts


def resolve_name(identifier: str, contacts: dict[str, str]) -> str:
    """Resolve a phone/email to a contact name."""
    if not identifier:
        return "Unknown"
    # Normalize phone
    normalized = "".join(c for c in identifier if c.isdigit() or c == "+")
    if not normalized.startswith("+") and len(normalized) == 10:
        normalized = "+1" + normalized
    return contacts.get(normalized, identifier)


def pull_conversations(limit: int = 5) -> list[dict]:
    """Pull recent conversations with message counts."""
    conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        WITH chat_stats AS (
            SELECT cmj.chat_id, COUNT(*) as msg_count,
                   MAX(m.date) as last_date
            FROM chat_message_join cmj
            JOIN message m ON cmj.message_id = m.ROWID
            GROUP BY cmj.chat_id
            HAVING msg_count > 30
        )
        SELECT c.ROWID, c.guid, c.display_name, c.chat_identifier,
               cs.msg_count, c.style
        FROM chat c
        JOIN chat_stats cs ON c.ROWID = cs.chat_id
        WHERE c.style = 45
        ORDER BY cs.last_date DESC
        LIMIT ?
    """,
        (limit,),
    )

    results = []
    for row in cur.fetchall():
        results.append(
            {
                "rowid": row["ROWID"],
                "guid": row["guid"],
                "display_name": row["display_name"],
                "chat_identifier": row["chat_identifier"],
                "msg_count": row["msg_count"],
            }
        )
    conn.close()
    return results


def pull_messages(chat_rowid: int, limit: int = 50) -> list[dict]:
    """Pull recent messages from a chat."""
    conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT m.ROWID, m.text, m.is_from_me, m.date,
               h.id as handle_id
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE cmj.chat_id = ?
          AND m.text IS NOT NULL
          AND m.text != ''
        ORDER BY m.date DESC
        LIMIT ?
    """,
        (chat_rowid, limit),
    )

    messages = []
    for row in cur.fetchall():
        messages.append(
            {
                "id": row["ROWID"],
                "text": row["text"],
                "is_from_me": bool(row["is_from_me"]),
                "handle": row["handle_id"] or "",
                "date": row["date"],
            }
        )
    conn.close()
    # Reverse to chronological order
    messages.reverse()
    return messages


def format_segment(
    messages: list[dict], contacts: dict[str, str], user_name: str = "Jwalin"
) -> str:
    """Format messages as a segment for model input."""
    lines = []
    for msg in messages:
        if msg["is_from_me"]:
            speaker = user_name
        else:
            speaker = resolve_name(msg["handle"], contacts)
            # Use first name only
            speaker = speaker.split()[0] if " " in speaker else speaker
        text = msg["text"].replace("\n", " ").strip()
        # Skip empty or media-only messages
        if not text or text == "\ufffc":
            continue
        lines.append(f"{speaker} says: {text}")
    return "\n".join(lines)


def find_fact_rich_segments(messages: list[dict], window: int = 8) -> list[list[dict]]:
    """Find segments likely to contain facts (health, jobs, locations, etc.)."""
    fact_keywords = [
        "moved",
        "moving",
        "live in",
        "lives in",
        "from",
        "job",
        "work",
        "hired",
        "started",
        "company",
        "quit",
        "doctor",
        "hospital",
        "sick",
        "allergic",
        "surgery",
        "married",
        "engaged",
        "dating",
        "broke up",
        "divorced",
        "graduated",
        "school",
        "college",
        "degree",
        "major",
        "sister",
        "brother",
        "mom",
        "dad",
        "cousin",
        "uncle",
        "birthday",
        "pregnant",
        "baby",
        "born",
        "bought",
        "apartment",
        "house",
        "lease",
    ]

    segments = []
    for i, msg in enumerate(messages):
        text_lower = (msg["text"] or "").lower()
        if any(kw in text_lower for kw in fact_keywords):
            # Grab window around the match
            start = max(0, i - window // 2)
            end = min(len(messages), i + window // 2)
            segment = messages[start:end]
            # Avoid duplicates (overlapping windows)
            if segments and segments[-1][-1]["id"] >= segment[0]["id"]:
                continue
            segments.append(segment)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Pull conversation segments from iMessage")
    parser.add_argument("--limit", type=int, default=5, help="Number of conversations to scan")
    parser.add_argument("--msgs", type=int, default=100, help="Messages per conversation to pull")
    parser.add_argument("--chat-id", type=str, help="Specific chat ROWID to pull from")
    parser.add_argument("--output", type=str, help="Save segments to JSON file")
    args = parser.parse_args()

    contacts = load_contacts()

    if args.chat_id:
        chat_rowid = int(args.chat_id)
        messages = pull_messages(chat_rowid, args.msgs)
        segments = find_fact_rich_segments(messages)
        print(f"\nFound {len(segments)} fact-rich segments in chat {chat_rowid}\n", flush=True)
        for i, seg in enumerate(segments):
            formatted = format_segment(seg, contacts)
            print(f"--- Segment {i + 1} ({len(seg)} msgs) ---", flush=True)
            print(formatted, flush=True)
            print(flush=True)
    else:
        convos = pull_conversations(args.limit)
        print(f"\nTop {len(convos)} conversations:\n", flush=True)

        all_segments = []
        for c in convos:
            name = c["display_name"] or resolve_name(
                c["chat_identifier"].replace("iMessage;-;", ""), contacts
            )
            print(f"  {name:20} | {c['msg_count']} msgs | ROWID {c['rowid']}", flush=True)

            messages = pull_messages(c["rowid"], args.msgs)
            segments = find_fact_rich_segments(messages)
            print(f"    -> {len(segments)} fact-rich segments found", flush=True)

            for seg in segments[:3]:  # Max 3 segments per conversation
                formatted = format_segment(seg, contacts)
                all_segments.append(
                    {
                        "chat": name,
                        "message_count": len(seg),
                        "formatted": formatted,
                    }
                )
                print(f"\n    --- Segment ({len(seg)} msgs) ---", flush=True)
                print(f"    {formatted[:200]}...", flush=True)
            print(flush=True)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_segments, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(all_segments)} segments to {args.output}", flush=True)

        print(
            f"\nTotal: {len(all_segments)} segments across {len(convos)} conversations", flush=True
        )


if __name__ == "__main__":
    main()
