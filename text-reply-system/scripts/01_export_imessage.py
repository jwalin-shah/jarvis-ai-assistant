#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def apple_ns_to_iso(value: int | float | None) -> str | None:
    if value in (None, 0):
        return None
    try:
        seconds = float(value) / 1_000_000_000
        dt = APPLE_EPOCH + timedelta(seconds=seconds)
        return dt.astimezone().isoformat()
    except Exception:
        return None


def decode_attributed_body(blob: bytes | None) -> str | None:
    if not blob:
        return None
    try:
        text = blob.decode("utf-8", errors="ignore")
        filtered = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        filtered = " ".join(filtered.split())
        return filtered.strip() or None
    except Exception:
        return None


def export_messages(chat_db: Path, output_path: Path) -> int:
    query = """
    SELECT
        COALESCE(h.id, 'unknown') AS contact,
        m.text,
        m.attributedBody,
        m.is_from_me,
        m.date,
        c.display_name,
        c.room_name,
        c.guid
    FROM message m
    LEFT JOIN handle h ON h.ROWID = m.handle_id
    LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
    LEFT JOIN chat c ON c.ROWID = cmj.chat_id
    ORDER BY m.date ASC
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with sqlite3.connect(str(chat_db)) as conn, output_path.open("w", encoding="utf-8") as out:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query)
        for row in cur:
            text = row["text"]
            if text is None:
                text = decode_attributed_body(row["attributedBody"])
            if not text:
                continue

            # Skip likely group chats for first pass
            display_name = row["display_name"]
            room_name = row["room_name"]
            guid = row["guid"] or ""
            if display_name or room_name or "chat" in guid and ";-;" in guid:
                continue

            ts = apple_ns_to_iso(row["date"])
            if not ts:
                continue

            rec = {
                "contact": row["contact"],
                "text": text,
                "is_from_me": bool(row["is_from_me"]),
                "timestamp": ts,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export iMessage chat.db to JSONL")
    parser.add_argument(
        "--chat-db",
        default=str(Path.home() / "Library" / "Messages" / "chat.db"),
        help="Path to chat.db",
    )
    parser.add_argument(
        "--output",
        default="data/raw/messages.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    chat_db = Path(args.chat_db)
    output = Path(args.output)

    print(f"[01] Reading chat DB: {chat_db}")
    if not chat_db.exists():
        raise SystemExit(f"chat.db not found at {chat_db}")

    exported = export_messages(chat_db, output)
    print(f"[01] Exported {exported} messages -> {output}")


if __name__ == "__main__":
    main()
