import sqlite3
from pathlib import Path

from integrations.imessage.parser import extract_text_from_row
from jarvis.db import get_db
from jarvis.topics.segment_storage import get_segments_for_chat

db = get_db()
chat_id = "iMessage;-;+17204963920"  # Sangati Shah
CHAT_DB_PATH = Path.home() / "Library/Messages/chat.db"


def get_message_text(mid):
    try:
        conn = sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT text, attributedBody, is_from_me FROM message WHERE ROWID = ?", (mid,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return "[MISSING ROW]", None
        txt = extract_text_from_row(row)
        return txt if txt else "[EMPTY TEXT]", row["is_from_me"]
    except Exception as e:
        return f"Error: {e}", None


with db.connection() as conn:
    segments = get_segments_for_chat(conn, chat_id, limit=5)

print(f"--- Topic Segmentation for {chat_id} ---")
if not segments:
    print("No segments found in jarvis.db!")

for i, seg in enumerate(segments):
    print(f"\nSEGMENT {i + 1}: {seg['topic_label']}")
    print(f"Keywords: {seg['keywords_json']}")
    print(f"Messages ({seg['message_count']}):")

    # Sort by position
    msg_details = sorted(seg["messages"], key=lambda x: x["position"])
    for m in msg_details[:5]:  # Show first 5
        text, is_from_me = get_message_text(m["message_rowid"])
        sender = "Me" if is_from_me else "Them"
        print(f"  [{sender}] (ID:{m['message_rowid']}) {text[:100]}")
    if len(msg_details) > 5:
        print(f"  ... (+{len(msg_details) - 5} more)")
