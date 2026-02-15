from jarvis.db import get_db

target_chat_id = "iMessage;-;+15629643639"
db = get_db()
with db.connection() as conn:
    cursor = conn.execute("DELETE FROM contact_facts WHERE contact_id = ?", (target_chat_id,))
    print(f"Deleted {cursor.rowcount} facts for {target_chat_id}")
