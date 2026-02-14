from integrations.imessage.reader import get_connection_pool

def find_shanay_messages():
    pool = get_connection_pool()
    with pool.connection() as conn:
        # Search for 'Shanay' in messages to find the chat
        rows = conn.execute("""
            SELECT m.text, c.guid, c.display_name
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text LIKE '%Shanay%'
            LIMIT 10
        """).fetchall()
        for r in rows:
            print(dict(r))

if __name__ == "__main__":
    find_shanay_messages()
