from integrations.imessage import ChatDBReader

target_chat_id = 'iMessage;-;+15629643639'

with ChatDBReader() as reader:
    messages = reader.get_messages(target_chat_id, limit=500)

    keywords = ["neuropathy", "ultimate frisbee"]

    print(f"--- Message Sender Check for {target_chat_id} ---")
    for m in messages:
        text = (m.text or "").lower()
        if any(k in text for k in keywords):
            sender_display = "Me" if m.is_from_me else "Radhika"
            print(f"Sender: {sender_display} (is_from_me: {m.is_from_me})")
            print(f"Text: {m.text}")
            print("-" * 20)
