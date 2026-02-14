from integrations.imessage.reader import ChatDBReader

def find_lavanya():
    with ChatDBReader() as reader:
        convs = reader.get_conversations(limit=200)
        print(f"Checking {len(convs)} conversations...")
        for c in convs:
            if c.display_name:
                print(f"Name: {c.display_name} | ID: {c.chat_id} | Participants: {c.participants}")
            if c.display_name and "Lavanya" in c.display_name:
                print(f"FOUND LAVANYA: {c.chat_id}")
            if any("14084643141" in p for p in c.participants):
                print(f"FOUND PARTICIPANT MATCH (+14084643141): {c.display_name} | {c.chat_id}")

if __name__ == "__main__":
    find_lavanya()
