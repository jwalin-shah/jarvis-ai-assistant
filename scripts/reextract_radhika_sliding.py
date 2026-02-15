import logging

from integrations.imessage.reader import ChatDBReader
from jarvis.contacts.fact_storage import delete_facts_for_contact, save_facts
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.db import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

CHAT_ID = "iMessage;-;+15629643639"

class MockSegment:
    def __init__(self, messages):
        self.messages = messages
        self.text = "\n".join([" ".join((m.text or "").splitlines()) for m in messages])

def sliding_window_extraction_with_names():
    db = get_db()

    # 1. Get Real Names
    user_name = "Jwalin" # Assuming Jwalin is the user
    contact_name = "Radhika"
    with db.connection() as conn:
        row = conn.execute("SELECT display_name FROM contacts WHERE chat_id = ?", (CHAT_ID,)).fetchone()
        if row and row[0]:
            contact_name = row[0].split()[0] # Use first name

    print(f"--- Running Extraction for {contact_name} (User: {user_name}) ---")

    # 2. Clear old facts
    delete_facts_for_contact(CHAT_ID)

    # 3. Fetch messages
    reader = ChatDBReader()
    messages = reader.get_messages(CHAT_ID, limit=300)
    messages.reverse()

    # 4. Create Sliding Windows
    window_size = 25
    overlap = 5
    windows = []
    for i in range(0, len(messages), window_size - overlap):
        window = messages[i:i + window_size]
        if len(window) < 5: break
        windows.append(MockSegment(window))

    print(f"Created {len(windows)} sliding windows.")

    # 5. Extract with 1.2b (Identity-Keyed + Self-Correction)
    extractor = get_instruction_extractor(tier="1.2b")

    all_facts = []
    for i, seg in enumerate(windows):
        print(f"\nProcessing window {i+1}/{len(windows)} ({len(seg.messages)} msgs)...")
        # PASS REAL NAMES HERE
        facts = extractor.extract_facts_from_segment(
            seg,
            contact_id=CHAT_ID,
            contact_name=contact_name,
            user_name=user_name
        )
        if facts:
            print(f"  Verified {len(facts)} facts.")
            for f in facts:
                print(f"    - {f.subject}: {f.value} (NLI: {f.confidence:.2f})")
            all_facts.extend(facts)
            save_facts(facts, CHAT_ID)

    print(f"\nTotal verified facts saved: {len(all_facts)}")

if __name__ == "__main__":
    sliding_window_extraction_with_names()
