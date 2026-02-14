import logging
import os
from jarvis.db import get_db
from integrations.imessage.reader import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.contacts.fact_storage import save_facts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

TARGET_CHATS = [
    {"chat_id": "iMessage;-;+14087867207", "default_name": "Mateo"},
    {"chat_id": "iMessage;-;+14084643141", "default_name": "Friend"}
]

class MockSegment:
    def __init__(self, messages):
        self.messages = messages
        # Safer join to avoid syntax issues in prompt rendering
        self.text = "\n".join([" ".join((m.text or "").splitlines()) for m in messages])

def process_top_contacts():
    reader = ChatDBReader()
    user_name = "Jwalin"
    extractor = get_instruction_extractor(tier="1.2b")
    
    for target in TARGET_CHATS:
        chat_id = target["chat_id"]
        
        # 1. Resolve Contact Name
        contact_name = target["default_name"]
        conv = reader.get_conversation(chat_id)
        if conv and conv.display_name:
            contact_name = conv.display_name.split()[0]
        
        print(f"\n" + "="*60)
        print(f"PROCESSING: {contact_name} (Chat ID: {chat_id})")
        print("="*60)
        
        # 2. Fetch messages
        messages = reader.get_messages(chat_id, limit=300)
        messages.reverse()
        if not messages:
            print(f"No messages found for {chat_id}")
            continue
            
        # 3. Create Sliding Windows
        WINDOW_SIZE = 25
        OVERLAP = 5
        windows = []
        for i in range(0, len(messages), WINDOW_SIZE - OVERLAP):
            window = messages[i:i + WINDOW_SIZE]
            if len(window) < 5: break
            windows.append(MockSegment(window))
        
        print(f"Created {len(windows)} windows from {len(messages)} messages.")
        
        # 4. Extract
        all_facts = []
        for i, seg in enumerate(windows[:5]):
            print(f"\nProcessing window {i+1}/5...")
            facts = extractor.extract_facts_from_segment(
                seg, 
                contact_id=chat_id,
                contact_name=contact_name,
                user_name=user_name
            )
            if facts:
                print(f"  Verified {len(facts)} facts.")
                for f in facts:
                    print(f"    - {f.subject}: {f.value}")
                all_facts.extend(facts)
                save_facts(facts, chat_id)
        
        print(f"\nCompleted {contact_name}. Total verified facts: {len(all_facts)}")

if __name__ == "__main__":
    process_top_contacts()
