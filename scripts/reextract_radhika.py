import logging

from integrations.imessage.reader import ChatDBReader
from jarvis.contacts.fact_storage import delete_facts_for_contact, save_facts
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.db import get_db
from jarvis.topics.topic_segmenter import segment_conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

CHAT_ID = "iMessage;-;+15629643639"


def reextract():
    _ = get_db()  # Ensure DB initialized

    # 1. Clear old bad facts
    print(f"Clearing old facts for {CHAT_ID}...")
    delete_facts_for_contact(CHAT_ID)

    # 2. Fetch messages
    print(f"Fetching messages for {CHAT_ID}...")
    reader = ChatDBReader()
    messages = reader.get_messages(CHAT_ID, limit=500)
    messages.reverse()

    # 3. Segment
    print("Segmenting...")
    segments = segment_conversation(messages, contact_id=CHAT_ID)
    print(f"Created {len(segments)} segments.")

    # 4. Extract with 1.2b
    print("Loading 1.2b extractor...")
    extractor = get_instruction_extractor(tier="1.2b")

    all_facts = []
    for i, seg in enumerate(segments):
        print(f"Processing segment {i + 1}/{len(segments)}...")
        facts = extractor.extract_facts_from_segment(seg, contact_id=CHAT_ID)
        if facts:
            print(f"  Found {len(facts)} facts.")
            for f in facts:
                print(f"    - {f.category}: {f.subject} {f.predicate} {f.value}")
            all_facts.extend(facts)
            save_facts(facts, CHAT_ID)

    print(f"\nTotal facts saved: {len(all_facts)}")


if __name__ == "__main__":
    reextract()
