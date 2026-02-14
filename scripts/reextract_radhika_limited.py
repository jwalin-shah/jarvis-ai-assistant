import logging
import os
from jarvis.db import get_db
from integrations.imessage.reader import ChatDBReader
from jarvis.topics.topic_segmenter import segment_conversation
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.contacts.fact_storage import save_facts, delete_facts_for_contact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

CHAT_ID = "iMessage;-;+15629643639"

def reextract_limited():
    db = get_db()
    
    # 1. Clear old facts
    print(f"Clearing old facts for {CHAT_ID}...")
    delete_facts_for_contact(CHAT_ID)
    
    # 2. Fetch messages (Higher limit to get more history for segmentation)
    print(f"Fetching messages for {CHAT_ID}...")
    reader = ChatDBReader()
    messages = reader.get_messages(CHAT_ID, limit=1000)
    messages.reverse()
    
    # 3. Segment (uses updated 48h gap logic and drift_threshold=0.6)
    print("Segmenting...")
    segments = segment_conversation(messages, contact_id=CHAT_ID, drift_threshold=0.6)
    print(f"Created {len(segments)} segments.")
    
    # 4. Limit to 50 segments
    target_segments = segments[:50]
    print(f"--- Running Extraction on {len(target_segments)} segments ---")
    
    # 5. Extract with 1.2b (High quality suggester) + NLI (Verifier)
    extractor = get_instruction_extractor(tier="1.2b")
    
    all_facts = []
    for i, seg in enumerate(target_segments):
        print(f"Processing segment {i+1}/{len(target_segments)} [Topic: {seg.topic_label}]...")
        # Every fact here will pass through NLI verification inside this call
        facts = extractor.extract_facts_from_segment(seg, contact_id=CHAT_ID)
        if facts:
            print(f"  Verified {len(facts)} facts.")
            for f in facts:
                print(f"    - {f.category}: {f.subject} {f.predicate} {f.value} (NLI Score: {f.confidence:.2f})")
            all_facts.extend(facts)
            save_facts(facts, CHAT_ID)

    print(f"\nTotal verified facts saved: {len(all_facts)}")

if __name__ == "__main__":
    reextract_limited()
