import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.contacts.contact_profile import get_contact_profile, ContactProfileBuilder
from integrations.imessage.reader import ChatDBReader
from jarvis.db import get_db

def check_profiles():
    reader = ChatDBReader()
    builder = ContactProfileBuilder(min_messages=5)
    db = get_db()
    
    # 1. Target Chats
    targets = [
        ("iMessage;-;+14084643141", "Lavanya"),
        ("iMessage;-;+14087867207", "Mateo"),
        ("iMessage;-;+16505397073", "Clera")
    ]
    
    for chat_id, name in targets:
        print(f"\n{'='*80}\nPROFILE CHECK: {name} ({chat_id})\n{'='*80}")
        
        # Build profile (this now includes DB fact fetching)
        messages = reader.get_messages(chat_id, limit=100)
        profile = builder.build_profile(chat_id, messages, contact_name=name)
        
        print(f"Relationship: {profile.relationship} (Conf: {profile.relationship_confidence:.2f})")
        print(f"Style: {profile.formality} (Score: {profile.formality_score:.2f})")
        
        print(f"\nExtracted Facts ({len(profile.extracted_facts)}):")
        if not profile.extracted_facts:
            print("  (No facts found in DB yet - run backfill script first)")
        for f in profile.extracted_facts[:10]:
            print(f"  ✓ [{f['category']}] {f['subject']} {f['predicate']} {f['value']} (Conf: {f['confidence']:.2f})")
            
        # Check topic labels
        if profile.top_topics:
            print(f"\nTop Topics: {', '.join(profile.top_topics)}")

if __name__ == "__main__":
    check_profiles()
