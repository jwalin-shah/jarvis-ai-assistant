import sys
import time
import logging
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loader import MLXModelLoader, ModelConfig
from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.contacts.fact_storage import save_facts
from jarvis.db import get_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill")

def resolve_name(db, chat_id, display_name):
    """Robust name resolution logic from test script."""
    contact_name = display_name
    
    try:
        if not contact_name or contact_name in ["Unknown", "None", "Contact"]:
            with db.connection() as conn:
                clean_id = chat_id.split(';')[-1] if ';' in chat_id else chat_id
                row = conn.execute(
                    "SELECT display_name FROM contacts WHERE chat_id LIKE ? OR phone_or_email LIKE ?", 
                    (f"%{clean_id}%", f"%{clean_id}%")
                ).fetchone()
                if row and row[0]:
                    contact_name = row[0]
    except:
        pass
    return contact_name if (contact_name and contact_name not in ["Unknown", "None"]) else "Contact"

def run_backfill():
    db = get_db()
    reader = ChatDBReader()
    extractor = get_instruction_extractor(tier="0.7b")
    
    # Load model once
    if not extractor.is_loaded():
        extractor.load()

    conversations = reader.get_conversations(limit=500)
    # Filter for real chats
    active_chats = [c for c in conversations if c.message_count >= 5 and ("iMessage" in c.chat_id or "RCS" in c.chat_id)]
    
    logger.info(f"Starting backfill for {len(active_chats)} conversations...")

    total_new_facts = 0
    user_name = reader.get_user_name()
    logger.info(f"Identity Anchor: {user_name}")
    
    for conv in active_chats:
        contact_name = resolve_name(db, conv.chat_id, conv.display_name)
        
        # Get last processed ID to avoid duplicates
        last_id = 0
        with db.connection() as conn:
            row = conn.execute("SELECT last_extracted_rowid FROM contacts WHERE chat_id = ?", (conv.chat_id,)).fetchone()
            if row and row[0]:
                last_id = row[0]

        messages = reader.get_messages_after(last_id, conv.chat_id, limit=500)
        if not messages:
            continue

        logger.info(f"Processing '{contact_name}': {len(messages)} new messages")
        
        # Create sliding windows of 25 messages
        max_msg_id = last_id
        for j in range(0, len(messages), 20):
            window = messages[j : j + 25]
            if not window: break
            
            # Update max_id for tracking
            for m in window:
                if m.id and m.id > max_msg_id: max_msg_id = m.id

            # Use extractor (it handles turn-grouping and NLI internally now)
            # Create a mock segment object
            class Seg:
                def __init__(self, msgs):
                    self.messages = msgs
                    self.text = "\n".join([(m.text or "") for m in msgs])
            
            extracted = extractor.extract_facts_from_segment(
                Seg(window), 
                contact_id=conv.chat_id,
                contact_name=contact_name,
                user_name=user_name
            )
            
            if extracted:
                logger.info(f"  ✓ Found {len(extracted)} facts")
                save_facts(extracted, conv.chat_id)
                total_new_facts += len(extracted)

        # Update progress in DB
        with db.connection() as conn:
            conn.execute(
                "UPDATE contacts SET last_extracted_rowid = ?, last_extracted_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
                (max_msg_id, conv.chat_id)
            )

    logger.info(f"Backfill complete. Total new facts saved: {total_new_facts}")
    extractor.unload()

if __name__ == "__main__":
    run_backfill()
