#!/usr/bin/env python3
"""Final Verification: LFM 0.7B Base with Anti-AI Prompts."""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, UTC

sys.path.insert(0, ".")

from jarvis.db import get_db
from integrations.imessage import ChatDBReader
from jarvis.reply_service import ReplyService
from jarvis.contracts.pipeline import MessageContext, ClassificationResult, IntentType, CategoryType, UrgencyLevel
from models.loader import ModelConfig
from models.generator import MLXGenerator

def run_bakeoff():
    db = get_db()
    reader = ChatDBReader()
    
    model_path = "models/lfm-0.7b-4bit"
    
    print("=" * 80)
    print(f"VERIFYING LFM 0.7B BASE")
    print("=" * 80)

    # Fetch a real conversation
    convs = reader.get_conversations(limit=1)
    if not convs: 
        print("No conversations found.")
        return
    
    conv = convs[0]
    messages = reader.get_messages(conv.chat_id, limit=5)
    # messages[0] is the most recent (incoming)
    incoming = messages[0].text or "Hey what's up?"
    # messages[1:] are previous messages
    thread = []
    for m in reversed(messages[1:]):
        if not m.text:
            continue
        prefix = "Me" if m.is_from_me else "Them"
        thread.append(f"{prefix}: {m.text}")
    
    # Clean up input for display
    print(f"Test Input: \"{incoming}\"\n")
    print(f"Thread Context ({len(thread)} msgs):")
    for t in thread:
        print(f"  {t}")

    message_context = MessageContext(
        chat_id=conv.chat_id,
        message_text=incoming,
        is_from_me=messages[0].is_from_me,
        timestamp=datetime.now(UTC),
        metadata={"thread": thread}
    )

    dummy_class = ClassificationResult(
        intent=IntentType.STATEMENT,
        category=CategoryType.FULL_RESPONSE,
        urgency=UrgencyLevel.LOW,
        confidence=1.0,
        requires_knowledge=False,
        metadata={"category_name": "statement"}
    )

    try:
        m_cfg = ModelConfig(model_path=model_path)
        gen = MLXGenerator(config=m_cfg, skip_templates=True)
        service = ReplyService(generator=gen)
        
        start = time.perf_counter()
        # Pass empty search_results to test zero-shot capabilities first
        response = service.generate_reply(message_context, classification=dummy_class, search_results=[])
        elapsed = (time.perf_counter() - start) * 1000
        
        # Show the actual prompt the model saw
        final_prompt = response.metadata.get("final_prompt", "(not captured)")
        print(f"\n--- PROMPT SENT TO MODEL ---")
        print(final_prompt)
        print(f"--- END PROMPT ---\n")
        
        print(f"Time: {elapsed:.0f}ms")
        print(f"Reply: \"{response.response}\"")
        
        if not response.response:
            print("ERROR: Response is still empty.")
            
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    run_bakeoff()
