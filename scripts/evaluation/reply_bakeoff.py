#!/usr/bin/env python3
"""Reply Bakeoff: Compare reply generation strategies side-by-side.

Evaluates how the presence/absence of the response classifier and different
RAG configurations affect the quality of generated replies.

Usage:
    uv run python scripts/reply_bakeoff.py
"""

from __future__ import annotations

import sys
import time
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

# Setup logging to show debug info from instrumented modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Reduce noise from noisy libraries
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

sys.path.insert(0, ".")

from jarvis.reply_service import get_reply_service
from jarvis.contracts.pipeline import MessageContext, ClassificationResult, CategoryType, IntentType, UrgencyLevel
from jarvis.db.models import Contact

class TestCase(TypedDict):
    id: str
    chat_id: str
    contact_name: str
    incoming: str
    thread: List[str]
    relationship: str
    last_is_from_me: bool

# --- TEST CASES ---
# A mix of inputs to test classification and RAG.
TEST_CASES: List[TestCase] = [
    {
        "id": "coffee_invite",
        "chat_id": "bakeoff_1",
        "contact_name": "Sarah",
        "incoming": "Hey! Are you free for coffee tomorrow afternoon?",
        "thread": ["Sarah: Hey! How's it going?"],
        "relationship": "friend",
        "last_is_from_me": False,
    },
    {
        "id": "urgent_work",
        "chat_id": "bakeoff_2",
        "contact_name": "Mike (Boss)",
        "incoming": "Did you finish the slide deck for the meeting at 4?",
        "thread": ["Mike (Boss): Just checking in on the deck."],
        "relationship": "coworker",
        "last_is_from_me": False,
    },
    {
        "id": "emotional_vent",
        "chat_id": "bakeoff_3",
        "contact_name": "Mom",
        "incoming": "I'm just feeling so overwhelmed with everything happening at home right now.",
        "thread": ["You: Hey mom, how are things?"],
        "relationship": "family",
        "last_is_from_me": False,
    },
    {
        "id": "followup_me",
        "chat_id": "bakeoff_4",
        "contact_name": "Alex",
        "incoming": "Wait, I forgot to ask about the concert tickets!",
        "thread": [
            "Alex: See you later!",
            "You: Sounds good! Can't wait."
        ],
        "relationship": "close friend",
        "last_is_from_me": True,
    },
    {
        "id": "minimal_context",
        "chat_id": "bakeoff_5",
        "contact_name": "Unknown",
        "incoming": "k",
        "thread": [],
        "relationship": "unknown",
        "last_is_from_me": False,
    }
]

def format_rag_metadata(metadata: Dict[str, Any]) -> str:
    """Format RAG metadata for readable output."""
    parts = []
    if "similarity_score" in metadata:
        parts.append(f"Sim: {metadata['similarity_score']:.3f}")
    if "confidence_label" in metadata:
        parts.append(f"Conf: {metadata['confidence_label']}")
    if "category" in metadata:
        parts.append(f"Cat: {metadata['category']}")
    if "vec_candidates" in metadata:
        parts.append(f"Docs: {metadata['vec_candidates']}")
    return " | ".join(parts)

def run_bakeoff() -> None:
    from integrations.imessage import ChatDBReader
    from jarvis.db import JarvisDB, get_db
    from models.loader import ModelConfig
    from models.generator import MLXGenerator
    
    # Force load config model_id
    import json
    from pathlib import Path
    config_path = Path.home() / ".jarvis" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_id = config.get("model_id", "models/lfm2-1.2b-extract-mlx-4bit")
    
    # Initialize fresh service with explicit model
    from jarvis.reply_service import ReplyService
    print(f"Loading Model: {model_id}...")
    cfg = ModelConfig(model_id=model_id)
    gen = MLXGenerator(config=cfg, skip_templates=True)
    service = ReplyService(generator=gen)
    
    db = get_db()
    reader = ChatDBReader()
    
    print("Initializing Bakeoff with REAL messages...")
    
    # 1. Fetch real conversations
    print("Fetching recent conversations...")
    conversations = reader.get_conversations(limit=5)
    
    if not conversations:
        print("No conversations found in chat.db. Make sure Full Disk Access is granted.")
        return

    print("=" * 80)

    for conv in conversations:
        chat_id = conv.chat_id
        
        # 2. Get recent messages for context
        print(f"\n[DEBUG] Fetching messages for {chat_id}...")
        t_msg_start = time.perf_counter()
        messages = reader.get_messages(chat_id, limit=10)
        t_msg_end = time.perf_counter()
        print(f"[DEBUG] Fetching messages took {(t_msg_end - t_msg_start)*1000:.1f}ms")
        
        if not messages:
            continue
            
        # The first message in the list is the most recent
        last_msg_obj = messages[0]
        incoming = last_msg_obj.text or ""
        if not incoming:
            continue
            
        # Build thread (reverse chronological for display, chronological for context)
        thread = [f"{'You' if m.is_from_me else (m.sender_name or 'Them')}: {m.text}" for m in reversed(messages)]
        last_is_from_me = last_msg_obj.is_from_me
        
        # 3. Look up real contact record
        print(f"[DEBUG] Looking up contact for {chat_id}...")
        t_contact_start = time.perf_counter()
        contact = db.get_contact_by_chat_id(chat_id)
        t_contact_end = time.perf_counter()
        print(f"[DEBUG] Contact lookup took {(t_contact_end - t_contact_start)*1000:.1f}ms")
        
        contact_name = conv.display_name or "Unknown"
        relationship = contact.relationship if contact else "unknown"

        print(f"\nREAL CHAT: {chat_id[:20]}... ({contact_name})")
        print(f"Relationship: {relationship}")
        print(f"Incoming: \"{incoming[:100]}{'...' if len(incoming) > 100 else ''}\"")
        if last_is_from_me:
            print("Note: You sent the last message.")
        print("-" * 80)

        message_context = MessageContext(
            chat_id=chat_id,
            message_text=incoming,
            is_from_me=last_is_from_me,
            timestamp=datetime.now(UTC),
            metadata={"thread": [m.text for m in reversed(messages) if m.text]},
        )

        # --- PATH A: CURRENT (WITH CLASSIFIER) ---
        print("\n>> PATH A: CURRENT (With Classifier)")
        
        print("   [DEBUG] Running classifier cascade...")
        t_class_start = time.perf_counter()
        from jarvis.classifiers.cascade import classify_with_cascade
        mobilization = classify_with_cascade(incoming)
        classification = service._build_classification_result(incoming, [m.text for m in reversed(messages) if m.text], mobilization)
        t_class_end = time.perf_counter()
        print(f"   [DEBUG] Classification took {(t_class_end - t_class_start)*1000:.1f}ms")
        
        print(f"   Classification: {classification.metadata.get('category_name')} (conf={classification.confidence:.2f})")
        
        print("   [DEBUG] Calling generate_reply (this often hangs)...")
        start_a = time.perf_counter()
        response_a = service.generate_reply(
            message_context,
            classification=classification,
            contact=contact
        )
        elapsed_a = (time.perf_counter() - start_a) * 1000
        print(f"   [DEBUG] generate_reply total time: {elapsed_a:.1f}ms")
        
        print(f"   RAG Stats: {format_rag_metadata(response_a.metadata)}")
        if response_a.used_kg_facts:
            print("   RAG Docs Used:")
            for doc in response_a.used_kg_facts[:2]:
                print(f"     - {doc[:100]}...")
        
        print(f"   REPLY: '{response_a.response}'")
        if not response_a.response:
            print("   [DEBUG] Empty response! Final Prompt:")
            print(f"--- PROMPT START ---\n{response_a.metadata.get('final_prompt')}\n--- PROMPT END ---")

        # --- PATH B: EXPERIMENTAL (NO CLASSIFIER) ---
        print("\n>> PATH B: EXPERIMENTAL (Skip Classifier - Forced 'statement')")
        start_b = time.perf_counter()
        
        forced_classification = ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.5,
            requires_knowledge=False,
            metadata={
                "category_name": "statement",
                "category_method": "bakeoff_forced"
            }
        )
        
        response_b = service.generate_reply(
            message_context,
            classification=forced_classification,
            contact=contact
        )
        elapsed_b = (time.perf_counter() - start_b) * 1000
        
        print(f"   REPLY ({elapsed_b:.0f}ms): '{response_b.response}'")
        
        # --- PATH C: RAG ENRICHED (Experimental) ---
        print("\n>> PATH C: RAG ENRICHED (Specific Instruction)")
        start_c = time.perf_counter()
        
        response_c = service.generate_reply(
            message_context,
            classification=classification,
            contact=contact,
            instruction="Match the user's slang and abbreviations exactly. Be brief."
        )
        elapsed_c = (time.perf_counter() - start_c) * 1000
        print(f"   REPLY ({elapsed_c:.0f}ms): '{response_c.response}'")

        # --- PATH D: ZERO-SHOT (Experimental) ---
        print("\n>> PATH D: ZERO-SHOT (No few-shot examples)")
        start_d = time.perf_counter()
        
        # We build the request manually to ensure empty examples
        from jarvis.reply_service_generation import build_generation_request
        zero_shot_request = build_generation_request(
            service,
            context=message_context,
            classification=classification,
            search_results=[], # No RAG docs
            contact=contact
        )
        # Clear any examples selected by default
        zero_shot_request.few_shot_examples = []
        
        response_d = service._generate_llm_reply(zero_shot_request, [], [])
        elapsed_d = (time.perf_counter() - start_d) * 1000
        print(f"   REPLY ({elapsed_d:.0f}ms): '{response_d.response}'")

        # --- PATH F: NO CLASSIFIER (Direct LLM) ---
        print("\n>> PATH F: NO CLASSIFIER (Skip logic, direct to LLM)")
        start_f = time.perf_counter()
        
        # This path skips the mobilization and category logic
        # It uses a generic 'statement' config but builds context normally
        direct_classification = ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=1.0,
            requires_knowledge=False,
            metadata={"category_name": "statement", "category_method": "direct"}
        )
        
        response_f = service.generate_reply(
            message_context,
            classification=direct_classification,
            contact=contact
        )
        elapsed_f = (time.perf_counter() - start_f) * 1000
        print(f"   REPLY ({elapsed_f:.0f}ms): '{response_f.response}'")

        # --- PATH G: NO RAG (Context Only) ---
        print("\n>> PATH G: NO RAG (Conversation context only, no facts/examples)")
        start_g = time.perf_counter()
        
        # We build the request manually to ensure empty RAG docs
        from jarvis.reply_service_generation import build_generation_request
        no_rag_request = build_generation_request(
            service,
            context=message_context,
            classification=classification,
            search_results=[], # No RAG docs
            contact=contact
        )
        # Clear any facts or examples selected by default
        no_rag_request.retrieved_docs = []
        no_rag_request.few_shot_examples = []
        
        response_g = service._generate_llm_reply(no_rag_request, [], [])
        elapsed_g = (time.perf_counter() - start_g) * 1000
        print(f"   REPLY ({elapsed_g:.0f}ms): '{response_g.response}'")

        # --- GROUND TRUTH COMPARISON ---
        # Find what the user actually said next in history (if available)
        try:
            # The next message in the reader (which is newest-first) would be "ahead" of last_msg_obj
            # But we'd need to fetch more messages or use a specific query.
            # For bakeoff, we'll just check if the last message WAS from me, then THAT is the truth.
            if last_is_from_me:
                print(f"\n[TRUTH] User actually said: \"{incoming}\"")
        except Exception:
            pass

        print("=" * 80)


if __name__ == "__main__":
    try:
        run_bakeoff()
    except KeyboardInterrupt:
        print("\nBakeoff cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nBakeoff failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
