"""
EXAMPLE: Clean generation pipeline (simplified from reply_service + router)

Before: reply_service.py (817 lines) + router.py (300+ lines) + generation.py
After: One clean pipeline with clear stages
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.pipelines.classify import Classification, Category, Urgency


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Reply:
    """A generated reply."""
    text: str
    confidence: float  # 0.0 - 1.0
    method: str        # "template", "slm", "fallback"
    
    # Optional metadata for debugging
    context: dict = field(default_factory=dict)


@dataclass
class Context:
    """Context for reply generation."""
    message: str
    thread: list[str]           # Recent conversation history
    similar_examples: list[tuple[str, str]]  # (incoming, reply) from RAG
    contact_name: str | None = None
    facts: list[str] = field(default_factory=list)  # Relevant KG facts


# ============================================================================
# Template Replies (for simple categories)
# ============================================================================

ACKNOWLEDGE_TEMPLATES = [
    "Got it!",
    "Okay, sounds good.",
    "Sure thing.",
    "👍",
    "Alright.",
]

CLOSING_TEMPLATES = [
    "Goodnight!",
    "Talk later!",
    "See you!",
    "Bye for now!",
]


def get_template_reply(category: Category) -> Reply:
    """Get a template reply for simple categories."""
    if category == Category.ACKNOWLEDGE:
        return Reply(
            text=random.choice(ACKNOWLEDGE_TEMPLATES),
            confidence=1.0,
            method="template_ack"
        )
    elif category == Category.CLOSING:
        return Reply(
            text=random.choice(CLOSING_TEMPLATES),
            confidence=1.0,
            method="template_closing"
        )
    else:
        raise ValueError(f"No template for category: {category}")


# ============================================================================
# Context Assembly (RAG + Profile lookup)
# ============================================================================

class ContextAssembler:
    """
    Assemble context for reply generation.
    
    This is pure data fetching - no ML here.
    """
    
    def __init__(self, db, embedder, searcher):
        self.db = db
        self.embedder = embedder
        self.searcher = searcher
    
    def assemble(
        self,
        message: str,
        chat_id: str | None,
        classification: Classification
    ) -> Context:
        """
        Gather all context needed for reply generation.
        
        Steps:
        1. Get recent thread history
        2. Search for similar past exchanges (RAG)
        3. Look up relevant facts from knowledge graph
        4. Get contact info
        """
        # 1. Thread history
        thread = self._get_thread_history(chat_id, limit=5)
        
        # 2. RAG search for similar examples
        similar = self._search_similar(message, top_k=3)
        
        # 3. Knowledge graph facts
        facts = self._get_relevant_facts(message, chat_id)
        
        # 4. Contact name
        contact_name = self._get_contact_name(chat_id)
        
        return Context(
            message=message,
            thread=thread,
            similar_examples=similar,
            contact_name=contact_name,
            facts=facts
        )
    
    def _get_thread_history(self, chat_id: str | None, limit: int) -> list[str]:
        """Get recent messages from thread."""
        if not chat_id:
            return []
        # Query DB for recent messages
        return []  # Placeholder
    
    def _search_similar(self, message: str, top_k: int) -> list[tuple[str, str]]:
        """RAG search for similar exchanges."""
        # Embed and search
        # Return (incoming_message, reply) pairs
        return []  # Placeholder
    
    def _get_relevant_facts(self, message: str, chat_id: str | None) -> list[str]:
        """Query knowledge graph for relevant facts."""
        # Extract entities from message, query KG
        return []  # Placeholder
    
    def _get_contact_name(self, chat_id: str | None) -> str | None:
        """Get contact name from chat_id."""
        return None  # Placeholder


# ============================================================================
# Prompt Building
# ============================================================================

SYSTEM_PROMPT = """You are helping draft a reply to a message.
Be concise and natural. Match the tone of the conversation.
Use the examples below as guidance for your style."""


def build_prompt(context: Context, classification: Classification) -> str:
    """
    Build the prompt for the SLM.
    
    Structure:
    1. System instruction
    2. Few-shot examples (from RAG)
    3. Current conversation context
    4. Message to reply to
    """
    lines = [SYSTEM_PROMPT]
    
    # Add category-specific hint
    if classification.category == Category.QUESTION:
        lines.append("The message is asking a question. Answer directly.")
    elif classification.category == Category.REQUEST:
        lines.append("The message is making a request. Respond helpfully.")
    elif classification.category == Category.EMOTION:
        lines.append("The message expresses emotion. Respond with empathy.")
    
    # Add relevant facts if available
    if context.facts:
        lines.append("\nRelevant context:")
        for fact in context.facts:
            lines.append(f"- {fact}")
    
    # Add few-shot examples
    if context.similar_examples:
        lines.append("\nExamples of how you've replied before:")
        for incoming, reply in context.similar_examples[:3]:
            lines.append(f"Them: {incoming}")
            lines.append(f"You: {reply}")
    
    # Add conversation history
    if context.thread:
        lines.append("\nRecent conversation:")
        for msg in context.thread[-3:]:
            lines.append(msg)
    
    # The actual message to reply to
    lines.append(f"\nThem: {context.message}")
    lines.append("You:")
    
    return "\n".join(lines)


# ============================================================================
# Generation Pipeline
# ============================================================================

class GenerationPipeline:
    """
    Main pipeline for generating replies.
    
    Stages:
    1. Decide if we should reply (from classification)
    2. Assemble context (RAG, thread, facts)
    3. Build prompt
    4. Call SLM (or use template)
    5. Post-process and return
    """
    
    def __init__(self, db, embedder, searcher, slm):
        self.context_assembler = ContextAssembler(db, embedder, searcher)
        self.slm = slm
        
        # Confidence thresholds
        self.min_confidence = 0.45
        self.high_confidence = 0.70
    
    def generate(
        self,
        message: str,
        classification: Classification,
        chat_id: str | None = None
    ) -> Reply:
        """
        Generate a reply to a message.
        
        Flow:
        1. If classification says no reply → return empty
        2. If template category → use template
        3. Otherwise → assemble context, build prompt, call SLM
        """
        start_time = time.time()
        
        # 1. Should we reply?
        if not classification.should_reply:
            return Reply(
                text="",
                confidence=classification.confidence,
                method="skip_no_reply_needed",
                context={"reason": "ack/closing category"}
            )
        
        # 2. Use template for simple categories
        if classification.use_template:
            return get_template_reply(classification.category)
        
        # 3. Assemble context
        context = self.context_assembler.assemble(message, chat_id, classification)
        
        # 4. Build prompt
        prompt = build_prompt(context, classification)
        
        # 5. Call SLM
        try:
            response_text = self.slm.generate(prompt, max_tokens=40)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                classification=classification,
                context=context,
                response=response_text
            )
            
            # Post-process
            response_text = self._post_process(response_text)
            
            return Reply(
                text=response_text,
                confidence=confidence,
                method="slm",
                context={
                    "latency_ms": (time.time() - start_time) * 1000,
                    "examples_used": len(context.similar_examples),
                }
            )
            
        except Exception as e:
            # Fallback on error
            return Reply(
                text="I'm having trouble generating a response.",
                confidence=0.3,
                method="fallback_error",
                context={"error": str(e)}
            )
    
    def _calculate_confidence(
        self,
        classification: Classification,
        context: Context,
        response: str
    ) -> float:
        """
        Calculate confidence in the generated reply.
        
        Factors:
        - Classification confidence
        - RAG similarity scores
        - Response length/coherence
        """
        base = classification.confidence
        
        # Boost for high-quality RAG matches
        if context.similar_examples:
            base = min(base * 1.1, 0.95)
        
        # Penalty for very short responses
        if len(response.split()) < 2:
            base *= 0.8
        
        return round(base, 2)
    
    def _post_process(self, text: str) -> str:
        """Clean up the generated response."""
        text = text.strip()
        
        # Remove "You:" prefix if model added it
        if text.lower().startswith("you:"):
            text = text[4:].strip()
        
        # Ensure proper ending
        if text and text[-1] not in ".!?":
            text += "."
        
        return text


# ============================================================================
# Public API
# ============================================================================

_pipeline: GenerationPipeline | None = None


def get_generation_pipeline() -> GenerationPipeline:
    """Get the singleton generation pipeline."""
    global _pipeline
    if _pipeline is None:
        from jarvis.db import get_db
        from jarvis.embedding_adapter import get_embedder
        from jarvis.search.semantic_search import get_semantic_searcher
        from models import get_generator
        
        _pipeline = GenerationPipeline(
            db=get_db(),
            embedder=get_embedder(),
            searcher=get_semantic_searcher(),
            slm=get_generator()
        )
    return _pipeline


def generate_reply(
    message: str,
    classification: Classification,
    chat_id: str | None = None
) -> Reply:
    """
    Generate a reply to a message.
    
    This is the main entry point.
    """
    return get_generation_pipeline().generate(message, classification, chat_id)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from jarvis.pipelines.classify import classify_message, Category, Urgency
    
    # Example flow
    message = "Want to grab lunch tomorrow?"
    
    # 1. Classify
    classification = classify_message(message)
    print(f"Classification: {classification.category.value}, urgency={classification.urgency.value}")
    
    # 2. Generate reply
    if classification.should_reply:
        reply = generate_reply(message, classification)
        print(f"Reply ({reply.method}, conf={reply.confidence}): {reply.text}")
    else:
        print("No reply needed")
    
    # Example output:
    # Classification: question, urgency=high
    # Reply (slm, conf=0.82): Sure! What time works for you?
