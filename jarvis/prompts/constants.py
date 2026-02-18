"""Constant strings, system prompts, template strings, and configuration.

This module contains all static data used by the prompt system:
- Version metadata
- Token limits
- Tone detection indicators and patterns
- Prompt templates (PromptTemplate instances)
- Category configuration and mappings
- Template responses
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# =============================================================================
# Prompt Metadata & Versioning
# =============================================================================

PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-01-26"


# =============================================================================
# Dataclasses (used across submodules)
# =============================================================================


@dataclass
class PromptMetadata:
    """Metadata for a prompt or example set.

    Attributes:
        name: Human-readable name for the prompt/example set
        version: Semantic version string (e.g., "1.0.0")
        last_updated: ISO date string of last update
        description: Brief description of the prompt's purpose
    """

    name: str
    version: str = PROMPT_VERSION
    last_updated: str = PROMPT_LAST_UPDATED
    description: str = ""


@dataclass
class FewShotExample:
    """A few-shot example for prompt engineering.

    Attributes:
        context: The conversation context
        output: The expected output/reply
        tone: The tone of the example (casual/professional)
    """

    context: str
    output: str
    tone: Literal["casual", "professional"] = "casual"


@dataclass
class PromptTemplate:
    """A prompt template with placeholders.

    Attributes:
        name: Template identifier
        system_message: Role/context for the model
        template: Format string with {placeholders}
        max_output_tokens: Suggested max tokens for response
    """

    name: str
    system_message: str
    template: str
    max_output_tokens: int = 100


@dataclass
class UserStyleAnalysis:
    """Analysis of user's texting style from message examples.

    Attributes:
        avg_length: Average message length in characters
        min_length: Minimum message length seen
        max_length: Maximum message length seen
        formality: Detected formality level
        uses_lowercase: Whether user typically uses lowercase
        uses_abbreviations: Whether user uses text abbreviations (u, ur, gonna)
        uses_minimal_punctuation: Whether user avoids excessive punctuation
        common_abbreviations: List of abbreviations user commonly uses
        emoji_frequency: Emojis per message
        exclamation_frequency: Exclamation marks per message
    """

    avg_length: float = 50.0
    min_length: int = 0
    max_length: int = 200
    formality: Literal["formal", "casual", "very_casual"] = "casual"
    uses_lowercase: bool = False
    uses_abbreviations: bool = False
    uses_minimal_punctuation: bool = False
    common_abbreviations: list[str] = field(default_factory=list)
    emoji_frequency: float = 0.0
    exclamation_frequency: float = 0.0


@dataclass
class CategoryConfig:
    """Configuration for a message category.

    Attributes:
        skip_slm: If True, skip SLM generation (use template response).
        prompt: Prompt template key to use (if not skipping SLM).
        context_depth: Number of messages to include in context.
        system_prompt: Optional category-specific system prompt.
    """

    skip_slm: bool
    prompt: str | None
    context_depth: int
    system_prompt: str | None = None


# =============================================================================
# Token Limits
# =============================================================================

# Token limit guidance for small models
MAX_PROMPT_TOKENS = 1500  # Reserve space for generation
MAX_CONTEXT_CHARS = 4000  # Approximate, ~4 chars per token


# =============================================================================
# Tone Detection
# =============================================================================

# Casual indicators
CASUAL_INDICATORS: set[str] = {
    # Emoji patterns are detected separately
    "lol",
    "haha",
    "hehe",
    "lmao",
    "omg",
    "btw",
    "brb",
    "ttyl",
    "idk",
    "ikr",
    "nvm",
    "tbh",
    "imo",
    "fyi",
    "np",
    "k",
    "kk",
    "ok",
    "yeah",
    "yep",
    "nope",
    "yup",
    "gonna",
    "wanna",
    "gotta",
    "cuz",
    "bc",
    "u",
    "ur",
    "r",
    "y",
    "thx",
    "ty",
    "pls",
    "plz",
    "omw",
    "wya",
    "wassup",
    "sup",
    "hey",
    "yo",
    "dude",
    "bro",
    "sis",
    "fam",
    "lit",
    "chill",
    "cool",
    "nice",
    "sick",
    "dope",
    "yay",
    "ooh",
    "ahh",
    "hmm",
    "meh",
    "ugh",
    "whoa",
    "wow",
    "aww",
    "oops",
    "whoops",
}

# Professional indicators
PROFESSIONAL_INDICATORS: set[str] = {
    "regarding",
    "pursuant",
    "attached",
    "please",
    "kindly",
    "sincerely",
    "regards",
    "cordially",
    "respectfully",
    "appreciate",
    "opportunity",
    "discussed",
    "confirmed",
    "scheduled",
    "deadline",
    "deliverable",
    "milestone",
    "stakeholder",
    "proposal",
    "presentation",
    "quarterly",
    "annual",
    "fiscal",
    "eod",
    "eow",
    "asap",
    "cc",
    "per",
    "via",
    "ensure",
    "verify",
    "confirm",
    "acknowledge",
    "proceed",
    "follow-up",
    "followup",
    "meeting",
    "conference",
    "agenda",
    "minutes",
    "action item",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "dear",
    "hello",
    "good morning",
    "good afternoon",
    "good evening",
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed characters
    "]+"
)

# Common text abbreviations to detect
TEXT_ABBREVIATIONS: set[str] = {
    "u",
    "ur",
    "r",
    "y",
    "n",
    "k",
    "kk",
    "ok",
    "bc",
    "cuz",
    "gonna",
    "wanna",
    "gotta",
    "thx",
    "ty",
    "pls",
    "plz",
    "idk",
    "nvm",
    "brb",
    "ttyl",
    "omw",
    "lol",
    "lmao",
    "omg",
    "tbh",
    "imo",
    "ikr",
    "rn",
    "atm",
    "btw",
    "fyi",
}


# =============================================================================
# Prompt Templates
# =============================================================================

# Static system prefix for KV cache reuse.
SYSTEM_PREFIX = (
    "You are Jwalin Shah, a tech founder. Text like a real person on an iPhone. "
    "Voice: busy but chill, direct, lowercase only. "
    "Style: use texting slang (idk, lowkey, bet, dw, omw, rn) naturally. "
    "No punctuation, no 'AI assistant' helpfulness. "
    "Match the energy of the sender. If they are brief, be briefer. "
    "If you don't know the context, just say something like 'idk what that is' or 'copy'. "
    "Just text back. 10 words max.\n"
)

REPLY_PROMPT = PromptTemplate(
    name="reply_generation",
    system_message="",
    template="{context}\nMe: ",
    max_output_tokens=25,
)

# Simple prompt - just conversation context, no extra stuff
SIMPLE_REPLY_PROMPT = PromptTemplate(
    name="simple_reply_generation",
    system_message="",
    template=(
        "<|im_start|>system\n" + SYSTEM_PREFIX + "{current_time}{instruction}<|im_end|>\n"
        "<|im_start|>user\n"
        "Context:\n{context}\n\n"
        "Last Message: {last_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    max_output_tokens=25,
)

RAG_REPLY_PROMPT = PromptTemplate(
    name="rag_reply_generation",
    system_message="",
    template=(
        "<|im_start|>system\n" + SYSTEM_PREFIX + "{instruction}<|im_end|>\n"
        "<|im_start|>user\n"
        "{extra_context}\n"
        "Context:\n{context}\n\n"
        "Last Message: {last_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    max_output_tokens=25,
)


SUMMARY_PROMPT = PromptTemplate(
    name="conversation_summary",
    system_message="You are summarizing a text message conversation. Extract key information "
    "concisely and highlight any action items or commitments.",
    template="""<system>
Summarize this conversation. Include:
- Key points discussed
- Any action items or commitments made
- Important dates, times, or locations mentioned
{focus_instruction}
</system>

<conversation>
{context}
</conversation>

<summary>""",
    max_output_tokens=150,
)


SEARCH_PROMPT = PromptTemplate(
    name="search_answer",
    system_message="You are answering a question about a text message conversation. "
    "Base your answer only on the provided messages.",
    template="""<system>
Answer the question based ONLY on the messages provided. Be specific and cite relevant details.
If the answer isn't in the messages, say so.
</system>

<conversation>
{context}
</conversation>

<question>{question}</question>

<answer>""",
    max_output_tokens=100,
)


THREADED_REPLY_PROMPT = PromptTemplate(
    name="threaded_reply",
    system_message=(
        "You are NOT an AI assistant. You are replying to a text message from your phone. "
        "Just text back. No helpfulness, no formality, no assistant behavior."
    ),
    template="""<system>
You are NOT an AI assistant. You are replying to a text message from your phone.
Just text back. No helpfulness, no formality, no assistant behavior.
Rules:
- Match their texting style exactly (length, formality, abbreviations, emoji, punctuation)
- Sound natural, never like an AI
- Match the thread's {response_style} tone and {length_guidance}
</system>

<thread>
Topic: {thread_topic}
State: {thread_state}
Your role: {user_role}
{participants_info}
</thread>

<conversation>
{context}
</conversation>

<instruction>
{additional_instructions}
{custom_instruction}
</instruction>

<last_message>{last_message}</last_message>

<reply>""",
    max_output_tokens=100,
)


# System prompt for chat-based reply generation
CHAT_SYSTEM_PROMPT = (
    "Generate a casual text message reply. One short sentence max. "
    'No AI phrases like "I understand" or "Let me know". '
    "No emojis unless they used them first."
)


# Category mapping: map old/runtime signals to new 6-category schema
CATEGORY_MAP: dict[str, str] = {
    # Old 5-category schema -> new 6-category
    "ack": "acknowledge",
    "info": "request",
    "emotional": "emotion",
    "social": "statement",
    # Old 4-category schema -> new 6-category
    "brief": "request",
    "warm": "emotion",
    # Runtime signals -> new categories
    "quick_exchange": "acknowledge",
    "logistics": "request",
    "emotional_support": "emotion",
    "catching_up": "statement",
    "planning": "request",
    # Tone -> category defaults
    "casual": "statement",
    "professional": "request",
    "mixed": "statement",
    "celebration": "social",
    # Ambiguous / low context
    "clarify": "clarify",
    "edge_case": "clarify",
    "unknown": "clarify",
    "information": "clarify",
}


# Category configurations (maps category -> routing behavior)
CATEGORY_CONFIGS: dict[str, CategoryConfig] = {
    # Note: We use a UNIVERSAL system prompt for all categories.
    # Research showed category-specific prompts hurt quality
    # See docs/research/CATEGORIZATION_ABLATION_FINDINGS.md
    # Category classification is kept for: context_depth, analytics, and routing (skip_slm)
    # But NOT for selecting different system prompts.
    "closing": CategoryConfig(
        skip_slm=True,  # Use template responses, not LLM
        prompt=None,
        context_depth=0,
        system_prompt=None,  # Uses universal SYSTEM_PREFIX
    ),
    "acknowledge": CategoryConfig(
        skip_slm=True,  # Use template responses, not LLM
        prompt=None,
        context_depth=0,
        system_prompt=None,  # Uses universal SYSTEM_PREFIX
    ),
    "question": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,  # Optimized via sweep
        system_prompt=None,  # Uses universal SYSTEM_PREFIX (was: "They asked a question...")
    ),
    "request": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,  # Optimized via sweep
        system_prompt=None,  # Uses universal SYSTEM_PREFIX (was: "They're asking you...")
    ),
    "emotion": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,  # Optimized via sweep
        # Uses universal SYSTEM_PREFIX (was: "They're sharing something emotional...")
        system_prompt=None,
    ),
    "statement": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,  # Optimized via sweep
        system_prompt=None,  # Uses universal SYSTEM_PREFIX (was: "They're sharing or chatting...")
    ),
}


# Template responses for acknowledge/closing categories (when skip_slm=True)
ACKNOWLEDGE_TEMPLATES: list[str] = [
    "ok",
    "sounds good",
    "got it",
    "thanks",
    "np",
    "\U0001f44d",
    "for sure",
    "alright",
    "bet",
    "cool",
]

CLOSING_TEMPLATES: list[str] = [
    "bye!",
    "see ya",
    "later!",
    "talk soon",
    "ttyl",
    "peace",
    "catch you later",
    "gn",
]
