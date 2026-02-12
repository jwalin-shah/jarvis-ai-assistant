"""Functions that build and format prompts for various use cases.

This module contains all prompt building functions:
- Tone detection and style analysis
- Example formatting (consolidated into a generic formatter)
- Reply, summary, search, and threaded reply prompt builders
- RAG-enhanced prompt builders
- Contact facts formatting
- Utility functions (token estimation, etc.)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from jarvis.contacts.contact_profile_context import ContactProfileContext

# Pre-compiled patterns for tone/style analysis functions
_WORD_RE = re.compile(r"\b\w+\b")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")

from jarvis.prompts.constants import (
    CASUAL_INDICATORS,
    EMOJI_PATTERN,
    MAX_CONTEXT_CHARS,
    MAX_PROMPT_TOKENS,
    PROFESSIONAL_INDICATORS,
    RAG_REPLY_PROMPT,
    REPLY_PROMPT,
    SEARCH_PROMPT,
    SUMMARY_PROMPT,
    TEXT_ABBREVIATIONS,
    THREADED_REPLY_PROMPT,
    UserStyleAnalysis,
)
from jarvis.prompts.examples import (
    CASUAL_REPLY_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    THREAD_EXAMPLES,
    FewShotExample,
)

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.classifiers.response_mobilization import MobilizationResult
    from jarvis.contacts.contact_profile import Fact
    from jarvis.contracts.pipeline import GenerationRequest as PipelineGenerationRequest
    from jarvis.relationships import RelationshipProfile
    from jarvis.threading import ThreadContext, ThreadedReplyConfig


# =============================================================================
# Tone Detection
# =============================================================================


def detect_tone(messages: list[str]) -> Literal["casual", "professional", "mixed"]:
    """Detect the conversational tone from a list of messages.

    Analyzes messages for casual indicators (slang, emoji, informal greetings)
    and professional indicators (formal language, business terminology).

    Args:
        messages: List of message strings to analyze

    Returns:
        "casual" if predominantly informal language
        "professional" if predominantly formal language
        "mixed" if both styles are present or unclear
    """
    if not messages:
        return "casual"  # Default to casual for empty input

    # Combine all messages for analysis
    combined = " ".join(messages).lower()
    words = set(_WORD_RE.findall(combined))

    # Count indicators
    casual_count = len(words & CASUAL_INDICATORS)
    professional_count = len(words & PROFESSIONAL_INDICATORS)

    # Check for emoji (strong casual indicator)
    emoji_count = len(EMOJI_PATTERN.findall(combined))
    casual_count += emoji_count * 2  # Weight emoji heavily

    # Check for exclamation marks (casual indicator)
    exclamation_count = combined.count("!")
    if exclamation_count > 2:
        casual_count += 1

    # Check for multi-character expressions like "hahahaha" or "looool"
    if _REPEATED_CHAR_RE.search(combined):  # 4+ repeated chars
        casual_count += 2

    # Determine tone based on counts
    total = casual_count + professional_count

    if total == 0:
        return "casual"  # Default to casual when no strong indicators

    casual_ratio = casual_count / total

    if casual_ratio >= 0.7:
        return "casual"
    elif casual_ratio <= 0.3:
        return "professional"
    else:
        return "mixed"


# =============================================================================
# User Style Analysis
# =============================================================================


def analyze_user_style(messages: list[str]) -> UserStyleAnalysis:
    """Analyze user's texting style from message examples.

    Examines the user's actual messages to extract style patterns including:
    - Average/min/max message length
    - Formality level
    - Use of lowercase, abbreviations, punctuation
    - Emoji and exclamation frequency

    Args:
        messages: List of message strings from the user

    Returns:
        UserStyleAnalysis with detected patterns
    """
    if not messages:
        return UserStyleAnalysis()

    # Filter to non-empty messages
    messages = [m for m in messages if m and m.strip()]
    if not messages:
        return UserStyleAnalysis()

    # Calculate length statistics
    lengths = [len(m) for m in messages]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    # Analyze case usage - check if messages are predominantly lowercase
    lowercase_count = 0
    for msg in messages:
        # Remove emojis and punctuation for case analysis
        letters_only = _NON_ALPHA_RE.sub("", msg)
        if letters_only:
            lowercase_ratio = sum(1 for c in letters_only if c.islower()) / len(letters_only)
            if lowercase_ratio > 0.9:  # 90%+ lowercase
                lowercase_count += 1
    uses_lowercase = lowercase_count / len(messages) > 0.7  # 70%+ of messages

    # Detect abbreviation usage
    combined_lower = " ".join(messages).lower()
    words = set(_WORD_RE.findall(combined_lower))
    found_abbreviations = list(words & TEXT_ABBREVIATIONS)
    uses_abbreviations = len(found_abbreviations) >= 2  # At least 2 different abbreviations

    # Analyze punctuation - minimal if low exclamation/period density
    total_chars = sum(len(m) for m in messages)
    total_exclamations = sum(m.count("!") for m in messages)
    total_periods = sum(m.count(".") for m in messages)
    exclamation_density = total_exclamations / max(total_chars, 1)
    period_density = total_periods / max(total_chars, 1)
    uses_minimal_punctuation = exclamation_density < 0.02 and period_density < 0.03

    # Calculate frequencies per message
    emoji_count = sum(len(EMOJI_PATTERN.findall(m)) for m in messages)
    emoji_frequency = emoji_count / len(messages)
    exclamation_frequency = total_exclamations / len(messages)

    # Determine formality level
    casual_count = len(words & CASUAL_INDICATORS)
    if casual_count >= 3 or (uses_abbreviations and uses_lowercase):
        formality: Literal["formal", "casual", "very_casual"] = "very_casual"
    elif casual_count >= 1 or uses_abbreviations or avg_length < 30:
        formality = "casual"
    else:
        formality = "formal"

    return UserStyleAnalysis(
        avg_length=round(avg_length, 1),
        min_length=min_length,
        max_length=max_length,
        formality=formality,
        uses_lowercase=uses_lowercase,
        uses_abbreviations=uses_abbreviations,
        uses_minimal_punctuation=uses_minimal_punctuation,
        common_abbreviations=found_abbreviations[:5],  # Top 5
        emoji_frequency=round(emoji_frequency, 2),
        exclamation_frequency=round(exclamation_frequency, 2),
    )


def build_style_instructions(style: UserStyleAnalysis) -> str:
    """Build style-matching instructions from a UserStyleAnalysis.

    Generates specific instructions for the model to match the user's
    texting style based on analyzed patterns.

    Args:
        style: UserStyleAnalysis from analyze_user_style()

    Returns:
        String with style-matching instructions for the prompt
    """
    instructions = []

    # Length guidance
    if style.avg_length < 15:
        instructions.append(
            f"Keep response VERY short (1-{max(5, int(style.avg_length * 1.5))} words)"
        )
    elif style.avg_length < 30:
        instructions.append(
            f"Keep response brief ({int(style.min_length)}-{int(style.avg_length * 1.2)} chars)"
        )
    elif style.avg_length < 60:
        instructions.append("Keep response concise (1 short sentence)")
    else:
        instructions.append("Response can be 1-2 sentences")

    # Case/formality guidance
    if style.uses_lowercase:
        instructions.append("Use lowercase (no capitalization)")

    if style.formality == "very_casual":
        instructions.append("Be very casual - no formal greetings like 'Hey!' or 'Hi there!'")
    elif style.formality == "casual":
        instructions.append("Keep it casual - skip formal greetings")

    # Abbreviation guidance
    if style.uses_abbreviations and style.common_abbreviations:
        abbrevs = ", ".join(style.common_abbreviations[:3])
        instructions.append(f"Use abbreviations like: {abbrevs}")

    # Punctuation guidance
    if style.uses_minimal_punctuation:
        instructions.append("Use minimal punctuation")

    # Emoji guidance
    if style.emoji_frequency < 0.1:
        instructions.append("Avoid emojis")
    elif style.emoji_frequency > 0.5:
        instructions.append("Feel free to use emojis")

    # Exclamation guidance
    if style.exclamation_frequency < 0.3:
        instructions.append("Avoid excessive exclamation marks")

    return "\n".join(f"- {inst}" for inst in instructions)


# =============================================================================
# Example Formatting (Consolidated)
# =============================================================================


def _format_examples(examples: list[FewShotExample]) -> str:
    """Format few-shot examples for prompt inclusion.

    Args:
        examples: List of FewShotExample objects

    Returns:
        Formatted string with examples
    """
    formatted = []
    for ex in examples:
        formatted.append(f"Context: {ex.context}\nReply: {ex.output}")
    return "\n\n".join(formatted)


def _format_summary_examples(examples: list[tuple[str, str]]) -> str:
    """Format summarization examples for prompt inclusion.

    Args:
        examples: List of (conversation, summary) tuples

    Returns:
        Formatted string with examples
    """
    formatted = []
    for conversation, summary in examples:
        formatted.append(f"Conversation:\n{conversation}\n\n{summary}")
    return "\n\n---\n\n".join(formatted)


def _format_search_examples(examples: list[tuple[str, str, str]]) -> str:
    """Format search/QA examples for prompt inclusion.

    Args:
        examples: List of (messages, question, answer) tuples

    Returns:
        Formatted string with examples
    """
    formatted = []
    for messages, question, answer in examples:
        formatted.append(f"Messages:\n{messages}\nQuestion: {question}\nAnswer: {answer}")
    return "\n\n---\n\n".join(formatted)


def _truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to fit within token limits.

    Keeps the most recent messages when truncating.

    Args:
        context: The conversation context
        max_chars: Maximum characters to keep

    Returns:
        Truncated context string
    """
    if len(context) <= max_chars:
        return context

    # Keep the most recent messages (end of context)
    truncated = context[-max_chars:]

    # Try to start at a message boundary (newline)
    first_newline = truncated.find("\n")
    if first_newline != -1 and first_newline < 200:
        truncated = truncated[first_newline + 1 :]

    return f"[Earlier messages truncated]\n{truncated}"


# =============================================================================
# Prompt Builder Functions
# =============================================================================


def _determine_effective_tone(
    tone: Literal["casual", "professional", "mixed"],
    relationship_profile: RelationshipProfile | None,
) -> Literal["casual", "professional", "mixed"]:
    """Determine effective tone from base tone and optional relationship profile.

    Args:
        tone: Base tone preference.
        relationship_profile: Optional profile with formality score.

    Returns:
        Resolved tone literal.
    """
    from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE

    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        formality = relationship_profile.tone_profile.formality_score
        if formality >= 0.7:
            return "professional"
        elif formality < 0.4:
            return "casual"
        else:
            return "mixed"
    return tone


def _select_examples_for_tone(
    effective_tone: Literal["casual", "professional", "mixed"],
    relationship_profile: RelationshipProfile | None,
) -> list[FewShotExample]:
    """Select few-shot examples matching the effective tone.

    Args:
        effective_tone: Resolved tone.
        relationship_profile: Optional profile for profile-based selection.

    Returns:
        List of FewShotExample instances.
    """
    from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE, select_matching_examples

    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        casual_tuples = [(ex.context, ex.output) for ex in CASUAL_REPLY_EXAMPLES]
        professional_tuples = [(ex.context, ex.output) for ex in PROFESSIONAL_REPLY_EXAMPLES]
        examples_list = select_matching_examples(
            relationship_profile, casual_tuples, professional_tuples,
        )
        return [FewShotExample(context=ctx, output=out) for ctx, out in examples_list]
    elif effective_tone == "professional":
        return PROFESSIONAL_REPLY_EXAMPLES[:3]
    else:
        return CASUAL_REPLY_EXAMPLES[:3]


def _build_custom_instructions(
    instruction: str | None,
    relationship_profile: RelationshipProfile | None,
) -> str:
    """Build custom instruction string, incorporating relationship style guidance.

    Args:
        instruction: Optional explicit instruction.
        relationship_profile: Optional profile for style guide generation.

    Returns:
        Combined custom instruction string.
    """
    from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE, generate_style_guide

    custom_instruction = instruction or ""

    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        style_guide = generate_style_guide(relationship_profile)
        if custom_instruction:
            custom_instruction += f"\n- Communication style: {style_guide}"
        else:
            custom_instruction = f"- Communication style: {style_guide}"

    return custom_instruction


def build_reply_prompt(
    context: str,
    last_message: str,
    instruction: str | None = None,
    tone: Literal["casual", "professional", "mixed"] = "casual",
    relationship_profile: RelationshipProfile | None = None,
    user_messages: list[str] | None = None,
    user_style: UserStyleAnalysis | None = None,
) -> str:
    """Build a prompt for generating iMessage replies.

    Args:
        context: The conversation history
        last_message: The most recent message to reply to
        instruction: Optional custom instruction for the reply
        tone: The desired tone for the reply
        relationship_profile: Optional relationship profile for personalized replies.
            If provided, the profile's communication patterns will be used to
            customize the reply style (emoji usage, formality, typical phrases).
        user_messages: Optional list of user's own messages for style analysis.
            If provided, the prompt will include explicit instructions to match
            the user's texting style (length, formality, abbreviations, etc.).
        user_style: Optional pre-computed UserStyleAnalysis to avoid recomputation.
            If not provided and user_messages is given, it will be computed.

    Returns:
        Formatted prompt string ready for model input
    """
    effective_tone = _determine_effective_tone(tone, relationship_profile)
    examples = _select_examples_for_tone(effective_tone, relationship_profile)

    tone_str = "professional/formal" if effective_tone == "professional" else "casual/friendly"

    style_instructions = ""
    if user_messages:
        style_analysis = user_style or analyze_user_style(user_messages)
        style_instructions = build_style_instructions(style_analysis)

    custom_instruction = _build_custom_instructions(instruction, relationship_profile)
    truncated_context = _truncate_context(context)

    prompt = REPLY_PROMPT.template.format(
        context=truncated_context,
        tone=tone_str,
        style_instructions=style_instructions,
        custom_instruction=custom_instruction,
        examples=_format_examples(examples),
        last_message=last_message,
    )

    return prompt


def build_summary_prompt(
    context: str,
    focus: str | None = None,
) -> str:
    """Build a prompt for summarizing conversations.

    Args:
        context: The conversation to summarize
        focus: Optional focus area (e.g., "action items", "decisions", "dates")

    Returns:
        Formatted prompt string ready for model input
    """
    # Format focus instruction
    focus_instruction = ""
    if focus:
        focus_instruction = f"- Focus especially on: {focus}"

    # Select a subset of examples
    examples = SUMMARIZATION_EXAMPLES[:2]

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = SUMMARY_PROMPT.template.format(
        context=truncated_context,
        focus_instruction=focus_instruction,
        examples=_format_summary_examples(examples),
    )

    return prompt


def build_search_answer_prompt(
    context: str,
    question: str,
) -> str:
    """Build a prompt for answering questions about conversations.

    Args:
        context: The relevant messages/conversation context
        question: The question to answer

    Returns:
        Formatted prompt string ready for model input
    """
    # Select a subset of examples
    examples = SEARCH_ANSWER_EXAMPLES[:2]

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = SEARCH_PROMPT.template.format(
        context=truncated_context,
        question=question,
        examples=_format_search_examples(examples),
    )

    return prompt


def _get_thread_examples(topic_name: str) -> list[FewShotExample]:
    """Get few-shot examples for a thread topic.

    Args:
        topic_name: The thread topic name (e.g., "logistics", "emotional_support")

    Returns:
        List of relevant FewShotExample instances
    """
    # Map topic enum values to example keys
    topic_map = {
        "logistics": "logistics",
        "planning": "planning",
        "social": "social",
        "catching_up": "social",
        "warm": "warm",
        "emotional_support": "warm",
        "brief": "brief",
        "quick_exchange": "brief",
        "information": "logistics",  # Use logistics as fallback
        "decision_making": "planning",  # Similar to planning
        "celebration": "social",  # Similar tone
        "unknown": "social",  # Default to conversational
        "clarify": "social",  # Clarify uses social examples as fallback
    }

    key = topic_map.get(topic_name, "social")
    return THREAD_EXAMPLES.get(key, CATCHING_UP_THREAD_EXAMPLES)


def _format_thread_context(messages: Sequence[Message | str]) -> str:
    """Format thread messages for prompt context.

    Args:
        messages: List of Message objects or strings

    Returns:
        Formatted string of messages
    """
    lines = []
    for msg in messages:
        # Handle Message objects (duck typing)
        if hasattr(msg, "date") and hasattr(msg, "text"):
            timestamp = msg.date.strftime("%H:%M") if hasattr(msg.date, "strftime") else ""
            sender = (
                "Me"
                if getattr(msg, "is_from_me", False)
                else getattr(msg, "sender_name", None) or getattr(msg, "sender", "Unknown")
            )
            text = msg.text or ""
            lines.append(f"[{timestamp}] {sender}: {text}")
        # Handle raw strings
        elif isinstance(msg, str):
            lines.append(msg)

    return "\n".join(lines)


def _get_length_guidance(response_style: str, max_length: int) -> str:
    """Get length guidance based on response style.

    Args:
        response_style: The recommended response style
        max_length: Maximum response length

    Returns:
        Human-readable length guidance string
    """
    if response_style in ("concise", "brief"):
        return "brief and to the point (1-2 sentences)"
    elif response_style == "empathetic":
        return "warm and supportive (2-3 sentences, show you care)"
    elif response_style == "detailed":
        return "complete but not lengthy (2-3 sentences with relevant details)"
    elif response_style == "enthusiastic":
        return "upbeat and celebratory (1-2 sentences)"
    else:
        return "natural and conversational (1-2 sentences)"


def _get_additional_instructions(
    topic_name: str,
    state_name: str,
    config: ThreadedReplyConfig,
) -> str:
    """Get additional instructions based on thread context.

    Args:
        topic_name: Thread topic name
        state_name: Thread state name
        config: Thread response configuration

    Returns:
        Additional instruction string
    """
    instructions = []

    # State-specific instructions
    if state_name == "open_question":
        instructions.append("- Answer the question directly")
    elif state_name == "awaiting_response":
        instructions.append("- Acknowledge their message and respond appropriately")

    # Topic-specific instructions
    if topic_name == "emotional_support":
        instructions.append("- Show empathy and understanding")
        instructions.append("- Offer support without being preachy")
    elif topic_name == "logistics":
        instructions.append("- Be clear and specific about times/places")
        instructions.append("- Confirm key details")
    elif topic_name == "planning":
        instructions.append("- Be constructive and suggest next steps")
        if config.include_action_items:
            instructions.append("- Include any commitments you're making")
    elif topic_name == "decision_making":
        instructions.append("- Provide your input clearly")
        instructions.append("- Help move the decision forward")

    # Config-specific instructions
    if config.suggest_follow_up:
        instructions.append("- End with a question or invitation to continue")

    return "\n".join(instructions) if instructions else ""


@dataclass
class ThreadedPromptComponents:
    """Holds the resolved pieces needed to build a threaded reply prompt."""

    topic_name: str
    state_name: str
    user_role: str
    context: str
    last_message: str
    participants_info: str
    examples: list[FewShotExample]
    length_guidance: str
    additional_instructions: str
    custom_instruction: str


def _build_threaded_components(
    thread_context: ThreadContext,
    config: ThreadedReplyConfig,
    instruction: str | None = None,
) -> ThreadedPromptComponents:
    """Extract and resolve all components for a threaded reply prompt.

    Args:
        thread_context: Analyzed thread context from ThreadAnalyzer.
        config: Response configuration based on thread type.
        instruction: Optional custom instruction.

    Returns:
        ThreadedPromptComponents with all resolved fields.
    """
    topic_name = thread_context.topic.value
    state_name = thread_context.state.value
    user_role = thread_context.user_role.value

    examples = _get_thread_examples(topic_name)[:2]

    relevant_msgs = thread_context.relevant_messages or thread_context.messages[-5:]
    context = _format_thread_context(relevant_msgs)

    last_message = ""
    if thread_context.messages:
        last_msg = thread_context.messages[-1]
        if hasattr(last_msg, "text"):
            last_message = last_msg.text or ""
        elif isinstance(last_msg, str):
            last_message = last_msg

    participants_info = ""
    if thread_context.participants_count > 1:
        participants_info = f"Group chat with {thread_context.participants_count} participants"

    return ThreadedPromptComponents(
        topic_name=topic_name,
        state_name=state_name,
        user_role=user_role,
        context=_truncate_context(context, max_chars=2000),
        last_message=last_message,
        participants_info=participants_info,
        examples=examples,
        length_guidance=_get_length_guidance(config.response_style, config.max_response_length),
        additional_instructions=_get_additional_instructions(topic_name, state_name, config),
        custom_instruction=instruction or "",
    )


def build_threaded_reply_prompt(
    thread_context: ThreadContext,
    config: ThreadedReplyConfig,
    instruction: str | None = None,
    tone: Literal["casual", "professional", "mixed"] = "casual",
) -> str:
    """Build a prompt for thread-aware reply generation.

    Constructs a prompt that includes thread context, topic-specific examples,
    and appropriate instructions for the thread type and state.

    Args:
        thread_context: Analyzed thread context from ThreadAnalyzer
        config: Response configuration based on thread type
        instruction: Optional custom instruction for the reply
        tone: Overall tone preference (default: casual)

    Returns:
        Formatted prompt string ready for model input
    """
    c = _build_threaded_components(thread_context, config, instruction)

    prompt = THREADED_REPLY_PROMPT.template.format(
        thread_topic=c.topic_name.replace("_", " ").title(),
        thread_state=c.state_name.replace("_", " ").title(),
        user_role=c.user_role.replace("_", " ").title(),
        participants_info=c.participants_info,
        context=c.context,
        response_style=config.response_style,
        length_guidance=c.length_guidance,
        additional_instructions=c.additional_instructions,
        custom_instruction=c.custom_instruction,
        examples=_format_examples(c.examples),
        last_message=c.last_message,
    )

    return prompt


def get_thread_max_tokens(config: ThreadedReplyConfig) -> int:
    """Get max tokens for generation based on thread config.

    Args:
        config: Thread response configuration

    Returns:
        Recommended max tokens for generation
    """
    # Rough estimate: ~4 chars per token
    base_tokens = config.max_response_length // 4

    # Add buffer for safety
    return max(30, min(base_tokens + 20, 150))


# =============================================================================
# Contact Facts Formatting
# =============================================================================


def format_facts_for_prompt(facts: list[Fact], max_facts: int = 10) -> str:
    """Format contact facts compactly for prompt injection.

    Takes extracted facts about a contact and produces a compact string
    suitable for including in the reply generation prompt (~2-3 tokens per fact).

    Args:
        facts: List of Fact objects, typically from get_facts_for_contact().
        max_facts: Maximum number of facts to include (default 10).

    Returns:
        Compact formatted string like "lives_in: Austin, works_at: Google".
        Empty string if no qualifying facts.
    """
    if not facts:
        return ""

    # Filter to confident facts and cap count
    qualified = [f for f in facts if f.confidence >= 0.5][:max_facts]
    if not qualified:
        return ""

    # Group by category for readability
    by_category: dict[str, list[str]] = {}
    for fact in qualified:
        entry = f"{fact.predicate}: {fact.subject}"
        if fact.value:
            entry += f" ({fact.value})"
        by_category.setdefault(fact.category, []).append(entry)

    # Flatten into compact format
    parts: list[str] = []
    for entries in by_category.values():
        parts.extend(entries)

    return ", ".join(parts)


# =============================================================================
# RAG-Enhanced Prompt Builders
# =============================================================================


def _format_similar_exchanges(exchanges: list[tuple[str, str]]) -> str:
    """Format similar past exchanges for RAG prompt.

    Args:
        exchanges: List of (context, response) tuples from past conversations

    Returns:
        Formatted string with examples
    """
    if not exchanges:
        return "(No similar past exchanges found)"

    formatted = []
    for i, (ctx, response) in enumerate(exchanges[:3], 1):
        # Truncate long contexts
        ctx_preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
        formatted.append(f"Example {i}:\nContext: {ctx_preview}\nYour reply: {response}")
    return "\n\n".join(formatted)


def _format_relationship_context(
    contact_context: ContactProfileContext | None,
    tone: str,
    avg_length: float,
    response_patterns: dict[str, float | int] | None = None,
    user_messages: list[str] | None = None,
    user_style: UserStyleAnalysis | None = None,
) -> str:
    """Format relationship context for RAG prompt.

    Returns a compact single-line style description (~30 tokens) instead of
    verbose bullet points (~80 tokens). Small models parse dense text better.

    Args:
        contact_context: Optional typed contact profile context.
        tone: Typical communication tone.
        avg_length: Average message length.
        response_patterns: Optional response pattern statistics.
        user_messages: Optional list of user's messages for style analysis.
        user_style: Optional pre-computed UserStyleAnalysis to avoid recomputation.
            If not provided and user_messages is given, it will be computed.

    Returns:
        Compact relationship context string.
    """
    parts: list[str] = []

    # If we have user messages, analyze their style directly
    if user_messages:
        style = user_style or analyze_user_style(user_messages)

        # Tone
        formality_labels = {
            "very_casual": "very casual",
            "casual": "casual",
            "formal": "formal",
        }
        parts.append(f"Tone: {formality_labels.get(style.formality, 'casual')}")
        parts.append(f"Avg length: {int(style.avg_length)} chars")

        # Style traits as comma-separated list
        traits: list[str] = []
        if style.uses_lowercase:
            traits.append("lowercase")
        if style.uses_abbreviations and style.common_abbreviations:
            abbrevs = ", ".join(style.common_abbreviations[:3])
            traits.append(f"abbreviations ({abbrevs})")
        if style.uses_minimal_punctuation:
            traits.append("minimal punctuation")
        if style.emoji_frequency < 0.1:
            traits.append("no emoji")
        elif style.emoji_frequency > 0.5:
            traits.append("uses emoji")

        if traits:
            parts.append(", ".join(traits))

        return ". ".join(parts) + "."

    # Fallback: use provided tone and avg_length
    tone_source = contact_context.tone if contact_context else tone
    parts.append(f"Tone: {tone_source}")

    effective_avg_length = contact_context.avg_message_length if contact_context else avg_length
    if effective_avg_length < 20:
        parts.append("very short messages (1-5 words)")
    elif effective_avg_length < 40:
        parts.append("brief messages (1 sentence)")
    elif effective_avg_length < 80:
        parts.append("moderate messages (1-2 sentences)")
    else:
        parts.append("longer messages (2-3 sentences)")

    # Contact profile extras (compact)
    if contact_context and contact_context.style_guide:
        parts.append(contact_context.style_guide)
    if contact_context and contact_context.greeting_style:
        greetings = ", ".join(contact_context.greeting_style[:2])
        parts.append(f"Common greetings: {greetings}")
    if contact_context and contact_context.signoff_style:
        signoffs = ", ".join(contact_context.signoff_style[:2])
        parts.append(f"Typical signoffs: {signoffs}")
    if contact_context and contact_context.top_topics:
        topics = ", ".join(contact_context.top_topics[:3])
        parts.append(f"Topics you often discuss: {topics}")

    return ". ".join(parts) + "."


def build_rag_reply_prompt(
    context: str,
    last_message: str,
    contact_name: str,
    similar_exchanges: list[tuple[str, str]] | None = None,
    relationship_profile: dict[str, Any] | None = None,
    contact_context: ContactProfileContext | None = None,
    instruction: str | None = None,
    user_messages: list[str] | None = None,
    contact_facts: str = "",
    relationship_graph: str = "",
    user_style: UserStyleAnalysis | None = None,
) -> str:
    """Build a RAG-enhanced prompt for generating personalized iMessage replies.

    Uses retrieved similar past exchanges and relationship profile to generate
    responses that match the user's typical communication style with the contact.

    Args:
        context: The current conversation history
        last_message: The most recent message to reply to
        contact_name: Name of the contact being messaged
        similar_exchanges: List of (context, response) tuples from similar past conversations
        relationship_profile: Dict with tone, avg_message_length, response_patterns, etc.
        contact_context: Optional typed contact profile context for richer guidance.
        instruction: Optional custom instruction for the reply
        user_messages: Optional list of user's own messages for style analysis.
            If provided, the prompt will include explicit instructions to match
            the user's texting style (length, formality, abbreviations, etc.).
        contact_facts: Pre-formatted facts string from format_facts_for_prompt().
        relationship_graph: Pre-formatted graph context from get_graph_context().

    Returns:
        Formatted prompt string ready for model input

    Example:
        >>> from jarvis.search.embeddings import find_similar_messages, get_relationship_profile
        >>> # Get similar exchanges from history
        >>> similar = find_similar_messages(last_message, contact_id="chat123", limit=3)
        >>> exchanges = [(m.text, "...response...") for m in similar]
        >>> # Get relationship profile
        >>> profile = get_relationship_profile("chat123")
        >>> prompt = build_rag_reply_prompt(
        ...     context=context,
        ...     last_message=last_message,
        ...     contact_name="John",
        ...     similar_exchanges=exchanges,
        ...     relationship_profile={"tone": profile.typical_tone, ...},
        ...     user_messages=["yeah", "k sounds good", "omw"],
        ... )
    """
    # Extract profile info
    profile_payload = relationship_profile or {}
    tone = str(profile_payload.get("tone", contact_context.tone if contact_context else "casual"))
    avg_length = float(
        profile_payload.get(
            "avg_message_length",
            contact_context.avg_message_length if contact_context else 50,
        )
    )
    response_patterns = profile_payload.get("response_patterns")

    # If user_messages provided, use style analysis for avg_length
    if user_messages:
        resolved_style = user_style or analyze_user_style(user_messages)
        avg_length = resolved_style.avg_length
    else:
        resolved_style = user_style

    # Format relationship context with user messages for style analysis
    relationship_context = _format_relationship_context(
        contact_context=contact_context,
        tone=tone,
        avg_length=avg_length,
        response_patterns=response_patterns if isinstance(response_patterns, dict) else None,
        user_messages=user_messages,
        user_style=resolved_style,
    )

    # Format similar exchanges
    exchanges = similar_exchanges or []
    similar_context = _format_similar_exchanges(exchanges)

    # Format custom instruction
    custom_instruction = ""
    if instruction:
        custom_instruction = instruction

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = RAG_REPLY_PROMPT.template.format(
        contact_name=contact_name,
        relationship_context=relationship_context,
        relationship_graph=relationship_graph or "(none)",
        contact_facts=contact_facts or "(none)",
        similar_exchanges=similar_context,
        context=truncated_context,
        custom_instruction=custom_instruction,
        last_message=last_message,
    )

    return prompt


def build_prompt_from_request(req: PipelineGenerationRequest) -> str:
    """Build a reply prompt from a typed pipeline generation request."""
    context_messages = req.context.metadata.get("context_messages")
    if isinstance(context_messages, list):
        formatted_context = "\n".join(str(msg) for msg in context_messages if isinstance(msg, str))
    else:
        thread_messages = req.context.metadata.get("thread", [])
        if isinstance(thread_messages, list):
            formatted_context = "\n".join(
                str(msg) for msg in thread_messages if isinstance(msg, str)
            )
        else:
            formatted_context = ""

    if not formatted_context:
        formatted_context = req.context.message_text

    similar_exchanges: list[tuple[str, str]] = []
    for doc in req.retrieved_docs:
        response_text = str(doc.metadata.get("response_text", "")).strip()
        if doc.content.strip() and response_text:
            similar_exchanges.append((doc.content.strip(), response_text))

    for example in req.few_shot_examples:
        input_text = str(example.get("input") or example.get("context") or "").strip()
        output_text = str(example.get("output") or example.get("response") or "").strip()
        pair = (input_text, output_text)
        if input_text and output_text and pair not in similar_exchanges:
            similar_exchanges.append(pair)

    relationship_profile = req.context.metadata.get("relationship_profile")
    if not isinstance(relationship_profile, dict):
        relationship_profile = None

    contact_context_raw = req.context.metadata.get("contact_context")
    contact_context = (
        contact_context_raw if isinstance(contact_context_raw, ContactProfileContext) else None
    )

    user_messages_raw = req.context.metadata.get("user_messages")
    user_messages = (
        [msg for msg in user_messages_raw if isinstance(msg, str)]
        if isinstance(user_messages_raw, list)
        else None
    )

    instruction_raw = req.context.metadata.get("instruction")
    instruction = instruction_raw if isinstance(instruction_raw, str) and instruction_raw else None

    contact_name_raw = req.context.metadata.get("contact_name") or req.context.sender_id or "them"
    contact_name = str(contact_name_raw)

    contact_facts_raw = req.context.metadata.get("contact_facts")
    contact_facts = contact_facts_raw if isinstance(contact_facts_raw, str) else ""

    relationship_graph_raw = req.context.metadata.get("relationship_graph")
    relationship_graph = relationship_graph_raw if isinstance(relationship_graph_raw, str) else ""

    return build_rag_reply_prompt(
        context=formatted_context,
        last_message=req.context.message_text,
        contact_name=contact_name,
        similar_exchanges=similar_exchanges[:5],
        relationship_profile=relationship_profile,
        contact_context=contact_context,
        instruction=instruction,
        user_messages=user_messages,
        contact_facts=contact_facts,
        relationship_graph=relationship_graph,
    )


def build_rag_reply_prompt_from_embeddings(
    context: str,
    last_message: str,
    contact_id: str,
    contact_name: str | None = None,
    instruction: str | None = None,
) -> str:
    """Build a RAG-enhanced prompt using the embedding store directly.

    Convenience function that fetches similar exchanges and relationship
    profile automatically from the embedding store.

    Args:
        context: The current conversation history
        last_message: The most recent message to reply to
        contact_id: Chat ID for the contact
        contact_name: Optional display name (fetched from profile if not provided)
        instruction: Optional custom instruction for the reply

    Returns:
        Formatted prompt string ready for model input
    """
    # Import here to avoid circular imports
    from jarvis.contacts.contact_profile import get_contact_profile
    from jarvis.search.vec_search import get_vec_searcher

    # Get relationship profile
    profile = get_contact_profile(contact_id)

    if profile:
        name = contact_name or profile.contact_name or "this person"
        contact_context = ContactProfileContext.from_contact_profile(profile)
        # Create dict for compatibility
        profile_dict = {
            "tone": contact_context.tone,
            "avg_message_length": profile.avg_message_length,
            "response_patterns": {},  # ContactProfile doesn't store this yet
        }
    else:
        name = contact_name or "this person"
        contact_context = None
        profile_dict = {}

    # Find similar past messages
    searcher = get_vec_searcher()
    results = searcher.search(query=last_message, chat_id=contact_id, limit=5)

    # Build exchanges: for each similar incoming message, try to find your response
    # This is a simplified approach - we include the similar message as context
    # In practice, you'd want to fetch the actual response that followed
    exchanges: list[tuple[str, str]] = []
    for res in results:
        if not res.is_from_me and res.text:
            # This was a message to you - include it as context
            # The response would need to be fetched from subsequent messages
            exchanges.append((res.text, "(your typical response style)"))

    # Build the prompt
    return build_rag_reply_prompt(
        context=context,
        last_message=last_message,
        contact_name=name,
        similar_exchanges=exchanges,
        relationship_profile=profile_dict,
        contact_context=contact_context,
        instruction=instruction,
    )


# =============================================================================
# Category Resolution
# =============================================================================


def resolve_category(
    last_message: str,
    context: list[str] | None = None,
    tone: str = "casual",
    mobilization: MobilizationResult | None = None,
) -> str:
    """Classify a message into an optimization category.

    Uses the SVM-based category classifier when available, falling back
    to the static CATEGORY_MAP for graceful degradation.

    Args:
        last_message: The incoming message text.
        context: Recent conversation messages.
        tone: Detected tone (casual/professional/mixed).
        mobilization: Pre-computed mobilization result.

    Returns:
        Optimization category name.
    """
    from jarvis.prompts.constants import CATEGORY_MAP

    try:
        from jarvis.classifiers.category_classifier import classify_category

        result = classify_category(last_message, context=context, mobilization=mobilization)
        return result.category
    except Exception:
        # Graceful fallback to static mapping
        return CATEGORY_MAP.get(tone, "statement")


def get_category_config(category: str) -> CategoryConfig:
    """Get routing configuration for a category.

    Args:
        category: Category name (closing, acknowledge, question, request, emotion, statement).

    Returns:
        CategoryConfig for the category, or default (statement) if unknown.
    """
    from jarvis.prompts.constants import CATEGORY_CONFIGS

    return CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.

    Uses a simple heuristic of ~4 characters per token.
    This is an approximation; actual token counts vary by model.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def is_within_token_limit(prompt: str, limit: int = MAX_PROMPT_TOKENS) -> bool:
    """Check if a prompt is within the token limit.

    Args:
        prompt: The prompt to check
        limit: Maximum allowed tokens (default: MAX_PROMPT_TOKENS)

    Returns:
        True if prompt is within limit, False otherwise
    """
    return estimate_tokens(prompt) <= limit
