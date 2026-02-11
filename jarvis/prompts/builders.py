"""Core prompt builder functions for iMessage reply generation.

Contains the main prompt construction functions for reply generation,
summarization, search answers, and threaded replies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from jarvis.prompts.examples import (
    CASUAL_REPLY_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    THREAD_EXAMPLES,
    FewShotExample,
)
from jarvis.prompts.templates import (
    MAX_CONTEXT_CHARS,
    REPLY_PROMPT,
    SEARCH_PROMPT,
    SUMMARY_PROMPT,
    THREADED_REPLY_PROMPT,
)
from jarvis.prompts.tone import analyze_user_style, build_style_instructions

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.contacts.contact_profile import Fact
    from jarvis.relationships import RelationshipProfile
    from jarvis.threading import ThreadContext, ThreadedReplyConfig


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


def build_reply_prompt(
    context: str,
    last_message: str,
    instruction: str | None = None,
    tone: Literal["casual", "professional", "mixed"] = "casual",
    relationship_profile: RelationshipProfile | None = None,
    user_messages: list[str] | None = None,
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

    Returns:
        Formatted prompt string ready for model input
    """
    # Import here to avoid circular imports
    from jarvis.relationships import (
        MIN_MESSAGES_FOR_PROFILE,
        generate_style_guide,
        select_matching_examples,
    )

    # Determine effective tone from profile if available
    effective_tone = tone
    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        formality = relationship_profile.tone_profile.formality_score
        if formality >= 0.7:
            effective_tone = "professional"
        elif formality < 0.4:
            effective_tone = "casual"
        else:
            effective_tone = "mixed"

    # Select appropriate examples based on tone (and profile if available)
    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        # Use profile to select matching examples
        # Convert FewShotExample to tuple format for the selector function
        casual_tuples = [(ex.context, ex.output) for ex in CASUAL_REPLY_EXAMPLES]
        professional_tuples = [(ex.context, ex.output) for ex in PROFESSIONAL_REPLY_EXAMPLES]
        examples_list = select_matching_examples(
            relationship_profile,
            casual_tuples,
            professional_tuples,
        )
        examples = [FewShotExample(context=ctx, output=out) for ctx, out in examples_list]
    elif effective_tone == "professional":
        examples = PROFESSIONAL_REPLY_EXAMPLES[:3]
    else:
        examples = CASUAL_REPLY_EXAMPLES[:3]

    # Determine tone string
    if effective_tone == "professional":
        tone_str = "professional/formal"
    else:
        tone_str = "casual/friendly"

    # Build style instructions from user's actual messages
    style_instructions = ""
    if user_messages:
        style_analysis = analyze_user_style(user_messages)
        style_instructions = build_style_instructions(style_analysis)

    # Format custom instruction
    custom_instruction = ""
    if instruction:
        custom_instruction = instruction

    # Add relationship-based style guidance if profile is available
    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        style_guide = generate_style_guide(relationship_profile)
        if custom_instruction:
            custom_instruction += f"\n- Communication style: {style_guide}"
        else:
            custom_instruction = f"- Communication style: {style_guide}"

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
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
    # Get topic and state names
    topic_name = thread_context.topic.value
    state_name = thread_context.state.value
    user_role = thread_context.user_role.value

    # Get appropriate examples for this thread type
    examples = _get_thread_examples(topic_name)[:2]

    # Format the relevant messages (not all messages)
    relevant_msgs = thread_context.relevant_messages or thread_context.messages[-5:]
    context = _format_thread_context(relevant_msgs)

    # Get the last message to reply to
    last_message = ""
    if thread_context.messages:
        last_msg = thread_context.messages[-1]
        if hasattr(last_msg, "text"):
            last_message = last_msg.text or ""
        elif isinstance(last_msg, str):
            last_message = last_msg

    # Build participants info for group chats
    participants_info = ""
    if thread_context.participants_count > 1:
        participants_info = f"Group chat with {thread_context.participants_count} participants"

    # Get length guidance
    length_guidance = _get_length_guidance(config.response_style, config.max_response_length)

    # Get additional instructions
    additional_instructions = _get_additional_instructions(topic_name, state_name, config)

    # Format custom instruction
    custom_instruction = ""
    if instruction:
        custom_instruction = instruction

    # Truncate context if needed
    truncated_context = _truncate_context(context, max_chars=2000)

    # Build the prompt
    prompt = THREADED_REPLY_PROMPT.template.format(
        thread_topic=topic_name.replace("_", " ").title(),
        thread_state=state_name.replace("_", " ").title(),
        user_role=user_role.replace("_", " ").title(),
        participants_info=participants_info,
        context=truncated_context,
        response_style=config.response_style,
        length_guidance=length_guidance,
        additional_instructions=additional_instructions,
        custom_instruction=custom_instruction,
        examples=_format_examples(examples),
        last_message=last_message,
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
