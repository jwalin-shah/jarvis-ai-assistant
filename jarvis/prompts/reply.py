from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from jarvis.prompts.constants import (
    REPLY_PROMPT,
    THREADED_REPLY_PROMPT,
    FewShotExample,
)
from jarvis.prompts.examples import (
    CASUAL_REPLY_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    THREAD_EXAMPLES,
)
from jarvis.prompts.tone import (
    analyze_user_style,
    build_style_instructions,
    determine_effective_tone,
)
from jarvis.prompts.utils import truncate_context

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.relationships import RelationshipProfile
    from jarvis.threading import ThreadContext, ThreadedReplyConfig


def select_examples_for_tone(
    effective_tone: Literal["casual", "professional", "mixed"],
    relationship_profile: RelationshipProfile | None,
) -> list[FewShotExample]:
    """Select few-shot examples matching the effective tone."""
    from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE, select_matching_examples

    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        casual_tuples = [(ex.context, ex.output) for ex in CASUAL_REPLY_EXAMPLES]
        professional_tuples = [(ex.context, ex.output) for ex in PROFESSIONAL_REPLY_EXAMPLES]
        examples_list = select_matching_examples(
            relationship_profile,
            casual_tuples,
            professional_tuples,
        )
        return [FewShotExample(context=ctx, output=out) for ctx, out in examples_list]
    elif effective_tone == "professional":
        return PROFESSIONAL_REPLY_EXAMPLES[:3]
    else:
        return CASUAL_REPLY_EXAMPLES[:3]


def build_custom_instructions(
    instruction: str | None,
    relationship_profile: RelationshipProfile | None,
) -> str:
    """Build custom instruction string, incorporating relationship style guidance."""
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
    user_style: Any | None = None,
) -> str:
    """Build a prompt for generating iMessage replies."""
    from jarvis.prompts.utils import format_examples

    effective_tone = determine_effective_tone(tone, relationship_profile)

    tone_str = "professional/formal" if effective_tone == "professional" else "casual/friendly"

    style_instructions = ""
    examples_str = ""
    if user_messages:
        style_analysis = user_style or analyze_user_style(user_messages)
        style_instructions = build_style_instructions(style_analysis)
        
        # Include a few examples matching the tone
        examples = select_examples_for_tone(effective_tone, relationship_profile)
        if examples:
            examples_str = f"\n<examples>\n{format_examples(examples)}\n</examples>"

    truncated_context = truncate_context(context)

    prompt = REPLY_PROMPT.template.format(
        context=truncated_context,
        tone=tone_str,
        style_instructions=style_instructions + examples_str,
        last_message=last_message,
    )

    return prompt


def _get_thread_examples(topic_name: str) -> list[FewShotExample]:
    """Get few-shot examples for a thread topic."""
    topic_map = {
        "logistics": "logistics",
        "planning": "planning",
        "social": "social",
        "catching_up": "social",
        "warm": "warm",
        "emotional_support": "warm",
        "brief": "brief",
        "quick_exchange": "brief",
        "information": "logistics",
        "decision_making": "planning",
        "celebration": "social",
        "unknown": "social",
        "clarify": "social",
    }
    key = topic_map.get(topic_name, "social")
    return THREAD_EXAMPLES.get(key, CATCHING_UP_THREAD_EXAMPLES)


def _format_thread_context(messages: Sequence[Message | str]) -> str:
    """Format thread messages for prompt context."""
    lines = []
    for msg in messages:
        if hasattr(msg, "date") and hasattr(msg, "text"):
            timestamp = msg.date.strftime("%H:%M") if hasattr(msg.date, "strftime") else ""
            sender = (
                "Me"
                if getattr(msg, "is_from_me", False)
                else getattr(msg, "sender_name", None) or getattr(msg, "sender", "Unknown")
            )
            text = msg.text or ""
            lines.append(f"[{timestamp}] {sender}: {text}")
        elif isinstance(msg, str):
            lines.append(msg)
    return "\n".join(lines)


def _get_length_guidance(response_style: str, max_length: int) -> str:
    """Get length guidance based on response style."""
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
    """Get additional instructions based on thread context."""
    instructions = []
    if state_name == "open_question":
        instructions.append("- Answer the question directly")
    elif state_name == "awaiting_response":
        instructions.append("- Acknowledge their message and respond appropriately")

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
    """Extract and resolve all components for a threaded reply prompt."""
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
        context=truncate_context(context, max_chars=2000),
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
    """Build a prompt for thread-aware reply generation."""
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
        last_message=c.last_message,
    )

    return prompt


def get_thread_max_tokens(config: ThreadedReplyConfig) -> int:
    """Get max tokens for generation based on thread config."""
    base_tokens = config.max_response_length // 4
    return max(30, min(base_tokens + 20, 150))
