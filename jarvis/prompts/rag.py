from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.contacts.contact_profile_context import ContactProfileContext
from jarvis.prompts.constants import RAG_REPLY_PROMPT
from jarvis.prompts.tone import analyze_user_style
from jarvis.prompts.utils import truncate_context

if TYPE_CHECKING:
    from jarvis.prompts.constants import UserStyleAnalysis


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
        resolved_style = user_style or analyze_user_style([])

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
    custom_instruction = instruction or ""

    # Truncate context if needed
    truncated_context = truncate_context(context)

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


def build_prompt_from_request(req: Any) -> str:
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
    """Build a RAG-enhanced prompt using the embedding store directly."""
    from jarvis.contacts.contact_profile import get_contact_profile
    from jarvis.search.vec_search import get_vec_searcher

    # Get relationship profile
    profile = get_contact_profile(contact_id)

    if profile:
        name = contact_name or profile.contact_name or "this person"
        contact_context = ContactProfileContext.from_contact_profile(profile)
        profile_dict = {
            "tone": contact_context.tone,
            "avg_message_length": profile.avg_message_length,
            "response_patterns": {},
        }
    else:
        name = contact_name or "this person"
        contact_context = None
        profile_dict = {}

    # Find similar past messages
    searcher = get_vec_searcher()
    results = searcher.search(query=last_message, chat_id=contact_id, limit=5)

    exchanges: list[tuple[str, str]] = []
    for res in results:
        if not res.is_from_me and res.text:
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
