"""Smart prompt builder V2 - improved few-shot retrieval + style tuning.

Improvements over V1:
1. Uses improved retriever with multi-signal matching
2. Better style instructions based on cluster analysis
3. Message type-aware prompting
4. Dynamic example selection based on query type
"""

from dataclasses import dataclass

from core.generation.style_prompter import get_style_context, CLUSTER_STYLES
from core.generation.fewshot_retriever_v2 import (
    get_retriever_v2,
    FewShotExample,
    MatchCriteria,
    detect_message_type,
    extract_style_features,
)


# Expected response styles by message type
RESPONSE_HINTS = {
    "reaction": "lol ok",  # Short casual acknowledgment
    "greeting": "hey",  # Simple greeting back
    "question": "sure",  # Short affirmative
    "statement": "",  # No hint, use input matching
}


@dataclass
class SmartPromptV2:
    """A fully constructed prompt for reply generation."""
    prompt: str
    style_context: str
    few_shot_examples: list[FewShotExample]
    contact: str
    cluster_name: str
    message_type: str  # New: type of message being replied to


def build_smart_prompt_v2(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 3,
    min_similarity: float = 0.5,
) -> SmartPromptV2:
    """Build an improved smart prompt with better example matching.

    Args:
        contact_name: Name of the contact
        conversation: List of {"text": "...", "is_from_me": bool}
        n_examples: Number of few-shot examples to include
        min_similarity: Minimum similarity for examples

    Returns:
        SmartPromptV2 with the full prompt and metadata
    """
    # Get style context
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    # Get the last message to reply to
    last_their_msg = ""
    for msg in reversed(conversation):
        if not msg.get("is_from_me"):
            last_their_msg = msg.get("text", "")
            break

    # Detect message type
    msg_type = detect_message_type(last_their_msg)

    # Build conversation context (last 3 messages)
    conv_lines = []
    for msg in conversation[-8:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            conv_lines.append(f"{prefix} {text}")
    conversation_text = "\n".join(conv_lines)
    context_for_matching = " ".join(conv_lines[-3:])

    # Get examples with improved matching - use HYBRID retrieval
    criteria = MatchCriteria(
        prefer_cluster=style.cluster_id,
        target_length=avg_len,
        use_lol=style.use_lol,
        min_similarity=min_similarity,
    )

    retriever = get_retriever_v2()
    retriever.build_embeddings()

    # Get expected response hint based on message type
    expected_style = RESPONSE_HINTS.get(msg_type, "")

    # Use hybrid retrieval for better example matching
    examples = retriever.find_hybrid(
        query=last_their_msg,
        expected_style=expected_style,
        criteria=criteria,
        n=n_examples,
    )

    # Build style instruction - concise and specific
    style_parts = [f"~{avg_len} chars", "casual"]

    # Add message-type specific guidance
    if msg_type == "question":
        style_parts.append("answer directly")
    elif msg_type == "reaction":
        style_parts.append("brief reaction")
    elif msg_type == "greeting":
        style_parts.append("casual greeting back")

    # Style markers
    style_parts.append("no period at end")
    if style.use_lol:
        style_parts.append("lol ok")

    style_instruction = f"[{', '.join(style_parts)}]"

    # Build examples text - prioritize high-similarity examples
    examples_text = ""
    if examples:
        # Filter to only very similar examples for inclusion
        good_examples = [ex for ex in examples if ex.similarity > 0.55][:n_examples]

        if good_examples:
            examples_text = "\nExamples:\n"
            for ex in good_examples:
                # Get last "them:" line from conversation
                conv = ex.conversation
                lines = conv.strip().split("\n")
                their_msg = ""
                for line in reversed(lines):
                    if line.startswith("them:"):
                        their_msg = line.replace("them:", "").strip()[:45]
                        break
                if their_msg:
                    examples_text += f"them: {their_msg}\n"
                    examples_text += f"me: {ex.your_reply}\n"

    # Build final prompt
    if examples_text:
        prompt = f"{style_instruction}\n{examples_text}\n{conversation_text}\nme:"
    else:
        prompt = f"{style_instruction}\n\n{conversation_text}\nme:"

    return SmartPromptV2(
        prompt=prompt.strip(),
        style_context=style_instruction,
        few_shot_examples=examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
        message_type=msg_type,
    )


def build_reaction_prompt(
    contact_name: str,
    conversation: list[dict],
) -> SmartPromptV2:
    """Build a specialized prompt for reaction-type messages.

    Reactions are short responses like "haha", "nice", "yea fr", etc.
    """
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})

    # Get the last message
    last_their_msg = ""
    for msg in reversed(conversation):
        if not msg.get("is_from_me"):
            last_their_msg = msg.get("text", "")
            break

    # Format conversation (minimal for reactions)
    conv_lines = []
    for msg in conversation[-4:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            conv_lines.append(f"{prefix} {text}")
    conversation_text = "\n".join(conv_lines)

    # Get reaction-style examples
    criteria = MatchCriteria(
        prefer_cluster=style.cluster_id,
        target_length=15,  # Reactions are short
        min_similarity=0.4,
    )

    retriever = get_retriever_v2()
    retriever.build_embeddings()
    examples = retriever.find_similar(
        query=last_their_msg,
        criteria=criteria,
        n=5,
    )

    # Filter to only short reactions
    reaction_examples = [ex for ex in examples if ex.reply_length < 25][:3]

    # Build prompt for reactions
    style_instruction = "[5-15 chars, casual reaction"
    if style.use_lol:
        style_instruction += ", lol ok"
    style_instruction += "]"

    examples_text = ""
    if reaction_examples:
        examples_text = "\nReaction examples:\n"
        for ex in reaction_examples:
            examples_text += f"→ {ex.your_reply}\n"

    prompt = f"{style_instruction}\n{examples_text}\n{conversation_text}\nme:"

    return SmartPromptV2(
        prompt=prompt.strip(),
        style_context=style_instruction,
        few_shot_examples=reaction_examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
        message_type="reaction",
    )


# Test
if __name__ == "__main__":
    test_conv = [
        {"text": "hey what are you up to", "is_from_me": False},
        {"text": "nm just working", "is_from_me": True},
        {"text": "wanna grab dinner later", "is_from_me": False},
    ]

    contacts = ["Mihir Shah", "Mom", "Faith", "Test Contact"]

    for contact in contacts:
        print(f"\n{'='*60}")
        print(f"Contact: {contact}")
        print("=" * 60)

        result = build_smart_prompt_v2(contact, test_conv, n_examples=3)
        print(f"Cluster: {result.cluster_name}")
        print(f"Message type: {result.message_type}")
        print(f"Style: {result.style_context}")
        print(f"Examples: {len(result.few_shot_examples)}")
        if result.few_shot_examples:
            print("Top examples:")
            for ex in result.few_shot_examples[:2]:
                print(f"  [{ex.similarity:.2f}] \"{ex.your_reply}\"")
        print(f"\nPrompt:\n{result.prompt}")
