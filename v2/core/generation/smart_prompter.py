"""Smart prompt builder combining style awareness + few-shot retrieval.

This is the main entry point for generating personalized reply prompts.
"""

from dataclasses import dataclass

from core.generation.style_prompter import get_style_context, CLUSTER_STYLES, USER_STYLE
from core.generation.fewshot_retriever import get_retriever, FewShotExample


@dataclass
class SmartPrompt:
    """A fully constructed prompt for reply generation."""
    prompt: str
    style_context: str
    few_shot_examples: list[FewShotExample]
    contact: str
    cluster_name: str


def build_smart_prompt(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 2,
    include_examples: bool = True,
) -> SmartPrompt:
    """Build a smart prompt with style + few-shot examples.

    Args:
        contact_name: Name of the contact
        conversation: List of {"text": "...", "is_from_me": bool}
        n_examples: Number of few-shot examples to include
        include_examples: Whether to include few-shot examples

    Returns:
        SmartPrompt with the full prompt and metadata
    """
    # Get style context
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    # Build style instruction - more specific
    style_parts = []

    # Length guidance
    if style.target_length == "brief":
        style_parts.append(f"~{avg_len} chars")
    elif style.target_length == "longer":
        style_parts.append(f"~{avg_len} chars")
    else:
        style_parts.append(f"~{avg_len} chars")

    # Tone
    if style.cluster_name == "family":
        style_parts.append("warm")
    elif style.cluster_name in ["playful_friends"]:
        style_parts.append("casual playful")
    else:
        style_parts.append("casual")

    # Specific style markers
    style_parts.append("no period at end")

    if style.use_lol:
        style_parts.append("lol ok")

    style_instruction = f"[{', '.join(style_parts)}]"

    # Get few-shot examples with better filtering
    examples = []
    examples_text = ""

    if include_examples:
        last_their_msg = ""
        for msg in reversed(conversation):
            if not msg.get("is_from_me"):
                last_their_msg = msg.get("text", "")
                break

        if last_their_msg:
            retriever = get_retriever()
            retriever.build_embeddings()
            # Get more candidates, filter by similarity
            candidates = retriever.find_similar(last_their_msg, n=n_examples * 2, same_cluster=style.cluster_id)
            examples = [ex for ex in candidates if ex.similarity > 0.65][:n_examples]

            if examples:
                examples_text = "\nExamples:\n"
                for ex in examples:
                    their_msg = ex.conversation.split("\n")[-1].replace("them: ", "")[:40]
                    examples_text += f'them: {their_msg}\n'
                    examples_text += f'me: {ex.your_reply}\n'

    # Format conversation
    conv_lines = []
    for msg in conversation[-10:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            conv_lines.append(f"{prefix} {text}")

    conversation_text = "\n".join(conv_lines)

    # Build final prompt - cleaner
    if examples_text:
        prompt = f"{style_instruction}\n{examples_text}\n{conversation_text}\nme:"
    else:
        prompt = f"{style_instruction}\n\n{conversation_text}\nme:"

    return SmartPrompt(
        prompt=prompt.strip(),
        style_context=style_instruction,
        few_shot_examples=examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
    )


def build_detailed_prompt(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 3,
) -> str:
    """Build a more detailed system-style prompt.

    This version is more explicit and works better with instruction-following models.
    """
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    # Get few-shot examples (with better filtering)
    last_their_msg = ""
    for msg in reversed(conversation):
        if not msg.get("is_from_me"):
            last_their_msg = msg.get("text", "")
            break

    examples = []
    if last_their_msg:
        retriever = get_retriever()
        retriever.build_embeddings()
        # Get more examples, filter to high similarity only
        candidates = retriever.find_similar(last_their_msg, n=n_examples * 2, same_cluster=style.cluster_id)
        # Only keep examples with similarity > 0.7
        examples = [ex for ex in candidates if ex.similarity > 0.65][:n_examples]

    # Build examples section
    examples_section = ""
    if examples:
        examples_section = "Examples of actual replies in this style:\n"
        for ex in examples:
            last_line = ex.conversation.strip().split("\n")[-1]
            # Clean up the example
            their_msg = last_line.replace("them: ", "").strip()
            examples_section += f'them: {their_msg}\n'
            examples_section += f'me: {ex.your_reply}\n\n'

    # Format conversation
    conv_lines = []
    for msg in conversation[-10:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            conv_lines.append(f"{prefix} {text}")

    conversation_text = "\n".join(conv_lines)

    # Build the prompt - more specific instructions
    prompt = f"""Reply to this text message. Match the style EXACTLY.

STYLE RULES (follow strictly):
1. Length: ~{avg_len} chars (short and casual)
2. NO periods at the end
3. Lowercase "i" is ok, natural texting
4. Abbreviations ONLY where natural: "u" for "you", "ur" for "your", "yea" not "yeah"
5. {"Can say 'lol' or 'haha' if funny" if style.use_lol else "Don't use lol/haha"}
6. DON'T start with "U" alone - that's weird
7. DON'T add uwu, emojis, or be overly cute
8. Sound like a normal person texting, not an AI

{examples_section}CONVERSATION:
{conversation_text}

Reply (just the text, nothing else):"""

    return prompt


# Quick test
if __name__ == "__main__":
    test_conv = [
        {"text": "hey what are you up to", "is_from_me": False},
        {"text": "nm just working", "is_from_me": True},
        {"text": "wanna grab dinner later", "is_from_me": False},
    ]

    contacts = ["Mihir Shah", "Mom", "Faith"]

    for contact in contacts:
        print(f"\n{'='*60}")
        print(f"Contact: {contact}")
        print("=" * 60)

        result = build_smart_prompt(contact, test_conv, n_examples=2)
        print(f"Cluster: {result.cluster_name}")
        print(f"Style: {result.style_context}")
        print(f"Examples: {len(result.few_shot_examples)}")
        print(f"\nPrompt:\n{result.prompt}")
