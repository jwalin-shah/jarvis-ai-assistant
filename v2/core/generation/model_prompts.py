"""Model-specific prompt builders for optimal reply generation.

Each model has different behaviors and needs tailored prompts:
- qwen3-0.6b: Has thinking mode, needs /no_think and completion-style prompts
- lfm2.5-1.2b: Tends to output meta-commentary, needs strict "just reply" instructions
- lfm2-2.6b-exp: Best performer, benefits from clean few-shot format
"""

from dataclasses import dataclass
from typing import Literal

from core.generation.style_prompter import get_style_context, CLUSTER_STYLES
from core.generation.fewshot_retriever import get_retriever, FewShotExample


@dataclass
class ModelPrompt:
    """A model-specific prompt for reply generation."""
    prompt: str
    model_id: str
    style_context: str
    few_shot_examples: list[FewShotExample]
    contact: str
    cluster_name: str


def get_few_shot_examples(
    conversation: list[dict],
    cluster_id: int,
    n_examples: int = 2,
    min_similarity: float = 0.6,
) -> tuple[list[FewShotExample], str]:
    """Get few-shot examples for the conversation."""
    last_their_msg = ""
    for msg in reversed(conversation):
        if not msg.get("is_from_me"):
            last_their_msg = msg.get("text", "")
            break

    examples = []
    if last_their_msg:
        retriever = get_retriever()
        retriever.build_embeddings()
        candidates = retriever.find_similar(
            last_their_msg, n=n_examples * 3, same_cluster=cluster_id
        )
        examples = [ex for ex in candidates if ex.similarity > min_similarity][:n_examples]

    return examples, last_their_msg


def format_conversation(conversation: list[dict], max_messages: int = 8) -> str:
    """Format conversation for prompt."""
    conv_lines = []
    for msg in conversation[-max_messages:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            conv_lines.append(f"{prefix} {text}")
    return "\n".join(conv_lines)


# =============================================================================
# QWEN3-0.6B: NOT RECOMMENDED - thinking mode can't be disabled
# =============================================================================

def build_qwen3_prompt(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 3,
) -> ModelPrompt:
    """Build prompt for qwen3-0.6b.

    WARNING: Qwen3-0.6b has a fundamental issue - it's trained to think before
    responding and the /no_think token doesn't reliably disable this. The model
    often gets stuck in an infinite thinking loop or outputs nothing after
    the thinking is stripped.

    NOT RECOMMENDED for reply generation. Use lfm2-2.6b-exp instead.

    This prompt is kept for completeness but will likely produce empty results.
    """
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    examples, _ = get_few_shot_examples(conversation, style.cluster_id, n_examples, 0.55)
    conversation_text = format_conversation(conversation)

    # Build style tag - very brief
    style_tag = f"[{avg_len}ch, casual"
    if style.use_lol:
        style_tag += ", lol ok"
    style_tag += "]"

    # Pure completion format with examples - won't help much
    prompt_parts = ["/no_think", style_tag, ""]

    # Add examples as direct completions
    for ex in examples:
        their_msg = ex.conversation.split("\n")[-1].replace("them: ", "")[:50]
        prompt_parts.append(f"them: {their_msg}")
        prompt_parts.append(f"me: {ex.your_reply}")
        prompt_parts.append("")

    # Add current conversation
    prompt_parts.append(conversation_text)
    prompt_parts.append("me:")

    prompt = "\n".join(prompt_parts)

    return ModelPrompt(
        prompt=prompt,
        model_id="qwen3-0.6b",
        style_context=style_tag,
        few_shot_examples=examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
    )


# =============================================================================
# LFM2.5-1.2B: Needs pure completion format to avoid meta-commentary
# =============================================================================

def build_lfm25_prompt(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 3,
) -> ModelPrompt:
    """Build prompt for lfm2.5-1.2b.

    LFM2.5 tends to output meta-commentary like "Sure! Here's a casual version".
    Strategy:
    1. Use PURE completion format - no instructions at all
    2. More few-shot examples to establish the pattern
    3. Style as a minimal tag, not instructions
    4. Let the pattern speak for itself
    """
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    # Get more examples to establish pattern
    examples, _ = get_few_shot_examples(conversation, style.cluster_id, n_examples, 0.55)
    conversation_text = format_conversation(conversation)

    # Minimal style tag
    style_tag = f"[{avg_len}ch casual"
    if style.use_lol:
        style_tag += " lol"
    style_tag += "]"

    # Pure completion format - examples only, no instructions
    prompt_parts = [style_tag, ""]

    # Add multiple examples to establish pattern
    for ex in examples:
        their_msg = ex.conversation.split("\n")[-1].replace("them: ", "")[:40]
        prompt_parts.append(f"them: {their_msg}")
        prompt_parts.append(f"me: {ex.your_reply}")
        prompt_parts.append("")

    # Add conversation
    prompt_parts.append(conversation_text)
    prompt_parts.append("me:")

    prompt = "\n".join(prompt_parts)

    return ModelPrompt(
        prompt=prompt,
        model_id="lfm2.5-1.2b",
        style_context=style_tag,
        few_shot_examples=examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
    )


# =============================================================================
# LFM2-2.6B-EXP: Best performer - use same format as smart_prompter (it works!)
# =============================================================================

def build_lfm2_exp_prompt(
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 2,
) -> ModelPrompt:
    """Build prompt for lfm2-2.6b-exp.

    This model performs best with the original smart_prompter format.
    Keep it minimal - the key is:
    1. Brief style tag in brackets
    2. Clean few-shot examples
    3. Direct completion format (no "Reply to:" prefix)
    """
    style = get_style_context(contact_name)
    cluster_info = CLUSTER_STYLES.get(style.cluster_id, {})
    avg_len = cluster_info.get("avg_length", 28)

    # Get examples with high similarity threshold
    examples, _ = get_few_shot_examples(conversation, style.cluster_id, n_examples, 0.65)
    conversation_text = format_conversation(conversation)

    # Build concise style instruction (same format as smart_prompter)
    style_parts = [f"~{avg_len} chars", "casual", "no period at end"]
    if style.use_lol:
        style_parts.append("lol ok")
    style_instruction = f"[{', '.join(style_parts)}]"

    # Build prompt - EXACT same format as smart_prompter which works best
    prompt_parts = [style_instruction]

    # Add examples in the same format
    if examples:
        prompt_parts.append("")
        prompt_parts.append("Examples:")
        for ex in examples:
            their_msg = ex.conversation.split("\n")[-1].replace("them: ", "")[:40]
            prompt_parts.append(f"them: {their_msg}")
            prompt_parts.append(f"me: {ex.your_reply}")

    # Add conversation - no "Reply to:" prefix, just direct
    prompt_parts.append("")
    prompt_parts.append(conversation_text)
    prompt_parts.append("me:")

    prompt = "\n".join(prompt_parts)

    return ModelPrompt(
        prompt=prompt,
        model_id="lfm2-2.6b-exp",
        style_context=style_instruction,
        few_shot_examples=examples,
        contact=contact_name,
        cluster_name=style.cluster_name,
    )


# =============================================================================
# Unified interface
# =============================================================================

ModelType = Literal["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]

MODEL_PROMPT_BUILDERS = {
    "qwen3-0.6b": build_qwen3_prompt,
    "lfm2.5-1.2b": build_lfm25_prompt,
    "lfm2-2.6b-exp": build_lfm2_exp_prompt,
}


def build_model_prompt(
    model_id: str,
    contact_name: str,
    conversation: list[dict],
    n_examples: int = 2,
) -> ModelPrompt:
    """Build a model-specific prompt.

    Args:
        model_id: The model identifier
        contact_name: Name of the contact
        conversation: List of {"text": "...", "is_from_me": bool}
        n_examples: Number of few-shot examples

    Returns:
        ModelPrompt with the optimized prompt for that model
    """
    builder = MODEL_PROMPT_BUILDERS.get(model_id)
    if not builder:
        # Default to lfm2-exp style for unknown models
        builder = build_lfm2_exp_prompt

    return builder(contact_name, conversation, n_examples)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    test_conv = [
        {"text": "hey what are you up to", "is_from_me": False},
        {"text": "nm just working", "is_from_me": True},
        {"text": "wanna grab dinner later", "is_from_me": False},
    ]

    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print("=" * 70)

        result = build_model_prompt(model_id, "Mihir Shah", test_conv)
        print(f"Cluster: {result.cluster_name}")
        print(f"Examples: {len(result.few_shot_examples)}")
        print(f"\nPrompt:\n{result.prompt}")
