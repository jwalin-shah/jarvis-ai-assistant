#!/usr/bin/env python3
"""Convert SOC-2508 conversations into mlx-lm SFT training format.

Downloads the SOC-2508 dataset (1,180 conversations) and converts each
conversation turn into a JSONL chat-message training example suitable for
mlx_lm.lora fine-tuning.

Output: data/soc_sft/{train,valid,test}.jsonl

Usage:
    uv run python scripts/prepare_soc_data.py
    uv run python scripts/prepare_soc_data.py --min-context 3 --max-context 10
    uv run python scripts/prepare_soc_data.py --dry-run  # print stats only
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Media tag normalization
# ---------------------------------------------------------------------------

# SOC uses tags like <image>, <gif>, <video>, <link>, <delay X>
MEDIA_TAG_RE = re.compile(
    r"<(image|gif|video|sticker|audio|file)>",
    re.IGNORECASE,
)
LINK_TAG_RE = re.compile(r"<link>", re.IGNORECASE)
DELAY_TAG_RE = re.compile(r"<delay\s+[\d.]+>", re.IGNORECASE)

MEDIA_PLACEHOLDER_MAP = {
    "image": "[Photo]",
    "gif": "[GIF]",
    "video": "[Video]",
    "sticker": "[Sticker]",
    "audio": "[Voice Memo]",
    "file": "[File]",
}


def normalize_media_tags(text: str) -> str:
    """Replace SOC media tags with iMessage-style placeholders."""

    # Replace specific media tags
    def _replace_media(m: re.Match) -> str:
        tag = m.group(1).lower()
        return MEDIA_PLACEHOLDER_MAP.get(tag, f"[{tag.title()}]")

    text = MEDIA_TAG_RE.sub(_replace_media, text)
    text = LINK_TAG_RE.sub("[Link]", text)
    # Strip delay tags entirely
    text = DELAY_TAG_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Style guide formatting (matches runtime format_style_guide output)
# ---------------------------------------------------------------------------


def format_style_from_persona(persona: dict) -> str:
    """Format SOC persona chatting_style into style guide matching runtime format.

    Args:
        persona: SOC persona dict with 'chatting_style' field.

    Returns:
        Style guide string matching format_style_guide() output format.
    """
    style = persona.get("chatting_style", "")
    if not style:
        return ""

    # SOC chatting_style is already a natural language description
    # Wrap it in the same format as format_style_guide() output
    return style.strip()


# ---------------------------------------------------------------------------
# Conversation extraction
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = (
    "<system>\n"
    "You are NOT an AI assistant. You are replying to a text message from your phone.\n"
    "Just text back. No helpfulness, no formality, no assistant behavior.\n"
    "Rules:\n"
    "- Match their texting style exactly (length, formality, abbreviations, emoji, punctuation)\n"
    "- Sound natural, never like an AI\n"
    '- No phrases like "I hope this helps" or "Let me know"\n'
    "- No formal greetings unless they use them\n"
    '- If the message is unclear or you lack context to reply properly, respond with just "?"\n'
    "{style_section}"
    "</system>"
)


def extract_messages(chat_parts: list[dict]) -> list[dict]:
    """Flatten chat_parts into a list of {sender, text} dicts.

    Each chat_part has a 'sender' and 'messages' list. Each message
    can be a string or a dict with 'text' key.
    """
    flat: list[dict] = []
    for part in chat_parts:
        sender = part.get("sender", "unknown")
        messages = part.get("messages", [])
        for msg in messages:
            if isinstance(msg, str):
                text = msg
            elif isinstance(msg, dict):
                text = msg.get("text", "")
            else:
                continue
            text = normalize_media_tags(text)
            if text:
                flat.append({"sender": sender, "text": text})
    return flat


def build_training_examples(
    conversation: dict,
    min_context: int = 2,
    max_context: int = 10,
) -> list[dict]:
    """Extract SFT training examples from a single SOC conversation.

    For each reply turn, creates a training example with:
    - system: the system prompt + style guide
    - user: conversation context + last message
    - assistant: the actual reply

    Extracts from both directions (persona1->persona2, persona2->persona1).

    Args:
        conversation: Single SOC conversation record.
        min_context: Minimum context messages before the reply.
        max_context: Maximum context messages to include.

    Returns:
        List of training examples in chat message format.
    """
    experience = conversation.get("experience", {})
    persona1 = experience.get("persona1", {})
    persona2 = experience.get("persona2", {})
    relationship = experience.get("relationship", "friends")
    chat_parts = conversation.get("chat_parts", [])

    messages = extract_messages(chat_parts)
    if len(messages) < min_context + 1:
        return []

    examples = []

    for i in range(min_context, len(messages)):
        reply_msg = messages[i]
        reply_sender = reply_msg["sender"]
        reply_text = reply_msg["text"]

        # Skip very short or empty replies
        if len(reply_text.strip()) < 2:
            continue

        # Skip replies that are just media placeholders
        if reply_text.strip() in ("[Photo]", "[GIF]", "[Video]", "[Link]", "[Sticker]"):
            continue

        # Determine which persona is replying and use their style
        if reply_sender == "persona1":
            style_desc = format_style_from_persona(persona1)
        else:
            style_desc = format_style_from_persona(persona2)

        # Build style section
        style_section = ""
        if style_desc:
            style_section = f"\nStyle guide:\n{style_desc}\n"
        if relationship:
            rel_text = relationship.replace("_", " ")
            style_section += f"Relationship: {rel_text}\n"

        system_msg = SYSTEM_TEMPLATE.format(style_section=style_section)

        # Build context window (last N messages before the reply)
        context_start = max(0, i - max_context)
        context_msgs = messages[context_start:i]
        last_msg = context_msgs[-1] if context_msgs else reply_msg

        # Format context as conversation
        context_lines = []
        for cm in context_msgs[:-1]:  # all but last
            context_lines.append(f"{cm['sender']}: {cm['text']}")
        context_str = "\n".join(context_lines)

        # Build user message
        user_parts = []
        if context_str:
            user_parts.append(f"<conversation>\n{context_str}\n</conversation>")
        user_parts.append(f"<last_message>{last_msg['text']}</last_message>")
        user_msg = "\n".join(user_parts)

        example = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply_text},
            ]
        }
        examples.append(example)

    return examples


# ---------------------------------------------------------------------------
# Style archetype classification (for stratified splitting)
# ---------------------------------------------------------------------------

STYLE_KEYWORDS = {
    "formal": ["professional", "formal", "polite", "courteous", "proper"],
    "casual": ["casual", "chill", "relaxed", "laid-back", "easy-going"],
    "enthusiastic": ["enthusiastic", "energetic", "exclamation", "excited", "emoji"],
    "terse": ["short", "brief", "terse", "concise", "minimal", "dry"],
    "slang": ["slang", "abbreviat", "lol", "lmao", "text-speak", "informal"],
}


def classify_style_archetype(style_desc: str) -> str:
    """Classify a persona's chatting style into a broad archetype."""
    if not style_desc:
        return "unknown"
    lower = style_desc.lower()
    scores: dict[str, int] = {}
    for archetype, keywords in STYLE_KEYWORDS.items():
        scores[archetype] = sum(1 for kw in keywords if kw in lower)
    if not any(scores.values()):
        return "unknown"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_data(
    min_context: int = 2,
    max_context: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """Download SOC-2508 and convert to SFT training format.

    Returns:
        Dict with stats about the conversion.
    """
    from datasets import load_dataset

    print("Downloading SOC-2508 dataset...")
    ds = load_dataset("marcodsn/SOC-2508", split="train")
    print(f"Loaded {len(ds)} conversations")

    all_examples: list[dict] = []
    # Track archetype per example for stratified splitting
    archetypes: list[str] = []
    skipped = 0

    for i, conv in enumerate(ds):
        examples = build_training_examples(
            conv,
            min_context=min_context,
            max_context=max_context,
        )
        if not examples:
            skipped += 1
            continue

        # Classify style for stratification
        experience = conv.get("experience", {})
        p1_style = experience.get("persona1", {}).get("chatting_style", "")
        p2_style = experience.get("persona2", {}).get("chatting_style", "")
        archetype = classify_style_archetype(p1_style + " " + p2_style)

        for ex in examples:
            all_examples.append(ex)
            archetypes.append(archetype)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(ds)} convs, {len(all_examples)} examples")

    print(f"\nTotal examples: {len(all_examples)}")
    print(f"Skipped conversations: {skipped}")

    # Archetype distribution
    arch_counts = Counter(archetypes)
    print("\nStyle archetype distribution:")
    for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
        print(f"  {arch:15s} {count:5d} ({count / len(archetypes) * 100:.1f}%)")

    if dry_run:
        print("\nDry run - not saving files.")
        # Show a sample
        if all_examples:
            print("\nSample example:")
            print(json.dumps(all_examples[0], indent=2)[:500])
        return {
            "total": len(all_examples),
            "skipped": skipped,
            "archetypes": dict(arch_counts),
        }

    # Stratified train/valid/test split
    random.seed(seed)

    # Group by archetype for stratified splitting
    archetype_indices: dict[str, list[int]] = {}
    for idx, arch in enumerate(archetypes):
        archetype_indices.setdefault(arch, []).append(idx)

    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []

    for arch, indices in archetype_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_indices.extend(indices[:n_train])
        valid_indices.extend(indices[n_train : n_train + n_valid])
        test_indices.extend(indices[n_train + n_valid :])

    # Shuffle within splits
    random.shuffle(train_indices)
    random.shuffle(valid_indices)
    random.shuffle(test_indices)

    # Save
    output_dir = PROJECT_ROOT / "data" / "soc_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
    }

    for split_name, indices in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for idx in indices:
                f.write(json.dumps(all_examples[idx]) + "\n")
        print(f"Saved {len(indices)} examples to {path}")

    # Save metadata
    meta = {
        "source": "marcodsn/SOC-2508",
        "total_conversations": len(ds),
        "total_examples": len(all_examples),
        "skipped_conversations": skipped,
        "splits": {k: len(v) for k, v in splits.items()},
        "archetypes": dict(arch_counts),
        "params": {
            "min_context": min_context,
            "max_context": max_context,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "seed": seed,
        },
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata saved to {meta_path}")

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SOC-2508 to mlx-lm SFT format")
    parser.add_argument(
        "--min-context", type=int, default=2, help="Min context messages (default: 2)"
    )
    parser.add_argument(
        "--max-context", type=int, default=10, help="Max context messages (default: 10)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    prepare_data(
        min_context=args.min_context,
        max_context=args.max_context,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
