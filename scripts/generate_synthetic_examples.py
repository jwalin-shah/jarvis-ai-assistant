#!/usr/bin/env python3
"""Generate synthetic training examples for under-represented categories.

Creates realistic synthetic examples for `ack` category only.
Clarify and social synthetic examples removed due to causing label confusion.

Usage:
    uv run python scripts/generate_synthetic_examples.py
"""

from __future__ import annotations

import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_ack_examples(n: int = 1500) -> list[dict]:
    """Generate synthetic acknowledgment examples.

    Creates realistic ack messages with varying context to train the model
    on recognition of simple acknowledgments, reactions, and short responses.

    Args:
        n: Number of examples to generate.

    Returns:
        List of example dicts with text, last_message, context, label.
    """
    # Ack message templates
    ack_messages = [
        # Simple acknowledgments
        "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "yea",
        "no", "nope", "nah", "sure", "alright", "aight",
        # Reactions
        "lol", "lmao", "haha", "hehe", "😂", "🤣", "💀",
        "nice", "cool", "dope", "lit", "fire", "🔥",
        # Gratitude
        "thanks", "thx", "ty", "thank you", "np", "yw",
        # Confirmations
        "sounds good", "got it", "gotcha", "bet", "word", "for sure",
        "copy", "heard", "👍", "👌", "✓",
        # Closings
        "bye", "cya", "later", "ttyl", "see you", "gn", "gm",
        # Status updates
        "omw", "on my way", "be there soon", "almost there",
        # Emoji-only
        "👍", "👌", "🙏", "❤️", "😊", "😎", "🎉", "✨",
    ]

    # Context messages (what they're acknowledging)
    context_templates = [
        # Questions
        "Want to grab lunch?",
        "Can you pick up milk?",
        "Are you coming to the party?",
        "Did you see my message?",
        "Can you help me with this?",
        # Statements
        "I'll be there at 5",
        "Just sent you the file",
        "Meeting is at 3pm",
        "Thanks for your help",
        "See you tomorrow",
        # Plans
        "Let's meet at the cafe",
        "I'll pick you up at 7",
        "Dinner at 8?",
        "Coffee this afternoon?",
        "Movie tonight?",
        # News
        "I got the job!",
        "Check out this video",
        "Just finished the report",
        "Found a great restaurant",
        "Made it home safe",
    ]

    examples = []
    random.seed(42)

    for i in range(n):
        # Pick random ack message
        text = random.choice(ack_messages)

        # Add variation (capitalization, punctuation)
        if random.random() < 0.3:  # 30% chance of capitalization
            text = text.capitalize()
        if random.random() < 0.2 and not any(c in text for c in "!?.👍👌"):  # 20% add punctuation
            text += random.choice(["", "!", ".", "!!", "..."])

        # Pick random context
        last_msg = random.choice(context_templates)

        # Build context (0-3 previous messages)
        context_size = random.choice([0, 0, 1, 2, 3])  # Bias toward less context
        if context_size > 0:
            context = random.sample(context_templates, min(context_size, len(context_templates)))
        else:
            context = []

        examples.append({
            "text": text,
            "last_message": last_msg,
            "context": context,
            "label": "ack",
            "source": "synthetic_ack",
            "metadata": None,
        })

    return examples


def main() -> int:
    """Generate synthetic examples and report statistics."""
    print("Generating synthetic training examples...")

    ack_examples = generate_ack_examples(n=1500)

    print(f"\nGenerated {len(ack_examples)} ack examples")
    print(f"Total synthetic: {len(ack_examples)}")

    # Show samples
    print("\n--- Sample ACK examples ---")
    for ex in random.sample(ack_examples, 5):
        print(f"  '{ex['text']}' (responding to: '{ex['last_message']}')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
