#!/usr/bin/env python3
"""Sample messages from SAMSum + DailyDialog for manual labeling test.

Saves to JSON for Claude to label interactively.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset


def sample_samsum(n: int = 100, seed: int = 42) -> list[dict]:
    """Sample n turns from SAMSum conversations."""
    print(f"Loading SAMSum...")
    ds = load_dataset("knkarthick/samsum", split="train")

    # Extract all turns
    all_turns = []
    for conv in ds:
        dialogue_text = conv["dialogue"]
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]

        messages = []
        for line in lines:
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 30:
                text = line[colon_idx + 1 :].strip()
                if text and len(text) >= 3:
                    messages.append(text)

        if len(messages) >= 2:
            for i in range(1, len(messages)):
                all_turns.append(
                    {
                        "text": messages[i],
                        "previous": messages[i - 1],
                        "source": "samsum",
                    }
                )

    random.seed(seed)
    samples = random.sample(all_turns, min(n, len(all_turns)))
    print(f"  Sampled {len(samples)} SAMSum turns")
    return samples


def sample_dailydialog(n: int = 50, seed: int = 42) -> list[dict]:
    """Sample n utterances from DailyDialog."""
    print(f"Loading DailyDialog...")
    ds = load_dataset("OpenRL/daily_dialog", split="train")

    # Extract all utterances (ignore old labels)
    all_utterances = []
    for dialogue in ds:
        utterances = dialogue["dialog"]
        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) >= 3:
                all_utterances.append(
                    {
                        "text": text,
                        "previous": utterances[i - 1].strip(),
                        "source": "dailydialog",
                    }
                )

    random.seed(seed)
    samples = random.sample(all_utterances, min(n, len(all_utterances)))
    print(f"  Sampled {len(samples)} DailyDialog utterances")
    return samples


def main() -> int:
    print("=" * 80)
    print("SAMPLING MESSAGES FOR MANUAL LABELING")
    print("=" * 80)

    samsum_samples = sample_samsum(n=100)
    dd_samples = sample_dailydialog(n=50)

    all_samples = samsum_samples + dd_samples
    random.shuffle(all_samples)

    # Save to JSON
    output_path = PROJECT_ROOT / "manual_labeling_sample.json"
    output_path.write_text(
        json.dumps(
            {
                "instructions": """
Label each message with ONE category:

1. needs_answer - Expects factual info (has "?" + wh-word: what/when/where/who/how/why)
2. needs_confirmation - Expects yes/no/acknowledgment (imperative OR "can you/could you/will you/please")
3. needs_empathy - Needs emotional support (negative: sad/stressed/died/failed OR very positive: promoted/excited OR emojis 😭😊🎉)
4. conversational - Casual engagement (default, everything else)

For each message, consider:
- What kind of REPLY does this expect?
- What should the SLM's tone/style be?
""",
                "categories": [
                    "needs_answer",
                    "needs_confirmation",
                    "needs_empathy",
                    "conversational",
                ],
                "messages": all_samples,
            },
            indent=2,
        )
    )

    print(f"\nSaved {len(all_samples)} messages to: {output_path}")
    print("\nNext step: Label these messages and analyze for confusion/overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
