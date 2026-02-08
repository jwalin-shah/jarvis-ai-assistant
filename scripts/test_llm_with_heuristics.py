#!/usr/bin/env python3
"""Test ultra-strict heuristics + LLM labeling on 150 samples.

1. Sample 150 messages (50 DailyDialog, 100 SAMSum)
2. Apply ultra-strict heuristics (only obvious cases)
3. Send remaining to LLM for labeling
4. Save for manual review
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from datasets import load_dataset
from openai import OpenAI

from evals.judge_config import JUDGE_API_KEY_ENV, JUDGE_BASE_URL


def ultra_strict_heuristic(text: str) -> tuple[str | None, float]:
    """Apply ultra-strict heuristics. Only label 100% obvious cases.

    Returns:
        (category, confidence) or (None, 0.0) if should send to LLM
    """
    text_lower = text.lower().strip()

    # needs_answer: ONLY clear wh-questions
    if text.startswith(("What ", "When ", "Where ", "Who ", "How ", "Why ", "Which ")):
        return "needs_answer", 0.99
    if "?" in text and any(f" {w} " in f" {text_lower} " for w in ["what", "when", "where", "who", "how", "why"]):
        return "needs_answer", 0.95

    # needs_confirmation: ONLY explicit requests
    if any(p in text_lower for p in ["can you ", "could you ", "will you ", "would you please"]):
        return "needs_confirmation", 0.95
    if text_lower.startswith(("please ", "could you", "can you", "will you")):
        return "needs_confirmation", 0.95

    # needs_empathy: ONLY explicit emotions
    # Strong negative
    if any(p in text_lower for p in ["i'm so sad", "i'm so stressed", "i hate ", "this sucks", "i'm sorry to hear"]):
        return "needs_empathy", 0.95
    # Strong positive
    if any(p in text_lower for p in ["congrat", "i'm so excited", "i'm so happy", "that's amazing", "so proud"]):
        return "needs_empathy", 0.95
    # Clear emotion emojis
    if any(e in text for e in ["😭", "😢", "🎉", "❤️", "💪", "🥳"]):
        return "needs_empathy", 0.90
    # Multiple exclamation marks (excitement)
    if text.count("!") >= 3:
        return "needs_empathy", 0.85

    # Everything else → send to LLM
    return None, 0.0


def sample_messages(n_dd: int = 50, n_samsum: int = 100, seed: int = 123) -> list[dict]:
    """Sample messages from both datasets."""
    print(f"Sampling {n_dd} from DailyDialog, {n_samsum} from SAMSum...")

    # DailyDialog
    dd = load_dataset("OpenRL/daily_dialog", split="train")
    dd_utterances = []
    for dialogue in dd:
        utterances = dialogue["dialog"]
        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) >= 3:
                dd_utterances.append(
                    {
                        "text": text,
                        "previous": utterances[i - 1].strip(),
                        "source": "dailydialog",
                    }
                )

    # SAMSum
    samsum = load_dataset("knkarthick/samsum", split="train")
    samsum_turns = []
    for conv in samsum:
        lines = [l.strip() for l in conv["dialogue"].split("\n") if l.strip()]
        messages = []
        for line in lines:
            colon_idx = line.find(":")
            if 0 < colon_idx < 30:
                text = line[colon_idx + 1 :].strip()
                if len(text) >= 3:
                    messages.append(text)
        if len(messages) >= 2:
            for i in range(1, len(messages)):
                samsum_turns.append(
                    {
                        "text": messages[i],
                        "previous": messages[i - 1],
                        "source": "samsum",
                    }
                )

    random.seed(seed)
    dd_samples = random.sample(dd_utterances, min(n_dd, len(dd_utterances)))
    samsum_samples = random.sample(samsum_turns, min(n_samsum, len(samsum_turns)))

    all_samples = dd_samples + samsum_samples
    random.shuffle(all_samples)

    print(f"  Sampled {len(all_samples)} messages")
    return all_samples


def label_with_llm(messages: list[dict], model: str = "llama-3.3-70b") -> list[str]:
    """Label messages with LLM in batches."""
    api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not api_key or api_key == "your-key-here":
        raise ValueError(f"{JUDGE_API_KEY_ENV} not set")

    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

    prompt_template = """Classify each message into ONE category based on what kind of REPLY it needs:

1. needs_answer - Expects factual information (questions with what/when/where/who/how/why)
2. needs_confirmation - Expects yes/no or acknowledgment (requests, yes/no questions, directives)
3. needs_empathy - Needs emotional support (celebrating, comforting, validating feelings)
4. conversational - Casual engagement (statements, updates, casual chat)

Messages:
{messages}

Reply with ONLY the category numbers (1-4), one per line. No explanations."""

    # Batch into groups of 50
    batch_size = 50
    all_labels = []

    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]
        messages_str = "\n".join(f"{i + 1}. {m['text']}" for i, m in enumerate(batch))
        prompt = prompt_template.format(messages=messages_str)

        print(f"  Labeling batch {start // batch_size + 1} ({len(batch)} messages)...", flush=True)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=len(batch) * 3,
        )

        response_text = resp.choices[0].message.content.strip()

        # Parse labels
        label_map = {"1": "needs_answer", "2": "needs_confirmation", "3": "needs_empathy", "4": "conversational"}
        labels = []
        for line in response_text.strip().splitlines():
            line = line.strip().rstrip(".")
            for char in line:
                if char in "1234":
                    labels.append(label_map[char])
                    break

        # Pad if needed
        while len(labels) < len(batch):
            labels.append("conversational")

        all_labels.extend(labels[: len(batch)])

    return all_labels


def main() -> int:
    # Sample messages
    messages = sample_messages(n_dd=50, n_samsum=100, seed=999)  # Different seed from first test

    # Apply ultra-strict heuristics
    print("\nApplying ultra-strict heuristics...")
    heuristic_labeled = 0
    llm_needed = []

    for msg in messages:
        category, confidence = ultra_strict_heuristic(msg["text"])
        if category:
            msg["label"] = category
            msg["method"] = "heuristic"
            msg["confidence"] = confidence
            heuristic_labeled += 1
        else:
            msg["label"] = None
            msg["method"] = "llm"
            msg["confidence"] = 0.0
            llm_needed.append(msg)

    print(f"  Heuristic labeled: {heuristic_labeled}/{len(messages)} ({heuristic_labeled / len(messages) * 100:.1f}%)")
    print(f"  Need LLM: {len(llm_needed)}/{len(messages)} ({len(llm_needed) / len(messages) * 100:.1f}%)")

    # Label remaining with LLM
    if llm_needed:
        print(f"\nLabeling {len(llm_needed)} messages with LLM...")
        llm_labels = label_with_llm(llm_needed, model="llama-3.3-70b")

        # Assign LLM labels back
        llm_idx = 0
        for msg in messages:
            if msg["label"] is None:
                msg["label"] = llm_labels[llm_idx]
                llm_idx += 1

    # Save for manual review
    output_path = PROJECT_ROOT / "llm_test_results.json"
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(messages),
                    "heuristic_labeled": heuristic_labeled,
                    "llm_labeled": len(llm_needed),
                    "model": "llama-3.3-70b",
                },
                "messages": messages,
            },
            indent=2,
        )
    )

    print(f"\n✅ Results saved to {output_path}")
    print("\nNext: Manually review all 150 labels to check accuracy")

    # Show distribution
    from collections import Counter

    label_counts = Counter(msg["label"] for msg in messages)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(messages) * 100
        print(f"  {label:20s} {count:3d} ({pct:5.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
