#!/usr/bin/env python3
"""Test Qwen3-235B labeling on 150 samples vs gold standard."""

from __future__ import annotations

import json
import os
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

from openai import OpenAI

from evals.judge_config import JUDGE_API_KEY_ENV, JUDGE_BASE_URL


def ultra_strict_heuristic(text: str) -> tuple[str | None, float]:
    """Apply ultra-strict heuristics. Only label 100% obvious cases."""
    text_lower = text.lower().strip()

    # needs_answer: ONLY clear wh-questions
    if text.startswith(("What ", "When ", "Where ", "Who ", "How ", "Why ", "Which ")):
        return "needs_answer", 0.99
    if "?" in text and any(
        f" {w} " in f" {text_lower} " for w in ["what", "when", "where", "who", "how", "why"]
    ):
        return "needs_answer", 0.95

    # needs_confirmation: ONLY explicit requests
    if any(p in text_lower for p in ["can you ", "could you ", "will you ", "would you please"]):
        return "needs_confirmation", 0.95
    if text_lower.startswith(("please ", "could you", "can you", "will you")):
        return "needs_confirmation", 0.95

    # needs_empathy: ONLY explicit emotions
    if any(
        p in text_lower
        for p in [
            "i'm so sad",
            "i'm so stressed",
            "i hate ",
            "this sucks",
            "i'm sorry to hear",
        ]
    ):
        return "needs_empathy", 0.95
    if any(
        p in text_lower
        for p in [
            "congrat",
            "i'm so excited",
            "i'm so happy",
            "that's amazing",
            "so proud",
        ]
    ):
        return "needs_empathy", 0.95
    if any(e in text for e in ["😭", "😢", "🎉", "❤️", "💪", "🥳"]):
        return "needs_empathy", 0.90
    if text.count("!") >= 3:
        return "needs_empathy", 0.85

    return None, 0.0


def label_with_qwen(messages: list[dict], batch_size: int = 50) -> list[str]:
    """Label messages with Qwen3-235B."""
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

    all_labels = []

    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]
        messages_str = "\n".join(f"{i + 1}. {m['text_normalized']}" for i, m in enumerate(batch))
        prompt = prompt_template.format(messages=messages_str)

        print(
            f"  Labeling batch {start // batch_size + 1} ({len(batch)} messages)...", flush=True
        )

        resp = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=len(batch) * 3,
        )

        response_text = resp.choices[0].message.content.strip()

        # Parse labels
        label_map = {
            "1": "needs_answer",
            "2": "needs_confirmation",
            "3": "needs_empathy",
            "4": "conversational",
        }
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
    # Load gold standard
    gold_path = PROJECT_ROOT / "gold_standard_150.json"
    gold_data = json.load(open(gold_path))
    messages = gold_data["messages"]

    print("=" * 80)
    print("TESTING QWEN3-235B ON 150 MESSAGES")
    print("=" * 80)
    print()

    # Apply heuristics
    print("Applying ultra-strict heuristics...")
    heuristic_labeled = 0
    qwen_needed = []

    for msg in messages:
        category, confidence = ultra_strict_heuristic(msg["text_normalized"])
        if category:
            msg["qwen_label"] = category
            msg["qwen_method"] = "heuristic"
            msg["qwen_confidence"] = confidence
            heuristic_labeled += 1
        else:
            msg["qwen_label"] = None
            msg["qwen_method"] = "llm"
            msg["qwen_confidence"] = 0.0
            qwen_needed.append(msg)

    print(
        f"  Heuristic labeled: {heuristic_labeled}/{len(messages)} ({heuristic_labeled / len(messages) * 100:.1f}%)"
    )
    print(
        f"  Need Qwen: {len(qwen_needed)}/{len(messages)} ({len(qwen_needed) / len(messages) * 100:.1f}%)"
    )

    # Label with Qwen
    if qwen_needed:
        print(f"\nLabeling {len(qwen_needed)} messages with Qwen3-235B...")
        qwen_labels = label_with_qwen(qwen_needed, batch_size=50)

        # Assign labels back
        qwen_idx = 0
        for msg in messages:
            if msg["qwen_label"] is None:
                msg["qwen_label"] = qwen_labels[qwen_idx]
                qwen_idx += 1

    # Compare with gold standard
    print("\n" + "=" * 80)
    print("RESULTS: QWEN3-235B vs GOLD STANDARD")
    print("=" * 80)

    correct = 0
    wrong = 0
    errors = []

    for msg in messages:
        gold_label = msg["manual_label"]
        qwen_label = msg["qwen_label"]

        if gold_label == qwen_label:
            correct += 1
        else:
            wrong += 1
            errors.append(
                {
                    "text": msg["text_normalized"][:70],
                    "gold": gold_label,
                    "qwen": qwen_label,
                    "method": msg["qwen_method"],
                }
            )

    accuracy = correct / len(messages)

    print(f"\nTotal: {len(messages)}")
    print(f"Correct: {correct}/{len(messages)} ({accuracy * 100:.1f}%)")
    print(f"Wrong: {wrong}/{len(messages)} ({(1 - accuracy) * 100:.1f}%)")

    # Breakdown by method
    heuristic_errors = [e for e in errors if e["method"] == "heuristic"]
    qwen_errors = [e for e in errors if e["method"] == "llm"]

    print(f"\nBreakdown:")
    print(
        f"  Heuristic errors: {len(heuristic_errors)}/{heuristic_labeled} ({len(heuristic_errors) / max(heuristic_labeled, 1) * 100:.1f}% error rate)"
    )
    print(
        f"  Qwen errors: {len(qwen_errors)}/{len(qwen_needed)} ({len(qwen_errors) / max(len(qwen_needed), 1) * 100:.1f}% error rate)"
    )

    # Show errors
    print(f"\nFirst 15 errors:")
    for i, e in enumerate(errors[:15], 1):
        print(f'{i:2d}. [{e["method"]:10s}] Gold: {e["gold"]:20s} Qwen: {e["qwen"]:20s}')
        print(f'    "{e["text"]}"')
        print()

    # Save results
    output_path = PROJECT_ROOT / "qwen_test_results.json"
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(messages),
                    "correct": correct,
                    "wrong": wrong,
                    "accuracy": accuracy,
                    "heuristic_count": heuristic_labeled,
                    "qwen_count": len(qwen_needed),
                    "heuristic_errors": len(heuristic_errors),
                    "qwen_errors": len(qwen_errors),
                },
                "messages": messages,
                "errors": errors,
            },
            indent=2,
        )
    )

    print(f"\nResults saved to: {output_path}")

    # Compare with Llama
    print("\n" + "=" * 80)
    print("COMPARISON: Qwen3-235B vs Llama 3.3 70B")
    print("=" * 80)
    print(f"Qwen3-235B:     {accuracy * 100:.1f}%")
    print(f"Llama 3.3 70B:  50.0%")
    print(f"Improvement:    {(accuracy - 0.5) * 100:+.1f} percentage points")

    if accuracy >= 0.80:
        print("\n✅ Qwen3-235B is GOOD ENOUGH (≥80%)")
    elif accuracy >= 0.70:
        print("\n⚠️  Qwen3-235B is OK (70-80%)")
    else:
        print("\n❌ Qwen3-235B is NOT GOOD ENOUGH (<70%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
