#!/usr/bin/env python3
"""Test improved prompt with Qwen3-235B on 150 samples."""

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


IMPROVED_PROMPT = """You are labeling messages based on what KIND OF REPLY they need.

CRITICAL RULES:
- You're labeling the INCOMING message (what someone sent), not what to reply
- Look at the PREVIOUS message for context
- ANSWERS to questions are "conversational", NOT "needs_answer"
- RESPONSES to requests are "conversational", NOT "needs_confirmation"

Categories (pick ONE):

1. needs_answer - Message ASKS for factual information
   ✓ "What time is it?"
   ✓ "Where are you?"
   ✓ "How does this work?"
   ✗ "The meeting is at 3pm" (this is ANSWERING, label as conversational)
   ✗ "I'm at the store" (this is ANSWERING, label as conversational)

2. needs_confirmation - Message REQUESTS yes/no or action
   ✓ "Can you help?"
   ✓ "Want to grab lunch?"
   ✓ "Are you coming?" (yes/no question)
   ✓ "Off campus or on campus?" (either/or choice)
   ✗ "Yeah I'm coming" (this is ANSWERING a request, label as conversational)
   ✗ "No thanks" (this is DECLINING, label as conversational)

3. needs_empathy - Message expresses STRONG emotion (celebrating/venting/comforting)
   ✓ "I got the job!" (celebration)
   ✓ "I'm so stressed about this exam" (venting)
   ✓ "My dog just died" (grief)
   ✗ "That's cool" (mild acknowledgment, label as conversational)
   ✗ "Nice work" (mild praise, label as conversational)

4. conversational - Everything else (DEFAULT)
   ✓ Answers: "Yes, she's my friend", "The copier is over there"
   ✓ Responses: "ok see you", "sure", "no problem"
   ✓ Updates: "I'm heading out now", "just got home"
   ✓ Statements: "that's interesting", "sounds good"

Message: {message}
Previous: {previous}

Think step-by-step:
1. Is this ASKING a question? → needs_answer
2. Is this REQUESTING action/yes-no? → needs_confirmation
3. Is this expressing STRONG emotion? → needs_empathy
4. Otherwise → conversational

Category (1/2/3/4):"""


def label_with_improved_prompt(messages: list[dict], model: str = "qwen-3-235b-a22b-instruct-2507") -> list[str]:
    """Label messages ONE AT A TIME with improved prompt including context."""
    api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not api_key or api_key == "your-key-here":
        raise ValueError(f"{JUDGE_API_KEY_ENV} not set")

    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

    all_labels = []
    label_map = {
        "1": "needs_answer",
        "2": "needs_confirmation",
        "3": "needs_empathy",
        "4": "conversational",
    }

    print(f"  Labeling {len(messages)} messages individually (with context)...")

    # Process in batches of 10 to show progress
    for i, msg in enumerate(messages):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(messages)}", flush=True)

        prompt = IMPROVED_PROMPT.format(
            message=msg["text_normalized"],
            previous=msg["previous_normalized"]
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        response_text = resp.choices[0].message.content.strip()

        # Parse label
        label = "conversational"  # default
        for char in response_text:
            if char in "1234":
                label = label_map[char]
                break

        all_labels.append(label)

    return all_labels


def main() -> int:
    # Load gold standard
    gold_path = PROJECT_ROOT / "gold_standard_150.json"
    gold_data = json.load(open(gold_path))
    messages = gold_data["messages"]

    print("=" * 80)
    print("TESTING IMPROVED PROMPT WITH QWEN3-235B")
    print("=" * 80)
    print()

    # Apply heuristics
    print("Applying ultra-strict heuristics...")
    heuristic_labeled = 0
    llm_needed = []

    for msg in messages:
        category, confidence = ultra_strict_heuristic(msg["text_normalized"])
        if category:
            msg["improved_label"] = category
            msg["improved_method"] = "heuristic"
            heuristic_labeled += 1
        else:
            msg["improved_label"] = None
            msg["improved_method"] = "llm"
            llm_needed.append(msg)

    print(
        f"  Heuristic labeled: {heuristic_labeled}/{len(messages)} ({heuristic_labeled / len(messages) * 100:.1f}%)"
    )
    print(
        f"  Need LLM: {len(llm_needed)}/{len(messages)} ({len(llm_needed) / len(messages) * 100:.1f}%)"
    )

    # Label with improved prompt
    if llm_needed:
        print(f"\nLabeling {len(llm_needed)} messages with improved prompt...")
        improved_labels = label_with_improved_prompt(llm_needed)

        # Assign labels back
        llm_idx = 0
        for msg in messages:
            if msg["improved_label"] is None:
                msg["improved_label"] = improved_labels[llm_idx]
                llm_idx += 1

    # Compare with gold standard
    print("\n" + "=" * 80)
    print("RESULTS: IMPROVED PROMPT vs GOLD STANDARD")
    print("=" * 80)

    correct = 0
    wrong = 0
    errors = []

    for msg in messages:
        gold_label = msg["manual_label"]
        improved_label = msg["improved_label"]

        if gold_label == improved_label:
            correct += 1
        else:
            wrong += 1
            errors.append(
                {
                    "text": msg["text_normalized"][:70],
                    "previous": msg["previous_normalized"][:60],
                    "gold": gold_label,
                    "improved": improved_label,
                    "method": msg["improved_method"],
                }
            )

    accuracy = correct / len(messages)

    print(f"\nTotal: {len(messages)}")
    print(f"Correct: {correct}/{len(messages)} ({accuracy * 100:.1f}%)")
    print(f"Wrong: {wrong}/{len(messages)} ({(1 - accuracy) * 100:.1f}%)")

    # Breakdown by method
    heuristic_errors = [e for e in errors if e["method"] == "heuristic"]
    llm_errors = [e for e in errors if e["method"] == "llm"]

    print(f"\nBreakdown:")
    print(
        f"  Heuristic errors: {len(heuristic_errors)}/{heuristic_labeled} ({len(heuristic_errors) / max(heuristic_labeled, 1) * 100:.1f}% error rate)"
    )
    print(
        f"  LLM errors: {len(llm_errors)}/{len(llm_needed)} ({len(llm_errors) / max(len(llm_needed), 1) * 100:.1f}% error rate)"
    )

    # Show errors
    print(f"\nFirst 15 errors:")
    for i, e in enumerate(errors[:15], 1):
        print(f'{i:2d}. [{e["method"]:10s}] Gold: {e["gold"]:20s} Got: {e["improved"]:20s}')
        print(f'    Prev: "{e["previous"]}"')
        print(f'    Msg:  "{e["text"]}"')
        print()

    # Save results
    output_path = PROJECT_ROOT / "improved_prompt_results.json"
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(messages),
                    "correct": correct,
                    "wrong": wrong,
                    "accuracy": accuracy,
                    "heuristic_count": heuristic_labeled,
                    "llm_count": len(llm_needed),
                    "heuristic_errors": len(heuristic_errors),
                    "llm_errors": len(llm_errors),
                },
                "messages": messages,
                "errors": errors,
            },
            indent=2,
        )
    )

    print(f"\nResults saved to: {output_path}")

    # Compare with previous prompts
    print("\n" + "=" * 80)
    print("COMPARISON: Improved Prompt vs Original")
    print("=" * 80)
    print(f"Improved Prompt:  {accuracy * 100:.1f}%")
    print(f"Original Qwen:    44.0%")
    print(f"Original Llama:   50.0%")
    print(f"Improvement:      {(accuracy - 0.44) * 100:+.1f} points vs Qwen")

    if accuracy >= 0.80:
        print("\n✅ IMPROVED PROMPT WORKS! (≥80%)")
        print("   Safe to use for labeling training data")
    elif accuracy >= 0.70:
        print("\n⚠️  Better but not great (70-80%)")
    else:
        print("\n❌ Still not good enough (<70%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
