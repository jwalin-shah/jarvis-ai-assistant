#!/usr/bin/env python3
"""Test with extremely simple, unambiguous prompt."""

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
    """Apply ultra-strict heuristics."""
    text_lower = text.lower().strip()

    if text.startswith(("What ", "When ", "Where ", "Who ", "How ", "Why ", "Which ")):
        return "needs_answer", 0.99
    if "?" in text and any(
        f" {w} " in f" {text_lower} " for w in ["what", "when", "where", "who", "how", "why"]
    ):
        return "needs_answer", 0.95

    if any(p in text_lower for p in ["can you ", "could you ", "will you ", "would you please"]):
        return "needs_confirmation", 0.95
    if text_lower.startswith(("please ", "could you", "can you", "will you")):
        return "needs_confirmation", 0.95

    if any(
        p in text_lower
        for p in [
            "i'm so sad",
            "i'm so stressed",
            "i hate ",
            "this sucks",
            "i'm sorry to hear",
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


# DEAD SIMPLE PROMPT - No numbered lists, clear examples, simple format
SIMPLE_PROMPT = """Classify this message based on what reply it needs.

Previous message: "{previous}"
Current message: "{message}"

Categories:
A = needs_answer (message is ASKING for information using what/when/where/who/how/why)
B = needs_confirmation (message is REQUESTING yes/no or action)
C = needs_empathy (message expresses STRONG emotion - celebrating/venting/grief)
D = conversational (everything else - answers, updates, statements, casual chat)

IMPORTANT:
- If message is ANSWERING a question → choose D
- If message is RESPONDING to request → choose D
- Only choose A if message is ASKING (has ?)
- Only choose B if message is REQUESTING action/yes-no
- Only choose C if STRONG emotion (excited!!! or stressed or celebrating)
- Default to D when unsure

Examples:
"What time is it?" → A
"Can you help?" → B
"I got the job!!!" → C
"ok see you" → D
"I'm at the store" → D (answering "where are you?")
"Yeah I'm coming" → D (responding to "are you coming?")

Reply with ONLY the letter (A/B/C/D):"""


def label_batch_simple(messages: list[dict], batch_size: int = 20) -> list[str]:
    """Label in small batches with simple prompt."""
    api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not api_key:
        raise ValueError(f"{JUDGE_API_KEY_ENV} not set")

    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

    label_map = {
        "A": "needs_answer",
        "B": "needs_confirmation",
        "C": "needs_empathy",
        "D": "conversational",
    }

    all_labels = []

    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]

        # Create batch prompt
        batch_prompts = []
        for i, msg in enumerate(batch, 1):
            prompt = SIMPLE_PROMPT.format(
                previous=msg["previous_normalized"][:100],
                message=msg["text_normalized"][:100]
            )
            batch_prompts.append(f"Message {i}:\n{prompt}\n")

        full_prompt = "\n".join(batch_prompts) + f"\nReply with {len(batch)} letters (A/B/C/D), one per line:"

        if (start // batch_size + 1) % 5 == 0:
            print(f"    Batch {start // batch_size + 1}/{(len(messages) + batch_size - 1) // batch_size}", flush=True)

        resp = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            max_tokens=len(batch) * 2,
        )

        response_text = resp.choices[0].message.content.strip()

        # Parse - look for A/B/C/D
        labels = []
        for line in response_text.strip().splitlines():
            line = line.strip().upper()
            for char in line:
                if char in "ABCD":
                    labels.append(label_map[char])
                    break

        # Pad/truncate
        while len(labels) < len(batch):
            labels.append("conversational")
        labels = labels[:len(batch)]

        all_labels.extend(labels)

    return all_labels


def main() -> int:
    # Load gold standard
    gold_path = PROJECT_ROOT / "gold_standard_150.json"
    gold_data = json.load(open(gold_path))
    messages = gold_data["messages"]

    print("=" * 80)
    print("TESTING SIMPLE PROMPT (A/B/C/D format)")
    print("=" * 80)
    print()

    # Apply heuristics
    print("Applying ultra-strict heuristics...")
    heuristic_labeled = 0
    llm_needed = []

    for msg in messages:
        category, confidence = ultra_strict_heuristic(msg["text_normalized"])
        if category:
            msg["simple_label"] = category
            msg["simple_method"] = "heuristic"
            heuristic_labeled += 1
        else:
            msg["simple_label"] = None
            msg["simple_method"] = "llm"
            llm_needed.append(msg)

    print(f"  Heuristic: {heuristic_labeled}/{len(messages)}")
    print(f"  Need LLM: {len(llm_needed)}/{len(messages)}")

    # Label with simple prompt
    if llm_needed:
        print(f"\nLabeling {len(llm_needed)} messages with simple prompt...")
        simple_labels = label_batch_simple(llm_needed, batch_size=20)

        llm_idx = 0
        for msg in messages:
            if msg["simple_label"] is None:
                msg["simple_label"] = simple_labels[llm_idx]
                llm_idx += 1

    # Compare with gold
    print("\n" + "=" * 80)
    print("RESULTS vs GOLD STANDARD")
    print("=" * 80)

    correct = 0
    errors = []

    for msg in messages:
        if msg["manual_label"] == msg["simple_label"]:
            correct += 1
        else:
            errors.append({
                "text": msg["text_normalized"][:60],
                "gold": msg["manual_label"],
                "got": msg["simple_label"],
                "method": msg["simple_method"],
            })

    accuracy = correct / len(messages)

    print(f"\nAccuracy: {correct}/{len(messages)} ({accuracy * 100:.1f}%)")

    # Breakdown
    h_errors = [e for e in errors if e["method"] == "heuristic"]
    l_errors = [e for e in errors if e["method"] == "llm"]

    print(f"\nHeuristic errors: {len(h_errors)}/{heuristic_labeled}")
    print(f"LLM errors: {len(l_errors)}/{len(llm_needed)} ({len(l_errors) / max(len(llm_needed), 1) * 100:.1f}%)")

    print(f"\nFirst 10 errors:")
    for i, e in enumerate(errors[:10], 1):
        print(f'{i:2d}. Gold: {e["gold"]:20s} Got: {e["got"]:20s}')
        print(f'    "{e["text"]}"')

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Simple Prompt (A/B/C/D):  {accuracy * 100:.1f}%")
    print(f"Original Qwen:            44.0%")
    print(f"Original Llama:           50.0%")

    if accuracy >= 0.75:
        print("\n✅ SIMPLE PROMPT WORKS!")
    elif accuracy >= 0.60:
        print("\n⚠️  Getting better...")
    else:
        print("\n❌ Still not there")

    # Save
    output_path = PROJECT_ROOT / "simple_prompt_results.json"
    output_path.write_text(json.dumps({
        "summary": {"accuracy": accuracy, "correct": correct, "total": len(messages)},
        "messages": messages,
        "errors": errors,
    }, indent=2))
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
