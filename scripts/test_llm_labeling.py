#!/usr/bin/env python3
"""Test LLM labeling accuracy on DailyDialog ground truth.

Samples 50 messages from DailyDialog, labels with Llama 70B, compares accuracy.
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

# Label mapping (correct one)
LABEL_MAP = {0: "__dummy__", 1: "inform", 2: "question", 3: "directive", 4: "commissive"}
LABEL_TO_NUM = {"inform": "1", "question": "2", "directive": "3", "commissive": "4"}
NUM_TO_LABEL = {"1": "inform", "2": "question", "3": "directive", "4": "commissive"}

LABELING_PROMPT = """\
Classify each text message into ONE of these speech act categories:

1. inform - Statement providing information or facts
2. question - Asking for information (usually has "?")
3. directive - Request, command, or suggestion (asking someone to do something)
4. commissive - Commitment or promise (speaker commits to future action)

Rules:
- "?" usually means question, BUT "Can you help?" is directive (request)
- "I will/I'll/let me" usually means commissive
- Imperatives ("Go", "Please do") are directive
- Everything else is inform (default)

Messages:
{messages}

Reply with ONLY the category numbers (1-4), one per line. No explanations."""


def sample_dailydialog(n: int = 50, seed: int = 42) -> list[dict]:
    """Sample n messages from DailyDialog with ground truth labels."""
    print(f"Loading DailyDialog...")
    dd = load_dataset("OpenRL/daily_dialog", split="train")

    # Collect all messages with labels
    all_examples = []
    for dialogue in dd:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        for text, act in zip(utterances, acts):
            if act in LABEL_MAP and len(text.strip()) > 5:
                all_examples.append({"text": text.strip(), "label": LABEL_MAP[act]})

    # Filter out __dummy__
    all_examples = [ex for ex in all_examples if ex["label"] != "__dummy__"]

    # Stratified sample (equal from each category)
    random.seed(seed)
    by_label = {}
    for ex in all_examples:
        label = ex["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(ex)

    samples = []
    per_category = n // 4
    for label in ["inform", "question", "directive", "commissive"]:
        cat_samples = random.sample(by_label[label], per_category)
        samples.extend(cat_samples)

    random.shuffle(samples)
    print(f"Sampled {len(samples)} messages (stratified)")
    return samples


def label_with_llm(
    messages: list[str], model: str = "llama-3.1-70b", batch_size: int = 50
) -> list[str]:
    """Label messages using LLM."""
    api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not api_key or api_key == "your-key-here":
        raise ValueError(f"{JUDGE_API_KEY_ENV} not set in .env")

    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

    # Format messages
    messages_str = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    prompt = LABELING_PROMPT.format(messages=messages_str)

    print(f"\nLabeling {len(messages)} messages with {model}...", flush=True)

    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200
    )

    response_text = resp.choices[0].message.content.strip()

    # Parse labels
    labels = []
    for line in response_text.strip().splitlines():
        line = line.strip().rstrip(".")
        # Extract first digit 1-4
        for char in line:
            if char in "1234":
                labels.append(NUM_TO_LABEL[char])
                break

    # Pad if needed
    while len(labels) < len(messages):
        labels.append("inform")

    return labels[:len(messages)]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM labeling accuracy")
    parser.add_argument("--n", type=int, default=50, help="Number of messages to test")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b",
        help="Model to use (llama-3.1-70b, llama-3.1-8b, qwen-3-235b-a22b-instruct-2507)",
    )
    args = parser.parse_args()

    # Sample messages
    samples = sample_dailydialog(n=args.n)

    # Get ground truth
    ground_truth = [ex["label"] for ex in samples]
    texts = [ex["text"] for ex in samples]

    # Label with LLM
    predictions = label_with_llm(texts, model=args.model)

    # Calculate accuracy
    correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    accuracy = correct / len(ground_truth)

    print("\n" + "=" * 80)
    print(f"RESULTS: {args.model}")
    print("=" * 80)
    print(f"Total messages: {len(ground_truth)}")
    print(f"Correct: {correct}/{len(ground_truth)}")
    print(f"Accuracy: {accuracy:.1%}")

    # Per-category accuracy
    print("\nPer-category accuracy:")
    for label in ["inform", "question", "directive", "commissive"]:
        indices = [i for i, gt in enumerate(ground_truth) if gt == label]
        if indices:
            cat_correct = sum(1 for i in indices if predictions[i] == ground_truth[i])
            cat_acc = cat_correct / len(indices)
            print(f"  {label:12s} {cat_correct:2d}/{len(indices):2d}  ({cat_acc:.1%})")

    # Show errors
    errors = [
        (texts[i], ground_truth[i], predictions[i])
        for i in range(len(ground_truth))
        if ground_truth[i] != predictions[i]
    ]

    if errors:
        print(f"\nErrors ({len(errors)} total):")
        for text, gt, pred in errors[:10]:  # Show first 10
            print(f"  Text: {text[:70]}")
            print(f"  Expected: {gt}, Got: {pred}")
            print()

    # Save results
    results = {
        "model": args.model,
        "total": len(ground_truth),
        "correct": correct,
        "accuracy": accuracy,
        "per_category": {
            label: {
                "correct": sum(
                    1
                    for i in range(len(ground_truth))
                    if ground_truth[i] == label and predictions[i] == label
                ),
                "total": sum(1 for gt in ground_truth if gt == label),
            }
            for label in ["inform", "question", "directive", "commissive"]
        },
        "errors": [
            {"text": text, "expected": gt, "predicted": pred} for text, gt, pred in errors
        ],
    }

    output_path = PROJECT_ROOT / f"llm_labeling_test_{args.model}.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Decision
    print("\n" + "=" * 80)
    if accuracy >= 0.90:
        print(f"✅ {args.model} is GOOD ENOUGH (≥90% accuracy)")
        print(f"   Safe to use for labeling SAMSum (138k messages)")
    elif accuracy >= 0.80:
        print(f"⚠️  {args.model} is OKAY (80-90% accuracy)")
        print(f"   Consider using a larger model or heuristics")
    else:
        print(f"❌ {args.model} is TOO INACCURATE (<80%)")
        print(f"   Use a larger model or different approach")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
