#!/usr/bin/env python3
"""Pilot validation: test LLM category labeling on 200 examples.

Validates the classification prompt and model accuracy before full labeling.
Uses stratified sampling (40 examples per category via heuristic pre-screening).

Usage:
    uv run python scripts/validate_llm_categories.py
    uv run python scripts/validate_llm_categories.py --model zai-glm-4.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import get_judge_client, JUDGE_BASE_URL
from scripts.labeling_functions import get_registry, ABSTAIN
from scripts.label_aggregation import aggregate_labels

VALID_CATEGORIES = ["ack", "info", "emotional", "social", "clarify"]

CLASSIFICATION_PROMPT_TEMPLATE = """Classify each message into ONE category based on how an AI assistant should respond.

Categories:
- ack: Simple acknowledgment, reaction, emoji-only, "ok/yes/no/thanks/bye". No reply needed.
- info: Question, request, scheduling, logistics. Needs factual/action response.
- emotional: Expresses feelings, celebrates, vents. Needs empathy/support.
- social: Casual chat, banter, opinions, stories. Needs friendly engagement.
- clarify: Ambiguous, incomplete, context-dependent. Needs more context first.

Messages:
{messages_block}

Reply with ONLY the category name for each, one per line (e.g., "ack"). No numbers, no explanations."""


def load_stratified_sample(n_per_category: int = 40) -> list[dict]:
    """Load and stratify sample based on heuristic pre-screening.

    Args:
        n_per_category: Number of examples per category target.

    Returns:
        List of examples with text, context, last_message, label (heuristic).
    """
    from datasets import load_dataset
    from scripts.labeling_functions import get_registry

    print(f"Loading datasets for stratified sampling ({n_per_category} per category)...", flush=True)

    # Load both datasets
    dd_ds = load_dataset("OpenRL/daily_dialog", split="train")
    samsum_ds = load_dataset("knkarthick/samsum", split="train")

    # Extract examples
    all_examples: list[dict] = []

    # DailyDialog
    for dialogue in dd_ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, min(len(utterances), 6)):  # First 5 turns per dialogue
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            all_examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": {"act": int(acts[i]), "emotion": int(emotions[i])},
                "source": "dailydialog",
            })

    # SAMSum (first 500 conversations for speed)
    for conv in samsum_ds[:500]:
        dialogue_text = conv["dialogue"]
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]

        messages: list[tuple[str, str]] = []
        for line in lines:
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 30:
                sender = line[:colon_idx].strip()
                text = line[colon_idx + 1:].strip()
                if text:
                    messages.append((sender, text))

        if len(messages) < 2:
            continue

        for i in range(1, len(messages)):
            text = messages[i][1]
            if len(text.strip()) < 2:
                continue

            context = [m[1] for m in messages[max(0, i - 5):i]]
            last_msg = messages[i - 1][1]

            all_examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": None,
                "source": "samsum",
            })

    print(f"  Loaded {len(all_examples)} raw examples", flush=True)

    # Apply heuristic labeling for stratification
    print("  Applying heuristic labeling for stratification...", flush=True)
    registry = get_registry()
    labels, confidences = aggregate_labels(all_examples, registry, method="majority")

    for i, ex in enumerate(all_examples):
        ex["heuristic_label"] = labels[i]
        ex["confidence"] = confidences[i]

    # Stratified sampling
    import numpy as np
    rng = np.random.default_rng(42)

    stratified: list[dict] = []
    for category in VALID_CATEGORIES:
        cat_examples = [ex for ex in all_examples if ex["heuristic_label"] == category]

        if len(cat_examples) >= n_per_category:
            indices = rng.choice(len(cat_examples), n_per_category, replace=False)
            stratified.extend([cat_examples[i] for i in indices])
        else:
            # If not enough, take all
            stratified.extend(cat_examples)

    rng.shuffle(stratified)

    print(f"  Stratified sample: {len(stratified)} examples", flush=True)
    dist = Counter(ex["heuristic_label"] for ex in stratified)
    for cat, count in sorted(dist.items()):
        print(f"    {cat:10s} {count:3d}", flush=True)

    return stratified


def classify_batch_llm(
    examples: list[dict],
    model: str,
    client,
    batch_size: int = 20,
) -> list[str]:
    """Classify examples using LLM in batches.

    Args:
        examples: List of examples to classify.
        model: Model name.
        client: OpenAI client.
        batch_size: Number of examples per API call.

    Returns:
        List of predicted categories (same length as examples).
    """
    predictions: list[str] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]

        # Format messages block
        messages_block = "\n".join([
            f"{j + 1}. {ex['text']}"
            for j, ex in enumerate(batch)
        ])

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(messages_block=messages_block)

        # API call
        print(f"  Batch {i // batch_size + 1}/{(len(examples) + batch_size - 1) // batch_size} "
              f"(examples {i + 1}-{min(i + batch_size, len(examples))})", flush=True)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            # Validate and normalize
            for line in lines:
                # Remove leading numbers and periods if present
                clean = line.lstrip("0123456789. ").lower().strip()
                if clean in VALID_CATEGORIES:
                    predictions.append(clean)
                else:
                    # Fallback: try to extract category name
                    for cat in VALID_CATEGORIES:
                        if cat in clean:
                            predictions.append(cat)
                            break
                    else:
                        predictions.append("social")  # Default fallback

            # Ensure we have exactly batch_size predictions
            while len(predictions) < i + len(batch):
                predictions.append("social")  # Pad with fallback

            time.sleep(2)  # Rate limiting: 30 req/min = 2s between calls

        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            # Fallback to social for failed batch
            predictions.extend(["social"] * len(batch))

    return predictions[:len(examples)]  # Trim any excess


def compute_accuracy(predictions: list[str], examples: list[dict]) -> dict:
    """Compute accuracy metrics.

    Args:
        predictions: LLM predictions.
        examples: Examples with heuristic_label.

    Returns:
        Dict with accuracy, per-class accuracy, confusion matrix.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    y_true = [ex["heuristic_label"] for ex in examples]
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)

    print("\n=== Results ===")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, labels=VALID_CATEGORIES, zero_division=0))

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=VALID_CATEGORIES)
    print("        " + "  ".join(f"{cat:>4s}" for cat in VALID_CATEGORIES))
    for i, cat in enumerate(VALID_CATEGORIES):
        print(f"{cat:>8s} " + "  ".join(f"{cm[i, j]:4d}" for j in range(len(VALID_CATEGORIES))))

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot validation: test LLM category labeling"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-oss-120b",
        help="Model to use (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=40,
        help="Number of examples per category (default: 40)",
    )
    parser.add_argument(
        "--output", type=str, default="llm_pilot_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    print(f"=== LLM Category Labeling Pilot ===")
    print(f"Model: {args.model}")
    print(f"API: {JUDGE_BASE_URL}")
    print()

    # Get client
    client = get_judge_client()
    if not client:
        print("ERROR: CEREBRAS_API_KEY not set in .env")
        return 1

    # Load stratified sample
    examples = load_stratified_sample(n_per_category=args.n_per_category)

    # Classify with LLM
    print(f"\nClassifying {len(examples)} examples with {args.model}...", flush=True)
    t0 = time.perf_counter()
    predictions = classify_batch_llm(examples, args.model, client, batch_size=20)
    elapsed = time.perf_counter() - t0

    print(f"\nClassification complete in {elapsed:.1f}s", flush=True)

    # Compute accuracy
    results = compute_accuracy(predictions, examples)
    results["model"] = args.model
    results["n_examples"] = len(examples)
    results["elapsed_seconds"] = elapsed

    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Show sample comparisons (10 per category)
    print("\n=== Sample Comparisons ===")
    for category in VALID_CATEGORIES:
        cat_examples = [
            (i, ex, predictions[i])
            for i, ex in enumerate(examples)
            if ex["heuristic_label"] == category
        ][:10]

        if not cat_examples:
            continue

        print(f"\n{category.upper()} (showing {len(cat_examples)} examples):")
        for i, ex, pred in cat_examples:
            match = "✓" if pred == category else "✗"
            print(f"  {match} [{pred}] {ex['text'][:60]}")

    print(f"\n{'=' * 60}")
    if results["accuracy"] >= 0.80:
        print("✓ PASS: Accuracy >= 80%. Ready for full labeling.")
        return 0
    else:
        print("✗ FAIL: Accuracy < 80%. Review prompt or model before full labeling.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
