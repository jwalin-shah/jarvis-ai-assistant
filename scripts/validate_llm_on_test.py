#!/usr/bin/env python3
"""Validate LLM (Qwen 235B) accuracy on DailyDialog test set.

Compares LLM predictions vs ground truth labels to determine if we can
trust the LLM as a reference for production validation.

Usage:
    uv run python scripts/validate_llm_on_test.py --n-samples 200
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix

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


def sample_test_data(n_samples: int = 200) -> tuple[list[str], list[str]]:
    """Sample N examples from test set with ground truth labels."""
    DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"

    print(f"📂 Loading test data from {DATA_DIR}...")

    # Load test data
    test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)
    y_test = test_data["y"]

    # Load metadata for label mapping
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    labels = metadata["labels"]
    seed = metadata["seed"]

    # Load DailyDialog dataset directly
    print(f"📥 Downloading DailyDialog dataset...")
    from datasets import load_dataset
    dataset = load_dataset("OpenRL/daily_dialog", trust_remote_code=True)

    # Extract all texts and labels
    all_texts = []
    all_labels = []

    for split_name in ["train", "validation", "test"]:
        for example in dataset[split_name]:
            dialog = example["dialog"]
            acts = example.get("act", [])

            if not acts or len(dialog) != len(acts):
                continue

            for utterance, act in zip(dialog, acts):
                # Map DailyDialog act to our labels
                # DailyDialog: 1=inform, 2=question, 3=directive, 4=commissive
                act_map = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
                if act in act_map:
                    all_texts.append(utterance)
                    all_labels.append(act_map[act])

    # Split using same seed as training
    from sklearn.model_selection import train_test_split
    _, test_texts, _, test_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.2,
        random_state=seed,
        stratify=all_labels
    )

    print(f"✓ Loaded {len(test_texts)} test examples")

    # Sample random indices
    rng = np.random.RandomState(42)
    indices = rng.choice(len(test_texts), size=min(n_samples, len(test_texts)), replace=False)

    sampled_texts = [test_texts[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]

    print(f"✓ Sampled {len(sampled_texts)} test examples")
    print()

    # Show distribution
    from collections import Counter
    dist = Counter(sampled_labels)
    print("Ground Truth Distribution:")
    for label in labels:
        count = dist[label]
        pct = 100 * count / len(sampled_labels)
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    return sampled_texts, sampled_labels


def get_llm_labels(texts: list[str], labels: list[str]) -> list[str]:
    """Get labels from Cerebras Qwen 235B."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ CEREBRAS_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )

    # Construct prompt
    system_prompt = f"""You are a text classification expert. Classify each message into one of these categories:
- commissive: promises, commitments ("I'll do it", "sure, I can help")
- directive: requests, commands ("can you help?", "send me the file")
- inform: statements, facts ("I'm at the store", "the meeting is at 3pm")
- question: asking for information ("where are you?", "what time?")

Respond with ONLY the category name, nothing else."""

    predictions = []
    print(f"🤖 Getting LLM labels (Qwen 235B)...")
    print(f"   Cost: ~${len(texts) * 0.0001:.4f}")

    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(texts)}", flush=True)

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b",  # Fast, cheap model for classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            label = response.choices[0].message.content.strip().lower()

            # Validate label
            if label not in labels:
                # Try to extract valid label from response
                for valid_label in labels:
                    if valid_label in label:
                        label = valid_label
                        break
                else:
                    print(f"   ⚠️  Invalid label '{label}' for: {text[:50]}...")
                    label = "inform"  # Default to inform

            predictions.append(label)

        except Exception as e:
            print(f"   ❌ Error on text {i}: {e}")
            predictions.append("inform")  # Default

    print(f"✓ LLM labeling complete")
    print()
    return predictions


def analyze_results(texts, ground_truth, llm_preds, labels):
    """Analyze LLM accuracy against ground truth."""
    print("=" * 70)
    print("📊 LLM Accuracy Analysis")
    print("=" * 70)
    print()

    # Agreement rate
    agreements = sum(1 for gt, llm in zip(ground_truth, llm_preds) if gt == llm)
    agreement_rate = agreements / len(texts)

    print(f"Agreement Rate: {agreement_rate:.1%} ({agreements}/{len(texts)})")
    print()

    # LLM distribution
    from collections import Counter
    llm_dist = Counter(llm_preds)
    print("LLM Predictions:")
    for label in labels:
        count = llm_dist[label]
        pct = 100 * count / len(texts)
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Confusion matrix
    print("Confusion Matrix (Ground Truth=rows, LLM=cols):")
    cm = confusion_matrix(ground_truth, llm_preds, labels=labels)

    # Print header
    print(f"{'':12s}", end="")
    for label in labels:
        print(f"{label:12s}", end="")
    print()

    # Print matrix
    for i, label in enumerate(labels):
        print(f"{label:12s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:12d}", end="")
        print()
    print()

    # Per-class metrics (ground truth as reference)
    print("Per-Class Performance (Ground Truth as reference):")
    report = classification_report(
        ground_truth, llm_preds,
        labels=labels,
        target_names=labels,
        digits=3,
        zero_division=0,
    )
    print(report)

    # Disagreement examples
    print("=" * 70)
    print("📝 Disagreement Examples (first 20)")
    print("=" * 70)
    print()

    disagreements = [
        (text, gt, llm)
        for text, gt, llm in zip(texts, ground_truth, llm_preds)
        if gt != llm
    ]

    for i, (text, gt, llm) in enumerate(disagreements[:20]):
        print(f"{i+1}. Text: {text[:70]}")
        print(f"   Ground Truth: {gt:12s}  LLM: {llm:12s}")
        print()

    return agreement_rate


def main():
    parser = argparse.ArgumentParser(description="Validate LLM on test set")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM Validation on DailyDialog Test Set")
    print("=" * 70)
    print()

    # Load metadata for labels
    metadata_path = PROJECT_ROOT / "data" / "dailydialog_native" / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    labels = metadata["labels"]

    # Sample test data
    texts, ground_truth = sample_test_data(n_samples=args.n_samples)

    # Get LLM predictions
    llm_preds = get_llm_labels(texts, labels)

    # Analyze
    agreement_rate = analyze_results(texts, ground_truth, llm_preds, labels)

    # Save results
    if args.output:
        results = {
            "n_samples": len(texts),
            "agreement_rate": agreement_rate,
            "samples": [
                {
                    "text": text,
                    "ground_truth": gt,
                    "llm_pred": llm,
                    "agree": gt == llm,
                }
                for text, gt, llm in zip(texts, ground_truth, llm_preds)
            ],
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {args.output}")

    print()
    print("=" * 70)
    print("🎯 Verdict")
    print("=" * 70)
    print()

    if agreement_rate >= 0.85:
        print(f"✅ LLM is RELIABLE ({agreement_rate:.1%} agreement)")
        print("   → Can trust LLM labels on production data")
        print("   → LightGBM needs improvement (68% vs LLM on production)")
    elif agreement_rate >= 0.75:
        print(f"⚠️  LLM is SOMEWHAT RELIABLE ({agreement_rate:.1%} agreement)")
        print("   → LLM is better than random but not perfect")
        print("   → Need to check disagreement patterns before trusting it")
    else:
        print(f"❌ LLM is UNRELIABLE ({agreement_rate:.1%} agreement)")
        print("   → Cannot trust LLM labels")
        print("   → LightGBM might actually be better than we thought")


if __name__ == "__main__":
    main()
