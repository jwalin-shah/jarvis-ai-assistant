#!/usr/bin/env python3
"""Manual labeling tool for production messages.

Shows each message with LightGBM and LLM predictions, user provides ground truth.
Saves progress continuously so you can stop and resume.

Usage:
    uv run python scripts/label_production_messages.py
    uv run python scripts/label_production_messages.py --resume  # Continue from where you left off
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Progress file
PROGRESS_FILE = PROJECT_ROOT / "human_labels_progress.json"


def load_validation_data():
    """Load the 200 production messages with predictions."""
    print("=" * 70)
    print("Loading Production Messages with Model Predictions")
    print("=" * 70)
    print()

    # We need to recreate the validation data
    # Import the validation script functions
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from validate_on_production import (
        sample_messages,
        extract_features_batch,
        get_llm_labels,
    )

    # Load model
    model_path = PROJECT_ROOT / "models" / "lightgbm_category_final.joblib"
    model = joblib.load(model_path)

    metadata_path = PROJECT_ROOT / "models" / "lightgbm_category_final.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    labels = metadata["labels"]

    print("📂 Sampling 200 messages...")
    messages = sample_messages(n_samples=200)

    print("🔮 Getting LightGBM predictions...")
    features = extract_features_batch(messages)
    lgbm_pred_raw = model.predict(features)

    # Handle both string labels and numeric indices
    if isinstance(lgbm_pred_raw[0], (int, np.integer)):
        lgbm_preds = [labels[i] for i in lgbm_pred_raw]
    else:
        lgbm_preds = list(lgbm_pred_raw)

    print("🤖 Getting LLM predictions...")
    llm_preds = get_llm_labels(messages, labels)

    print(f"✓ Loaded {len(messages)} messages with predictions")
    print()

    return messages, lgbm_preds, llm_preds, labels


def load_progress():
    """Load saved progress if exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return None


def save_progress(data):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def display_message(idx, total, text, lgbm_pred, llm_pred, labels):
    """Display a message with predictions."""
    print()
    print("=" * 70)
    print(f"Message {idx + 1}/{total}")
    print("=" * 70)
    print()
    print(f"Text: {text}")
    print()
    print(f"LightGBM predicted: {lgbm_pred}")
    print(f"LLM predicted:      {llm_pred}")
    print()
    print("What is the correct label?")
    print()
    for i, label in enumerate(labels, 1):
        marker = ""
        if label == lgbm_pred and label == llm_pred:
            marker = " ✓✓ (both agree)"
        elif label == lgbm_pred:
            marker = " ✓ (LightGBM)"
        elif label == llm_pred:
            marker = " ✓ (LLM)"

        print(f"  {i}. {label}{marker}")
    print()
    print(f"  s. Skip this message")
    print(f"  q. Quit and save progress")
    print()


def get_user_label(labels):
    """Get label from user input."""
    while True:
        choice = input("Your choice (1-4, s, q): ").strip().lower()

        if choice == "q":
            return "quit"
        elif choice == "s":
            return "skip"
        elif choice.isdigit() and 1 <= int(choice) <= len(labels):
            return labels[int(choice) - 1]
        else:
            print(f"Invalid choice. Please enter 1-{len(labels)}, s, or q.")


def analyze_results(messages, human_labels, lgbm_preds, llm_preds, labels):
    """Analyze agreement between human labels and model predictions."""
    print()
    print("=" * 70)
    print("📊 Labeling Complete - Analysis")
    print("=" * 70)
    print()

    # Filter out skipped messages
    labeled_data = [
        (msg, human, lgbm, llm)
        for msg, human, lgbm, llm in zip(messages, human_labels, lgbm_preds, llm_preds)
        if human is not None
    ]

    if not labeled_data:
        print("No messages were labeled.")
        return

    messages_labeled = [d[0] for d in labeled_data]
    human = [d[1] for d in labeled_data]
    lgbm = [d[2] for d in labeled_data]
    llm = [d[3] for d in labeled_data]

    n_labeled = len(human)

    print(f"Total messages labeled: {n_labeled}")
    print()

    # Agreement rates
    lgbm_agreement = sum(1 for h, l in zip(human, lgbm) if h == l)
    llm_agreement = sum(1 for h, l in zip(human, llm) if h == l)
    both_agreement = sum(1 for h, lg, ll in zip(human, lgbm, llm) if h == lg == ll)

    lgbm_rate = lgbm_agreement / n_labeled
    llm_rate = llm_agreement / n_labeled
    both_rate = both_agreement / n_labeled

    print(f"LightGBM Agreement: {lgbm_rate:.1%} ({lgbm_agreement}/{n_labeled})")
    print(f"LLM Agreement:      {llm_rate:.1%} ({llm_agreement}/{n_labeled})")
    print(f"Both Agree:         {both_rate:.1%} ({both_agreement}/{n_labeled})")
    print()

    # Human label distribution
    from collections import Counter
    human_dist = Counter(human)
    print("Your Label Distribution:")
    for label in labels:
        count = human_dist[label]
        pct = 100 * count / n_labeled
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Confusion matrices
    print("=" * 70)
    print("LightGBM Confusion Matrix (Human=rows, LightGBM=cols)")
    print("=" * 70)
    cm_lgbm = confusion_matrix(human, lgbm, labels=labels)
    print(f"{'':12s}", end="")
    for label in labels:
        print(f"{label:12s}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:12s}", end="")
        for j in range(len(labels)):
            print(f"{cm_lgbm[i][j]:12d}", end="")
        print()
    print()

    print("=" * 70)
    print("LLM Confusion Matrix (Human=rows, LLM=cols)")
    print("=" * 70)
    cm_llm = confusion_matrix(human, llm, labels=labels)
    print(f"{'':12s}", end="")
    for label in labels:
        print(f"{label:12s}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:12s}", end="")
        for j in range(len(labels)):
            print(f"{cm_llm[i][j]:12d}", end="")
        print()
    print()

    # Per-class performance
    print("=" * 70)
    print("LightGBM Per-Class Performance (Human as ground truth)")
    print("=" * 70)
    print(classification_report(human, lgbm, labels=labels, target_names=labels, digits=3, zero_division=0))

    print("=" * 70)
    print("LLM Per-Class Performance (Human as ground truth)")
    print("=" * 70)
    print(classification_report(human, llm, labels=labels, target_names=labels, digits=3, zero_division=0))

    # Winner
    print("=" * 70)
    print("🏆 Verdict")
    print("=" * 70)
    print()

    if lgbm_rate > llm_rate + 0.05:
        print(f"✅ LightGBM WINS ({lgbm_rate:.1%} vs {llm_rate:.1%})")
        print(f"   LightGBM is {lgbm_rate - llm_rate:.1%} more accurate than LLM on your messages")
        print(f"   → Use LightGBM for production")
    elif llm_rate > lgbm_rate + 0.05:
        print(f"✅ LLM WINS ({llm_rate:.1%} vs {lgbm_rate:.1%})")
        print(f"   LLM is {llm_rate - lgbm_rate:.1%} more accurate than LightGBM on your messages")
        print(f"   → Need to retrain LightGBM on iMessage-like data")
    else:
        print(f"🤝 TIE ({lgbm_rate:.1%} vs {llm_rate:.1%})")
        print(f"   Both models perform similarly on your messages")
        print(f"   → Either model is acceptable for production")

    print()

    # Error examples
    print("=" * 70)
    print("📝 LightGBM Errors (first 10)")
    print("=" * 70)
    lgbm_errors = [
        (msg, h, lg)
        for msg, h, lg in zip(messages_labeled, human, lgbm)
        if h != lg
    ]
    for i, (msg, human_label, lgbm_label) in enumerate(lgbm_errors[:10]):
        print(f"{i+1}. Text: {msg['text'][:70]}")
        print(f"   You: {human_label:12s}  LightGBM: {lgbm_label:12s}")
        print()

    print("=" * 70)
    print("📝 LLM Errors (first 10)")
    print("=" * 70)
    llm_errors = [
        (msg, h, ll)
        for msg, h, ll in zip(messages_labeled, human, llm)
        if h != ll
    ]
    for i, (msg, human_label, llm_label) in enumerate(llm_errors[:10]):
        print(f"{i+1}. Text: {msg['text'][:70]}")
        print(f"   You: {human_label:12s}  LLM: {llm_label:12s}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Manual labeling tool")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    args = parser.parse_args()

    # Load or create data
    if args.resume and load_progress():
        print("📂 Resuming from saved progress...")
        progress = load_progress()
        messages = progress["messages"]
        lgbm_preds = progress["lgbm_preds"]
        llm_preds = progress["llm_preds"]
        labels = progress["labels"]
        human_labels = progress["human_labels"]
        start_idx = progress["next_idx"]
        print(f"✓ Resuming from message {start_idx + 1}/200")
    else:
        messages, lgbm_preds, llm_preds, labels = load_validation_data()
        human_labels = [None] * len(messages)
        start_idx = 0

    # Labeling loop
    try:
        for idx in range(start_idx, len(messages)):
            msg = messages[idx]
            lgbm_pred = lgbm_preds[idx]
            llm_pred = llm_preds[idx]

            display_message(idx, len(messages), msg["text"], lgbm_pred, llm_pred, labels)

            user_label = get_user_label(labels)

            if user_label == "quit":
                print()
                print("Saving progress...")
                save_progress({
                    "messages": messages,
                    "lgbm_preds": lgbm_preds,
                    "llm_preds": llm_preds,
                    "labels": labels,
                    "human_labels": human_labels,
                    "next_idx": idx,
                })
                print(f"✓ Progress saved. Resume with: --resume flag")
                print(f"   Labeled: {sum(1 for h in human_labels if h is not None)}/{len(messages)}")
                sys.exit(0)
            elif user_label == "skip":
                human_labels[idx] = None
                print("Skipped.")
            else:
                human_labels[idx] = user_label
                print(f"✓ Labeled as: {user_label}")

            # Auto-save every 10 messages
            if (idx + 1) % 10 == 0:
                save_progress({
                    "messages": messages,
                    "lgbm_preds": lgbm_preds,
                    "llm_preds": llm_preds,
                    "labels": labels,
                    "human_labels": human_labels,
                    "next_idx": idx + 1,
                })
                print(f"   [Auto-saved at {idx + 1}/200]")

    except KeyboardInterrupt:
        print()
        print("Interrupted. Saving progress...")
        save_progress({
            "messages": messages,
            "lgbm_preds": lgbm_preds,
            "llm_preds": llm_preds,
            "labels": labels,
            "human_labels": human_labels,
            "next_idx": idx,
        })
        print(f"✓ Progress saved. Resume with: --resume flag")
        sys.exit(0)

    # Analysis
    analyze_results(messages, human_labels, lgbm_preds, llm_preds, labels)

    # Save final results
    final_results = {
        "messages": messages,
        "lgbm_preds": lgbm_preds,
        "llm_preds": llm_preds,
        "human_labels": human_labels,
        "labels": labels,
    }

    output_file = PROJECT_ROOT / "human_labeled_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"💾 Results saved to: {output_file}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == "__main__":
    main()
