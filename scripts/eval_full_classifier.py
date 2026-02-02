#!/usr/bin/env python3
"""Evaluate the response classifier on the full dataset.

Usage:
    uv run python -m scripts.eval_full_classifier           # Full distribution
    uv run python -m scripts.eval_full_classifier --validate 200  # Sample 200 for manual review
"""

import argparse
import json
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path

from jarvis.db import get_db
from jarvis.response_classifier import get_response_classifier, reset_response_classifier


def validate_sample(n_samples: int = 200):
    """Sample messages for manual validation.

    Outputs a JSON file with samples grouped by predicted class.
    Review and mark 'correct': true/false for each.
    """
    reset_response_classifier()

    db = get_db()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.execute("SELECT trigger_text, response_text FROM pairs")
    all_pairs = cursor.fetchall()
    conn.close()

    print(f"Loaded {len(all_pairs)} pairs")

    # Sample
    random.seed(42)
    sample = random.sample(all_pairs, min(n_samples * 2, len(all_pairs)))

    hybrid = get_response_classifier()

    # Classify and group by prediction
    by_type = {}
    for trigger, response in sample:
        result = hybrid.classify(response)
        label = result.label.value
        if label not in by_type:
            by_type[label] = []
        by_type[label].append(
            {
                "trigger": trigger[:100] if trigger else "",
                "response": response[:100] if response else "",
                "predicted": label,
                "confidence": round(result.confidence, 2),
                "method": result.method,
                "correct": None,  # Fill this in manually
            }
        )

    # Take proportional samples from each type
    samples_per_type = max(10, n_samples // len(by_type))
    final_samples = []
    for label, items in by_type.items():
        final_samples.extend(random.sample(items, min(samples_per_type, len(items))))

    # Shuffle
    random.shuffle(final_samples)
    final_samples = final_samples[:n_samples]

    # Save
    output_file = Path.home() / ".jarvis" / "classifier_validation.json"
    with open(output_file, "w") as f:
        json.dump(final_samples, f, indent=2)

    print(f"\nSaved {len(final_samples)} samples to {output_file}")
    print("\nSamples by predicted type:")
    type_counts = Counter(s["predicted"] for s in final_samples)
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    print("\n" + "=" * 70)
    print("INSTRUCTIONS")
    print("=" * 70)
    print("1. Open the JSON file")
    print("2. For each sample, set 'correct': true or false")
    print("3. Run: uv run python -m scripts.eval_full_classifier --score")
    print("=" * 70)


def score_validation():
    """Score the manual validation results."""
    input_file = Path.home() / ".jarvis" / "classifier_validation.json"

    if not input_file.exists():
        print(f"No validation file found at {input_file}")
        print("Run with --validate first")
        return

    with open(input_file) as f:
        samples = json.load(f)

    # Count correct/incorrect per type
    by_type = {}
    total_correct = 0
    total_judged = 0

    for s in samples:
        label = s["predicted"]
        correct = s.get("correct")

        if correct is None:
            continue

        if label not in by_type:
            by_type[label] = {"correct": 0, "total": 0}

        by_type[label]["total"] += 1
        total_judged += 1

        if correct:
            by_type[label]["correct"] += 1
            total_correct += 1

    if total_judged == 0:
        print("No samples have been judged yet.")
        print("Edit the JSON file and set 'correct': true or false for each sample.")
        return

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n{'Category':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)

    for label in sorted(by_type.keys(), key=lambda x: -by_type[x]["total"]):
        stats = by_type[label]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"{label:<20} {stats['correct']:<10} {stats['total']:<10} {acc:5.1f}% {bar}")

    print("-" * 50)
    overall_acc = total_correct / total_judged * 100
    print(f"{'OVERALL':<20} {total_correct:<10} {total_judged:<10} {overall_acc:5.1f}%")
    print()

    # Show some incorrect examples
    incorrect = [s for s in samples if s.get("correct") is False]
    if incorrect:
        print("=" * 70)
        print(f"SAMPLE ERRORS ({len(incorrect)} total)")
        print("=" * 70)
        for s in incorrect[:10]:
            print(f"\n  Predicted: {s['predicted']} ({s['method']})")
            print(f"  Response:  {s['response'][:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Evaluate response classifier")
    parser.add_argument(
        "--validate", type=int, metavar="N", help="Sample N messages for manual validation"
    )
    parser.add_argument("--score", action="store_true", help="Score the manual validation results")
    args = parser.parse_args()

    if args.validate:
        validate_sample(args.validate)
        return

    if args.score:
        score_validation()
        return

    # Default: full evaluation
    reset_response_classifier()

    db = get_db()
    conn = sqlite3.connect(db.db_path)

    # Load all responses first for batched processing
    print("Loading all pairs...")
    load_start = time.time()
    cursor = conn.execute("SELECT response_text FROM pairs")
    all_responses = [row[0] for row in cursor.fetchall()]
    conn.close()
    load_elapsed = time.time() - load_start
    print(f"Loaded {len(all_responses):,} pairs in {load_elapsed:.1f}s")

    hybrid = get_response_classifier()

    # Force lazy load before timing (initialization overhead)
    print("Initializing classifier...")
    init_start = time.time()
    _ = hybrid.classify("test warmup")
    init_elapsed = time.time() - init_start
    print(f"Classifier initialized in {init_elapsed:.1f}s")

    print("Evaluating with batched embedding...")

    counts = Counter()
    method_counts = Counter()
    total = len(all_responses)

    # Process in batches for efficiency
    BATCH_SIZE = 512  # Can increase to 1024 on 16GB+ systems

    start = time.time()
    processed = 0
    last_report = 0
    REPORT_INTERVAL = 2000  # Report every 2000 items for visible progress

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = all_responses[batch_start:batch_end]

        # Classify batch (embeddings computed together)
        results = hybrid.classify_batch(batch, batch_size=BATCH_SIZE)

        for result in results:
            counts[result.label.value] += 1
            method_counts[result.method] += 1

        processed = batch_end

        # Report progress every REPORT_INTERVAL items
        if processed - last_report >= REPORT_INTERVAL:
            elapsed = time.time() - start
            rate = processed / elapsed
            pct = processed / total * 100
            print(f"  {processed:,}/{total:,} ({pct:.1f}%) - {rate:.0f}/sec")
            last_report = processed

    elapsed = time.time() - start

    print(f"\nCompleted {total} pairs in {elapsed:.1f}s ({total / elapsed:.0f}/sec)")
    print()
    print("=" * 70)
    print("FULL DATASET DISTRIBUTION")
    print("=" * 70)

    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"{label:20} {count:6} ({pct:5.1f}%) {bar}")

    print()
    print("=" * 70)
    print("CLASSIFICATION METHODS")
    print("=" * 70)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:30} {count:6} ({count / total * 100:.1f}%)")

    # Summary stats
    structural = sum(c for m, c in method_counts.items() if "structural" in m)
    filtered = method_counts.get("da_filtered", 0)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total pairs:           {total:,}")
    print(f"  Structural matches:    {structural:,} ({structural / total * 100:.1f}%)")
    print(f"  Filtered to ANSWER:    {filtered:,} ({filtered / total * 100:.1f}%)")
    print(f"  STATEMENT %:           {counts.get('STATEMENT', 0) / total * 100:.1f}%")
    print(f"  ANSWER %:              {counts.get('ANSWER', 0) / total * 100:.1f}%")


if __name__ == "__main__":
    main()
