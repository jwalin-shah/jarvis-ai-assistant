#!/usr/bin/env python3
"""Evaluate DA classifier accuracy on holdout data.

Measures classification accuracy before/after improvements:
- STATEMENT % (target: <50%, was 78%)
- AGREE/DECLINE/DEFER recall (target: >80%)
- QUESTION recall (target: >85%)

Also compares hybrid classifier (structural + DA) vs DA-only.

Usage:
    uv run python -m scripts.eval_da_classifier                   # Full evaluation
    uv run python -m scripts.eval_da_classifier --holdout         # Holdout only
    uv run python -m scripts.eval_da_classifier --compare-hybrid  # Compare hybrid vs DA
    uv run python -m scripts.eval_da_classifier --sample 100      # Sample 100 for manual review
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# Target metrics
TARGETS = {
    "STATEMENT_PCT": 50,       # STATEMENT should be <50% (was 78%)
    "AGREE_RECALL": 80,        # AGREE recall >80%
    "DECLINE_RECALL": 80,      # DECLINE recall >80%
    "DEFER_RECALL": 80,        # DEFER recall >80%
    "QUESTION_RECALL": 85,     # QUESTION recall >85%
}


def evaluate_da_classifier(
    use_holdout: bool = True,
    limit: int = 5000,
):
    """Evaluate the DA classifier on pairs.

    Args:
        use_holdout: If True, use holdout pairs only.
        limit: Maximum pairs to evaluate.

    Returns:
        Dict with evaluation metrics.
    """
    from jarvis.db import get_db

    print("=" * 70)
    print("DA CLASSIFIER EVALUATION")
    print("=" * 70)

    db = get_db()
    db.init_schema()

    # Get pairs
    if use_holdout:
        pairs = db.get_holdout_pairs(min_quality=0.0)
        print(f"\nUsing holdout pairs: {len(pairs)}")
    else:
        pairs = db.get_training_pairs(min_quality=0.0)
        print(f"\nUsing training pairs: {len(pairs)}")

    if limit:
        pairs = pairs[:limit]
        print(f"Limited to: {len(pairs)}")

    # Load classifier
    try:
        from scripts.build_da_classifier import DialogueActClassifier
        response_clf = DialogueActClassifier("response")
        trigger_clf = DialogueActClassifier("trigger")
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        print("Run 'uv run python -m scripts.build_da_classifier --build' first")
        return None

    # Classify responses
    print("\nClassifying responses...")
    responses = [p.response_text for p in pairs]
    response_results = response_clf.classify_batch(responses)

    # Response distribution
    print("\n" + "-" * 50)
    print("RESPONSE TYPE DISTRIBUTION")
    print("-" * 50)

    response_counts = Counter(r.label for r in response_results)
    total = len(response_results)

    for label, count in response_counts.most_common():
        pct = count / total * 100
        avg_conf = sum(r.confidence for r in response_results if r.label == label) / count
        status = ""
        if label == "STATEMENT":
            result = "[PASS]" if pct < TARGETS["STATEMENT_PCT"] else "[FAIL]"
            status = f" {result} target <{TARGETS['STATEMENT_PCT']}%"
        print(f"  {label:20} {count:5} ({pct:5.1f}%){status}  avg_conf: {avg_conf:.2f}")

    statement_pct = response_counts.get("STATEMENT", 0) / total * 100

    # Trigger distribution
    print("\n" + "-" * 50)
    print("TRIGGER TYPE DISTRIBUTION")
    print("-" * 50)

    triggers = [p.trigger_text for p in pairs]
    trigger_results = trigger_clf.classify_batch(triggers)

    trigger_counts = Counter(r.label for r in trigger_results)
    for label, count in trigger_counts.most_common():
        pct = count / total * 100
        print(f"  {label:20} {count:5} ({pct:5.1f}%)")

    # Cross-tabulation
    print("\n" + "-" * 50)
    print("TRIGGER -> RESPONSE CROSS-TAB")
    print("-" * 50)

    cross_tab: dict[str, Counter] = {}
    for tr, rr in zip(trigger_results, response_results):
        if tr.label not in cross_tab:
            cross_tab[tr.label] = Counter()
        cross_tab[tr.label][rr.label] += 1

    for trigger_type in sorted(cross_tab.keys()):
        print(f"\n  {trigger_type}:")
        for resp_type, count in cross_tab[trigger_type].most_common(5):
            pct = count / sum(cross_tab[trigger_type].values()) * 100
            print(f"    -> {resp_type:20} {count:4} ({pct:5.1f}%)")

    return {
        "total_pairs": len(pairs),
        "statement_pct": statement_pct,
        "response_distribution": dict(response_counts),
        "trigger_distribution": dict(trigger_counts),
    }


def compare_hybrid_vs_da(limit: int = 1000):
    """Compare hybrid classifier vs DA-only on responses.

    The hybrid classifier uses structural patterns first, then
    falls back to DA classifier.

    Args:
        limit: Maximum pairs to compare.
    """
    from jarvis.db import get_db
    from jarvis.response_classifier import get_response_classifier

    print("=" * 70)
    print("HYBRID vs DA-ONLY COMPARISON")
    print("=" * 70)

    db = get_db()
    db.init_schema()

    pairs = db.get_holdout_pairs(min_quality=0.0)[:limit]
    print(f"\nComparing on {len(pairs)} holdout pairs")

    # Load classifiers
    try:
        from scripts.build_da_classifier import DialogueActClassifier
        da_clf = DialogueActClassifier("response")
    except Exception as e:
        print(f"Failed to load DA classifier: {e}")
        return

    hybrid_clf = get_response_classifier()

    # Classify with both
    da_results = []
    hybrid_results = []
    agreements = 0
    structural_used = 0

    for p in pairs:
        da_result = da_clf.classify(p.response_text)
        hybrid_result = hybrid_clf.classify(p.response_text)

        da_results.append(da_result.label)
        hybrid_results.append(hybrid_result.label.value)

        if da_result.label == hybrid_result.label.value:
            agreements += 1
        if hybrid_result.structural_match:
            structural_used += 1

    # Compare distributions
    print("\n" + "-" * 50)
    print("DA-ONLY DISTRIBUTION")
    print("-" * 50)
    da_counts = Counter(da_results)
    for label, count in da_counts.most_common():
        pct = count / len(pairs) * 100
        print(f"  {label:20} {count:5} ({pct:5.1f}%)")

    print("\n" + "-" * 50)
    print("HYBRID DISTRIBUTION")
    print("-" * 50)
    hybrid_counts = Counter(hybrid_results)
    for label, count in hybrid_counts.most_common():
        pct = count / len(pairs) * 100
        print(f"  {label:20} {count:5} ({pct:5.1f}%)")

    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"  Agreement rate: {agreements}/{len(pairs)} ({agreements/len(pairs)*100:.1f}%)")
    struct_pct = structural_used / len(pairs) * 100
    print(f"  Structural matches: {structural_used}/{len(pairs)} ({struct_pct:.1f}%)")

    da_statement_pct = da_counts.get("STATEMENT", 0) / len(pairs) * 100
    hybrid_statement_pct = hybrid_counts.get("STATEMENT", 0) / len(pairs) * 100
    print(f"\n  STATEMENT % (DA-only):   {da_statement_pct:.1f}%")
    print(f"  STATEMENT % (Hybrid):    {hybrid_statement_pct:.1f}%")
    improvement = da_statement_pct - hybrid_statement_pct
    print(f"  Improvement:             {improvement:.1f} percentage points")

    # Show disagreements
    print("\n" + "-" * 50)
    print("SAMPLE DISAGREEMENTS")
    print("-" * 50)
    disagreements = [
        (p.response_text, da, hybrid)
        for p, da, hybrid in zip(pairs, da_results, hybrid_results)
        if da != hybrid
    ]
    for text, da_label, hybrid_label in disagreements[:10]:
        print(f"  Text: {text[:50]}...")
        print(f"    DA: {da_label} -> Hybrid: {hybrid_label}")
        print()


def sample_for_manual_review(n: int = 100, output_file: Path | None = None):
    """Sample pairs for manual accuracy review.

    Args:
        n: Number of pairs to sample.
        output_file: File to save samples for review.
    """
    from jarvis.db import get_db
    from jarvis.response_classifier import get_response_classifier

    print("=" * 70)
    print(f"SAMPLING {n} PAIRS FOR MANUAL REVIEW")
    print("=" * 70)

    db = get_db()
    db.init_schema()

    pairs = db.get_holdout_pairs(min_quality=0.0)
    if len(pairs) > n:
        pairs = random.sample(pairs, n)

    hybrid_clf = get_response_classifier()

    samples = []
    for p in pairs:
        result = hybrid_clf.classify(p.response_text)
        samples.append({
            "trigger": p.trigger_text,
            "response": p.response_text,
            "predicted_type": result.label.value,
            "confidence": round(result.confidence, 2),
            "method": result.method,
            "correct": None,  # To be filled by reviewer
        })

    # Group by predicted type
    by_type: dict[str, list] = {}
    for s in samples:
        t = s["predicted_type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(s)

    print("\nSamples by predicted type:")
    for t, ss in sorted(by_type.items()):
        print(f"  {t}: {len(ss)}")

    # Save for review
    if output_file is None:
        output_file = Path.home() / ".jarvis" / "da_review_samples.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nSaved {len(samples)} samples to {output_file}")
    print("Review and fill in 'correct' field (true/false) for each sample.")


def score_manual_review(review_file: Path | None = None):
    """Score results of manual review.

    Args:
        review_file: File with reviewed samples.
    """
    if review_file is None:
        review_file = Path.home() / ".jarvis" / "da_review_samples.json"

    if not review_file.exists():
        print(f"No review file found at {review_file}")
        return

    with open(review_file) as f:
        samples = json.load(f)

    # Filter to reviewed samples
    reviewed = [s for s in samples if s.get("correct") is not None]
    if not reviewed:
        print("No samples have been reviewed yet.")
        print("Edit the JSON file and set 'correct' to true or false for each sample.")
        return

    print("=" * 70)
    print("MANUAL REVIEW RESULTS")
    print("=" * 70)

    # Overall accuracy
    correct = sum(1 for s in reviewed if s["correct"])
    total = len(reviewed)
    print(f"\nOverall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Accuracy by type
    print("\nAccuracy by predicted type:")
    by_type: dict[str, list] = {}
    for s in reviewed:
        t = s["predicted_type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(s["correct"])

    for t in sorted(by_type.keys()):
        type_correct = sum(by_type[t])
        type_total = len(by_type[t])
        pct = type_correct / type_total * 100 if type_total > 0 else 0
        status = "[PASS]" if pct >= 80 else "[FAIL]"
        print(f"  {t:20} {type_correct:3}/{type_total:3} ({pct:5.1f}%) {status}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DA classifier")
    parser.add_argument("--holdout", action="store_true", help="Use holdout pairs only")
    parser.add_argument("--compare-hybrid", action="store_true", help="Compare hybrid vs DA-only")
    parser.add_argument("--sample", type=int, help="Sample N pairs for manual review")
    parser.add_argument("--score-review", action="store_true", help="Score manual review results")
    parser.add_argument("--limit", type=int, default=5000, help="Limit pairs to evaluate")
    parser.add_argument("--output", type=Path, help="Output file for samples")

    args = parser.parse_args()

    if args.sample:
        sample_for_manual_review(args.sample, args.output)
    elif args.score_review:
        score_manual_review(args.output)
    elif args.compare_hybrid:
        compare_hybrid_vs_da(args.limit)
    else:
        evaluate_da_classifier(use_holdout=args.holdout, limit=args.limit)


if __name__ == "__main__":
    main()
