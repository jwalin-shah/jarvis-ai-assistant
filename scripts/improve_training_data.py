#!/usr/bin/env python3
"""Improve DA classifier training data using validation feedback.

Strategies:
1. Mine hard negatives - examples that look like X but are actually Y
2. Mine from validation errors - learn from mistakes
3. Add user-specific patterns from their message history
4. Bootstrap from high-confidence structural matches

Usage:
    uv run python -m scripts.improve_training_data --analyze     # Analyze errors
    uv run python -m scripts.improve_training_data --mine-hard   # Mine hard negatives
    uv run python -m scripts.improve_training_data --bootstrap   # Bootstrap from structural
    uv run python -m scripts.improve_training_data --rebuild     # Rebuild classifier with new data
"""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Hard Negative Mining
# =============================================================================

# Patterns that LOOK like one class but are actually another
# Format: (pattern, looks_like, actually_is, example)
HARD_NEGATIVE_PATTERNS = [
    # DECLINE look-alikes that are actually other things
    (r"^i (can't|couldn't|cannot) (believe|imagine|wait)", "DECLINE", "REACT_POSITIVE",
     "I can't believe it!"),
    (r"^(no way|no freaking way)[\s!]*$", "DECLINE", "REACT_POSITIVE",
     "No way!"),
    (r"^(i really couldn't|couldn't have done)", "DECLINE", "REACT_POSITIVE",
     "I really couldn't have done it without you"),
    (r"^can't (even|deal|handle)", "DECLINE", "REACT_POSITIVE",
     "Can't even"),
    (r"^(noooo+|nooo+)\s*(way)?[\s!]*$", "DECLINE", "REACT_POSITIVE",
     "Nooooo way!"),

    # AGREE look-alikes that are actually statements
    (r"^(yeah|yes|yea).{20,}", "AGREE", "STATEMENT",
     "Yeah I was thinking the same thing about the weather"),
    (r"^(true|facts).{15,}", "AGREE", "STATEMENT",
     "True, I've been meaning to check that out"),
    (r"^(for sure|definitely).{20,}", "AGREE", "STATEMENT",
     "For sure, that's what I was planning to do anyway"),

    # QUESTION look-alikes that are actually statements
    (r"^you know (what|how|when)\b.{0,10}$", "QUESTION", "STATEMENT",
     "You know what"),
    (r"\?$", "QUESTION", "STATEMENT",  # Rhetorical questions
     "You've seen their data?"),  # Statement disguised as question

    # ACKNOWLEDGE look-alikes that are actually agreement
    (r"^(sounds good|works for me|perfect)[\s!.]*$", "ACKNOWLEDGE", "AGREE",
     "Sounds good!"),

    # REACT_POSITIVE look-alikes that are actually acknowledgments
    (r"^(nice|cool|great|awesome)[\s!.]*$", "REACT_POSITIVE", "ACKNOWLEDGE",
     "Nice"),  # Context-dependent - could be either
]


def analyze_validation_errors():
    """Analyze validation errors to find patterns."""
    validation_file = Path.home() / ".jarvis" / "classifier_validation.json"

    if not validation_file.exists():
        print("No validation file found. Run eval_full_classifier --validate first.")
        return

    with open(validation_file) as f:
        samples = json.load(f)

    # Find errors
    errors = [s for s in samples if s.get("correct") is False]

    if not errors:
        print("No errors found in validation data.")
        return

    print("=" * 70)
    print(f"VALIDATION ERROR ANALYSIS ({len(errors)} errors)")
    print("=" * 70)

    # Group by predicted type
    by_predicted = defaultdict(list)
    for e in errors:
        by_predicted[e["predicted"]].append(e)

    print("\nErrors by predicted type:")
    for pred, items in sorted(by_predicted.items(), key=lambda x: -len(x[1])):
        print(f"\n{pred} ({len(items)} errors):")
        for item in items[:5]:
            print(f"  T: {item['trigger'][:50]}...")
            print(f"  R: {item['response'][:50]}...")
            print(f"     method: {item['method']}, conf: {item['confidence']}")
            print()

    # Analyze error patterns
    print("\n" + "=" * 70)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 70)

    # Check if any hard negative patterns match
    print("\nHard negative pattern matches in errors:")
    for e in errors:
        response = e["response"].lower().strip()
        for pattern, looks_like, actually_is, example in HARD_NEGATIVE_PATTERNS:
            if e["predicted"] == looks_like and re.search(pattern, response, re.IGNORECASE):
                print(f"  [{looks_like} -> {actually_is}] \"{e['response'][:50]}...\"")
                break

    return errors


def mine_hard_negatives_from_data():
    """Mine hard negatives from the user's message history."""
    from jarvis.db import get_db

    print("=" * 70)
    print("MINING HARD NEGATIVES FROM DATA")
    print("=" * 70)

    db = get_db()
    pairs = db.get_all_pairs(min_quality=0.0)

    hard_negatives = defaultdict(list)  # actually_is -> [(text, looks_like)]

    for p in pairs:
        response = p.response_text.lower().strip()

        for pattern, looks_like, actually_is, example in HARD_NEGATIVE_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                # This response looks like `looks_like` but is actually `actually_is`
                hard_negatives[actually_is].append({
                    "text": p.response_text,
                    "looks_like": looks_like,
                    "pattern": pattern,
                })

    print("\nHard negatives found:")
    for actually_is, items in sorted(hard_negatives.items(), key=lambda x: -len(x[1])):
        print(f"\n{actually_is} (looks like something else):")

        # Group by looks_like
        by_looks_like = defaultdict(list)
        for item in items:
            by_looks_like[item["looks_like"]].append(item["text"])

        for looks_like, texts in by_looks_like.items():
            unique_texts = list(set(texts))[:5]
            print(f"  Looks like {looks_like} ({len(texts)} total):")
            for t in unique_texts:
                print(f"    - {t[:60]}")

    return hard_negatives


def bootstrap_from_structural():
    """Bootstrap additional exemplars using high-precision structural patterns."""
    from jarvis.db import get_db
    from jarvis.response_classifier import STRUCTURAL_PATTERNS, _COMPILED_PATTERNS

    print("=" * 70)
    print("BOOTSTRAPPING FROM STRUCTURAL PATTERNS")
    print("=" * 70)

    db = get_db()
    pairs = db.get_all_pairs(min_quality=0.3)  # Higher quality threshold

    # Find responses that match structural patterns
    structural_matches = defaultdict(list)

    for p in pairs:
        response = p.response_text.strip()
        response_lower = response.lower()

        for response_type, patterns in _COMPILED_PATTERNS.items():
            for compiled in patterns:
                if compiled.search(response_lower):
                    structural_matches[response_type.value].append(response)
                    break

    print("\nStructural pattern matches:")
    for da_type, texts in sorted(structural_matches.items(), key=lambda x: -len(x[1])):
        unique = list(set(texts))
        print(f"  {da_type}: {len(unique)} unique matches")

    # Save as new exemplars
    output_dir = Path.home() / ".jarvis" / "da_exemplars"
    output_dir.mkdir(parents=True, exist_ok=True)

    structural_file = output_dir / "structural_mined.json"

    # Deduplicate and limit per class
    final_exemplars = {}
    for da_type, texts in structural_matches.items():
        unique = list(set(texts))
        # Sample up to 500 per class
        if len(unique) > 500:
            import random
            random.seed(42)
            unique = random.sample(unique, 500)
        final_exemplars[da_type] = unique

    with open(structural_file, "w") as f:
        json.dump(final_exemplars, f, indent=2)

    print(f"\nSaved structural-mined exemplars to {structural_file}")

    return final_exemplars


def create_improved_exemplars():
    """Create improved exemplars combining all sources."""
    print("=" * 70)
    print("CREATING IMPROVED EXEMPLARS")
    print("=" * 70)

    output_dir = Path.home() / ".jarvis" / "da_exemplars"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing mined exemplars
    existing = {}
    combined_file = output_dir / "all_mined_exemplars.json"
    if combined_file.exists():
        with open(combined_file) as f:
            existing = json.load(f)
        print(f"Loaded {sum(len(v) for v in existing.values())} existing exemplars")

    # 2. Load structural-mined exemplars
    structural = {}
    structural_file = output_dir / "structural_mined.json"
    if structural_file.exists():
        with open(structural_file) as f:
            structural = json.load(f)
        print(f"Loaded {sum(len(v) for v in structural.values())} structural exemplars")

    # 3. Merge
    improved = defaultdict(set)

    for da_type, texts in existing.items():
        improved[da_type].update(texts)

    for da_type, texts in structural.items():
        improved[da_type].update(texts)

    # Convert to lists
    improved = {k: list(v) for k, v in improved.items()}

    print("\nImproved exemplar counts:")
    for da_type, texts in sorted(improved.items(), key=lambda x: -len(x[1])):
        print(f"  {da_type}: {len(texts)}")

    # Save
    improved_file = output_dir / "improved_exemplars.json"
    with open(improved_file, "w") as f:
        json.dump(improved, f, indent=2)

    print(f"\nSaved improved exemplars to {improved_file}")

    return improved


def rebuild_classifier_with_improved_data():
    """Rebuild the DA classifier with improved training data."""
    print("=" * 70)
    print("REBUILDING CLASSIFIER")
    print("=" * 70)

    # First bootstrap from structural if not done
    structural_file = Path.home() / ".jarvis" / "da_exemplars" / "structural_mined.json"
    if not structural_file.exists():
        print("\nBootstrapping from structural patterns first...")
        bootstrap_from_structural()

    # Create improved exemplars
    create_improved_exemplars()

    # Rebuild classifier
    print("\nRebuilding classifier with proportional sampling...")
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "-m", "scripts.build_da_classifier",
        "--build", "--proportional"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Reset the classifier singleton
    from jarvis.response_classifier import reset_response_classifier
    reset_response_classifier()

    print("\nClassifier rebuilt. Run validation to check accuracy:")
    print("  uv run python -m scripts.eval_full_classifier --score")


def interactive_labeling(n_samples: int = 20):
    """Interactive labeling of uncertain samples to improve training data."""
    from jarvis.db import get_db
    from jarvis.response_classifier import get_response_classifier, ResponseType

    print("=" * 70)
    print("INTERACTIVE LABELING")
    print("=" * 70)
    print("\nThis will show samples where the classifier is uncertain.")
    print("Label them correctly to improve the training data.\n")

    db = get_db()
    pairs = db.get_all_pairs(min_quality=0.3)

    classifier = get_response_classifier()

    # Find uncertain samples (confidence between 0.3 and 0.6)
    uncertain = []
    for p in pairs:
        result = classifier.classify(p.response_text)
        if 0.3 <= result.confidence <= 0.6:
            uncertain.append({
                "trigger": p.trigger_text,
                "response": p.response_text,
                "predicted": result.label.value,
                "confidence": result.confidence,
                "method": result.method,
            })

    print(f"Found {len(uncertain)} uncertain samples")

    # Sample for labeling
    import random
    random.seed(42)
    to_label = random.sample(uncertain, min(n_samples, len(uncertain)))

    # Save for labeling
    output_file = Path.home() / ".jarvis" / "uncertain_samples.json"
    with open(output_file, "w") as f:
        json.dump(to_label, f, indent=2)

    print(f"\nSaved {len(to_label)} samples to {output_file}")
    print("\nInstructions:")
    print("1. Open the JSON file")
    print("2. Add 'correct_label': 'AGREE' (or other DA type) to each sample")
    print("3. Run: uv run python -m scripts.improve_training_data --apply-labels")

    # Show first few
    print("\nSample uncertain predictions:")
    for s in to_label[:5]:
        print(f"\n  T: {s['trigger'][:50]}...")
        print(f"  R: {s['response'][:50]}...")
        print(f"  Predicted: {s['predicted']} ({s['confidence']:.2f})")


def show_improvement_recommendations():
    """Show specific recommendations for improving each problem category."""
    print("=" * 70)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 70)

    recommendations = {
        "AGREE": {
            "problem": "Confused with STATEMENT - long responses starting with 'yeah'",
            "solutions": [
                "Add length constraint: AGREE should be short (< 5 words typically)",
                "Add context: AGREE usually follows INVITATION/REQUEST triggers",
                "Add hard negatives: 'Yeah I was thinking...' -> STATEMENT",
            ],
            "example_fixes": [
                ("Yeah I figured I was like when did she get back", "STATEMENT"),
                ("And there's courts at schools nearby", "STATEMENT"),
            ]
        },
        "DECLINE": {
            "problem": "Confused with emotional expressions",
            "solutions": [
                "Add hard negatives: 'I can't believe it!' -> REACT_POSITIVE",
                "Add context: DECLINE follows commitment requests",
                "Filter: DECLINE shouldn't contain positive emotion markers",
            ],
            "example_fixes": [
                ("Ily dude I really couldn't get through this w out you", "REACT_POSITIVE"),
                ("I ate some more but it's too much dude", "STATEMENT"),
            ]
        },
        "QUESTION": {
            "problem": "Rhetorical questions classified as real questions",
            "solutions": [
                "Add hard negatives: 'You've seen their data?' (rhetorical) -> STATEMENT",
                "Check for question mark + length: very short = real question",
                "Add context: real questions expect information in response",
            ],
            "example_fixes": [
                ("You've seen their data", "STATEMENT"),
                ("Son of a facebook", "REACT_POSITIVE"),  # Not a question at all
            ]
        },
    }

    for da_type, info in recommendations.items():
        print(f"\n{'='*60}")
        print(f"{da_type}")
        print(f"{'='*60}")
        print(f"Problem: {info['problem']}")
        print("\nSolutions:")
        for i, sol in enumerate(info['solutions'], 1):
            print(f"  {i}. {sol}")
        print("\nExample fixes needed:")
        for text, correct in info['example_fixes']:
            print(f"  \"{text[:50]}...\" -> {correct}")

    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)
    print("""
1. Run: uv run python -m scripts.improve_training_data --bootstrap
   (Mine additional exemplars from structural patterns)

2. Run: uv run python -m scripts.improve_training_data --mine-hard
   (Find hard negatives in your data)

3. Run: uv run python -m scripts.improve_training_data --rebuild
   (Rebuild classifier with improved data)

4. Run: uv run python -m scripts.eval_full_classifier --score
   (Check new accuracy)
""")


def main():
    parser = argparse.ArgumentParser(description="Improve DA classifier training data")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze validation errors")
    parser.add_argument("--mine-hard", action="store_true",
                        help="Mine hard negatives from data")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Bootstrap from structural patterns")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild classifier with improved data")
    parser.add_argument("--label", type=int, metavar="N",
                        help="Interactive labeling of N uncertain samples")
    parser.add_argument("--recommend", action="store_true",
                        help="Show improvement recommendations")

    args = parser.parse_args()

    if args.analyze:
        analyze_validation_errors()

    if args.mine_hard:
        mine_hard_negatives_from_data()

    if args.bootstrap:
        bootstrap_from_structural()

    if args.rebuild:
        rebuild_classifier_with_improved_data()

    if args.label:
        interactive_labeling(args.label)

    if args.recommend:
        show_improvement_recommendations()

    if not any([args.analyze, args.mine_hard, args.bootstrap,
                args.rebuild, args.label, args.recommend]):
        # Default: show recommendations
        show_improvement_recommendations()


if __name__ == "__main__":
    main()
