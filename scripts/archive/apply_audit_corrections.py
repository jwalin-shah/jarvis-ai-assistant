#!/usr/bin/env python3
"""Apply batch corrections to trigger training data based on audit findings.

This script applies systematic corrections for common mislabeling patterns
identified in the audit.

Usage:
    uv run python -m scripts.apply_audit_corrections --preview    # Show what would change
    uv run python -m scripts.apply_audit_corrections --apply      # Apply corrections
    uv run python -m scripts.apply_audit_corrections --dedupe     # Remove duplicates only
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Data paths
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "trigger_training_full.jsonl"
OUTPUT_FILE = DATA_DIR / "trigger_training_corrected.jsonl"

# Text field can be "text" or "trigger_text" depending on file format
TEXT_FIELDS = ["text", "trigger_text"]


def get_text(sample: dict) -> str:
    """Get text from sample, handling both field names."""
    for field in TEXT_FIELDS:
        if field in sample and sample[field]:
            return sample[field]
    return ""


@dataclass
class CorrectionRule:
    """A rule for correcting labels."""

    name: str
    current_label: str | None  # None = any label
    pattern: re.Pattern | None  # None = no pattern check
    new_label: str
    reason: str


# =============================================================================
# Correction Rules
# =============================================================================

# These rules are based on audit findings where the model disagreed with
# high confidence and manual review confirmed the model was right.

CORRECTION_RULES = [
    # -------------------------------------------------------------------------
    # Reaction -> Ack: Emphatic acknowledgments mislabeled as reactions
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="emphatic_ack",
        current_label="reaction",
        pattern=re.compile(r"^(yes+|yea+|yeah+|yup|yep|no+|nope|nah+|i know|ik|ikr|right+)[\s!]*$", re.I),
        new_label="ack",
        reason="Emphatic acknowledgment, not reaction",
    ),
    CorrectionRule(
        name="laughter_ack",
        current_label="reaction",
        pattern=re.compile(r"^(lol|lmao|haha+|hehe+|😂|🤣|💀)+[\s!]*$", re.I),
        new_label="ack",
        reason="Laughter is acknowledgment, not reaction",
    ),
    CorrectionRule(
        name="caps_ack",
        current_label="reaction",
        pattern=re.compile(r"^[A-Z]{2,10}$"),  # Short all-caps like "YESSSS"
        new_label="ack",
        reason="Short emphatic response is acknowledgment",
    ),

    # -------------------------------------------------------------------------
    # Bad_news -> Statement: Messages with "unfortunately" aren't always bad news
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="unfortunately_statement",
        current_label="bad_news",
        pattern=re.compile(r"^unfortunately\b", re.I),
        new_label="statement",
        reason="'Unfortunately' prefix doesn't make it bad news",
    ),
    CorrectionRule(
        name="lost_not_bad",
        current_label="bad_news",
        pattern=re.compile(r"\bi lost\b(?!.*(wallet|keys|phone|job|someone|mom|dad|brother|sister|friend))", re.I),
        new_label="statement",
        reason="'I lost' (games, weight, etc.) is not bad news",
    ),
    CorrectionRule(
        name="bruh_lmao",
        current_label="bad_news",
        pattern=re.compile(r"\b(bruh|lmao|lol)\b.*\b(bruh|lmao|lol)\b", re.I),
        new_label="statement",
        reason="Joking tone indicates not serious bad news",
    ),

    # -------------------------------------------------------------------------
    # Good_news -> Statement: Neutral statements mislabeled as good news
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="wont_good_news",
        current_label="good_news",
        pattern=re.compile(r"^i (won't|wont|will not)\b", re.I),
        new_label="statement",
        reason="Negative statement is not good news",
    ),
    CorrectionRule(
        name="think_good_news",
        current_label="good_news",
        pattern=re.compile(r"^i think\b", re.I),
        new_label="statement",
        reason="Opinion/thought is not good news",
    ),
    CorrectionRule(
        name="hypothetical_good",
        current_label="good_news",
        pattern=re.compile(r"\b(would|could|might|maybe)\b", re.I),
        new_label="statement",
        reason="Hypothetical is not news",
    ),

    # -------------------------------------------------------------------------
    # Greeting adjustments
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="how_are_you_greeting",
        current_label="info_question",
        pattern=re.compile(r"^how (are|r) (you|u|ya)\b", re.I),
        new_label="greeting",
        reason="'How are you' is a greeting, not info question",
    ),
    CorrectionRule(
        name="whats_up_greeting",
        current_label="info_question",
        pattern=re.compile(r"^what'?s up\b", re.I),
        new_label="greeting",
        reason="'What's up' is a greeting, not info question",
    ),

    # -------------------------------------------------------------------------
    # Invitation adjustments: Responses to invitations, not invitations
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="im_down_response",
        current_label="invitation",
        pattern=re.compile(r"^(i'?m|im) down\b", re.I),
        new_label="ack",
        reason="This is accepting an invitation, not making one",
    ),
    CorrectionRule(
        name="confirmation_question",
        current_label="invitation",
        pattern=re.compile(r"\bthen\?$", re.I),
        new_label="yn_question",
        reason="Confirmation question, not new invitation",
    ),

    # -------------------------------------------------------------------------
    # Request adjustments: Fragments and advice, not requests
    # -------------------------------------------------------------------------
    CorrectionRule(
        name="conditional_fragment",
        current_label="request",
        pattern=re.compile(r"^(if|when)\s+\w{1,20}$", re.I),
        new_label="statement",
        reason="Conditional fragment, needs context",
    ),
    CorrectionRule(
        name="advice_not_request",
        current_label="request",
        pattern=re.compile(r"\b(try to|should|you should)\b", re.I),
        new_label="statement",
        reason="Advice/suggestion, not direct request",
    ),
]


def apply_rules(text: str, label: str) -> tuple[str, str | None]:
    """Apply correction rules to a sample.

    Returns:
        Tuple of (new_label, reason) or (original_label, None) if no change.
    """
    for rule in CORRECTION_RULES:
        # Check if rule applies to this label
        if rule.current_label is not None and rule.current_label != label:
            continue

        # Check pattern if specified
        if rule.pattern is not None and not rule.pattern.search(text):
            continue

        # Rule matches - apply correction
        return rule.new_label, rule.reason

    return label, None


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate samples, keeping the first occurrence."""
    seen = set()
    unique = []

    for sample in samples:
        text = get_text(sample).strip().lower()
        if text not in seen:
            seen.add(text)
            unique.append(sample)

    return unique


def undersample(samples: list[dict], max_per_class: int | None = None, target_ratio: float = 3.0) -> list[dict]:
    """Undersample majority classes to balance the dataset.

    Args:
        samples: List of samples with 'label' field
        max_per_class: Maximum samples per class (if None, computed from target_ratio)
        target_ratio: Max ratio between largest and smallest class

    Returns:
        Undersampled list of samples
    """
    import random

    # Group by label
    by_label: dict[str, list[dict]] = {}
    for sample in samples:
        label = sample.get("label", "")
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)

    # Find min class size
    min_size = min(len(v) for v in by_label.values())

    # Compute max per class
    if max_per_class is None:
        max_per_class = int(min_size * target_ratio)

    # Undersample
    result = []
    for label, label_samples in by_label.items():
        if len(label_samples) > max_per_class:
            # Random sample without replacement
            random.seed(42)  # Reproducible
            sampled = random.sample(label_samples, max_per_class)
            result.extend(sampled)
        else:
            result.extend(label_samples)

    return result


def load_data(filepath: Path) -> list[dict]:
    """Load JSONL data."""
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_data(samples: list[dict], filepath: Path) -> None:
    """Save samples to JSONL."""
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def preview_corrections(samples: list[dict]) -> dict:
    """Preview what corrections would be made."""
    corrections = Counter()
    examples = {}

    for sample in samples:
        text = get_text(sample)
        label = sample.get("label", "")

        new_label, reason = apply_rules(text, label)
        if reason:
            key = f"{label} -> {new_label}"
            corrections[key] += 1
            if key not in examples:
                examples[key] = []
            if len(examples[key]) < 3:
                examples[key].append((text[:60], reason))

    return {"corrections": corrections, "examples": examples}


def apply_corrections(samples: list[dict]) -> tuple[list[dict], dict]:
    """Apply all corrections and return corrected samples with stats."""
    corrected = []
    stats = Counter()

    for sample in samples:
        text = get_text(sample)
        label = sample.get("label", "")

        new_label, reason = apply_rules(text, label)

        new_sample = sample.copy()
        if reason:
            new_sample["label"] = new_label
            new_sample["original_label"] = label
            new_sample["correction_reason"] = reason
            stats[f"{label} -> {new_label}"] += 1
        corrected.append(new_sample)

    return corrected, stats


def main():
    parser = argparse.ArgumentParser(description="Apply audit corrections to training data")
    parser.add_argument("--preview", action="store_true", help="Preview corrections without applying")
    parser.add_argument("--apply", action="store_true", help="Apply corrections and save")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicates only")
    parser.add_argument("--undersample", type=int, default=None, help="Undersample majority classes to N per class")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Input file")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output file")

    args = parser.parse_args()

    if not any([args.preview, args.apply, args.dedupe]):
        args.preview = True

    # Load data
    print(f"Loading data from {args.input}...")
    samples = load_data(args.input)
    print(f"Loaded {len(samples)} samples")

    # Show original distribution
    print("\nOriginal label distribution:")
    orig_dist = Counter(s.get("label") for s in samples)
    for label, count in orig_dist.most_common():
        print(f"  {label}: {count}")

    if args.dedupe or args.apply:
        # Deduplicate
        print(f"\nDeduplicating...")
        before_count = len(samples)
        samples = deduplicate(samples)
        removed = before_count - len(samples)
        print(f"Removed {removed} duplicates, {len(samples)} remaining")

    if args.preview:
        # Preview corrections
        print("\nPreviewing corrections...")
        preview = preview_corrections(samples)

        print("\nCorrections that would be made:")
        for key, count in preview["corrections"].most_common():
            print(f"\n  {key}: {count} samples")
            for text, reason in preview["examples"][key]:
                print(f"    - {text}...")
                print(f"      Reason: {reason}")

        total = sum(preview["corrections"].values())
        print(f"\nTotal corrections: {total}")

    if args.apply:
        # Apply corrections
        print("\nApplying corrections...")
        samples, stats = apply_corrections(samples)

        print("\nCorrections applied:")
        for key, count in stats.most_common():
            print(f"  {key}: {count}")

        # Show new distribution
        print("\nNew label distribution:")
        new_dist = Counter(s.get("label") for s in samples)
        for label, count in new_dist.most_common():
            diff = count - orig_dist.get(label, 0)
            diff_str = f" ({diff:+d})" if diff != 0 else ""
            print(f"  {label}: {count}{diff_str}")

        # Undersample if requested
        if args.undersample:
            print(f"\nUndersampling to max {args.undersample} per class...")
            before_count = len(samples)
            samples = undersample(samples, max_per_class=args.undersample)
            print(f"Reduced from {before_count} to {len(samples)} samples")

            print("\nUndersampled distribution:")
            under_dist = Counter(s.get("label") for s in samples)
            for label, count in under_dist.most_common():
                print(f"  {label}: {count}")

        # Save
        print(f"\nSaving to {args.output}...")
        save_data(samples, args.output)
        print(f"Saved {len(samples)} samples")


if __name__ == "__main__":
    main()
