#!/usr/bin/env python3
"""Audit trigger training data for quality issues.

Identifies:
1. Duplicate texts
2. Model predictions that disagree with labels
3. Low-confidence predictions
4. Pattern-based red flags (responses labeled as triggers, fragments, etc.)

Usage:
    uv run python -m scripts.audit_trigger_training_data --analyze
    uv run python -m scripts.audit_trigger_training_data --export issues.jsonl
    uv run python -m scripts.audit_trigger_training_data --review 50  # Interactive review
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Data paths
DATA_DIR = Path("data")
TRAINING_FILES = [
    DATA_DIR / "trigger_training_confident.jsonl",
    DATA_DIR / "trigger_training_full.jsonl",
]


@dataclass
class AuditIssue:
    """A potential issue with a training sample."""

    text: str
    label: str
    issue_type: str  # duplicate, prediction_mismatch, low_confidence, pattern_flag
    details: str
    severity: str  # high, medium, low
    suggested_label: str | None = None
    confidence: float = 0.0
    source_file: str = ""
    line_number: int = 0


@dataclass
class AuditResults:
    """Collection of audit results."""

    duplicates: list[AuditIssue] = field(default_factory=list)
    prediction_mismatches: list[AuditIssue] = field(default_factory=list)
    low_confidence: list[AuditIssue] = field(default_factory=list)
    pattern_flags: list[AuditIssue] = field(default_factory=list)

    @property
    def all_issues(self) -> list[AuditIssue]:
        return self.duplicates + self.prediction_mismatches + self.low_confidence + self.pattern_flags

    def summary(self) -> dict:
        return {
            "duplicates": len(self.duplicates),
            "prediction_mismatches": len(self.prediction_mismatches),
            "low_confidence": len(self.low_confidence),
            "pattern_flags": len(self.pattern_flags),
            "total": len(self.all_issues),
        }


# =============================================================================
# Pattern-based red flags
# =============================================================================

# Patterns that suggest a sample is a RESPONSE, not a trigger
RESPONSE_PATTERNS = [
    (re.compile(r"^(i'?m|im) down\b", re.I), "Sounds like accepting an invitation"),
    (re.compile(r"^(i can|i could|i'?ll)\b.*\byou\b", re.I), "Offering to do something (response)"),
    (re.compile(r"^(sure|bet|ok|okay|yeah|yes|yep|nah|nope)\b", re.I), "Simple acknowledgment response"),
    (re.compile(r"^(sounds good|works for me|that works)\b", re.I), "Acceptance response"),
    (re.compile(r"^(thanks|thank you|thx|ty)\b", re.I), "Gratitude response"),
    (re.compile(r"^(lol|lmao|haha|😂|🤣)+\s*$", re.I), "Reaction response"),
]

# Patterns that suggest incomplete/fragment messages
FRAGMENT_PATTERNS = [
    (re.compile(r"^(if|when|but|and|or|like|also)\s+\w{1,15}$", re.I), "Looks like sentence fragment"),
    (re.compile(r"^\w{1,3}$"), "Too short to classify"),
    (re.compile(r"^[^a-zA-Z]*$"), "No alphabetic characters"),
]

# Patterns that suggest wrong label
LABEL_MISMATCH_PATTERNS = {
    "invitation": [
        (re.compile(r"^(i'?m|im) down\b", re.I), "This is ACCEPTING an invitation, not making one"),
        (re.compile(r"^(i can|i'?ll) (pick|get|grab)\b", re.I), "This is an OFFER, consider 'request' or 'statement'"),
        (re.compile(r"\bthen\?$", re.I), "Confirmation question, not new invitation"),
    ],
    "request": [
        (re.compile(r"^(if|when)\s", re.I), "Conditional statement, not request"),
        (re.compile(r"^(also|and)\s", re.I), "Continuation, needs context"),
        (re.compile(r"try to be|should be", re.I), "Advice/suggestion, not direct request"),
    ],
    "good_news": [
        (re.compile(r"^(you|u)\s", re.I), "About someone else, not self"),
        (re.compile(r"\?$"), "Question, not news"),
    ],
    "bad_news": [
        (re.compile(r"^(you|u)\s", re.I), "About someone else, not self"),
        (re.compile(r"\?$"), "Question, not news"),
        (re.compile(r"^(bruh|lmao|lol)\b", re.I), "May be joking/reaction, not actual bad news"),
    ],
    "statement": [
        (re.compile(r"\?$"), "Has question mark, might be question"),
    ],
}


def check_pattern_flags(text: str, label: str) -> list[tuple[str, str]]:
    """Check for pattern-based red flags.

    Returns list of (issue_type, details) tuples.
    """
    flags = []

    # Check if it looks like a response
    for pattern, reason in RESPONSE_PATTERNS:
        if pattern.search(text):
            flags.append(("response_as_trigger", reason))

    # Check for fragments
    for pattern, reason in FRAGMENT_PATTERNS:
        if pattern.match(text):
            flags.append(("fragment", reason))

    # Check label-specific mismatches
    if label in LABEL_MISMATCH_PATTERNS:
        for pattern, reason in LABEL_MISMATCH_PATTERNS[label]:
            if pattern.search(text):
                flags.append(("label_mismatch", f"{label}: {reason}"))

    return flags


# =============================================================================
# Duplicate detection
# =============================================================================


def find_duplicates(samples: list[dict]) -> dict[str, list[int]]:
    """Find duplicate texts and return mapping of text -> list of indices."""
    text_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, sample in enumerate(samples):
        text = sample.get("text", "").strip().lower()
        text_to_indices[text].append(i)

    # Return only duplicates (more than one occurrence)
    return {text: indices for text, indices in text_to_indices.items() if len(indices) > 1}


# =============================================================================
# Model-based checks
# =============================================================================


def load_classifier():
    """Load the trained trigger classifier."""
    try:
        from jarvis.trigger_classifier import get_trigger_classifier

        classifier = get_trigger_classifier()
        if not classifier._svm_loaded:
            print("Warning: No trained SVM model found, using centroid-only")
        return classifier
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}")
        return None


def get_model_predictions(
    classifier, samples: list[dict], batch_size: int = 64
) -> list[tuple[str, float]]:
    """Get model predictions for all samples.

    Returns list of (predicted_label, confidence) tuples.
    """
    predictions = []

    # Map TriggerType to consolidated labels
    TRIGGER_TO_CONSOLIDATED = {
        "invitation": "commitment",
        "request": "request",
        "yn_question": "question",
        "info_question": "question",
        "good_news": "reaction",
        "bad_news": "reaction",
        "reaction": "reaction",
        "statement": "statement",
        "greeting": "greeting",
        "ack": "ack",
        "unknown": "statement",
    }

    for sample in samples:
        text = sample.get("text", "")
        try:
            result = classifier.classify(text)
            # Map to consolidated label
            trigger_type = result.trigger_type.value
            consolidated = TRIGGER_TO_CONSOLIDATED.get(trigger_type, "statement")
            predictions.append((consolidated, result.confidence))
        except Exception:
            predictions.append(("statement", 0.0))

    return predictions


# =============================================================================
# Main audit logic
# =============================================================================


def load_training_data() -> list[dict]:
    """Load all training data from JSONL files."""
    samples = []

    for filepath in TRAINING_FILES:
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample["_source_file"] = str(filepath)
                    sample["_line_number"] = line_num
                    samples.append(sample)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at {filepath}:{line_num}")

    return samples


def run_audit(samples: list[dict], use_model: bool = True) -> AuditResults:
    """Run full audit on training samples."""
    results = AuditResults()

    # 1. Find duplicates
    print("Checking for duplicates...")
    duplicates = find_duplicates(samples)
    for text, indices in duplicates.items():
        # Get the first sample's info
        first_sample = samples[indices[0]]
        results.duplicates.append(
            AuditIssue(
                text=first_sample.get("text", ""),
                label=first_sample.get("label", ""),
                issue_type="duplicate",
                details=f"Found {len(indices)} copies at indices {indices}",
                severity="medium",
                source_file=first_sample.get("_source_file", ""),
                line_number=first_sample.get("_line_number", 0),
            )
        )

    # 2. Check pattern-based flags
    print("Checking pattern-based flags...")
    for sample in samples:
        text = sample.get("text", "")
        label = sample.get("label", "")

        flags = check_pattern_flags(text, label)
        for issue_type, details in flags:
            severity = "high" if issue_type == "label_mismatch" else "medium"
            results.pattern_flags.append(
                AuditIssue(
                    text=text,
                    label=label,
                    issue_type=issue_type,
                    details=details,
                    severity=severity,
                    source_file=sample.get("_source_file", ""),
                    line_number=sample.get("_line_number", 0),
                )
            )

    # 3. Model-based checks
    if use_model:
        print("Loading classifier for prediction checks...")
        classifier = load_classifier()

        if classifier:
            print("Getting model predictions...")
            predictions = get_model_predictions(classifier, samples)

            # Map original labels to consolidated for comparison
            LABEL_TO_CONSOLIDATED = {
                "invitation": "commitment",
                "request": "request",
                "yn_question": "question",
                "info_question": "question",
                "good_news": "reaction",
                "bad_news": "reaction",
                "reaction": "reaction",
                "statement": "statement",
                "greeting": "greeting",
                "ack": "ack",
            }

            for i, (sample, (pred_label, confidence)) in enumerate(zip(samples, predictions)):
                text = sample.get("text", "")
                true_label = sample.get("label", "")
                true_consolidated = LABEL_TO_CONSOLIDATED.get(true_label, true_label)

                # Check for prediction mismatch
                if pred_label != true_consolidated and confidence > 0.5:
                    results.prediction_mismatches.append(
                        AuditIssue(
                            text=text,
                            label=true_label,
                            issue_type="prediction_mismatch",
                            details=f"Model predicts '{pred_label}' with {confidence:.2f} confidence",
                            severity="high" if confidence > 0.7 else "medium",
                            suggested_label=pred_label,
                            confidence=confidence,
                            source_file=sample.get("_source_file", ""),
                            line_number=sample.get("_line_number", 0),
                        )
                    )

                # Check for low confidence (even when matching)
                if confidence < 0.4:
                    results.low_confidence.append(
                        AuditIssue(
                            text=text,
                            label=true_label,
                            issue_type="low_confidence",
                            details=f"Model confidence only {confidence:.2f}",
                            severity="low",
                            confidence=confidence,
                            source_file=sample.get("_source_file", ""),
                            line_number=sample.get("_line_number", 0),
                        )
                    )

    return results


def print_summary(results: AuditResults) -> None:
    """Print audit summary."""
    summary = results.summary()

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Duplicates:            {summary['duplicates']:>5}")
    print(f"Prediction mismatches: {summary['prediction_mismatches']:>5}")
    print(f"Low confidence:        {summary['low_confidence']:>5}")
    print(f"Pattern flags:         {summary['pattern_flags']:>5}")
    print("-" * 60)
    print(f"TOTAL ISSUES:          {summary['total']:>5}")
    print("=" * 60)

    # Break down pattern flags by type
    if results.pattern_flags:
        print("\nPattern flag breakdown:")
        flag_types = Counter(issue.issue_type for issue in results.pattern_flags)
        for flag_type, count in flag_types.most_common():
            print(f"  {flag_type}: {count}")

    # Break down by label
    print("\nIssues by label:")
    label_counts = Counter(issue.label for issue in results.all_issues)
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")


def print_high_priority_issues(results: AuditResults, limit: int = 30) -> None:
    """Print high-priority issues for review."""
    print("\n" + "=" * 60)
    print("HIGH PRIORITY ISSUES (for manual review)")
    print("=" * 60)

    # Combine and sort by severity
    all_issues = results.all_issues
    high_severity = [i for i in all_issues if i.severity == "high"]

    print(f"\nFound {len(high_severity)} high-severity issues. Showing first {limit}:\n")

    for i, issue in enumerate(high_severity[:limit], 1):
        print(f"{i}. [{issue.issue_type}] {issue.label}")
        print(f"   Text: {issue.text[:80]}{'...' if len(issue.text) > 80 else ''}")
        print(f"   Issue: {issue.details}")
        if issue.suggested_label:
            print(f"   Suggested: {issue.suggested_label}")
        print()


def export_issues(results: AuditResults, output_path: Path) -> None:
    """Export issues to JSONL for review."""
    with open(output_path, "w") as f:
        for issue in results.all_issues:
            record = {
                "text": issue.text,
                "current_label": issue.label,
                "issue_type": issue.issue_type,
                "details": issue.details,
                "severity": issue.severity,
                "suggested_label": issue.suggested_label,
                "confidence": issue.confidence,
                "source_file": issue.source_file,
                "line_number": issue.line_number,
                "corrected_label": None,  # To be filled in during review
                "action": None,  # keep, relabel, remove
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nExported {len(results.all_issues)} issues to {output_path}")


def interactive_review(results: AuditResults, limit: int = 50) -> None:
    """Interactive review of issues."""
    # Prioritize high-severity issues
    all_issues = sorted(results.all_issues, key=lambda x: (x.severity != "high", x.severity != "medium"))

    corrections = []
    reviewed = 0

    print("\nInteractive Review Mode")
    print("Commands: [k]eep, [r]elabel, [d]elete, [s]kip, [q]uit")
    print("-" * 60)

    for issue in all_issues[:limit]:
        print(f"\n[{issue.issue_type}] Current label: {issue.label}")
        print(f"Text: {issue.text}")
        print(f"Issue: {issue.details}")
        if issue.suggested_label:
            print(f"Suggested: {issue.suggested_label}")

        while True:
            choice = input("\nAction (k/r/d/s/q): ").strip().lower()

            if choice == "q":
                print(f"\nReviewed {reviewed} samples")
                return corrections
            elif choice == "s":
                break
            elif choice == "k":
                corrections.append({
                    "text": issue.text,
                    "action": "keep",
                    "original_label": issue.label,
                    "new_label": issue.label,
                })
                reviewed += 1
                break
            elif choice == "d":
                corrections.append({
                    "text": issue.text,
                    "action": "delete",
                    "original_label": issue.label,
                    "new_label": None,
                })
                reviewed += 1
                break
            elif choice == "r":
                new_label = input("New label: ").strip()
                if new_label:
                    corrections.append({
                        "text": issue.text,
                        "action": "relabel",
                        "original_label": issue.label,
                        "new_label": new_label,
                    })
                    reviewed += 1
                    break
            else:
                print("Invalid choice. Use k/r/d/s/q")

    print(f"\nReviewed {reviewed} samples")

    # Save corrections
    if corrections:
        output_path = Path("data/trigger_corrections.jsonl")
        with open(output_path, "w") as f:
            for correction in corrections:
                f.write(json.dumps(correction) + "\n")
        print(f"Saved {len(corrections)} corrections to {output_path}")

    return corrections


def main():
    parser = argparse.ArgumentParser(description="Audit trigger training data")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--export", type=str, help="Export issues to JSONL file")
    parser.add_argument("--review", type=int, help="Interactive review of N issues")
    parser.add_argument("--no-model", action="store_true", help="Skip model-based checks")
    parser.add_argument("--high-only", action="store_true", help="Show only high-severity issues")

    args = parser.parse_args()

    if not any([args.analyze, args.export, args.review]):
        args.analyze = True  # Default to analyze

    # Load data
    print("Loading training data...")
    samples = load_training_data()
    print(f"Loaded {len(samples)} samples")

    # Run audit
    results = run_audit(samples, use_model=not args.no_model)

    # Print summary
    print_summary(results)

    if args.analyze or args.high_only:
        print_high_priority_issues(results, limit=50 if args.high_only else 30)

    if args.export:
        export_issues(results, Path(args.export))

    if args.review:
        interactive_review(results, limit=args.review)


if __name__ == "__main__":
    main()
