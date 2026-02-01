#!/usr/bin/env python3
"""Auto-mine structural patterns from response data.

This script discovers common short responses that could be added as
high-precision structural patterns to the classifier.

Usage:
    uv run python -m scripts.mine_patterns                    # Discover patterns
    uv run python -m scripts.mine_patterns --min-count 30     # Higher threshold
    uv run python -m scripts.mine_patterns --generate         # Generate regex code
    uv run python -m scripts.mine_patterns --apply            # Apply to classifier
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from jarvis.db import get_db
from jarvis.response_classifier import (
    STRUCTURAL_PATTERNS,
    get_response_classifier,
    reset_response_classifier,
)


@dataclass
class PatternCandidate:
    """A candidate pattern discovered from the data."""

    text: str
    count: int
    predicted_class: str
    confidence: float
    already_covered: bool

    @property
    def regex_pattern(self) -> str:
        """Generate regex pattern for this text."""
        # Escape special regex characters
        escaped = re.escape(self.text.lower())
        return f"^({escaped})[\\s!.]*$"


def get_existing_patterns() -> set[str]:
    """Get set of texts already covered by structural patterns."""
    covered = set()

    for response_type, patterns in STRUCTURAL_PATTERNS.items():
        for pattern_str, is_regex in patterns:
            if is_regex:
                # Extract literal alternatives from pattern like "^(yes|yeah|yep)..."
                match = re.search(r'\(([^)]+)\)', pattern_str)
                if match:
                    alternatives = match.group(1).split('|')
                    for alt in alternatives:
                        # Clean up regex syntax
                        clean = re.sub(r"['\"]", "", alt)
                        clean = re.sub(r"\s*\+", "", clean)  # Remove + quantifiers
                        covered.add(clean.lower().strip())

    return covered


def mine_candidates(
    min_count: int = 20,
    max_length: int = 20,
    min_confidence: float = 0.6,
) -> list[PatternCandidate]:
    """Mine pattern candidates from the database.

    Args:
        min_count: Minimum occurrences to consider.
        max_length: Maximum response length.
        min_confidence: Minimum classifier confidence.

    Returns:
        List of PatternCandidate objects.
    """
    reset_response_classifier()
    classifier = get_response_classifier()
    existing = get_existing_patterns()

    db = get_db()
    conn = sqlite3.connect(db.db_path)

    # Get common short responses
    cursor = conn.execute("""
        SELECT response_text, COUNT(*) as cnt
        FROM pairs
        WHERE length(response_text) <= ?
        GROUP BY lower(trim(response_text))
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (max_length, min_count))

    candidates = []

    print(f"Mining patterns (min_count={min_count}, max_length={max_length})...\n")

    for text, count in cursor:
        text = text.strip()
        text_lower = text.lower()

        # Skip very short
        if len(text) < 2:
            continue

        # Check if already covered
        already_covered = text_lower in existing

        # Classify
        result = classifier.classify(text)

        # Skip if low confidence (ambiguous)
        if result.confidence < min_confidence:
            continue

        # Skip if already handled by structural/tapback
        if result.method.startswith("structural") or result.method.startswith("tapback"):
            already_covered = True

        candidates.append(PatternCandidate(
            text=text,
            count=count,
            predicted_class=result.label.value,
            confidence=result.confidence,
            already_covered=already_covered,
        ))

    conn.close()
    return candidates


def display_candidates(candidates: list[PatternCandidate], show_covered: bool = False):
    """Display candidates grouped by class."""
    # Group by class
    by_class: dict[str, list[PatternCandidate]] = defaultdict(list)
    for c in candidates:
        if not c.already_covered or show_covered:
            by_class[c.predicted_class].append(c)

    # Sort classes by total count
    class_totals = {
        cls: sum(c.count for c in cands)
        for cls, cands in by_class.items()
    }

    print("=" * 70)
    print("DISCOVERED PATTERN CANDIDATES")
    print("=" * 70)

    for cls in sorted(class_totals.keys(), key=lambda x: -class_totals[x]):
        cands = by_class[cls]
        if not cands:
            continue

        total = sum(c.count for c in cands)
        print(f"\n{cls} ({len(cands)} patterns, {total} total occurrences):")
        print("-" * 50)

        for c in sorted(cands, key=lambda x: -x.count)[:15]:
            status = "  " if not c.already_covered else "✓ "
            print(f"  {status}{c.count:4}x  '{c.text:20}' (conf: {c.confidence:.2f})")

    # Summary
    new_patterns = [c for c in candidates if not c.already_covered]
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(new_patterns)} new patterns discovered")
    print(f"         {len(candidates) - len(new_patterns)} already covered")
    print("=" * 70)


def generate_regex_code(candidates: list[PatternCandidate]) -> str:
    """Generate Python code for adding patterns to classifier."""
    # Group new patterns by class
    by_class: dict[str, list[PatternCandidate]] = defaultdict(list)
    for c in candidates:
        if not c.already_covered and c.confidence >= 0.7:
            by_class[c.predicted_class].append(c)

    lines = [
        "# Auto-generated patterns from mine_patterns.py",
        "# Review before adding to response_classifier.py",
        "",
    ]

    for cls in sorted(by_class.keys()):
        cands = by_class[cls]
        if not cands:
            continue

        # Sort by count
        cands = sorted(cands, key=lambda x: -x.count)

        # Group similar patterns
        texts = [c.text.lower() for c in cands[:10]]

        lines.append(f"# {cls} - {len(cands)} patterns")
        lines.append(f"# Top candidates: {', '.join(repr(t) for t in texts[:5])}")

        # Generate combined regex
        escaped = [re.escape(t) for t in texts]
        combined = "|".join(escaped)
        lines.append(f'(r"^({combined})[\\s!.]*$", True),')
        lines.append("")

    return "\n".join(lines)


def save_candidates(candidates: list[PatternCandidate], output_path: Path):
    """Save candidates to JSON for review."""
    data = [
        {
            "text": c.text,
            "count": c.count,
            "predicted_class": c.predicted_class,
            "confidence": c.confidence,
            "already_covered": c.already_covered,
            "approve": None,  # For manual review
        }
        for c in candidates
        if not c.already_covered
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(data)} candidates to {output_path}")
    print("Review and set 'approve': true/false for each pattern.")


def main():
    parser = argparse.ArgumentParser(description="Mine structural patterns from data")
    parser.add_argument("--min-count", type=int, default=20,
                        help="Minimum occurrences to consider (default: 20)")
    parser.add_argument("--max-length", type=int, default=20,
                        help="Maximum response length (default: 20)")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum classifier confidence (default: 0.6)")
    parser.add_argument("--show-covered", action="store_true",
                        help="Show already-covered patterns too")
    parser.add_argument("--generate", action="store_true",
                        help="Generate regex code for new patterns")
    parser.add_argument("--save", type=str, metavar="FILE",
                        help="Save candidates to JSON file for review")
    args = parser.parse_args()

    # Mine candidates
    candidates = mine_candidates(
        min_count=args.min_count,
        max_length=args.max_length,
        min_confidence=args.min_confidence,
    )

    # Display
    display_candidates(candidates, show_covered=args.show_covered)

    # Generate code
    if args.generate:
        print("\n" + "=" * 70)
        print("GENERATED REGEX CODE")
        print("=" * 70 + "\n")
        print(generate_regex_code(candidates))

    # Save for review
    if args.save:
        save_candidates(candidates, Path(args.save))


if __name__ == "__main__":
    main()
