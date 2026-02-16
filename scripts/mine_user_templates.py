#!/usr/bin/env python3
"""Adaptive Template Mining System - Mines user patterns while avoiding ambiguity.

This system:
1. Mines message→reply pairs from user's chat.db
2. Filters out ambiguous/context-dependent patterns
3. Keeps only clear, high-confidence patterns
4. Generates personalized templates
5. Stays within 8GB RAM constraint

Usage:
    uv run python scripts/mine_user_templates.py --min-frequency 5 --output templates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Patterns that are too ambiguous (context-dependent meanings)
AMBIGUOUS_SINGLE_WORDS = {"mwah", "oof", "hmm", "uh", "um", "er", "wow", "oh", "ah", "eh", "huh"}

# Patterns that require context to understand
CONTEXT_DEPENDENT = {
    "nice",
    "cool",
    "ok",
    "okay",
    "sure",
    "right",
    "really",
    "seriously",
    "literally",
    "actually",
}

# Clear, unambiguous patterns (always mean the same thing)
CLEAR_PATTERNS = {
    "lol",
    "lmao",
    "haha",
    "hehe",  # Always laughter
    "thanks",
    "thank you",
    "ty",
    "thx",  # Always gratitude
    "yes",
    "yeah",
    "yea",
    "yep",
    "yup",  # Always affirmation
    "no",
    "nope",
    "nah",  # Always negation
    "ok",
    "kk",
    "okie",  # Simple acknowledgment (not emotional)
    "bye",
    "cya",
    "ttyl",
    "gn",  # Always farewell
    "omw",
    "on my way",
    "coming",  # Always ETA
    "idk",
    "i don't know",  # Always uncertainty
}


@dataclass
class UserPattern:
    """A mined pattern from user's messages."""

    incoming: str  # What the other person said
    response: str  # What user replied
    frequency: int  # How many times seen
    unique_contacts: int  # How many different people
    confidence: float  # 0-1, based on consistency


class AmbiguityFilter:
    """Filter out ambiguous or context-dependent patterns."""

    @staticmethod
    def is_ambiguous(text: str) -> bool:
        """Check if text meaning depends on context."""
        text_lower = text.lower().strip()

        # Single ambiguous words
        if text_lower in AMBIGUOUS_SINGLE_WORDS:
            return True

        # Context-dependent words without modifiers
        words = text_lower.split()
        if len(words) == 1 and text_lower in CONTEXT_DEPENDENT:
            return True

        # Emoji-only (very ambiguous)
        if all(ord(c) > 127 for c in text if c.isalnum()):
            return True

        return False

    @staticmethod
    def is_too_specific(text: str) -> bool:
        """Check if pattern is too specific to user's life."""
        text_lower = text.lower()

        # Names (basic check)
        if any(name in text_lower for name in ["jwalin", "mahi", "kishan"]):
            return True

        # Specific references
        specific_refs = [
            r"\d+ million",  # "287 million"
            r"\b\d{4}\b",  # Years like "2024"
            r"khris|lebron|mahomes",  # Specific athletes
            r"india|fremont|austin",  # Specific places (unless user lives there)
        ]

        for pattern in specific_refs:
            if re.search(pattern, text_lower):
                return True

        return False

    @staticmethod
    def has_consistent_response(incoming: str, responses: list[str]) -> tuple[bool, str]:
        """Check if user responds consistently to this pattern."""
        if len(responses) < 3:
            return False, ""

        # Normalize responses
        normalized = [r.lower().strip() for r in responses]

        # Check if 70%+ are the same
        counter = Counter(normalized)
        most_common, count = counter.most_common(1)[0]

        consistency = count / len(responses)

        # Require 70% consistency
        if consistency >= 0.7:
            return True, most_common

        return False, ""


class UserTemplateMiner:
    """Mines templates from user's message history."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path.home() / "Library" / "Messages" / "chat.db"
        self.ambiguity_filter = AmbiguityFilter()

    def mine_patterns(
        self, min_frequency: int = 5, min_contacts: int = 2, max_patterns: int = 50
    ) -> list[UserPattern]:
        """Mine patterns from user's chat history."""

        if not self.db_path.exists():
            print(f"❌ chat.db not found at {self.db_path}")
            return []

        print(f"🔍 Mining patterns from {self.db_path}...")

        # Connect to database
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get message-reply pairs
        print("  Extracting message-reply pairs...")
        pairs = self._extract_pairs(cursor)
        print(f"  Found {len(pairs):,} pairs")

        # Group by incoming message
        print("  Grouping by incoming message...")
        grouped = self._group_by_incoming(pairs)
        print(f"  {len(grouped)} unique incoming messages")

        # Filter and score
        print("  Filtering ambiguous patterns...")
        patterns = self._filter_patterns(grouped, min_frequency, min_contacts)
        print(f"  {len(patterns)} patterns passed filters")

        # Sort by confidence and take top N
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        top_patterns = patterns[:max_patterns]

        conn.close()

        print(f"✅ Mined {len(top_patterns)} high-quality patterns")
        return top_patterns

    def _extract_pairs(self, cursor) -> list[tuple[str, str, str]]:
        """Extract incoming message + reply + chat_id."""
        cursor.execute("""
            SELECT
                m1.text,
                m2.text,
                cj.chat_id
            FROM message m1
            JOIN message m2 ON m2.ROWID = m1.ROWID + 1
            JOIN chat_message_join cj ON cj.message_id = m1.ROWID
            WHERE m1.is_from_me = 0
              AND m2.is_from_me = 1
              AND m1.text IS NOT NULL
              AND m2.text IS NOT NULL
              AND length(m1.text) <= 40
              AND length(m2.text) <= 40
              AND m1.text NOT LIKE '%http%'
              AND m2.text NOT LIKE '%http%'
            ORDER BY m1.date
        """)

        return [(row[0].strip(), row[1].strip(), row[2]) for row in cursor.fetchall()]

    def _group_by_incoming(
        self, pairs: list[tuple[str, str, str]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Group replies by normalized incoming message."""
        grouped = defaultdict(list)

        for incoming, reply, chat_id in pairs:
            # Normalize incoming message
            normalized = self._normalize(incoming)

            if normalized:  # Skip empty
                grouped[normalized].append((reply, chat_id))

        return grouped

    def _normalize(self, text: str) -> str:
        """Normalize text for grouping."""
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation at end (but keep internal)
        text = text.rstrip(".!?")

        return text

    def _filter_patterns(
        self, grouped: dict[str, list[tuple[str, str]]], min_frequency: int, min_contacts: int
    ) -> list[UserPattern]:
        """Filter patterns by quality criteria."""
        patterns = []

        for incoming, replies_chats in grouped.items():
            # Check frequency
            if len(replies_chats) < min_frequency:
                continue

            # Check for ambiguity
            if self.ambiguity_filter.is_ambiguous(incoming):
                continue

            # Check for too-specific content
            if self.ambiguity_filter.is_too_specific(incoming):
                continue

            # Extract replies and chats
            replies = [r for r, _ in replies_chats]
            chats = set(c for _, c in replies_chats)

            # Check multi-contact (avoids inside jokes)
            if len(chats) < min_contacts:
                continue

            # Check response consistency
            is_consistent, common_response = self.ambiguity_filter.has_consistent_response(
                incoming, replies
            )

            if not is_consistent:
                continue

            # Calculate confidence
            response_counts = Counter(r.lower().strip() for r in replies)
            most_common_count = response_counts.most_common(1)[0][1]
            confidence = most_common_count / len(replies)

            pattern = UserPattern(
                incoming=incoming,
                response=common_response,
                frequency=len(replies_chats),
                unique_contacts=len(chats),
                confidence=confidence,
            )

            patterns.append(pattern)

        return patterns


def generate_templates(patterns: list[UserPattern]) -> list[dict]:
    """Convert mined patterns to ResponseTemplate format."""
    templates = []

    for i, pattern in enumerate(patterns):
        # Generate variations of the incoming pattern
        variations = generate_variations(pattern.incoming)

        template = {
            "name": f"user_pattern_{i}",
            "patterns": variations,
            "response": pattern.response,
            "frequency": pattern.frequency,
            "confidence": round(pattern.confidence, 2),
            "contacts": pattern.unique_contacts,
        }

        templates.append(template)

    return templates


def generate_variations(pattern: str) -> list[str]:
    """Generate common variations of a pattern."""
    variations = {pattern}

    # Add common suffixes/prefixes
    if pattern in ["ok", "okay"]:
        variations.update(["ok", "okay", "okie", "k", "kk"])
    elif pattern == "yes":
        variations.update(["yes", "yeah", "yea", "yep", "yup", "ye"])
    elif pattern == "no":
        variations.update(["no", "nope", "nah"])
    elif pattern in ["lol", "haha"]:
        variations.update(["lol", "lmao", "haha", "hehe", "hahaha", "rofl"])
    elif pattern in ["thanks", "thank you"]:
        variations.update(["thanks", "thank you", "ty", "thx", "tysm"])
    elif pattern == "bye":
        variations.update(["bye", "cya", "see ya", "ttyl", "gn", "goodnight"])
    else:
        # For other patterns, just add punctuation variations
        variations.add(pattern + "!")
        variations.add(pattern + "?")

    return list(variations)


def main():
    parser = argparse.ArgumentParser(description="Mine user-specific templates")
    parser.add_argument(
        "--min-frequency", type=int, default=5, help="Minimum times pattern must appear"
    )
    parser.add_argument(
        "--min-contacts", type=int, default=2, help="Minimum different contacts for pattern"
    )
    parser.add_argument("--max-patterns", type=int, default=50, help="Maximum patterns to generate")
    parser.add_argument("--output", type=str, default="user_templates.json", help="Output file")
    parser.add_argument(
        "--db", type=str, default=None, help="Path to chat.db (default: ~/Library/Messages/chat.db)"
    )

    args = parser.parse_args()

    # Mine patterns
    db_path = Path(args.db) if args.db else None
    miner = UserTemplateMiner(db_path)

    patterns = miner.mine_patterns(
        min_frequency=args.min_frequency,
        min_contacts=args.min_contacts,
        max_patterns=args.max_patterns,
    )

    if not patterns:
        print("\n❌ No patterns found. Try lowering --min-frequency")
        return

    # Generate templates
    templates = generate_templates(patterns)

    # Save
    output_path = Path(args.output)
    output_path.write_text(json.dumps(templates, indent=2))

    print(f"\n💾 Saved {len(templates)} templates to {output_path}")

    # Show examples
    print("\n📋 Top 10 templates:")
    print("-" * 80)
    for t in templates[:10]:
        print(f"\n{t['name']}:")
        print(f"  Patterns: {', '.join(t['patterns'][:3])}")
        print(f"  Response: {t['response']}")
        print(f"  Confidence: {t['confidence'] * 100:.0f}% ({t['frequency']} times)")


if __name__ == "__main__":
    main()
