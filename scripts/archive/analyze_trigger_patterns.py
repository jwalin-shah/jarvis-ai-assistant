#!/usr/bin/env python3
"""Analyze trigger patterns from labeled data to improve regex patterns.

Extracts common patterns for each trigger type to help refine structural patterns.

Usage:
    uv run python -m scripts.analyze_trigger_patterns
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_data(path: Path) -> list[dict]:
    """Load labeled trigger data."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("text") and row.get("label"):
                    data.append(row)
    return data


def analyze_patterns(data: list[dict]) -> dict:
    """Analyze patterns in each label category."""
    by_label = defaultdict(list)
    for row in data:
        by_label[row["label"].lower()].append(row["text"])

    analysis = {}

    for label, texts in by_label.items():
        # Starting words
        start_words = Counter()
        for text in texts:
            words = text.lower().split()
            if words:
                start_words[words[0]] += 1

        # Starting bigrams
        start_bigrams = Counter()
        for text in texts:
            words = text.lower().split()
            if len(words) >= 2:
                start_bigrams[f"{words[0]} {words[1]}"] += 1

        # Ends with question mark
        ends_question = sum(1 for t in texts if t.strip().endswith("?"))

        # Short messages (< 5 words)
        short_msgs = sum(1 for t in texts if len(t.split()) < 5)

        # Contains common patterns
        patterns = {
            "wanna/want to": sum(1 for t in texts if re.search(r"\b(wanna|want to)\b", t, re.I)),
            "can/could you": sum(1 for t in texts if re.search(r"\b(can|could) (you|u)\b", t, re.I)),
            "let's/lets": sum(1 for t in texts if re.search(r"\blet'?s\b", t, re.I)),
            "how/what/when/where/why": sum(1 for t in texts if re.search(r"^(how|what|when|where|why)\b", t, re.I)),
            "are you/do you": sum(1 for t in texts if re.search(r"^(are|do|did|is|was|were|have|has|can|could|will|would) (you|u)\b", t, re.I)),
            "tapback": sum(1 for t in texts if re.search(r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]', t, re.I)),
            "greeting words": sum(1 for t in texts if re.search(r"^(hey|hi|hello|yo|sup|what'?s up|hola|gm|gn)\b", t, re.I)),
            "ack words": sum(1 for t in texts if re.search(r"^(ok|okay|sure|bet|got it|sounds good|cool|alright|yeah|yea|yep|nah|thanks|lol|haha)\b", t, re.I)),
            "emotional": sum(1 for t in texts if re.search(r"\b(omg|wow|damn|fuck|shit|crazy|insane|amazing|awesome|terrible|sad|happy|excited)\b", t, re.I)),
            "i'm/i am": sum(1 for t in texts if re.search(r"^i('m| am)\b", t, re.I)),
            "url": sum(1 for t in texts if re.search(r"https?://", t, re.I)),
        }

        analysis[label] = {
            "count": len(texts),
            "ends_with_question": ends_question,
            "ends_with_question_pct": round(ends_question / len(texts) * 100, 1),
            "short_messages": short_msgs,
            "short_messages_pct": round(short_msgs / len(texts) * 100, 1),
            "top_start_words": start_words.most_common(15),
            "top_start_bigrams": start_bigrams.most_common(10),
            "pattern_counts": patterns,
        }

    return analysis


def print_analysis(analysis: dict):
    """Print analysis results."""
    for label in ["commitment", "question", "reaction", "social", "statement"]:
        if label not in analysis:
            continue

        info = analysis[label]
        print("\n" + "=" * 70)
        print(f"{label.upper()} ({info['count']} examples)")
        print("=" * 70)

        print(f"\nEnds with '?': {info['ends_with_question']} ({info['ends_with_question_pct']}%)")
        print(f"Short (<5 words): {info['short_messages']} ({info['short_messages_pct']}%)")

        print("\nPattern matches:")
        for pattern, count in info["pattern_counts"].items():
            if count > 0:
                pct = round(count / info["count"] * 100, 1)
                print(f"  {pattern}: {count} ({pct}%)")

        print("\nTop starting words:")
        for word, count in info["top_start_words"][:10]:
            pct = round(count / info["count"] * 100, 1)
            print(f"  '{word}': {count} ({pct}%)")

        print("\nTop starting bigrams:")
        for bigram, count in info["top_start_bigrams"][:8]:
            pct = round(count / info["count"] * 100, 1)
            print(f"  '{bigram}': {count} ({pct}%)")


def suggest_patterns(analysis: dict):
    """Suggest regex patterns based on analysis."""
    print("\n" + "=" * 70)
    print("SUGGESTED PATTERN IMPROVEMENTS")
    print("=" * 70)

    # COMMITMENT patterns
    commit = analysis.get("commitment", {})
    print("\n--- COMMITMENT ---")
    print("High-signal patterns to add/verify:")
    patterns = commit.get("pattern_counts", {})
    if patterns.get("wanna/want to", 0) > 10:
        print("  ✓ 'wanna/want to' - strong signal")
    if patterns.get("can/could you", 0) > 10:
        print("  ✓ 'can/could you' - strong signal")
    if patterns.get("let's/lets", 0) > 5:
        print("  ✓ 'let's/lets' - invitation signal")

    # QUESTION patterns
    quest = analysis.get("question", {})
    print("\n--- QUESTION ---")
    print(f"  {quest.get('ends_with_question_pct', 0)}% end with '?' - use as strong signal")
    if quest.get("pattern_counts", {}).get("how/what/when/where/why", 0) > 20:
        print("  ✓ WH-questions are good indicators")

    # SOCIAL patterns
    social = analysis.get("social", {})
    print("\n--- SOCIAL ---")
    if social.get("pattern_counts", {}).get("tapback", 0) > 20:
        print("  ✓ Tapbacks are reliable (already handled)")
    if social.get("pattern_counts", {}).get("greeting words", 0) > 10:
        print("  ✓ Greeting words are reliable")
    if social.get("pattern_counts", {}).get("ack words", 0) > 10:
        print("  ✓ Ack words (ok, sure, bet, etc.) are reliable")

    # REACTION patterns
    react = analysis.get("reaction", {})
    print("\n--- REACTION ---")
    if react.get("pattern_counts", {}).get("emotional", 0) > 20:
        print("  ✓ Emotional words (omg, damn, crazy, etc.) are signals")

    # STATEMENT patterns
    stmt = analysis.get("statement", {})
    print("\n--- STATEMENT ---")
    print("  This is the fallback category - focus on excluding other patterns")
    im_count = stmt.get("pattern_counts", {}).get("i'm/i am", 0)
    if im_count > 50:
        print(f"  'I'm/I am' starts: {im_count} - often statements")


def main():
    path = Path("data/trigger_labeling.jsonl")
    print(f"Loading data from {path}...")
    data = load_data(path)
    print(f"Loaded {len(data)} labeled examples")

    print("\nAnalyzing patterns...")
    analysis = analyze_patterns(data)

    print_analysis(analysis)
    suggest_patterns(analysis)

    # Save analysis
    output_path = Path("results/trigger_pattern_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Counter objects for JSON
    json_analysis = {}
    for label, info in analysis.items():
        json_analysis[label] = {
            **info,
            "top_start_words": [[w, c] for w, c in info["top_start_words"]],
            "top_start_bigrams": [[b, c] for b, c in info["top_start_bigrams"]],
        }

    output_path.write_text(json.dumps(json_analysis, indent=2))
    print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()
