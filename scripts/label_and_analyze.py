#!/usr/bin/env python3
"""Label sampled messages and analyze category clarity."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Category definitions with heuristics
CATEGORIES = {
    "needs_answer": {
        "desc": "Expects factual info",
        "heuristics": [
            lambda t: "?" in t
            and any(w in t.lower() for w in ["what", "when", "where", "who", "how", "why"]),
            lambda t: t.lower().startswith(
                ("what", "when", "where", "who", "how", "why", "which")
            ),
        ],
    },
    "needs_confirmation": {
        "desc": "Expects yes/no/acknowledgment",
        "heuristics": [
            lambda t: any(
                p in t.lower()
                for p in ["can you", "could you", "will you", "would you", "please"]
            ),
            lambda t: t.strip().endswith("?")
            and not any(w in t.lower() for w in ["what", "when", "where", "who", "how", "why"]),
            lambda t: t.split()[0].lower() in ["go", "come", "get", "grab", "send", "call"]
            if t.split()
            else False,
        ],
    },
    "needs_empathy": {
        "desc": "Needs emotional support",
        "heuristics": [
            lambda t: any(
                w in t.lower()
                for w in [
                    "sad",
                    "stressed",
                    "died",
                    "failed",
                    "sorry",
                    "hate",
                    "suck",
                    "terrible",
                    "awful",
                ]
            ),
            lambda t: any(
                w in t.lower()
                for w in ["amazing", "excited", "promoted", "congrat", "love", "awesome", "great!"]
            ),
            lambda t: any(e in t for e in ["😭", "😢", "😔", "😊", "🎉", "❤️", "💪"]),
            lambda t: "!" * 2 in t or "!!" in t,  # Multiple exclamation marks
        ],
    },
    "conversational": {"desc": "Casual engagement (default)", "heuristics": []},
}


def label_message(text: str, previous: str) -> tuple[str, str, bool]:
    """Label a message and return (category, reason, is_confident).

    Returns:
        category: The assigned category
        reason: Why this category was chosen
        is_confident: True if clear-cut, False if ambiguous
    """
    text_clean = text.strip()

    # Try heuristics in order
    for category, info in CATEGORIES.items():
        if category == "conversational":
            continue  # Skip default for now
        for heuristic in info["heuristics"]:
            if heuristic(text_clean):
                return category, f"Heuristic match: {info['desc']}", True

    # Manual judgment for edge cases
    # Emotional reactions
    if text_clean.lower() in [
        "omg",
        "wow",
        "nice",
        "cool",
        "awesome",
        "yay",
        "ugh",
        "damn",
        "shit",
    ]:
        if "!" in text_clean or text_clean.lower() in ["yay", "omg"]:
            return "needs_empathy", "Emotional reaction (positive)", True
        else:
            return "conversational", "Casual reaction", True

    # Short confirmations
    if text_clean.lower() in [
        "ok",
        "okay",
        "yeah",
        "yep",
        "sure",
        "alright",
        "got it",
        "thanks",
        "thank you",
    ]:
        return "conversational", "Acknowledgment/confirmation", True

    # Statements about plans
    if any(
        p in text_clean.lower()
        for p in ["i will", "i'll", "gonna", "going to", "let me", "i can", "i'm"]
    ):
        return "conversational", "Statement about plans/status", True

    # Bare questions without wh-words
    if text_clean == "?" or text_clean.lower() in [
        "really?",
        "seriously?",
        "right?",
        "you sure?",
    ]:
        return "needs_answer", "Clarification question", False  # AMBIGUOUS

    # Rhetorical questions
    if text_clean.endswith("?") and any(
        p in text_clean.lower() for p in ["isn't it", "don't you", "wouldn't you"]
    ):
        return "conversational", "Rhetorical question (tag question)", False  # AMBIGUOUS

    # Default
    return "conversational", "Default (no clear category)", False


def main():
    # Load samples
    sample_path = PROJECT_ROOT / "manual_labeling_sample.json"
    data = json.loads(sample_path.read_text())
    messages = data["messages"]

    # Label all messages
    labeled = []
    uncertain = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    heuristic_count = 0

    for i, msg in enumerate(messages):
        text = msg["text"]
        previous = msg["previous"]
        source = msg["source"]

        category, reason, is_confident = label_message(text, previous)
        category_counts[category] += 1
        if is_confident:
            heuristic_count += 1
        else:
            uncertain.append({"index": i, "text": text, "category": category, "reason": reason})

        labeled.append(
            {
                **msg,
                "label": category,
                "reason": reason,
                "confident": is_confident,
            }
        )

    # Report
    print("=" * 80)
    print("MANUAL LABELING RESULTS")
    print("=" * 80)
    print(f"\nTotal messages: {len(messages)}")
    print(f"Confident labels: {heuristic_count} ({heuristic_count / len(messages) * 100:.1f}%)")
    print(
        f"Uncertain labels: {len(uncertain)} ({len(uncertain) / len(messages) * 100:.1f}%)"
    )

    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(messages) * 100
        print(f"  {cat:20s} {count:3d} ({pct:5.1f}%)")

    print(f"\nUncertain cases ({len(uncertain)} total):")
    for item in uncertain[:20]:  # Show first 20
        print(f"  [{item['category']}] {item['text'][:60]}")
        print(f"      Reason: {item['reason']}")
        print()

    # Analyze confusion patterns
    print("\n" + "=" * 80)
    print("CONFUSION ANALYSIS")
    print("=" * 80)

    # Group uncertain by category
    uncertain_by_cat = {}
    for item in uncertain:
        cat = item["category"]
        if cat not in uncertain_by_cat:
            uncertain_by_cat[cat] = []
        uncertain_by_cat[cat].append(item)

    for cat, items in uncertain_by_cat.items():
        print(f"\n{cat} ({len(items)} uncertain):")
        for item in items[:5]:
            print(f"  - {item['text'][:70]}")

    # Save labeled data
    output_path = PROJECT_ROOT / "manual_labeling_results.json"
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(messages),
                    "confident": heuristic_count,
                    "uncertain": len(uncertain),
                    "distribution": category_counts,
                },
                "labeled_messages": labeled,
                "uncertain_cases": uncertain,
            },
            indent=2,
        )
    )

    print(f"\n\nResults saved to: {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    uncertain_pct = len(uncertain) / len(messages) * 100
    if uncertain_pct < 10:
        print(f"✅ Categories are CLEAR ({uncertain_pct:.1f}% uncertain)")
        print("   Safe to use for training with heuristics + LLM for edge cases")
    elif uncertain_pct < 20:
        print(f"⚠️  Some ambiguity ({uncertain_pct:.1f}% uncertain)")
        print("   Consider refining category definitions or merging categories")
    else:
        print(f"❌ Too much confusion ({uncertain_pct:.1f}% uncertain)")
        print("   Categories need clearer boundaries or should be simplified")


if __name__ == "__main__":
    main()
