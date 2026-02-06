#!/usr/bin/env python3
"""Test tiered structural + NLI approach.

Insight from hypothesis testing:
- NLI hypotheses that catch musings over-classify real questions
- NLI hypotheses that catch questions miss musings

Solution: Structural rules handle clear cases, NLI handles ambiguous ones.

Tier 1: Structural patterns (high confidence)
- "?" at end -> likely question
- "can you", "wanna", imperatives -> likely request
- "!!" -> likely reactive
- Short acks -> acknowledgeable

Tier 2: NLI for ambiguous cases
- "I wonder if..." (no ?)
- Opinions that might be questions
- Statements that might need reaction

Usage:
    uv run python scripts/benchmark_tiered_approach.py
"""

from __future__ import annotations

import re

import numpy as np

TEXTS = [
    # Musings - no ? but could be confused with questions
    ("I wonder if bulls make another trade", "ACKNOWLEDGEABLE"),
    ("Kind of curious what they will offer", "ACKNOWLEDGEABLE"),
    ("I wonder what happened", "ACKNOWLEDGEABLE"),
    ("Curious how they'll handle it", "ACKNOWLEDGEABLE"),
    ("Wonder if anyone else noticed", "ACKNOWLEDGEABLE"),
    # Opinions with "no way" - could be confused with reactive
    ("No way Dallas could've gotten lottery picks", "ACKNOWLEDGEABLE"),
    ("No way that's real", "ACKNOWLEDGEABLE"),
    ("No chance they win tonight", "ACKNOWLEDGEABLE"),
    ("I don't think it's gonna happen", "ACKNOWLEDGEABLE"),
    ("Doubt they'll actually do it", "ACKNOWLEDGEABLE"),
    # Rhetorical questions - look like questions but aren't
    ("Why do dads text like that", "ACKNOWLEDGEABLE"),
    ("How does that even work", "ACKNOWLEDGEABLE"),
    ("Who even says that anymore", "ACKNOWLEDGEABLE"),
    # Actual questions WITH question marks
    ("What time is the game?", "ANSWERABLE"),
    ("Where are you?", "ANSWERABLE"),
    ("Did you get my text?", "ANSWERABLE"),
    ("What happened at the meeting?", "ANSWERABLE"),
    # Actual questions WITHOUT question marks (informal)
    ("What time is the game", "ANSWERABLE"),
    ("Where are you", "ANSWERABLE"),
    ("Did you get my text", "ANSWERABLE"),
    # Requests
    ("Can you pick me up?", "ACTIONABLE"),
    ("Can you pick me up", "ACTIONABLE"),
    ("Wanna grab lunch?", "ACTIONABLE"),
    ("Wanna grab lunch", "ACTIONABLE"),
    ("Let me know when you're free", "ACTIONABLE"),
    ("Text me when you get there", "ACTIONABLE"),
    # Reactive
    ("Omg I got the job!!", "REACTIVE"),
    ("That's so sad", "REACTIVE"),
    ("This is amazing!!", "REACTIVE"),
    ("I can't believe it!!!", "REACTIVE"),
    # Statements
    ("I went to the store", "ACKNOWLEDGEABLE"),
    ("The game starts at 7", "ACKNOWLEDGEABLE"),
    ("I'm heading out now", "ACKNOWLEDGEABLE"),
    ("Traffic was bad today", "ACKNOWLEDGEABLE"),
]


def classify_structural(text: str) -> tuple[str, float] | None:
    """Structural rules for clear cases. Returns (category, confidence) or None."""
    text_lower = text.lower().strip()
    words = text_lower.split()

    # === Tier 1a: High-confidence patterns ===

    # Request patterns (ACTIONABLE)
    if re.match(r"^(can|could|would|will)\s+(you|u|ya)\s+", text_lower):
        return ("ACTIONABLE", 0.95)
    if re.match(r"^(wanna|want to|down to)\s+", text_lower):
        return ("ACTIONABLE", 0.95)
    if re.match(r"^(let'?s|lets)\s+", text_lower):
        return ("ACTIONABLE", 0.90)
    if re.match(r"^(let me know|text me|call me|send me)\b", text_lower):
        return ("ACTIONABLE", 0.90)

    # Imperative verbs at start
    imperatives = {
        "send",
        "give",
        "bring",
        "take",
        "get",
        "grab",
        "pick",
        "call",
        "text",
        "help",
        "come",
        "tell",
        "show",
    }
    if words and words[0] in imperatives:
        return ("ACTIONABLE", 0.85)

    # Direct WH-questions with ? (ANSWERABLE)
    if text.rstrip().endswith("?"):
        wh_question = re.match(r"^(what|where|when|who|how|which|why)\s", text_lower)
        yesno_question = re.match(
            r"^(is|are|was|were|do|does|did|can|could|will|would|have|has)\s", text_lower
        )
        if wh_question or yesno_question:
            return ("ANSWERABLE", 0.90)

    # Strong emotional markers (REACTIVE)
    if text.count("!") >= 2:
        return ("REACTIVE", 0.85)
    if re.match(r"^(omg|oh my god|yay|congrats|congratulations|i got|we got|i'm so)\b", text_lower):
        return ("REACTIVE", 0.85)
    if re.search(r"\b(so happy|so sad|so excited|can'?t believe)\b", text_lower):
        return ("REACTIVE", 0.80)

    # Short acknowledgment words (ACKNOWLEDGEABLE)
    ack_words = {
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "yeah",
        "yea",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
        "sure",
        "bet",
        "cool",
        "nice",
        "good",
        "great",
        "fine",
        "alright",
        "word",
        "true",
        "facts",
        "lol",
        "lmao",
        "haha",
        "hi",
        "hey",
        "hello",
        "yo",
        "sup",
        "bye",
        "cya",
        "later",
        "thanks",
        "thx",
        "ty",
        "np",
    }
    if len(words) == 1 and text_lower.rstrip("!?.") in ack_words:
        return ("ACKNOWLEDGEABLE", 0.95)

    # === Tier 1b: Musing patterns (ACKNOWLEDGEABLE) ===
    # These look like questions but are just thinking out loud
    if re.match(r"^i wonder\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.90)  # "I wonder if..." = musing
    if re.match(r"^(kind of |kinda )?(curious|wondering)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.85)
    if re.match(r"^wonder\s+(if|what|how|why|who)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.85)  # "Wonder if..." = musing

    # Opinions with "no way", "no chance" (not reactive, just opinion)
    if re.match(r"^no (way|chance)\b", text_lower) and "!" not in text:
        return ("ACKNOWLEDGEABLE", 0.80)

    # "I don't think", "Doubt" = opinion statements
    if re.match(r"^(i don'?t think|doubt)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.80)

    # Past tense statements
    if re.match(r"^i (went|did|got|saw|had|made|took|came|finished)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.85)
    if re.match(r"^(the|my|our) \w+ (is|was|got|starts|ends)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.80)
    if re.match(r"^i'?m (going|heading|leaving|about to)\b", text_lower):
        return ("ACKNOWLEDGEABLE", 0.80)

    # Rhetorical questions - no ? and phrased as wondering
    # "Why do X" without ? = rhetorical
    if re.match(
        r"^(why|how|who)\s+(do|does|did|would|could|even)\b", text_lower
    ) and not text.endswith("?"):
        return ("ACKNOWLEDGEABLE", 0.75)

    return None


def classify_nli(model, text: str) -> tuple[str, dict[str, float]]:
    """NLI fallback for ambiguous cases.

    Uses hypotheses optimized for ambiguous cases (not clear questions/requests).
    """
    # Hypotheses tuned for ambiguous cases
    # Since structural rules catch clear questions/requests,
    # NLI mainly sees statements and potential musings
    hypotheses = {
        "ACTIONABLE": "The listener must decide whether to do something.",
        "ANSWERABLE": "The listener is being asked a direct question.",
        "REACTIVE": "This news deserves an emotional response.",
        "ACKNOWLEDGEABLE": "This is a thought or observation, no response needed.",
    }

    pairs = [(text, h) for h in hypotheses.values()]
    logits = model.predict(pairs)

    scores = {}
    for i, cat in enumerate(hypotheses.keys()):
        exp = np.exp(logits[i] - np.max(logits[i]))
        probs = exp / exp.sum()
        scores[cat] = float(probs[1])

    return max(scores, key=scores.get), scores


def classify_tiered(model, text: str) -> tuple[str, float, str]:
    """Tiered classification: structural first, then NLI.

    Returns (category, confidence, method).
    """
    # Try structural first
    structural = classify_structural(text)
    if structural:
        return structural[0], structural[1], "structural"

    # Fall back to NLI
    category, scores = classify_nli(model, text)
    confidence = scores[category]
    return category, confidence, "nli"


def main():
    print("Tiered Classification (Structural + NLI)")
    print("=" * 80)
    print(f"Testing on {len(TEXTS)} texts")
    print()

    # Load model
    print("Loading model...")
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
    print()

    # Run classification
    correct = 0
    by_method = {"structural": {"correct": 0, "total": 0}, "nli": {"correct": 0, "total": 0}}
    misclassifications = []

    for text, expected in TEXTS:
        predicted, confidence, method = classify_tiered(model, text)

        by_method[method]["total"] += 1
        if predicted == expected:
            correct += 1
            by_method[method]["correct"] += 1
        else:
            misclassifications.append((text, expected, predicted, confidence, method))

    accuracy = correct / len(TEXTS)

    print(f"Overall Accuracy: {accuracy:.1%}")
    print()

    print("By method:")
    for method, stats in by_method.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {method:12s}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    if misclassifications:
        print(f"\nMisclassifications ({len(misclassifications)}):")
        for text, expected, got, conf, method in misclassifications:
            print(f"  '{text[:45]:<45s}' exp={expected:14s} got={got:14s} ({method}, {conf:.2f})")

    # Analyze which patterns are being missed
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group misclassifications by pattern
    musing_errors = [
        m for m in misclassifications if "wonder" in m[0].lower() or "curious" in m[0].lower()
    ]
    question_errors = [m for m in misclassifications if m[1] == "ANSWERABLE"]
    request_errors = [m for m in misclassifications if m[1] == "ACTIONABLE"]
    reactive_errors = [m for m in misclassifications if m[1] == "REACTIVE"]
    ack_errors = [m for m in misclassifications if m[1] == "ACKNOWLEDGEABLE"]

    print(f"Musing errors: {len(musing_errors)}")
    print(f"Question errors: {len(question_errors)}")
    print(f"Request errors: {len(request_errors)}")
    print(f"Reactive errors: {len(reactive_errors)}")
    print(f"Other ACK errors: {len(ack_errors)}")


if __name__ == "__main__":
    main()
