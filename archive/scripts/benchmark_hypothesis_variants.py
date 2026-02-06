#!/usr/bin/env python3
"""Test different hypothesis phrasings for NLI response classification.

The core problem: "I wonder if..." is linguistically an indirect speech act.
- Surface form: wondering/musing (statement)
- Could be interpreted as: indirect question

The key insight: in casual chat, "I wonder if..." is thinking out loud,
not actually expecting an answer. The hypothesis must capture RESPONSE EXPECTATION.

Usage:
    uv run python scripts/benchmark_hypothesis_variants.py
"""

from __future__ import annotations

import numpy as np

# Test texts with ground truth
TEXTS = [
    # Musings - look like questions but are just thinking out loud
    ("I wonder if bulls make another trade", "ACKNOWLEDGEABLE"),
    ("Kind of curious what they will offer", "ACKNOWLEDGEABLE"),
    ("I wonder what happened", "ACKNOWLEDGEABLE"),
    ("Curious how they'll handle it", "ACKNOWLEDGEABLE"),
    ("Wonder if anyone else noticed", "ACKNOWLEDGEABLE"),
    # Opinions/disbelief - NOT emotional reactions
    ("No way Dallas could've gotten lottery picks", "ACKNOWLEDGEABLE"),
    ("No way that's real", "ACKNOWLEDGEABLE"),
    ("No chance they win tonight", "ACKNOWLEDGEABLE"),
    ("I don't think it's gonna happen", "ACKNOWLEDGEABLE"),
    ("Doubt they'll actually do it", "ACKNOWLEDGEABLE"),
    # Rhetorical questions - don't expect answers
    ("Why do dads text like that", "ACKNOWLEDGEABLE"),
    ("How does that even work", "ACKNOWLEDGEABLE"),
    ("Who even says that anymore", "ACKNOWLEDGEABLE"),
    # Actual questions - DO expect answers
    ("What time is the game", "ANSWERABLE"),
    ("Where are you", "ANSWERABLE"),
    ("Did you get my text", "ANSWERABLE"),
    ("What happened at the meeting", "ANSWERABLE"),
    # Requests - DO require commitment
    ("Can you pick me up", "ACTIONABLE"),
    ("Wanna grab lunch", "ACTIONABLE"),
    ("Let me know when you're free", "ACTIONABLE"),
    ("Text me when you get there", "ACTIONABLE"),
    # Reactions - DO warrant emotional response
    ("Omg I got the job!!", "REACTIVE"),
    ("That's so sad", "REACTIVE"),
    ("This is amazing!!", "REACTIVE"),
    # Statements
    ("I went to the store", "ACKNOWLEDGEABLE"),
    ("The game starts at 7", "ACKNOWLEDGEABLE"),
    ("I'm heading out now", "ACKNOWLEDGEABLE"),
]

# =============================================================================
# Hypothesis Variants to Test
# =============================================================================

# Current hypotheses (baseline - response_nli.py)
HYPOTHESES_V0 = {
    "ACTIONABLE": "This requires a yes or no decision about doing something.",
    "ANSWERABLE": "This is a question asking for specific information.",
    "REACTIVE": "This shares exciting or upsetting news that deserves a reaction.",
    "ACKNOWLEDGEABLE": "This is a statement or observation, not a question or request.",
}

# V1: Focus on RESPONSE EXPECTATION
HYPOTHESES_V1 = {
    "ACTIONABLE": "The speaker expects me to say yes or no to doing something.",
    "ANSWERABLE": "The speaker is waiting for me to provide specific information.",
    "REACTIVE": "The speaker wants me to react emotionally to exciting or sad news.",
    "ACKNOWLEDGEABLE": "The speaker is just sharing a thought, no specific response is needed.",
}

# V2: Contrastive phrasing (what it is NOT)
HYPOTHESES_V2 = {
    "ACTIONABLE": "This directly asks me to commit to doing something.",
    "ANSWERABLE": "This directly asks me a question that I should answer.",
    "REACTIVE": "This is announcing news that deserves congratulations or sympathy.",
    "ACKNOWLEDGEABLE": "This is thinking out loud or sharing info, not asking for anything.",
}

# V3: Musing-specific handling
HYPOTHESES_V3 = {
    "ACTIONABLE": "The speaker is making a request or invitation I need to accept or decline.",
    "ANSWERABLE": "The speaker is asking ME directly for information I should provide.",
    "REACTIVE": "The speaker is sharing emotional news and wants empathy or celebration.",
    "ACKNOWLEDGEABLE": "The speaker is expressing a thought, musing, or observation - response optional.",
}

# V4: Explicit musing exclusion
HYPOTHESES_V4 = {
    "ACTIONABLE": "This is a request for me to do something specific.",
    "ANSWERABLE": "This is a direct question expecting a factual answer from me.",
    "REACTIVE": "This is exciting or sad news that calls for an emotional response.",
    "ACKNOWLEDGEABLE": "This is a musing, opinion, wonder, or statement - no answer expected.",
}

# V5: Response-required framing
HYPOTHESES_V5 = {
    "ACTIONABLE": "I must respond with yes/no or a commitment to this request.",
    "ANSWERABLE": "I must respond with information to answer this direct question.",
    "REACTIVE": "I should respond with empathy, congratulations, or comfort to this news.",
    "ACKNOWLEDGEABLE": "I can simply acknowledge this or say nothing - no response required.",
}

# V6: Listener obligation focus
HYPOTHESES_V6 = {
    "ACTIONABLE": "As the listener, I am obligated to accept, decline, or negotiate this.",
    "ANSWERABLE": "As the listener, I am obligated to provide the requested information.",
    "REACTIVE": "As the listener, I should acknowledge this emotionally significant content.",
    "ACKNOWLEDGEABLE": "As the listener, I have no obligation to respond substantively.",
}

# V7: Indirect speech act aware
HYPOTHESES_V7 = {
    "ACTIONABLE": "This is a request or invitation, not just wondering about something.",
    "ANSWERABLE": "This is a direct question to me, not just curiosity or musing.",
    "REACTIVE": "This is news that calls for celebration, sympathy, or shared emotion.",
    "ACKNOWLEDGEABLE": "This is just sharing a thought, wondering aloud, or making an observation.",
}

# V8: Binary response test
HYPOTHESES_V8 = {
    "ACTIONABLE": "Responding requires me to say yes, no, or maybe to a request.",
    "ANSWERABLE": "Responding requires me to provide specific facts or information.",
    "REACTIVE": "Responding appropriately means sharing in the speaker's emotion.",
    "ACKNOWLEDGEABLE": "Responding is optional - a simple 'ok' or silence would be appropriate.",
}

HYPOTHESIS_VARIANTS = {
    "v0_baseline": HYPOTHESES_V0,
    "v1_expectation": HYPOTHESES_V1,
    "v2_contrastive": HYPOTHESES_V2,
    "v3_musing_aware": HYPOTHESES_V3,
    "v4_musing_explicit": HYPOTHESES_V4,
    "v5_response_required": HYPOTHESES_V5,
    "v6_obligation": HYPOTHESES_V6,
    "v7_indirect_aware": HYPOTHESES_V7,
    "v8_binary_response": HYPOTHESES_V8,
}


def load_model(model_id: str = "cross-encoder/nli-deberta-v3-small"):
    """Load NLI model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=512)


def classify(model, text: str, hypotheses: dict[str, str]) -> tuple[str, dict[str, float]]:
    """Classify using NLI."""
    pairs = [(text, h) for h in hypotheses.values()]
    logits = model.predict(pairs)

    scores = {}
    for i, cat in enumerate(hypotheses.keys()):
        exp = np.exp(logits[i] - np.max(logits[i]))
        probs = exp / exp.sum()
        scores[cat] = float(probs[1])

    return max(scores, key=scores.get), scores


def evaluate_hypotheses(model, hypotheses: dict[str, str], name: str) -> dict:
    """Evaluate a hypothesis set."""
    correct = 0
    correct_by_cat = {cat: 0 for cat in hypotheses.keys()}
    total_by_cat = {cat: 0 for cat in hypotheses.keys()}
    misclassifications = []

    for text, expected in TEXTS:
        total_by_cat[expected] += 1
        predicted, scores = classify(model, text, hypotheses)

        if predicted == expected:
            correct += 1
            correct_by_cat[expected] += 1
        else:
            misclassifications.append((text, expected, predicted, scores))

    accuracy = correct / len(TEXTS)
    accuracy_by_cat = {
        cat: correct_by_cat[cat] / total_by_cat[cat] if total_by_cat[cat] > 0 else 0.0
        for cat in hypotheses.keys()
    }

    return {
        "name": name,
        "accuracy": accuracy,
        "accuracy_by_cat": accuracy_by_cat,
        "misclassifications": misclassifications,
    }


def main():
    print("Hypothesis Template Comparison")
    print("=" * 80)
    print(f"Testing {len(HYPOTHESIS_VARIANTS)} variants on {len(TEXTS)} texts")
    print("Model: cross-encoder/nli-deberta-v3-small")
    print()

    # Load model once
    print("Loading model...")
    model = load_model()
    print()

    results = []
    for name, hypotheses in HYPOTHESIS_VARIANTS.items():
        result = evaluate_hypotheses(model, hypotheses, name)
        results.append(result)

        print(f"\n{'='*70}")
        print(f"Variant: {name}")
        print(f"{'='*70}")
        print(f"Overall: {result['accuracy']:.1%}")
        print(
            "By category:", ", ".join(f"{k}:{v:.0%}" for k, v in result["accuracy_by_cat"].items())
        )

        # Show hypothesis text
        print("\nHypotheses:")
        for cat, h in hypotheses.items():
            print(f"  {cat:15s}: {h[:60]}...")

        # Show misclassifications on musing texts
        musing_errors = [
            m
            for m in result["misclassifications"]
            if "wonder" in m[0].lower() or "curious" in m[0].lower()
        ]
        if musing_errors:
            print(f"\nMusing errors ({len(musing_errors)}):")
            for text, expected, got, scores in musing_errors[:5]:
                print(f"  '{text[:40]:<40s}' -> {got} (wanted {expected})")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - Sorted by Accuracy")
    print("=" * 80)
    results.sort(key=lambda x: -x["accuracy"])

    print(f"{'Variant':<25s} {'Overall':>10s} {'ACK':>10s} {'ANS':>10s} {'ACT':>10s} {'REA':>10s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<25s} "
            f"{r['accuracy']:>9.1%} "
            f"{r['accuracy_by_cat'].get('ACKNOWLEDGEABLE', 0):>9.1%} "
            f"{r['accuracy_by_cat'].get('ANSWERABLE', 0):>9.1%} "
            f"{r['accuracy_by_cat'].get('ACTIONABLE', 0):>9.1%} "
            f"{r['accuracy_by_cat'].get('REACTIVE', 0):>9.1%}"
        )

    # Best variant details
    best = results[0]
    print(f"\n{'='*80}")
    print(f"BEST: {best['name']} ({best['accuracy']:.1%})")
    print(f"{'='*80}")
    print("Hypotheses:")
    for cat, h in HYPOTHESIS_VARIANTS[best["name"]].items():
        print(f"  {cat}: {h}")

    if best["misclassifications"]:
        print(f"\nRemaining misclassifications ({len(best['misclassifications'])}):")
        for text, expected, got, scores in best["misclassifications"]:
            score_str = ", ".join(
                f"{k[:3]}:{v:.2f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])[:2]
            )
            print(f"  '{text[:45]:<45s}' exp={expected:14s} got={got:14s} [{score_str}]")


if __name__ == "__main__":
    main()
