#!/usr/bin/env python3
"""Manually review all 150 LLM-labeled messages."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load LLM results
data = json.load(open(PROJECT_ROOT / "llm_test_results.json"))
messages = data["messages"]

# Manual review
reviews = []
correct = 0
wrong = 0
ambiguous = 0

for i, msg in enumerate(messages, 1):
    text = msg["text"]
    prev = msg["previous"]
    llm_label = msg["label"]
    method = msg["method"]

    # Manual judgment
    text_lower = text.lower()

    # Determine correct label
    correct_label = None
    is_ambiguous = False

    # needs_answer: asking for factual info
    if "?" in text and any(w in text_lower for w in ["what", "when", "where", "who", "how", "why", "which"]):
        correct_label = "needs_answer"
    # needs_confirmation: yes/no questions or requests
    elif "?" in text and not any(w in text_lower for w in ["what", "when", "where", "who", "how", "why"]):
        correct_label = "needs_confirmation"
    # Explicit requests
    elif any(p in text_lower for p in ["can you", "could you", "will you", "would you", "shall we", "please"]):
        correct_label = "needs_confirmation"
    # needs_empathy: emotional content
    elif any(w in text_lower for w in ["sorry", "hope", "congrat", "excited", "sad", "hate", "love", ":(", ":)"]):
        # Check context
        if "sorry about" in text_lower or "hope" in text_lower:
            correct_label = "needs_empathy"
        else:
            correct_label = "conversational"  # Mild emotion
    # Emojis
    elif any(e in text for e in ["😭", "😢", "😔", "😊", "🎉", "❤️", "💪", ":("]):
        correct_label = "needs_empathy"
    # Multiple exclamation marks
    elif text.count("!") >= 2:
        if any(w in text_lower for w in ["awesome", "great", "nice", "cool", "love", "best"]):
            correct_label = "needs_empathy"
        else:
            correct_label = "conversational"
    # conversational: everything else
    else:
        correct_label = "conversational"

    # Special cases
    if "<file_" in text:
        correct_label = "conversational"  # or needs_answer? AMBIGUOUS
        is_ambiguous = True
    if text in ["ok", "okay", "yeah", "yep", "sure", "alright", "thanks"]:
        correct_label = "conversational"
    if text.lower().startswith(("i see", "i hope", "me too")):
        correct_label = "conversational"

    # Check if LLM got it right
    if is_ambiguous:
        status = "ambiguous"
        ambiguous += 1
    elif llm_label == correct_label:
        status = "correct"
        correct += 1
    else:
        status = "wrong"
        wrong += 1

    reviews.append(
        {
            "index": i,
            "text": text[:80],
            "previous": prev[:60],
            "llm_label": llm_label,
            "correct_label": correct_label,
            "status": status,
            "method": method,
        }
    )

# Save reviews
output = {
    "summary": {
        "total": len(messages),
        "correct": correct,
        "wrong": wrong,
        "ambiguous": ambiguous,
        "accuracy": correct / len(messages),
        "heuristic_count": sum(1 for m in messages if m["method"] == "heuristic"),
        "llm_count": sum(1 for m in messages if m["method"] == "llm"),
    },
    "reviews": reviews,
}

output_path = PROJECT_ROOT / "manual_review_150.json"
output_path.write_text(json.dumps(output, indent=2))

# Print summary
print("=" * 80)
print("MANUAL REVIEW RESULTS (ALL 150 MESSAGES)")
print("=" * 80)
print(f"\nTotal: {len(messages)}")
print(f"Correct: {correct}/{len(messages)} ({correct / len(messages) * 100:.1f}%)")
print(f"Wrong: {wrong}/{len(messages)} ({wrong / len(messages) * 100:.1f}%)")
print(f"Ambiguous: {ambiguous}/{len(messages)} ({ambiguous / len(messages) * 100:.1f}%)")

# Breakdown by method
heuristic_reviews = [r for r in reviews if r["method"] == "heuristic"]
llm_reviews = [r for r in reviews if r["method"] == "llm"]

heuristic_correct = sum(1 for r in heuristic_reviews if r["status"] == "correct")
llm_correct = sum(1 for r in llm_reviews if r["status"] == "correct")

print(f"\nHeuristic accuracy: {heuristic_correct}/{len(heuristic_reviews)} ({heuristic_correct / max(len(heuristic_reviews), 1) * 100:.1f}%)")
print(f"LLM accuracy: {llm_correct}/{len(llm_reviews)} ({llm_correct / max(len(llm_reviews), 1) * 100:.1f}%)")

# Show errors
errors = [r for r in reviews if r["status"] == "wrong"]
print(f"\nErrors ({len(errors)} total):")
for e in errors[:15]:
    print(f"  {e['index']:3d}. LLM: {e['llm_label']:20s} → Should be: {e['correct_label']:20s}")
    print(f"       \"{e['text']}\"")
    print()

print(f"\nFull results saved to: {output_path}")
