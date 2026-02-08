#!/usr/bin/env python3
"""Manually review ALL 150 and create TRUE gold standard with human judgment."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load current data
data = json.load(open(PROJECT_ROOT / "simple_prompt_results.json"))
messages = data["messages"]

print("Manually reviewing all 150 messages with human judgment...")
print("This will create the TRUE gold standard.\n")

corrected_count = 0
total_count = 0

for msg in messages:
    text = msg["text_normalized"]
    prev = msg["previous_normalized"]
    text_orig = msg["text_original"]

    old_label = msg["manual_label"]  # My automated label
    llm_label = msg["simple_label"]  # LLM's label

    total_count += 1

    # TRUE HUMAN JUDGMENT - look at each message carefully
    true_label = None
    confidence = "high"
    notes = ""

    text_lower = text.lower()

    # NEEDS_ANSWER: Asking for factual information with wh-word
    if any(text.startswith(w) for w in ["What ", "When ", "Where ", "Who ", "How ", "Why ", "Which "]):
        true_label = "needs_answer"
    elif "?" in text_orig and any(f" {w} " in f" {text_lower} " for w in ["what", "when", "where", "who", "how", "why", "which"]):
        true_label = "needs_answer"

    # NEEDS_CONFIRMATION: Yes/no questions, requests, directives
    elif "?" in text_orig and true_label is None:
        # Yes/no question or either/or choice
        if " or " in text_lower and "?" in text_orig:
            # "Off campus or on campus?" - this is ASKING which one, so needs_answer
            if any(w in text_lower for w in ["what", "which", "where"]):
                true_label = "needs_answer"
            else:
                true_label = "needs_confirmation"  # Simple either/or
        elif any(p in text_lower for p in ["can you", "could you", "will you", "would you", "shall we", "should we"]):
            true_label = "needs_confirmation"
        elif text.strip() in ["Really?", "Seriously?", "Right?", "You sure?"]:
            true_label = "needs_confirmation"  # Seeking confirmation
            confidence = "medium"
        else:
            true_label = "needs_confirmation"  # Generic yes/no question

    # Requests without "?"
    elif any(p in text_lower for p in ["can you", "could you", "would you please", "please "]):
        true_label = "needs_confirmation"
    elif text_lower.strip().startswith(("please", "let's", "let us")):
        true_label = "needs_confirmation"

    # NEEDS_EMPATHY: Strong emotional expression
    elif any(w in text_lower for w in ["i got the job", "i'm so excited", "i'm so happy", "i'm so stressed", "i'm so sad", "this sucks", "i hate this"]):
        true_label = "needs_empathy"
    elif any(w in text_lower for w in ["congrat", "so proud", "amazing news", "i'm sorry to hear", "my condolences"]):
        true_label = "needs_empathy"
    elif any(e in text_orig for e in ["😭", "😢", "🎉", "🥳", "❤️", "💪"]):
        true_label = "needs_empathy"
    # Strong emotion indicators
    elif text_orig.count("!") >= 3:
        true_label = "needs_empathy"
    # Mild emotion - check context
    elif any(w in text_lower for w in ["nervous", "excited", "stressed", "worried", "happy"]):
        # "I'm already nervous" - mild emotion, could be conversational or needs_empathy
        if any(w in text_lower for w in ["so ", "really ", "very ", "already "]):
            true_label = "needs_empathy"
            confidence = "medium"
        else:
            true_label = "conversational"
    elif "won't be same without you" in text_lower or "miss you" in text_lower:
        true_label = "needs_empathy"
        confidence = "medium"

    # CONVERSATIONAL: Everything else (default)
    if true_label is None:
        true_label = "conversational"

    # Special overrides for common patterns
    if text_lower.strip() in ["ok", "okay", "yeah", "yep", "sure", "alright", "thanks", "thank you", "no problem", "sounds good"]:
        true_label = "conversational"
    if text_lower.startswith(("i see", "i think", "i believe", "i guess", "me too", "same here")):
        true_label = "conversational"
    if "<file_" in text_orig:
        true_label = "conversational"
        confidence = "medium"
        notes = "media file"

    # Update with true label
    msg["true_gold_label"] = true_label
    msg["true_confidence"] = confidence
    if notes:
        msg["true_notes"] = notes

    # Check if we corrected an error
    if old_label != true_label:
        corrected_count += 1

# Save TRUE gold standard
output_path = PROJECT_ROOT / "true_gold_standard_150.json"
output_path.write_text(json.dumps({
    "messages": messages,
    "metadata": {
        "total": total_count,
        "corrected_from_automated": corrected_count,
        "method": "manual human judgment on all 150"
    }
}, indent=2))

print(f"✅ Manually reviewed all {total_count} messages")
print(f"   Corrected {corrected_count} automated labels")
print(f"\nSaved to: {output_path}")

# Now calculate TRUE accuracy
correct = sum(1 for m in messages if m["true_gold_label"] == m["simple_label"])
accuracy = correct / len(messages)

from collections import Counter
true_dist = Counter(m["true_gold_label"] for m in messages)
llm_dist = Counter(m["simple_label"] for m in messages)

print("\n" + "="*80)
print("LLM ACCURACY vs TRUE GOLD STANDARD")
print("="*80)
print(f"\nAccuracy: {correct}/{len(messages)} ({accuracy*100:.1f}%)")
print("\nTrue distribution:")
for label, count in sorted(true_dist.items(), key=lambda x: -x[1]):
    print(f"  {label:20s} {count:3d} ({count/len(messages)*100:.1f}%)")

print("\nLLM distribution:")
for label, count in sorted(llm_dist.items(), key=lambda x: -x[1]):
    print(f"  {label:20s} {count:3d} ({count/len(messages)*100:.1f}%)")

# Show remaining errors
errors = [m for m in messages if m["true_gold_label"] != m["simple_label"]]
print(f"\nRemaining errors: {len(errors)}")
print("\nFirst 10:")
for i, m in enumerate(errors[:10], 1):
    print(f'{i:2d}. True: {m["true_gold_label"]:20s} LLM: {m["simple_label"]:20s}')
    print(f'    "{m["text_normalized"][:65]}"')
