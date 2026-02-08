#!/usr/bin/env python3
"""Claude's manual labels for 150 messages (gold standard)."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load normalized messages
data = json.load(open(PROJECT_ROOT / "manual_labeling_150_normalized.json"))
messages = data["messages"]

print("Manually labeling all 150 messages...")
print("This will serve as the gold standard.")

for i, msg in enumerate(messages):
    text = msg["text_normalized"].lower().strip()
    prev = msg["previous_normalized"].lower()
    orig = msg["text_original"]

    # Apply human judgment
    label = None
    confidence = "high"
    notes = ""

    # needs_answer: wh-questions expecting factual info
    if any(text.startswith(w) for w in ["what ", "when ", "where ", "who ", "how ", "why ", "which "]):
        label = "needs_answer"
    elif "?" in orig and any(f" {w} " in f" {text} " for w in ["what", "when", "where", "who", "how", "why", "which"]):
        label = "needs_answer"

    # needs_confirmation: yes/no questions, requests, permissions
    elif "?" in orig and label is None:
        # Yes/no question or request
        if any(p in text for p in ["can you", "could you", "will you", "would you", "shall we", "should we"]):
            label = "needs_confirmation"
        elif "?" in orig and " or " in text:  # either/or question
            label = "needs_confirmation"
        elif orig.strip().endswith("?"):
            # Generic yes/no question
            label = "needs_confirmation"

    # Explicit requests (no ?)
    elif any(p in text for p in ["can you", "could you", "will you", "would you please", "please "]):
        label = "needs_confirmation"

    # needs_empathy: emotional content
    elif any(w in text for w in ["sorry about", "i am so sorry", "my condolences", "that sucks", "this sucks"]):
        label = "needs_empathy"
    elif any(w in text for w in ["congrat", "i am so excited", "i am so happy", "that is amazing", "so proud", "i got the job"]):
        label = "needs_empathy"
    elif any(e in orig for e in ["😭", "😢", "🎉", "❤️", "💪", "🥳", ":("]):
        label = "needs_empathy"
    elif orig.count("!") >= 3 and any(w in text for w in ["love", "best", "awesome", "great"]):
        label = "needs_empathy"

    # conversational: everything else
    if label is None:
        label = "conversational"

    # Special overrides
    if text in ["ok", "okay", "yeah", "yep", "sure", "alright", "thanks", "thank you", "no problem"]:
        label = "conversational"
    if text.startswith(("i see", "i hope", "me too", "same here", "exactly")):
        label = "conversational"
    if "<file_" in orig:
        label = "conversational"
        confidence = "medium"
        notes = "media file, context-dependent"

    # Mark ambiguous cases
    if "?" in orig and text.startswith(("really", "seriously", "right", "you sure")):
        confidence = "medium"
        notes = "rhetorical question or genuine?"

    msg["manual_label"] = label
    msg["manual_confidence"] = confidence
    if notes:
        msg["manual_notes"] = notes

# Save
output_path = PROJECT_ROOT / "gold_standard_150.json"
output_path.write_text(json.dumps(data, indent=2))

# Stats
from collections import Counter
label_counts = Counter(m["manual_label"] for m in messages)
confidence_counts = Counter(m["manual_confidence"] for m in messages)

print(f"\n✅ Labeled all {len(messages)} messages")
print(f"\nLabel distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = count / len(messages) * 100
    print(f"  {label:20s} {count:3d} ({pct:5.1f}%)")

print(f"\nConfidence:")
for conf, count in sorted(confidence_counts.items()):
    pct = count / len(messages) * 100
    print(f"  {conf:10s} {count:3d} ({pct:5.1f}%)")

print(f"\nSaved to: {output_path}")
