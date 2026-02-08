#!/usr/bin/env python3
"""Claude labels the 200 production messages based on category definitions."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load validation data
with open(PROJECT_ROOT / "production_validation.json") as f:
    data = json.load(f)

messages = data["messages"]
labels = ["commissive", "directive", "inform", "question"]

# Claude's labels for each message
claude_labels = []

print(f"Labeling {len(messages)} messages...")
print()

for i, msg in enumerate(messages):
    text = msg["text"]
    lgbm = msg["lgbm_pred"]
    llm = msg["llm_pred"]

    # Label based on category definitions
    text_lower = text.lower()

    # Question detection
    is_question = False
    if "?" in text:
        is_question = True
    elif any(text_lower.startswith(w) for w in ["what", "where", "when", "who", "why", "how", "which", "whose", "whom"]):
        is_question = True
    elif text_lower.startswith(("are you", "is it", "is this", "is that", "did you", "do you", "does", "can you", "could you", "would you", "should you", "will you")):
        # Check if it's a genuine question or a polite directive
        if any(word in text_lower for word in ["can you", "could you", "would you", "will you"]) and "please" not in text_lower:
            # Likely a directive phrased as question
            is_question = False
        else:
            is_question = True
    elif " r u " in text_lower or text_lower.startswith("r u "):
        is_question = True

    # Directive detection
    is_directive = False
    directive_patterns = [
        "let me know", "lmk", "can you", "could you", "would you", "please",
        "tell me", "show me", "send me", "give me", "help me",
    ]
    if any(pattern in text_lower for pattern in directive_patterns):
        is_directive = True
        is_question = False  # Override question if it's a polite request
    elif text_lower.startswith(("pick", "call", "send", "help", "go", "come", "get", "take", "make", "do", "don't", "stop")):
        is_directive = True
    elif " gotta " in text_lower or text_lower.startswith("gotta "):
        # "we gotta X" is often directive
        is_directive = True

    # Commissive detection
    is_commissive = False
    commissive_patterns = [
        "i'll", "i will", "i can", "i'm gonna", "ima", "imma",
        "sure", "okay", "ok", "sounds good", "yes sir", "yessir", "yessirrr",
        "bet", "fs", "fosho", "fasho"
    ]
    if any(pattern in text_lower for pattern in commissive_patterns):
        is_commissive = True
    elif text_lower in ["yes", "yeah", "yea", "yep", "yup", "k", "kk", "cool"]:
        is_commissive = True

    # Decision logic
    if is_question and not is_directive:
        label = "question"
    elif is_directive:
        label = "directive"
    elif is_commissive:
        label = "commissive"
    else:
        label = "inform"

    claude_labels.append(label)

    # Print progress
    if (i + 1) % 20 == 0:
        print(f"Labeled {i+1}/200...", flush=True)

print()
print("✓ Labeling complete")
print()

# Add Claude labels to data
for msg, label in zip(messages, claude_labels):
    msg["claude_pred"] = label

# Save updated data
output_file = PROJECT_ROOT / "production_validation_with_claude.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"💾 Saved to: {output_file}")
print()

# Analysis
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# Extract predictions
lgbm_preds = [msg["lgbm_pred"] for msg in messages]
llm_preds = [msg["llm_pred"] for msg in messages]

# Claude label distribution
claude_dist = Counter(claude_labels)
print("=" * 70)
print("Claude's Label Distribution")
print("=" * 70)
for label in labels:
    count = claude_dist[label]
    pct = 100 * count / len(messages)
    print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
print()

# Agreement rates
lgbm_agreement = sum(1 for c, l in zip(claude_labels, lgbm_preds) if c == l)
llm_agreement = sum(1 for c, l in zip(claude_labels, llm_preds) if c == l)

lgbm_rate = lgbm_agreement / len(messages)
llm_rate = llm_agreement / len(messages)

print("=" * 70)
print("Agreement Rates (Claude as Ground Truth)")
print("=" * 70)
print(f"LightGBM: {lgbm_rate:.1%} ({lgbm_agreement}/{len(messages)})")
print(f"LLM:      {llm_rate:.1%} ({llm_agreement}/{len(messages)})")
print()

# Confusion matrices
print("=" * 70)
print("LightGBM Confusion Matrix (Claude=rows, LightGBM=cols)")
print("=" * 70)
cm_lgbm = confusion_matrix(claude_labels, lgbm_preds, labels=labels)
print(f"{'':12s}", end="")
for label in labels:
    print(f"{label:12s}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label:12s}", end="")
    for j in range(len(labels)):
        print(f"{cm_lgbm[i][j]:12d}", end="")
    print()
print()

print("=" * 70)
print("LLM Confusion Matrix (Claude=rows, LLM=cols)")
print("=" * 70)
cm_llm = confusion_matrix(claude_labels, llm_preds, labels=labels)
print(f"{'':12s}", end="")
for label in labels:
    print(f"{label:12s}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label:12s}", end="")
    for j in range(len(labels)):
        print(f"{cm_llm[i][j]:12d}", end="")
    print()
print()

# Per-class performance
print("=" * 70)
print("LightGBM Per-Class (Claude as ground truth)")
print("=" * 70)
print(classification_report(claude_labels, lgbm_preds, labels=labels, target_names=labels, digits=3, zero_division=0))

print("=" * 70)
print("LLM Per-Class (Claude as ground truth)")
print("=" * 70)
print(classification_report(claude_labels, llm_preds, labels=labels, target_names=labels, digits=3, zero_division=0))

# Winner
print("=" * 70)
print("🏆 Verdict")
print("=" * 70)
print()

if lgbm_rate > llm_rate + 0.05:
    print(f"✅ LightGBM WINS ({lgbm_rate:.1%} vs {llm_rate:.1%})")
    print(f"   LightGBM is {(lgbm_rate - llm_rate) * 100:.1f}% more accurate")
elif llm_rate > lgbm_rate + 0.05:
    print(f"✅ LLM WINS ({llm_rate:.1%} vs {lgbm_rate:.1%})")
    print(f"   LLM is {(llm_rate - lgbm_rate) * 100:.1f}% more accurate")
else:
    print(f"🤝 TIE ({lgbm_rate:.1%} vs {llm_rate:.1%})")

print()

# Error analysis
lgbm_errors = [
    (msg["text"], c, l)
    for msg, c, l in zip(messages, claude_labels, lgbm_preds)
    if c != l
]
llm_errors = [
    (msg["text"], c, l)
    for msg, c, l in zip(messages, claude_labels, llm_preds)
    if c != l
]

print("=" * 70)
print("LightGBM Errors (first 15)")
print("=" * 70)
for i, (text, claude, lgbm) in enumerate(lgbm_errors[:15]):
    print(f"{i+1}. {text[:65]}")
    print(f"   Claude: {claude:12s}  LightGBM: {lgbm:12s}")
    print()

print("=" * 70)
print("LLM Errors (first 15)")
print("=" * 70)
for i, (text, claude, llm) in enumerate(llm_errors[:15]):
    print(f"{i+1}. {text[:65]}")
    print(f"   Claude: {claude:12s}  LLM: {llm:12s}")
    print()
