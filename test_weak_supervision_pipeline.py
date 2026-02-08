#!/usr/bin/env python3
"""Manual test of the weak supervision pipeline end-to-end."""

from jarvis.classifiers.category_classifier import classify_category
from jarvis.prompts import get_category_config, ACK_TEMPLATES

# Test messages for each category
test_messages = [
    ("ok", [], "ack"),
    ("thanks", [], "ack"),
    ("What time is the meeting?", [], "info"),
    ("Can you pick up milk?", [], "info"),
    ("I'm so stressed out", [], "emotional"),
    ("We won!", [], "emotional"),
    ("Hey how's it going", [], "social"),
    ("What do you think about the new movie?", [], "social"),
    ("?", [], "clarify"),
    ("huh", ["Check this out"], "clarify"),
]

print("Testing weak supervision pipeline:\n")
correct = 0
total = len(test_messages)

for text, context, expected in test_messages:
    result = classify_category(text, context=context)
    category = result.category
    conf = result.confidence
    method = result.method

    status = "✓" if category == expected else "✗"
    print(f"{status} '{text:30s}' → {category:10s} (expected: {expected:10s}) "
          f"[conf={conf:.2f}, method={method}]")

    if category == expected:
        correct += 1

print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")

# Test category configs
print("\n\nTesting category configs:")
for cat in ["ack", "info", "emotional", "social", "clarify"]:
    config = get_category_config(cat)
    print(f"  {cat:10s}: skip_slm={config.skip_slm}, context_depth={config.context_depth}")

# Test ack templates
print(f"\n\nACK templates ({len(ACK_TEMPLATES)} available):")
print(f"  {ACK_TEMPLATES[:5]}")
