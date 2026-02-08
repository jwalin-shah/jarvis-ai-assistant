#!/usr/bin/env python3
"""Test ack labeling - check if acknowledgments are being correctly labeled."""

from scripts.label_aggregation import aggregate_labels
from scripts.labeling_functions import get_registry

# Examples that SHOULD be labeled as 'ack'
ack_examples = [
    {"text": "ok", "context": [], "last_message": "See you at 5", "metadata": None},
    {"text": "lol", "context": ["That was so funny"], "last_message": "Did you see that video?", "metadata": None},
    {"text": "thanks", "context": [], "last_message": "Here's the file", "metadata": None},
    {"text": "Liked \"hey there\"", "context": [], "last_message": "hey there", "metadata": None},
    {"text": "👍", "context": ["Sounds good"], "last_message": "Let's meet at 3", "metadata": None},
    {"text": "np", "context": [], "last_message": "Thanks!", "metadata": None},
    {"text": "nice", "context": ["I got an A"], "last_message": "I got an A", "metadata": None},
]

registry = get_registry()

labels, confidences = aggregate_labels(ack_examples, registry, method="majority")

print("Ack labeling test:")
for i, ex in enumerate(ack_examples):
    label = labels[i]
    conf = confidences[i]
    status = "✓" if label == "ack" else "✗"
    print(f"{status} '{ex['text']:20s}' → {label:10s} (conf={conf:.2f})")

accuracy = sum(1 for lbl in labels if lbl == "ack") / len(labels)
print(f"\nAccuracy: {accuracy:.1%}")
