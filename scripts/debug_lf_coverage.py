#!/usr/bin/env python3
"""Debug LF coverage - check why abstain rate is so high."""

from scripts.labeling_functions import get_registry, ABSTAIN

# Test examples
examples = [
    {"text": "ok", "context": [], "last_message": "See you at 5?", "metadata": None},
    {"text": "What time?", "context": [], "last_message": "Let's meet", "metadata": None},
    {"text": "I'm so happy!", "context": [], "last_message": "You got the job!", "metadata": None},
    {"text": "Hey how's it going", "context": [], "last_message": "", "metadata": None},
    {"text": "?", "context": [], "last_message": "Did you see my message", "metadata": None},
]

registry = get_registry()

for ex in examples:
    print(f"\nText: '{ex['text']}'")
    labels = registry.apply_all(ex["text"], ex["context"], ex["last_message"], ex["metadata"])

    votes = {}
    for i, (lf, label) in enumerate(zip(registry.lfs, labels)):
        if label != ABSTAIN:
            votes[lf.name] = label

    print(f"  Votes ({len(votes)}/{len(registry.lfs)}):")
    for lf_name, label in votes.items():
        print(f"    {lf_name}: {label}")

    if not votes:
        print("    (all abstained)")
