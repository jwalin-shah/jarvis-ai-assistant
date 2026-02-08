#!/usr/bin/env python3
"""Interactive labeling tool for 150 messages."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load messages
data_path = PROJECT_ROOT / "manual_labeling_150_normalized.json"
data = json.load(open(data_path))
messages = data["messages"]

# Resume from where we left off
labeled_count = sum(1 for m in messages if "manual_label" in m)
print(f"\nResuming... {labeled_count}/150 already labeled")

categories = ["needs_answer", "needs_confirmation", "needs_empathy", "conversational"]

print("\n" + "="*80)
print("INTERACTIVE LABELING")
print("="*80)
print("\nCategories:")
print("  1 = needs_answer (expects factual info)")
print("  2 = needs_confirmation (expects yes/no/acknowledgment)")
print("  3 = needs_empathy (needs emotional support)")
print("  4 = conversational (casual engagement)")
print("\nCommands: 1-4 to label, s=skip, q=quit, b=back\n")

i = labeled_count
while i < len(messages):
    msg = messages[i]

    print(f"\n[{i+1}/150] ----------------------------------------")
    print(f"Previous: {msg['previous_normalized'][:70]}")
    print(f"Message:  {msg['text_normalized'][:70]}")
    if msg['text_original'] != msg['text_normalized']:
        print(f"Original: {msg['text_original'][:70]}")

    # Show current label if exists
    if "manual_label" in msg:
        current = msg["manual_label"]
        print(f"\nCurrent label: {current}")

    choice = input("\nLabel (1-4, s, q, b): ").strip().lower()

    if choice == 'q':
        break
    elif choice == 's':
        i += 1
        continue
    elif choice == 'b':
        i = max(0, i - 1)
        continue
    elif choice in ['1', '2', '3', '4']:
        label_idx = int(choice) - 1
        msg["manual_label"] = categories[label_idx]
        msg["manual_confidence"] = "high"  # can edit later if needed

        # Save after each label
        data_path.write_text(json.dumps(data, indent=2))
        i += 1
    else:
        print("Invalid choice")

print(f"\n✅ Labeled {sum(1 for m in messages if 'manual_label' in m)}/150")
print(f"Saved to {data_path}")
