#!/usr/bin/env python3
"""Prepare 150 samples for manual labeling with normalization."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.text_normalizer import normalize_text

# Load samples
data = json.load(open(PROJECT_ROOT / "llm_test_results.json"))
messages = data["messages"]

# Normalize each message
print("Normalizing 150 messages...")
for msg in messages:
    msg["text_original"] = msg["text"]
    msg["text_normalized"] = normalize_text(msg["text"], expand_slang=True, spell_check=False)
    msg["previous_normalized"] = normalize_text(msg["previous"], expand_slang=True, spell_check=False)

# Save for manual labeling
output = {
    "instructions": """
MANUAL LABELING INSTRUCTIONS
=============================

For each message, assign ONE category based on what kind of REPLY the SLM should generate:

1. needs_answer
   - Expects factual information
   - Examples: "What time is it?", "Where are you?", "How does this work?"

2. needs_confirmation
   - Expects yes/no or acknowledgment
   - Examples: "Can you help?", "Are you coming?", "Want to grab lunch?"

3. needs_empathy
   - Needs emotional support (celebrating, comforting, validating)
   - Examples: "I got the job!", "I'm so stressed", "This sucks"

4. conversational
   - Casual engagement, statements, updates
   - Examples: "I'm at the store", "That's cool", "Yeah haha"

GUIDELINES:
- Use normalized text for labeling (slang expanded)
- Consider context (previous message)
- Think: "What should JARVIS say back?"
- If truly ambiguous, note it

For each message, add:
  "manual_label": "category_name",
  "manual_confidence": "high" | "medium" | "low",
  "manual_notes": "any ambiguity or reasoning"
""",
    "categories": ["needs_answer", "needs_confirmation", "needs_empathy", "conversational"],
    "messages": messages,
}

output_path = PROJECT_ROOT / "manual_labeling_150_normalized.json"
output_path.write_text(json.dumps(output, indent=2))

print(f"\n✅ Saved {len(messages)} normalized messages to:")
print(f"   {output_path}")
print()
print("Next: Manually add labels to each message:")
print('  "manual_label": "category",')
print('  "manual_confidence": "high|medium|low",')
print('  "manual_notes": "optional"')
