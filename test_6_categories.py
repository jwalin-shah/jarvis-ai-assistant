#!/usr/bin/env python3
"""Test the new 6-category schema with clear examples."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import get_judge_client

PROMPT = """Classify each message into ONE category. Use the decision tree below - check categories in order, take the FIRST match.

Categories (check in this order):

1. closing - Ending the conversation
   Examples: "bye", "ttyl", "see you later", "gotta go", "talk soon"

2. acknowledge - Minimal agreement/acknowledgment (≤3 words, no question)
   Examples: "ok", "thanks", "yeah", "gotcha", "sure", "yup", "👍"

3. request - Seeking action (has "can you"/"could you"/"would you"/"please" + action verb OR imperative verb OR "I suggest"/"let's")
   Examples: "Can you send the file?", "Please call me", "Send it over", "I suggest we meet", "Let's go"

4. question - Seeking information (has "?" OR starts with: what, when, where, who, why, how, is, are, do, does, will, should)
   Examples: "What time?", "Where are you?", "Is it ready?", "How are you?"

5. emotion - Expressing feelings (contains: happy, sad, angry, stressed, excited, frustrated, love, hate, amazing, terrible OR multiple "!" OR CAPS)
   Examples: "I'm so stressed!", "This is AMAZING", "Ugh so frustrated", "Love it!"

6. statement - Everything else (opinions, facts, stories, answers, comments)
   Examples: "It's raining", "I think so", "The meeting went well", "That's cool"

For each message, consider the conversation context (previous message) when classifying.

Message 1:
Previous: "How was your day?"
Current: "bye"

Message 2:
Previous: "Want to grab lunch?"
Current: "ok"

Message 3:
Previous: "I'm at the store"
Current: "What time will you be back?"

Message 4:
Previous: "I need help with this"
Current: "Can you send me the file?"

Message 5:
Previous: "How did the interview go?"
Current: "I'm so excited! It went AMAZING!"

Message 6:
Previous: "What do you think about the project?"
Current: "I think it looks pretty good overall"

Reply with ONLY the category name for each, one per line (e.g., "acknowledge"). No numbers, no explanations."""

def main():
    client = get_judge_client()
    if not client:
        return 1

    print("Testing 6-category schema with clear examples...")
    print()

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.0,
        max_tokens=400,
    )

    message = response.choices[0].message
    content = message.content or message.reasoning

    print("=" * 60)
    print("RAW RESPONSE:")
    print("=" * 60)
    print(content)
    print("=" * 60)
    print()

    # Parse
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    # Skip explanatory lines
    predictions = []
    for line in lines:
        if any(skip in line.lower() for skip in ["classify", "let's", "need to", "categories"]):
            continue
        # Extract category
        if "->" in line:
            parts = line.split("->")
            clean = parts[1].split("(")[0].strip().lower() if len(parts) >= 2 else line
        else:
            clean = line.lstrip("0123456789. \"").lower().strip()
        clean = clean.split("(")[0].strip()

        valid_cats = ["closing", "acknowledge", "question", "request", "emotion", "statement"]
        if clean in valid_cats:
            predictions.append(clean)

    # Expected
    expected = ["closing", "acknowledge", "question", "request", "emotion", "statement"]
    examples = [
        "bye",
        "ok",
        "What time will you be back?",
        "Can you send me the file?",
        "I'm so excited! It went AMAZING!",
        "I think it looks pretty good overall"
    ]

    print("RESULTS:")
    print()
    correct = 0
    for i, (exp, ex) in enumerate(zip(expected, examples)):
        pred = predictions[i] if i < len(predictions) else "???"
        match = "✓" if pred == exp else "✗"
        if pred == exp:
            correct += 1
        print(f"{match} {i + 1}. {ex[:50]:50s} → {pred:12s} (expected: {exp})")

    print()
    print(f"Accuracy: {correct}/{len(expected)} = {correct/len(expected)*100:.1f}%")

    return 0 if correct >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
