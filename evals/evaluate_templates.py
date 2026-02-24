#!/usr/bin/env python3
"""Evaluate template responses against real iMessage data.

This script:
1. Fetches real acknowledge/closing messages from chat.db
2. Shows what templates would be sent
3. Evaluates if templates are appropriate using Cerebras judge
4. Suggests better templates based on user's actual reply style

Usage:
    uv run python evals/evaluate_templates.py [--limit 50]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import JUDGE_MODEL, get_judge_client

from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES


def fetch_real_messages(limit: int = 50) -> list[dict]:
    """Fetch real acknowledge/closing messages from chat.db."""
    import sqlite3

    chat_db = Path.home() / "Library" / "Messages" / "chat.db"
    if not chat_db.exists():
        print(f"❌ chat.db not found at {chat_db}")
        return []

    try:
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Look for short messages that might be acks/closings
        # Heuristic: short messages (2-15 chars) or common patterns
        cursor.execute(
            """
            SELECT
                m.text,
                m.date,
                c.display_name,
                m.is_from_me
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL
              AND length(m.text) <= 20
              AND m.text NOT LIKE '%http%'
            ORDER BY m.date DESC
            LIMIT ?
        """,
            (limit * 3,),
        )  # Get more to filter

        messages = []
        for row in cursor.fetchall():
            text, date, display_name, is_from_me = row
            if text and not is_from_me:  # Only incoming messages
                messages.append(
                    {
                        "text": text.strip(),
                        "contact": display_name or "Unknown",
                    }
                )
                if len(messages) >= limit:
                    break

        conn.close()
        return messages

    except Exception as e:
        print(f"❌ Error reading chat.db: {e}")
        return []


def categorize_message(text: str) -> str:
    """Simple heuristic to categorize message."""
    text_lower = text.lower().strip()

    # Closing patterns
    closing_words = ["bye", "goodbye", "gn", "night", "ttyl", "later", "peace", "cya"]
    if any(word in text_lower for word in closing_words):
        return "closing"

    # Acknowledge patterns
    ack_words = [
        "ok",
        "okay",
        "k",
        "yes",
        "yeah",
        "yep",
        "sure",
        "sounds good",
        "got it",
        "thanks",
        "ty",
        "np",
    ]
    if any(word in text_lower for word in ack_words) or len(text_lower) <= 5:
        return "acknowledge"

    return "other"


def judge_template_appropriateness(message: str, template: str, client) -> dict:
    """Use Cerebras judge to evaluate if template is appropriate."""
    prompt = f"""You are evaluating if a template response is appropriate for a text message.

Incoming message: "{message}"
Proposed response: "{template}"

Rate this response on a scale of 1-10:
- 9-10: Perfect natural response
- 7-8: Good response, appropriate
- 5-6: Acceptable but could be better
- 3-4: Poor, somewhat awkward or inappropriate
- 1-2: Very bad, completely wrong

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>", "better_alternative": "<suggested better response or null>"}}
"""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content.strip())
        return result
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge error: {e}", "better_alternative": None}


def analyze_user_style(messages: list[dict]) -> dict:
    """Analyze user's actual reply style from their messages."""
    all_texts = [m["text"].lower() for m in messages]

    # Count common patterns
    from collections import Counter

    # Find acknowledgments they use
    ack_patterns = [
        "ok",
        "okay",
        "k",
        "sounds good",
        "got it",
        "thanks",
        "ty",
        "np",
        "bet",
        "cool",
        "sure",
        "yep",
        "yeah",
        "alright",
        "👍",
        "🙏",
    ]
    found_acks = []
    for text in all_texts:
        for pattern in ack_patterns:
            if pattern in text:
                found_acks.append(text)
                break

    # Find closings they use
    closing_patterns = ["bye", "later", "ttyl", "gn", "night", "peace", "cya", "talk soon"]
    found_closings = []
    for text in all_texts:
        for pattern in closing_patterns:
            if pattern in text:
                found_closings.append(text)
                break

    return {
        "ack_examples": Counter(found_acks).most_common(10),
        "closing_examples": Counter(found_closings).most_common(10),
        "avg_ack_length": sum(len(t) for t in found_acks) / len(found_acks) if found_acks else 0,
        "avg_closing_length": sum(len(t) for t in found_closings) / len(found_closings)
        if found_closings
        else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate template responses")
    parser.add_argument("--limit", type=int, default=30, help="Number of messages to evaluate")
    parser.add_argument(
        "--judge", action="store_true", help="Use Cerebras judge (slower but better)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TEMPLATE EVALUATION")
    print("=" * 70)
    print()
    print("Current Templates:")
    print(f"  ACKNOWLEDGE ({len(ACKNOWLEDGE_TEMPLATES)}): {ACKNOWLEDGE_TEMPLATES}")
    print(f"  CLOSING ({len(CLOSING_TEMPLATES)}): {CLOSING_TEMPLATES}")
    print()

    # Fetch real messages
    print("Fetching real messages from chat.db...")
    messages = fetch_real_messages(args.limit)
    if not messages:
        print("❌ No messages found")
        return

    print(f"✅ Found {len(messages)} messages")
    print()

    # Categorize
    categorized = {"acknowledge": [], "closing": [], "other": []}
    for m in messages:
        cat = categorize_message(m["text"])
        categorized[cat].append(m)

    print("Message Distribution:")
    for cat, msgs in categorized.items():
        print(f"  {cat:12}: {len(msgs)} messages")
    print()

    # Judge templates if requested
    if args.judge:
        print("Judging template appropriateness with Cerebras...")
        print("(This will take ~2 seconds per evaluation due to rate limits)")
        print()

        client = get_judge_client()

        # Sample a few from each category
        samples = []
        for cat in ["acknowledge", "closing"]:
            if categorized[cat]:
                samples.extend([(m, cat) for m in categorized[cat][:5]])

        results = []
        for msg, cat in samples:
            templates = ACKNOWLEDGE_TEMPLATES if cat == "acknowledge" else CLOSING_TEMPLATES
            template = random.choice(templates)

            result = judge_template_appropriateness(msg["text"], template, client)
            results.append(
                {"incoming": msg["text"], "category": cat, "template": template, **result}
            )

            print(f"Message: '{msg['text']}'")
            print(f"Template: '{template}'")
            print(f"Score: {result['score']}/10 - {result['reasoning']}")
            if result.get("better_alternative"):
                print(f"Better: '{result['better_alternative']}'")
            print()

            time.sleep(2.1)  # Rate limit

        # Summary
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"Average Template Score: {avg_score:.1f}/10")
        print()

    # Analyze user style
    print("Analyzing your actual reply style...")
    style = analyze_user_style(messages)

    print("\nYour Acknowledgment Patterns:")
    if style["ack_examples"]:
        for text, count in style["ack_examples"][:5]:
            print(f"  '{text}' (seen {count}x)")
    else:
        print("  No clear patterns found")

    print("\nYour Closing Patterns:")
    if style["closing_examples"]:
        for text, count in style["closing_examples"][:5]:
            print(f"  '{text}' (seen {count}x)")
    else:
        print("  No clear patterns found")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
Based on the analysis:

1. Check if your templates match your actual style
2. Consider adding emoji if you use them (👍, 🙏)
3. Add abbreviations you actually use ("bet", "alr", etc.)
4. Consider context-aware template selection (not random)

To improve templates:
- Edit jarvis/prompts/constants.py
- Add/remove from ACKNOWLEDGE_TEMPLATES and CLOSING_TEMPLATES
- Test with: uv run python evals/evaluate_templates.py --judge
""")


if __name__ == "__main__":
    main()
