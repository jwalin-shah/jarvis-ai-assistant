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
  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

# noqa: E402
from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES  # noqa: E402


  # noqa: E402
  # noqa: E402
def fetch_real_messages(limit: int = 50) -> list[dict]:  # noqa: E402
    """Fetch real acknowledge/closing messages from chat.db."""  # noqa: E402
    import sqlite3  # noqa: E402
  # noqa: E402
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"  # noqa: E402
    if not chat_db.exists():  # noqa: E402
        print(f"❌ chat.db not found at {chat_db}")  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)  # noqa: E402
        cursor = conn.cursor()  # noqa: E402
  # noqa: E402
        # Look for short messages that might be acks/closings  # noqa: E402
        # Heuristic: short messages (2-15 chars) or common patterns  # noqa: E402
        cursor.execute(  # noqa: E402
            """  # noqa: E402
            SELECT  # noqa: E402
                m.text,  # noqa: E402
                m.date,  # noqa: E402
                c.display_name,  # noqa: E402
                m.is_from_me  # noqa: E402
            FROM message m  # noqa: E402
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id  # noqa: E402
            JOIN chat c ON cmj.chat_id = c.ROWID  # noqa: E402
            WHERE m.text IS NOT NULL  # noqa: E402
              AND length(m.text) <= 20  # noqa: E402
              AND m.text NOT LIKE '%http%'  # noqa: E402
            ORDER BY m.date DESC  # noqa: E402
            LIMIT ?  # noqa: E402
        """,  # noqa: E402
            (limit * 3,),  # noqa: E402
        )  # Get more to filter  # noqa: E402
  # noqa: E402
        messages = []  # noqa: E402
        for row in cursor.fetchall():  # noqa: E402
            text, date, display_name, is_from_me = row  # noqa: E402
            if text and not is_from_me:  # Only incoming messages  # noqa: E402
                messages.append(  # noqa: E402
                    {  # noqa: E402
                        "text": text.strip(),  # noqa: E402
                        "contact": display_name or "Unknown",  # noqa: E402
                    }  # noqa: E402
                )  # noqa: E402
                if len(messages) >= limit:  # noqa: E402
                    break  # noqa: E402
  # noqa: E402
        conn.close()  # noqa: E402
        return messages  # noqa: E402
  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"❌ Error reading chat.db: {e}")  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
  # noqa: E402
def categorize_message(text: str) -> str:  # noqa: E402
    """Simple heuristic to categorize message."""  # noqa: E402
    text_lower = text.lower().strip()  # noqa: E402
  # noqa: E402
    # Closing patterns  # noqa: E402
    closing_words = ["bye", "goodbye", "gn", "night", "ttyl", "later", "peace", "cya"]  # noqa: E402
    if any(word in text_lower for word in closing_words):  # noqa: E402
        return "closing"  # noqa: E402
  # noqa: E402
    # Acknowledge patterns  # noqa: E402
    ack_words = [  # noqa: E402
        "ok",  # noqa: E402
        "okay",  # noqa: E402
        "k",  # noqa: E402
        "yes",  # noqa: E402
        "yeah",  # noqa: E402
        "yep",  # noqa: E402
        "sure",  # noqa: E402
        "sounds good",  # noqa: E402
        "got it",  # noqa: E402
        "thanks",  # noqa: E402
        "ty",  # noqa: E402
        "np",  # noqa: E402
    ]  # noqa: E402
    if any(word in text_lower for word in ack_words) or len(text_lower) <= 5:  # noqa: E402
        return "acknowledge"  # noqa: E402
  # noqa: E402
    return "other"  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_template_appropriateness(message: str, template: str, client) -> dict:  # noqa: E402
    """Use Cerebras judge to evaluate if template is appropriate."""  # noqa: E402
    prompt = f"""You are evaluating if a template response is appropriate for a text message.  # noqa: E402
  # noqa: E402
Incoming message: "{message}"  # noqa: E402
Proposed response: "{template}"  # noqa: E402
  # noqa: E402
Rate this response on a scale of 1-10:  # noqa: E402
- 9-10: Perfect natural response  # noqa: E402
- 7-8: Good response, appropriate  # noqa: E402
- 5-6: Acceptable but could be better  # noqa: E402
- 3-4: Poor, somewhat awkward or inappropriate  # noqa: E402
- 1-2: Very bad, completely wrong  # noqa: E402
  # noqa: E402
Respond with ONLY a JSON object:  # noqa: E402
{{"score": <number>, "reasoning": "<brief explanation>", "better_alternative": "<suggested better response or null>"}}  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        resp = client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=200,  # noqa: E402
        )  # noqa: E402
        content = resp.choices[0].message.content  # noqa: E402
        if "```json" in content:  # noqa: E402
            content = content.split("```json")[1].split("```")[0]  # noqa: E402
        elif "```" in content:  # noqa: E402
            content = content.split("```")[1].split("```")[0]  # noqa: E402
        result = json.loads(content.strip())  # noqa: E402
        return result  # noqa: E402
    except Exception as e:  # noqa: E402
        return {"score": 0, "reasoning": f"Judge error: {e}", "better_alternative": None}  # noqa: E402
  # noqa: E402
  # noqa: E402
def analyze_user_style(messages: list[dict]) -> dict:  # noqa: E402
    """Analyze user's actual reply style from their messages."""  # noqa: E402
    all_texts = [m["text"].lower() for m in messages]  # noqa: E402
  # noqa: E402
    # Count common patterns  # noqa: E402
    from collections import Counter  # noqa: E402
  # noqa: E402
    # Find acknowledgments they use  # noqa: E402
    ack_patterns = [  # noqa: E402
        "ok",  # noqa: E402
        "okay",  # noqa: E402
        "k",  # noqa: E402
        "sounds good",  # noqa: E402
        "got it",  # noqa: E402
        "thanks",  # noqa: E402
        "ty",  # noqa: E402
        "np",  # noqa: E402
        "bet",  # noqa: E402
        "cool",  # noqa: E402
        "sure",  # noqa: E402
        "yep",  # noqa: E402
        "yeah",  # noqa: E402
        "alright",  # noqa: E402
        "👍",  # noqa: E402
        "🙏",  # noqa: E402
    ]  # noqa: E402
    found_acks = []  # noqa: E402
    for text in all_texts:  # noqa: E402
        for pattern in ack_patterns:  # noqa: E402
            if pattern in text:  # noqa: E402
                found_acks.append(text)  # noqa: E402
                break  # noqa: E402
  # noqa: E402
    # Find closings they use  # noqa: E402
    closing_patterns = ["bye", "later", "ttyl", "gn", "night", "peace", "cya", "talk soon"]  # noqa: E402
    found_closings = []  # noqa: E402
    for text in all_texts:  # noqa: E402
        for pattern in closing_patterns:  # noqa: E402
            if pattern in text:  # noqa: E402
                found_closings.append(text)  # noqa: E402
                break  # noqa: E402
  # noqa: E402
    return {  # noqa: E402
        "ack_examples": Counter(found_acks).most_common(10),  # noqa: E402
        "closing_examples": Counter(found_closings).most_common(10),  # noqa: E402
        "avg_ack_length": sum(len(t) for t in found_acks) / len(found_acks) if found_acks else 0,  # noqa: E402
        "avg_closing_length": sum(len(t) for t in found_closings) / len(found_closings)  # noqa: E402
        if found_closings  # noqa: E402
        else 0,  # noqa: E402
    }  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    parser = argparse.ArgumentParser(description="Evaluate template responses")  # noqa: E402
    parser.add_argument("--limit", type=int, default=30, help="Number of messages to evaluate")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--judge", action="store_true", help="Use Cerebras judge (slower but better)"  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("TEMPLATE EVALUATION")  # noqa: E402
    print("=" * 70)  # noqa: E402
    print()  # noqa: E402
    print("Current Templates:")  # noqa: E402
    print(f"  ACKNOWLEDGE ({len(ACKNOWLEDGE_TEMPLATES)}): {ACKNOWLEDGE_TEMPLATES}")  # noqa: E402
    print(f"  CLOSING ({len(CLOSING_TEMPLATES)}): {CLOSING_TEMPLATES}")  # noqa: E402
    print()  # noqa: E402
  # noqa: E402
    # Fetch real messages  # noqa: E402
    print("Fetching real messages from chat.db...")  # noqa: E402
    messages = fetch_real_messages(args.limit)  # noqa: E402
    if not messages:  # noqa: E402
        print("❌ No messages found")  # noqa: E402
        return  # noqa: E402
  # noqa: E402
    print(f"✅ Found {len(messages)} messages")  # noqa: E402
    print()  # noqa: E402
  # noqa: E402
    # Categorize  # noqa: E402
    categorized = {"acknowledge": [], "closing": [], "other": []}  # noqa: E402
    for m in messages:  # noqa: E402
        cat = categorize_message(m["text"])  # noqa: E402
        categorized[cat].append(m)  # noqa: E402
  # noqa: E402
    print("Message Distribution:")  # noqa: E402
    for cat, msgs in categorized.items():  # noqa: E402
        print(f"  {cat:12}: {len(msgs)} messages")  # noqa: E402
    print()  # noqa: E402
  # noqa: E402
    # Judge templates if requested  # noqa: E402
    if args.judge:  # noqa: E402
        print("Judging template appropriateness with Cerebras...")  # noqa: E402
        print("(This will take ~2 seconds per evaluation due to rate limits)")  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
        client = get_judge_client()  # noqa: E402
  # noqa: E402
        # Sample a few from each category  # noqa: E402
        samples = []  # noqa: E402
        for cat in ["acknowledge", "closing"]:  # noqa: E402
            if categorized[cat]:  # noqa: E402
                samples.extend([(m, cat) for m in categorized[cat][:5]])  # noqa: E402
  # noqa: E402
        results = []  # noqa: E402
        for msg, cat in samples:  # noqa: E402
            templates = ACKNOWLEDGE_TEMPLATES if cat == "acknowledge" else CLOSING_TEMPLATES  # noqa: E402
            template = random.choice(templates)  # noqa: E402
  # noqa: E402
            result = judge_template_appropriateness(msg["text"], template, client)  # noqa: E402
            results.append(  # noqa: E402
                {"incoming": msg["text"], "category": cat, "template": template, **result}  # noqa: E402
            )  # noqa: E402
  # noqa: E402
            print(f"Message: '{msg['text']}'")  # noqa: E402
            print(f"Template: '{template}'")  # noqa: E402
            print(f"Score: {result['score']}/10 - {result['reasoning']}")  # noqa: E402
            if result.get("better_alternative"):  # noqa: E402
                print(f"Better: '{result['better_alternative']}'")  # noqa: E402
            print()  # noqa: E402
  # noqa: E402
            time.sleep(2.1)  # Rate limit  # noqa: E402
  # noqa: E402
        # Summary  # noqa: E402
        avg_score = sum(r["score"] for r in results) / len(results)  # noqa: E402
        print(f"Average Template Score: {avg_score:.1f}/10")  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
    # Analyze user style  # noqa: E402
    print("Analyzing your actual reply style...")  # noqa: E402
    style = analyze_user_style(messages)  # noqa: E402
  # noqa: E402
    print("\nYour Acknowledgment Patterns:")  # noqa: E402
    if style["ack_examples"]:  # noqa: E402
        for text, count in style["ack_examples"][:5]:  # noqa: E402
            print(f"  '{text}' (seen {count}x)")  # noqa: E402
    else:  # noqa: E402
        print("  No clear patterns found")  # noqa: E402
  # noqa: E402
    print("\nYour Closing Patterns:")  # noqa: E402
    if style["closing_examples"]:  # noqa: E402
        for text, count in style["closing_examples"][:5]:  # noqa: E402
            print(f"  '{text}' (seen {count}x)")  # noqa: E402
    else:  # noqa: E402
        print("  No clear patterns found")  # noqa: E402
  # noqa: E402
    # Recommendations  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("RECOMMENDATIONS")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    print("""  # noqa: E402
Based on the analysis:  # noqa: E402
  # noqa: E402
1. Check if your templates match your actual style  # noqa: E402
2. Consider adding emoji if you use them (👍, 🙏)  # noqa: E402
3. Add abbreviations you actually use ("bet", "alr", etc.)  # noqa: E402
4. Consider context-aware template selection (not random)  # noqa: E402
  # noqa: E402
To improve templates:  # noqa: E402
- Edit jarvis/prompts/constants.py  # noqa: E402
- Add/remove from ACKNOWLEDGE_TEMPLATES and CLOSING_TEMPLATES  # noqa: E402
- Test with: uv run python evals/evaluate_templates.py --judge  # noqa: E402
""")  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
