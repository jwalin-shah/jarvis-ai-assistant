#!/usr/bin/env python3  # noqa: E501
"""Evaluate template responses against real iMessage data.  # noqa: E501
  # noqa: E501
This script:  # noqa: E501
1. Fetches real acknowledge/closing messages from chat.db  # noqa: E501
2. Shows what templates would be sent  # noqa: E501
3. Evaluates if templates are appropriate using Cerebras judge  # noqa: E501
4. Suggests better templates based on user's actual reply style  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/evaluate_templates.py [--limit 50]  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import random  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E501

# noqa: E501
from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
def fetch_real_messages(limit: int = 50) -> list[dict]:  # noqa: E501
    """Fetch real acknowledge/closing messages from chat.db."""  # noqa: E501
    import sqlite3  # noqa: E501
  # noqa: E501
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"  # noqa: E501
    if not chat_db.exists():  # noqa: E501
        print(f"❌ chat.db not found at {chat_db}")  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)  # noqa: E501
        cursor = conn.cursor()  # noqa: E501
  # noqa: E501
        # Look for short messages that might be acks/closings  # noqa: E501
        # Heuristic: short messages (2-15 chars) or common patterns  # noqa: E501
        cursor.execute(  # noqa: E501
            """  # noqa: E501
            SELECT  # noqa: E501
                m.text,  # noqa: E501
                m.date,  # noqa: E501
                c.display_name,  # noqa: E501
                m.is_from_me  # noqa: E501
            FROM message m  # noqa: E501
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id  # noqa: E501
            JOIN chat c ON cmj.chat_id = c.ROWID  # noqa: E501
            WHERE m.text IS NOT NULL  # noqa: E501
              AND length(m.text) <= 20  # noqa: E501
              AND m.text NOT LIKE '%http%'  # noqa: E501
            ORDER BY m.date DESC  # noqa: E501
            LIMIT ?  # noqa: E501
        """,  # noqa: E501
            (limit * 3,),  # noqa: E501
        )  # Get more to filter  # noqa: E501
  # noqa: E501
        messages = []  # noqa: E501
        for row in cursor.fetchall():  # noqa: E501
            text, date, display_name, is_from_me = row  # noqa: E501
            if text and not is_from_me:  # Only incoming messages  # noqa: E501
                messages.append(  # noqa: E501
                    {  # noqa: E501
                        "text": text.strip(),  # noqa: E501
                        "contact": display_name or "Unknown",  # noqa: E501
                    }  # noqa: E501
                )  # noqa: E501
                if len(messages) >= limit:  # noqa: E501
                    break  # noqa: E501
  # noqa: E501
        conn.close()  # noqa: E501
        return messages  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"❌ Error reading chat.db: {e}")  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
  # noqa: E501
def categorize_message(text: str) -> str:  # noqa: E501
    """Simple heuristic to categorize message."""  # noqa: E501
    text_lower = text.lower().strip()  # noqa: E501
  # noqa: E501
    # Closing patterns  # noqa: E501
    closing_words = ["bye", "goodbye", "gn", "night", "ttyl", "later", "peace", "cya"]  # noqa: E501
    if any(word in text_lower for word in closing_words):  # noqa: E501
        return "closing"  # noqa: E501
  # noqa: E501
    # Acknowledge patterns  # noqa: E501
    ack_words = [  # noqa: E501
        "ok",  # noqa: E501
        "okay",  # noqa: E501
        "k",  # noqa: E501
        "yes",  # noqa: E501
        "yeah",  # noqa: E501
        "yep",  # noqa: E501
        "sure",  # noqa: E501
        "sounds good",  # noqa: E501
        "got it",  # noqa: E501
        "thanks",  # noqa: E501
        "ty",  # noqa: E501
        "np",  # noqa: E501
    ]  # noqa: E501
    if any(word in text_lower for word in ack_words) or len(text_lower) <= 5:  # noqa: E501
        return "acknowledge"  # noqa: E501
  # noqa: E501
    return "other"  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_template_appropriateness(message: str, template: str, client) -> dict:  # noqa: E501
    """Use Cerebras judge to evaluate if template is appropriate."""  # noqa: E501
    prompt = f"""You are evaluating if a template response is appropriate for a text message.  # noqa: E501
  # noqa: E501
Incoming message: "{message}"  # noqa: E501
Proposed response: "{template}"  # noqa: E501
  # noqa: E501
Rate this response on a scale of 1-10:  # noqa: E501
- 9-10: Perfect natural response  # noqa: E501
- 7-8: Good response, appropriate  # noqa: E501
- 5-6: Acceptable but could be better  # noqa: E501
- 3-4: Poor, somewhat awkward or inappropriate  # noqa: E501
- 1-2: Very bad, completely wrong  # noqa: E501
  # noqa: E501
Respond with ONLY a JSON object:  # noqa: E501
{{"score": <number>, "reasoning": "<brief explanation>", "better_alternative": "<suggested better response or null>"}}  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=200,  # noqa: E501
        )  # noqa: E501
        content = resp.choices[0].message.content  # noqa: E501
        if "```json" in content:  # noqa: E501
            content = content.split("```json")[1].split("```")[0]  # noqa: E501
        elif "```" in content:  # noqa: E501
            content = content.split("```")[1].split("```")[0]  # noqa: E501
        result = json.loads(content.strip())  # noqa: E501
        return result  # noqa: E501
    except Exception as e:  # noqa: E501
        return {"score": 0, "reasoning": f"Judge error: {e}", "better_alternative": None}  # noqa: E501
  # noqa: E501
  # noqa: E501
def analyze_user_style(messages: list[dict]) -> dict:  # noqa: E501
    """Analyze user's actual reply style from their messages."""  # noqa: E501
    all_texts = [m["text"].lower() for m in messages]  # noqa: E501
  # noqa: E501
    # Count common patterns  # noqa: E501
    from collections import Counter  # noqa: E501
  # noqa: E501
    # Find acknowledgments they use  # noqa: E501
    ack_patterns = [  # noqa: E501
        "ok",  # noqa: E501
        "okay",  # noqa: E501
        "k",  # noqa: E501
        "sounds good",  # noqa: E501
        "got it",  # noqa: E501
        "thanks",  # noqa: E501
        "ty",  # noqa: E501
        "np",  # noqa: E501
        "bet",  # noqa: E501
        "cool",  # noqa: E501
        "sure",  # noqa: E501
        "yep",  # noqa: E501
        "yeah",  # noqa: E501
        "alright",  # noqa: E501
        "👍",  # noqa: E501
        "🙏",  # noqa: E501
    ]  # noqa: E501
    found_acks = []  # noqa: E501
    for text in all_texts:  # noqa: E501
        for pattern in ack_patterns:  # noqa: E501
            if pattern in text:  # noqa: E501
                found_acks.append(text)  # noqa: E501
                break  # noqa: E501
  # noqa: E501
    # Find closings they use  # noqa: E501
    closing_patterns = ["bye", "later", "ttyl", "gn", "night", "peace", "cya", "talk soon"]  # noqa: E501
    found_closings = []  # noqa: E501
    for text in all_texts:  # noqa: E501
        for pattern in closing_patterns:  # noqa: E501
            if pattern in text:  # noqa: E501
                found_closings.append(text)  # noqa: E501
                break  # noqa: E501
  # noqa: E501
    return {  # noqa: E501
        "ack_examples": Counter(found_acks).most_common(10),  # noqa: E501
        "closing_examples": Counter(found_closings).most_common(10),  # noqa: E501
        "avg_ack_length": sum(len(t) for t in found_acks) / len(found_acks) if found_acks else 0,  # noqa: E501
        "avg_closing_length": sum(len(t) for t in found_closings) / len(found_closings)  # noqa: E501
        if found_closings  # noqa: E501
        else 0,  # noqa: E501
    }  # noqa: E501
  # noqa: E501
  # noqa: E501
def main():  # noqa: E501
    parser = argparse.ArgumentParser(description="Evaluate template responses")  # noqa: E501
    parser.add_argument("--limit", type=int, default=30, help="Number of messages to evaluate")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge", action="store_true", help="Use Cerebras judge (slower but better)"  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("TEMPLATE EVALUATION")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print()  # noqa: E501
    print("Current Templates:")  # noqa: E501
    print(f"  ACKNOWLEDGE ({len(ACKNOWLEDGE_TEMPLATES)}): {ACKNOWLEDGE_TEMPLATES}")  # noqa: E501
    print(f"  CLOSING ({len(CLOSING_TEMPLATES)}): {CLOSING_TEMPLATES}")  # noqa: E501
    print()  # noqa: E501
  # noqa: E501
    # Fetch real messages  # noqa: E501
    print("Fetching real messages from chat.db...")  # noqa: E501
    messages = fetch_real_messages(args.limit)  # noqa: E501
    if not messages:  # noqa: E501
        print("❌ No messages found")  # noqa: E501
        return  # noqa: E501
  # noqa: E501
    print(f"✅ Found {len(messages)} messages")  # noqa: E501
    print()  # noqa: E501
  # noqa: E501
    # Categorize  # noqa: E501
    categorized = {"acknowledge": [], "closing": [], "other": []}  # noqa: E501
    for m in messages:  # noqa: E501
        cat = categorize_message(m["text"])  # noqa: E501
        categorized[cat].append(m)  # noqa: E501
  # noqa: E501
    print("Message Distribution:")  # noqa: E501
    for cat, msgs in categorized.items():  # noqa: E501
        print(f"  {cat:12}: {len(msgs)} messages")  # noqa: E501
    print()  # noqa: E501
  # noqa: E501
    # Judge templates if requested  # noqa: E501
    if args.judge:  # noqa: E501
        print("Judging template appropriateness with Cerebras...")  # noqa: E501
        print("(This will take ~2 seconds per evaluation due to rate limits)")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
        client = get_judge_client()  # noqa: E501
  # noqa: E501
        # Sample a few from each category  # noqa: E501
        samples = []  # noqa: E501
        for cat in ["acknowledge", "closing"]:  # noqa: E501
            if categorized[cat]:  # noqa: E501
                samples.extend([(m, cat) for m in categorized[cat][:5]])  # noqa: E501
  # noqa: E501
        results = []  # noqa: E501
        for msg, cat in samples:  # noqa: E501
            templates = ACKNOWLEDGE_TEMPLATES if cat == "acknowledge" else CLOSING_TEMPLATES  # noqa: E501
            template = random.choice(templates)  # noqa: E501
  # noqa: E501
            result = judge_template_appropriateness(msg["text"], template, client)  # noqa: E501
            results.append(  # noqa: E501
                {"incoming": msg["text"], "category": cat, "template": template, **result}  # noqa: E501
            )  # noqa: E501
  # noqa: E501
            print(f"Message: '{msg['text']}'")  # noqa: E501
            print(f"Template: '{template}'")  # noqa: E501
            print(f"Score: {result['score']}/10 - {result['reasoning']}")  # noqa: E501
            if result.get("better_alternative"):  # noqa: E501
                print(f"Better: '{result['better_alternative']}'")  # noqa: E501
            print()  # noqa: E501
  # noqa: E501
            time.sleep(2.1)  # Rate limit  # noqa: E501
  # noqa: E501
        # Summary  # noqa: E501
        avg_score = sum(r["score"] for r in results) / len(results)  # noqa: E501
        print(f"Average Template Score: {avg_score:.1f}/10")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
    # Analyze user style  # noqa: E501
    print("Analyzing your actual reply style...")  # noqa: E501
    style = analyze_user_style(messages)  # noqa: E501
  # noqa: E501
    print("\nYour Acknowledgment Patterns:")  # noqa: E501
    if style["ack_examples"]:  # noqa: E501
        for text, count in style["ack_examples"][:5]:  # noqa: E501
            print(f"  '{text}' (seen {count}x)")  # noqa: E501
    else:  # noqa: E501
        print("  No clear patterns found")  # noqa: E501
  # noqa: E501
    print("\nYour Closing Patterns:")  # noqa: E501
    if style["closing_examples"]:  # noqa: E501
        for text, count in style["closing_examples"][:5]:  # noqa: E501
            print(f"  '{text}' (seen {count}x)")  # noqa: E501
    else:  # noqa: E501
        print("  No clear patterns found")  # noqa: E501
  # noqa: E501
    # Recommendations  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("RECOMMENDATIONS")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    print("""  # noqa: E501
Based on the analysis:  # noqa: E501
  # noqa: E501
1. Check if your templates match your actual style  # noqa: E501
2. Consider adding emoji if you use them (👍, 🙏)  # noqa: E501
3. Add abbreviations you actually use ("bet", "alr", etc.)  # noqa: E501
4. Consider context-aware template selection (not random)  # noqa: E501
  # noqa: E501
To improve templates:  # noqa: E501
- Edit jarvis/prompts/constants.py  # noqa: E501
- Add/remove from ACKNOWLEDGE_TEMPLATES and CLOSING_TEMPLATES  # noqa: E501
- Test with: uv run python evals/evaluate_templates.py --judge  # noqa: E501
""")  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
