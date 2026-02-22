#!/usr/bin/env python3
"""Evaluate semantic templates vs legacy regex templates."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES  # noqa: E402

# Configuration
NUM_EXAMPLES = 50
DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"


def load_recent_messages(limit: int = 100) -> list[dict]:
    """Load recent incoming messages from real iMessage DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                m.text,
                m.date,
                h.id as sender
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.is_from_me = 0
            AND m.text IS NOT NULL
            AND length(m.text) > 5
            ORDER BY m.date DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"text": r[0], "sender": r[2]} for r in rows]
    except Exception as e:
        print(f"Error reading chat.db: {e}")
        return [
            {"text": "Where are you?", "sender": "+15551234567"},
            {"text": "Sounds good!", "sender": "test@example.com"},
            {"text": "Can we talk later?", "sender": "+15559876543"},
        ] * 10


def batch_judge_templates(evaluations: list[dict], client, batch_size: int = 10) -> list[dict]:
    """Judge templates in batches to save time."""
    results = []

    for i in range(0, len(evaluations), batch_size):
        batch = evaluations[i : i + batch_size]
        prompt = "Judge the appropriateness of these response templates for the given incoming messages.\n\n"  # noqa: E501  # noqa: E501

        for idx, item in enumerate(batch):
            prompt += f"--- CASE {idx + 1} ---\n"
            prompt += f"Incoming: {item['incoming']}\n"
            prompt += f"Template: {item['template_text']}\n"
            prompt += f"Category: {item['category']}\n\n"

        prompt += """Respond with ONLY a JSON object:
{"score": <number>, "reasoning": "<brief explanation>",
 "better_alternative": "<suggested better response or null>"}
"""

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            batch_results = json.loads(content.strip())

            # Map back to original items
            for item, result in zip(batch, batch_results):
                item.update(result)
                results.append(item)

            print(
                f"  ✓ Batch {i // batch_size + 1}/"
                f"{(len(evaluations) + batch_size - 1) // batch_size} complete"
            )
            time.sleep(1.0)  # Rate limit

        except Exception as e:
            print(f"Batch failed: {e}")
            for item in batch:
                item.update({"score": 0, "reasoning": "Batch error", "better_alternative": None})
                results.append(item)

    return results


def run_evaluation(args):
    print(f"Loading {NUM_EXAMPLES} recent messages...")
    messages = load_recent_messages(NUM_EXAMPLES)

    print("Matching templates...")
    evaluations = []

    # Use existing templates
    templates = ACKNOWLEDGE_TEMPLATES + CLOSING_TEMPLATES

    for msg in tqdm(messages):
        text = msg["text"]
        # Basic matching logic for evaluation
        matched = next((t for t in templates if t in text), None)

        if matched:
            evaluations.append(
                {
                    "type": "template",
                    "incoming": text,
                    "template_text": matched,
                    "category": "basic_match",
                    "score": 1.0,
                }
            )

    # Judging
    if not args.no_judge and evaluations:
        client = get_judge_client()
        print(f"\n⚖️  Judging {len(evaluations)} template responses with Cerebras...")
        judge_results = batch_judge_templates(evaluations, client, args.batch_size)

        # Analyze results
        scores = [r["score"] for r in judge_results]
        avg_score = np.mean(scores) if scores else 0

        print("\nQuality Scores (1-10):")
        print(f"Average: {avg_score:.2f} (n={len(scores)})")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judging")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for judging")
    args = parser.parse_args()

    sys.exit(run_evaluation(args))
