#!/usr/bin/env python3
"""Evaluate semantic templates (1.2B) vs legacy regex templates."""

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

from models.template_defaults import get_minimal_fallback_templates  # noqa: E402
from models.templates import ResponseTemplate, TemplateMatcher  # noqa: E402

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


def judge_template_quality(client, incoming: str, template: ResponseTemplate) -> dict:
    """Judge if a template is appropriate for a message."""
    prompt = f"""You are judging the quality of a semantic response template.

Incoming Message: "{incoming}"
Selected Template: "{template.text}"
Category: {template.category}
Tone: {template.tone or 'neutral'}

Respond with ONLY a JSON array in this exact format:
[
  {{"score": <number>, "reasoning": "<brief explanation>",
    "better_alternative": "<suggested better response or null>"}},
]

Criteria:
- 10: Perfect fit, natural, matches context
- 7-9: Good fit, minor tone mismatch
- 4-6: Acceptable but generic or slightly off
- 1-3: Completely inappropriate or nonsensical
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
        return json.loads(content.strip())
    except Exception as e:
        return {"score": 0, "reasoning": str(e), "better_alternative": None}


def batch_judge_templates(
    evaluations: list[dict], client, batch_size: int = 10
) -> list[dict]:
    """Judge templates in batches to save time."""
    results = []

    for i in range(0, len(evaluations), batch_size):
        batch = evaluations[i : i + batch_size]
        prompt = (
            "Judge the appropriateness of these response templates for the given incoming messages.\n\n"  # noqa: E501  # noqa: E501
        )

        for idx, item in enumerate(batch):
            prompt += f"--- CASE {idx + 1} ---\n"
            prompt += f"Incoming: {item['incoming']}\n"
            prompt += f"Template: {item['template_text']}\n"
            prompt += f"Category: {item['category']}\n\n"

        prompt += """Respond with ONLY a JSON array in this exact format:
[
  {"score": <number>, "reasoning": "<brief explanation>",
   "better_alternative": "<suggested better response or null>"},
  ... (one object for each evaluation)
]
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


class LegacyMatcher:
    """Mock legacy regex matcher."""

    def match(self, text: str):
        text = text.lower()
        if "?" in text and ("where" in text or "when" in text):
            return "I'm not sure yet, let me check."
        if "sound" in text and "good" in text:
            return "Great!"
        return None


def run_evaluation(args):
    print(f"Loading {NUM_EXAMPLES} recent messages...")
    messages = load_recent_messages(NUM_EXAMPLES)

    print("Initializing matchers...")
    semantic_matcher = TemplateMatcher()

    # Load default templates
    defaults = get_minimal_fallback_templates()
    for t in defaults:
        semantic_matcher.add_template(t)

    legacy_matcher = LegacyMatcher()

    # Run matching
    print("Matching templates...")
    semantic_hits = 0
    legacy_hits = 0

    evaluations = []

    for msg in tqdm(messages):
        text = msg["text"]

        # Semantic Match
        semantic_result = semantic_matcher.find_match(text, threshold=0.65)

        # Legacy Match
        legacy_result = legacy_matcher.match(text)

        if semantic_result:
            semantic_hits += 1
            evaluations.append({
                "type": "semantic",
                "incoming": text,
                "template_text": semantic_result.text,
                "category": semantic_result.category,
                "score": semantic_result.score
            })

        if legacy_result:
            legacy_hits += 1
            evaluations.append({
                "type": "legacy",
                "incoming": text,
                "template_text": legacy_result,
                "category": "regex_match",
                "score": 1.0
            })

    print("\nMatch Rates:")
    print(f"Semantic: {semantic_hits}/{len(messages)} ({semantic_hits/len(messages)*100:.1f}%)")
    print(f"Legacy:   {legacy_hits}/{len(messages)} ({legacy_hits/len(messages)*100:.1f}%)")

    # Judging
    if not args.no_judge and evaluations:
        client = get_judge_client()
        print(f"\n⚖️  Judging {len(evaluations)} template responses with Cerebras...")
        print(
            f"   (Batch size: {args.batch_size}, Estimated time: "
            f"{len(evaluations) // args.batch_size * 2.1:.0f}s)"
        )
        judge_results = batch_judge_templates(evaluations, client, args.batch_size)

        # Analyze results
        semantic_scores = [r["score"] for r in judge_results if r["type"] == "semantic"]
        legacy_scores = [r["score"] for r in judge_results if r["type"] == "legacy"]

        avg_semantic = np.mean(semantic_scores) if semantic_scores else 0
        avg_legacy = np.mean(legacy_scores) if legacy_scores else 0

        print("\nQuality Scores (1-10):")
        print(f"Semantic: {avg_semantic:.2f} (n={len(semantic_scores)})")
        print(f"Legacy:   {avg_legacy:.2f} (n={len(legacy_scores)})")

        # Show low scores
        print("\nLow Scoring Semantic Matches (<6):")
        for r in judge_results:
            if r["type"] == "semantic" and r["score"] < 6:
                print(f"- Incoming: {r['incoming']}")
                print(f"  Template: {r['template_text']}")
                print(f"  Score: {r['score']} | Reason: {r['reasoning']}")
                print()

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judging")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for judging")
    args = parser.parse_args()

    sys.exit(run_evaluation(args))
