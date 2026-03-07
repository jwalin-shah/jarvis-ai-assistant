#!/usr/bin/env python3
"""Comprehensive batched evaluation of semantic templates using Cerebras judge.

This script:
1. Fetches real messages from chat.db
2. Tests all 91 semantic templates against them
3. Batches Cerebras API calls (10 evaluations per call)
4. Analyzes hit rates, coverage, and gaps
5. Provides actionable recommendations

Usage:
    uv run python evals/evaluate_semantic_templates.py [--limit 100] [--batch-size 10]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

from models.template_defaults import get_minimal_fallback_templates  # noqa: E402
from models.templates import ResponseTemplate, TemplateMatcher  # noqa: E402


def fetch_real_messages(limit: int = 100) -> list[dict]:
    """Fetch diverse real messages from chat.db."""
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"
    if not chat_db.exists():
        print(f"❌ chat.db not found at {chat_db}")  # noqa: E501
        return []

    try:
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get diverse messages of different lengths
        cursor.execute(
            """
            SELECT
                m.text,
                m.date,
                c.display_name,
                m.is_from_me,
                length(m.text) as msg_len
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL
              AND m.text NOT LIKE '%http%'
              AND m.text NOT LIKE '%http%'
              AND length(m.text) <= 100
              AND m.is_from_me = 0
            ORDER BY RANDOM()
            LIMIT ?
        """,
            (limit * 2,),
        )

        messages = []
        for row in cursor.fetchall():
            text, date, display_name, is_from_me, msg_len = row
            if text and len(messages) < limit:
                messages.append(
                    {
                        "text": text.strip(),
                        "contact": display_name or "Unknown",
                        "length": msg_len,
                    }
                )

        conn.close()
        return messages

    except Exception as e:
        print(f"❌ Error reading chat.db: {e}")  # noqa: E501
        return []


def get_all_templates() -> list[ResponseTemplate]:
    """Get all 91 semantic templates."""
    return get_minimal_fallback_templates()


def categorize_template(template: ResponseTemplate) -> str:
    """Categorize template by its purpose."""
    name = template.name.lower()
    if "group_" in name:
        return "group"
    elif any(x in name for x in ["summarize", "find_", "search", "show"]):
        return "assistant_query"
    elif any(x in name for x in ["quick_", "thank_you", "acknowledgment"]):
        return "quick_response"
    elif any(x in name for x in ["meeting", "schedule", "time", "plan"]):
        return "scheduling"
    else:
        return "general"


def batch_judge_templates(evaluations: list[dict], client, batch_size: int = 10) -> list[dict]:
    """Judge multiple template-message pairs in a single API call."""
    if not client:
        return []

    results = []

    # Process in batches
    for i in range(0, len(evaluations), batch_size):
        batch = evaluations[i : i + batch_size]

        # Build batched prompt
        eval_items = []
        for idx, eval_item in enumerate(batch):
            eval_items.append(f"""
{idx + 1}. Message: "{eval_item["message"]}"
   Template Response: "{eval_item["response"]}"
   Template Name: {eval_item["template_name"]}""")

        prompt = f"""You are evaluating if template responses are appropriate for text messages.

Rate each response on a scale of 1-10:
- 9-10: Perfect natural response, sounds like a real person texting
- 7-8: Good response, appropriate for the context
- 5-6: Acceptable but could be better
- 3-4: Poor, somewhat awkward or inappropriate
- 1-2: Very bad, completely wrong tone or content

{chr(10).join(eval_items)}

Respond with ONLY a JSON array in this exact format:
[

  ... (one object for each evaluation)
]
"""

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = resp.choices[0].message.content

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            batch_results = json.loads(content.strip())

            # Combine with original eval data
            for eval_item, result in zip(batch, batch_results):
                results.append(
                    {
                        **eval_item,
                        "score": result.get("score", 0),
                        "reasoning": result.get("reasoning", "No reasoning"),

                    }
                )

            print(  # noqa: E501
                f"  ✓ Batch {i // batch_size + 1}/{(len(evaluations) + batch_size - 1) // batch_size} complete"  # noqa: E501
            )

            # Rate limit: 30 req/min = 2 sec between calls
            if i + batch_size < len(evaluations):
                time.sleep(2.1)

        except Exception as e:
            print(f"  ⚠ Batch {i // batch_size + 1} failed: {e}")  # noqa: E501
            # Add failed items with 0 score
            for eval_item in batch:
                results.append(
                    {
                        **eval_item,
                        "score": 0,
                        "reasoning": f"API error: {e}",

                    }
                )

    return results


def find_template_matches(messages: list[dict], templates: list[ResponseTemplate]) -> dict:
    """Find which templates would match real messages."""
    matcher = TemplateMatcher(templates)

    matches = []
    unmatched = []

    for msg in messages:
        text = msg["text"]
        match = matcher.match(text, track_analytics=False)

        if match:
            matches.append(
                {
                    "message": text,
                    "template_name": match.template.name,
                    "matched_pattern": match.matched_pattern,
                    "similarity": match.similarity,
                    "response": match.template.response,
                    "category": categorize_template(match.template),
                }
            )
        else:
            unmatched.append(text)

    return {
        "matches": matches,
        "unmatched": unmatched,
        "hit_rate": len(matches) / len(messages) if messages else 0,
    }


def analyze_coverage(templates: list[ResponseTemplate]) -> dict:
    """Analyze template coverage and diversity."""
    categories = defaultdict(list)
    pattern_counts = defaultdict(int)

    for template in templates:
        cat = categorize_template(template)
        categories[cat].append(template)
        pattern_counts[cat] += len(template.patterns)

    return {
        "by_category": {
            cat: {
                "count": len(temps),
                "total_patterns": pattern_counts[cat],
                "avg_patterns_per_template": pattern_counts[cat] / len(temps) if temps else 0,
            }
            for cat, temps in categories.items()
        },
        "total_templates": len(templates),
        "total_patterns": sum(len(t.patterns) for t in templates),
    }


def generate_report(
    match_results: dict,
    judge_results: list[dict],
    coverage: dict,
    templates: list[ResponseTemplate],
) -> None:
    """Generate comprehensive evaluation report."""
    print("\n" + "=" * 80)  # noqa: E501
    print("SEMANTIC TEMPLATE EVALUATION REPORT")  # noqa: E501
    print("=" * 80)  # noqa: E501

    # 1. Coverage Analysis
    print("\n📊 TEMPLATE COVERAGE")  # noqa: E501
    print("-" * 80)  # noqa: E501
    print(f"Total Templates: {coverage['total_templates']}")  # noqa: E501
    print(f"Total Patterns: {coverage['total_patterns']}")  # noqa: E501
    print("\nBy Category:")  # noqa: E501
    for cat, stats in sorted(coverage["by_category"].items()):
        print(  # noqa: E501
            f"  {cat:20}: {stats['count']:3} templates, {stats['total_patterns']:3} patterns "
            f"(avg {stats['avg_patterns_per_template']:.1f} patterns/template)"
        )

    # 2. Hit Rate Analysis
    print("\n🎯 MATCH PERFORMANCE")  # noqa: E501
    print("-" * 80)  # noqa: E501
    total_msgs = len(match_results["matches"]) + len(match_results["unmatched"])
    print(f"Messages Tested: {total_msgs}")  # noqa: E501
    print(f"Templates Matched: {len(match_results['matches'])}")  # noqa: E501
    print(f"Hit Rate: {match_results['hit_rate'] * 100:.1f}%")  # noqa: E501

    if match_results["matches"]:
        similarities = [m["similarity"] for m in match_results["matches"]]
        print(f"Avg Similarity: {sum(similarities) / len(similarities):.3f}")  # noqa: E501
        print(f"Min Similarity: {min(similarities):.3f}")  # noqa: E501
        print(f"Max Similarity: {max(similarities):.3f}")  # noqa: E501

    # 3. Category Performance
    print("\n📈 CATEGORY BREAKDOWN")  # noqa: E501
    print("-" * 80)  # noqa: E501
    category_matches = defaultdict(list)
    for match in match_results["matches"]:
        category_matches[match["category"]].append(match)

    for cat, matches in sorted(category_matches.items()):
        cat_hit_rate = len(matches) / total_msgs
        avg_sim = sum(m["similarity"] for m in matches) / len(matches) if matches else 0
        print(  # noqa: E501
            f"  {cat:20}: {len(matches):3} hits, {cat_hit_rate * 100:5.1f}% hit rate, "
            f"{avg_sim:.3f} avg similarity"
        )

    # 4. Quality Judgment (Cerebras)
    if judge_results:
        print("\n⚖️  QUALITY ASSESSMENT (Cerebras Judge)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        scores = [r["score"] for r in judge_results if r["score"] > 0]
        if scores:
            print(f"Evaluations: {len(scores)}")  # noqa: E501
            print(f"Average Score: {sum(scores) / len(scores):.1f}/10")  # noqa: E501
            print("Score Distribution:")  # noqa: E501

            buckets = {
                "9-10 (Excellent)": 0,
                "7-8 (Good)": 0,
                "5-6 (Acceptable)": 0,
                "3-4 (Poor)": 0,
                "1-2 (Very Bad)": 0,
            }
            for score in scores:
                if score >= 9:
                    buckets["9-10 (Excellent)"] += 1
                elif score >= 7:
                    buckets["7-8 (Good)"] += 1
                elif score >= 5:
                    buckets["5-6 (Acceptable)"] += 1
                elif score >= 3:
                    buckets["3-4 (Poor)"] += 1
                else:
                    buckets["1-2 (Very Bad)"] += 1

            for bucket, count in buckets.items():
                pct = count / len(scores) * 100
                print(f"    {bucket:20}: {count:3} ({pct:5.1f}%)")  # noqa: E501

    # 5. Top Performers
    if match_results["matches"]:
        print("\n🏆 TOP MATCHING TEMPLATES")  # noqa: E501
        print("-" * 80)  # noqa: E501
        template_hit_counts = defaultdict(lambda: {"count": 0, "avg_sim": []})
        for match in match_results["matches"]:
            name = match["template_name"]
            template_hit_counts[name]["count"] += 1
            template_hit_counts[name]["avg_sim"].append(match["similarity"])

        top_templates = sorted(
            template_hit_counts.items(), key=lambda x: x[1]["count"], reverse=True
        )[:10]

        for name, data in top_templates:
            avg_sim = sum(data["avg_sim"]) / len(data["avg_sim"])
            print(f"  {name:40}: {data['count']:3} hits, {avg_sim:.3f} avg similarity")  # noqa: E501

    # 6. Problem Areas
    if judge_results:
        print("\n⚠️  TEMPLATES NEEDING IMPROVEMENT (Score < 6)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        poor_performers = [r for r in judge_results if r["score"] > 0 and r["score"] < 6]
        if poor_performers:
            for result in poor_performers[:10]:
                print(f"  Template: {result['template_name']}")  # noqa: E501
                print(f"  Message: '{result['message'][:50]}...'")  # noqa: E501
                print(f"  Score: {result['score']}/10 - {result['reasoning']}")  # noqa: E501


                print()  # noqa: E501
        else:
            print("  ✓ All evaluated templates scored well!")  # noqa: E501

    # 7. Unmatched Messages
    if match_results["unmatched"]:
        print("\n❌ UNMATCHED MESSAGES (Potential Gaps)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        print(f"Count: {len(match_results['unmatched'])}")  # noqa: E501
        print("\nSample unmatched messages (potential new patterns):")  # noqa: E501
        for msg in match_results["unmatched"][:15]:
            print(f"  - '{msg[:60]}...'" if len(msg) > 60 else f"  - '{msg}'")  # noqa: E501

    # 8. Recommendations
    print("\n💡 RECOMMENDATIONS")  # noqa: E501
    print("-" * 80)  # noqa: E501

    recommendations = []

    if match_results["hit_rate"] < 0.3:
        recommendations.append(
            "LOW HIT RATE: Consider adding more patterns to existing templates or "
            "creating new templates for common unmatched messages"
        )

    if judge_results:
        avg_score = sum(r["score"] for r in judge_results if r["score"] > 0) / len(
            [r for r in judge_results if r["score"] > 0]
        )
        if avg_score < 7:
            recommendations.append(
                f"LOW QUALITY SCORE ({avg_score:.1f}/10): Many responses sound unnatural. "
                "Consider revising template responses to be more casual and brief."
            )

    # Check for category gaps
    if "group" not in category_matches and len(templates) > 50:
        recommendations.append(
            "NO GROUP MATCHES: Group chat templates aren't being triggered. "
            "Check group size detection or add more group-specific patterns."
        )

    # Check pattern diversity
    for cat, stats in coverage["by_category"].items():
        if stats["avg_patterns_per_template"] < 3:
            recommendations.append(
                f"LOW PATTERN DIVERSITY in {cat}: Avg {stats['avg_patterns_per_template']:.1f} "
                "patterns per template. Add more variation to improve match rates."
            )

    if not recommendations:
        recommendations.append("✓ Templates are performing well overall!")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")  # noqa: E501

    # 9. Quick Wins
    print("\n🚀 QUICK WINS")  # noqa: E501
    print("-" * 80)  # noqa: E501
    print("To improve templates immediately:")  # noqa: E501
    print("  1. Add the unmatched messages above as patterns to existing templates")  # noqa: E501
    print("  2. Review poor-scoring templates and make responses more casual/brief")  # noqa: E501
    print("  3. Add emoji to templates where appropriate (👍, 😊, 🎉)")  # noqa: E501
    print("  4. Test with: uv run python evals/evaluate_semantic_templates.py --limit 200")  # noqa: E501
    print("\n" + "=" * 80)  # noqa: E501


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic templates with Cerebras judge")
    parser.add_argument("--limit", type=int, default=100, help="Number of messages to test")
    parser.add_argument("--batch-size", type=int, default=10, help="API calls per batch")
    parser.add_argument("--skip-judge", action="store_true", help="Skip Cerebras judging")
    args = parser.parse_args()

    print("=" * 80)  # noqa: E501
    print("SEMANTIC TEMPLATE EVALUATION")  # noqa: E501
    print("=" * 80)  # noqa: E501

    # 1. Load templates
    print("\n📦 Loading templates...")  # noqa: E501
    templates = get_all_templates()
    print(f"✓ Loaded {len(templates)} semantic templates")  # noqa: E501

    # 2. Fetch messages
    print(f"\n📱 Fetching {args.limit} real messages from chat.db...")  # noqa: E501
    messages = fetch_real_messages(args.limit)
    if not messages:
        print("❌ No messages found")  # noqa: E501
        return
    print(f"✓ Fetched {len(messages)} messages")  # noqa: E501

    # 3. Find matches
    print("\n🔍 Testing templates against messages...")  # noqa: E501
    match_results = find_template_matches(messages, templates)
    print(f"✓ Found {len(match_results['matches'])} template matches")  # noqa: E501

    # 4. Analyze coverage
    print("\n📊 Analyzing template coverage...")  # noqa: E501
    coverage = analyze_coverage(templates)

    # 5. Judge quality (batched)
    judge_results = []
    if not args.skip_judge:
        client = get_judge_client()
        if client:
            # Prepare evaluations for matched templates
            evaluations = []
            for match in match_results["matches"][:50]:  # Judge top 50 matches
                evaluations.append(
                    {
                        "message": match["message"],
                        "template_name": match["template_name"],
                        "response": match["response"],
                        "similarity": match["similarity"],
                    }
                )

            if evaluations:
                print(f"\n⚖️  Judging {len(evaluations)} template responses with Cerebras...")  # noqa: E501
                print(  # noqa: E501
                    f"   (Batch size: {args.batch_size}, Estimated time: {len(evaluations) // args.batch_size * 2.1:.0f}s)"  # noqa: E501
                )
                judge_results = batch_judge_templates(evaluations, client, args.batch_size)
        else:
            print("\n⚠️  Cerebras client not configured (set CEREBRAS_API_KEY)")  # noqa: E501

    # 6. Generate report
    generate_report(match_results, judge_results, coverage, templates)

    # Save detailed results
    output_file = PROJECT_ROOT / "evals" / "template_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "match_results": match_results,
                "judge_results": judge_results,
                "coverage": coverage,
                "config": {
                    "messages_tested": len(messages),
                    "templates_loaded": len(templates),
                    "batch_size": args.batch_size,
                },
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n💾 Detailed results saved to: {output_file}")  # noqa: E501


if __name__ == "__main__":
    main()
