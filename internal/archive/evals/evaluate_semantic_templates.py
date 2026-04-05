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
  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

# noqa: E402
from models.template_defaults import get_minimal_fallback_templates  # noqa: E402
from models.templates import ResponseTemplate, TemplateMatcher  # noqa: E402


  # noqa: E402
  # noqa: E402
def fetch_real_messages(limit: int = 100) -> list[dict]:  # noqa: E402
    """Fetch diverse real messages from chat.db."""  # noqa: E402
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"  # noqa: E402
    if not chat_db.exists():  # noqa: E402
        print(f"❌ chat.db not found at {chat_db}")  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)  # noqa: E402
        cursor = conn.cursor()  # noqa: E402
  # noqa: E402
        # Get diverse messages of different lengths  # noqa: E402
        cursor.execute(  # noqa: E402
            """  # noqa: E402
            SELECT  # noqa: E402
                m.text,  # noqa: E402
                m.date,  # noqa: E402
                c.display_name,  # noqa: E402
                m.is_from_me,  # noqa: E402
                length(m.text) as msg_len  # noqa: E402
            FROM message m  # noqa: E402
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id  # noqa: E402
            JOIN chat c ON cmj.chat_id = c.ROWID  # noqa: E402
            WHERE m.text IS NOT NULL  # noqa: E402
              AND m.text NOT LIKE '%http%'  # noqa: E402
              AND m.text NOT LIKE '%http%'  # noqa: E402
              AND length(m.text) <= 100  # noqa: E402
              AND m.is_from_me = 0  # noqa: E402
            ORDER BY RANDOM()  # noqa: E402
            LIMIT ?  # noqa: E402
        """,  # noqa: E402
            (limit * 2,),  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        messages = []  # noqa: E402
        for row in cursor.fetchall():  # noqa: E402
            text, date, display_name, is_from_me, msg_len = row  # noqa: E402
            if text and len(messages) < limit:  # noqa: E402
                messages.append(  # noqa: E402
                    {  # noqa: E402
                        "text": text.strip(),  # noqa: E402
                        "contact": display_name or "Unknown",  # noqa: E402
                        "length": msg_len,  # noqa: E402
                    }  # noqa: E402
                )  # noqa: E402
  # noqa: E402
        conn.close()  # noqa: E402
        return messages  # noqa: E402
  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"❌ Error reading chat.db: {e}")  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
  # noqa: E402
def get_all_templates() -> list[ResponseTemplate]:  # noqa: E402
    """Get all 91 semantic templates."""  # noqa: E402
    return get_minimal_fallback_templates()  # noqa: E402
  # noqa: E402
  # noqa: E402
def categorize_template(template: ResponseTemplate) -> str:  # noqa: E402
    """Categorize template by its purpose."""  # noqa: E402
    name = template.name.lower()  # noqa: E402
    if "group_" in name:  # noqa: E402
        return "group"  # noqa: E402
    elif any(x in name for x in ["summarize", "find_", "search", "show"]):  # noqa: E402
        return "assistant_query"  # noqa: E402
    elif any(x in name for x in ["quick_", "thank_you", "acknowledgment"]):  # noqa: E402
        return "quick_response"  # noqa: E402
    elif any(x in name for x in ["meeting", "schedule", "time", "plan"]):  # noqa: E402
        return "scheduling"  # noqa: E402
    else:  # noqa: E402
        return "general"  # noqa: E402
  # noqa: E402
  # noqa: E402
def batch_judge_templates(evaluations: list[dict], client, batch_size: int = 10) -> list[dict]:  # noqa: E402
    """Judge multiple template-message pairs in a single API call."""  # noqa: E402
    if not client:  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    # Process in batches  # noqa: E402
    for i in range(0, len(evaluations), batch_size):  # noqa: E402
        batch = evaluations[i : i + batch_size]  # noqa: E402
  # noqa: E402
        # Build batched prompt  # noqa: E402
        eval_items = []  # noqa: E402
        for idx, eval_item in enumerate(batch):  # noqa: E402
            eval_items.append(f"""  # noqa: E402
{idx + 1}. Message: "{eval_item["message"]}"  # noqa: E402
   Template Response: "{eval_item["response"]}"  # noqa: E402
   Template Name: {eval_item["template_name"]}""")  # noqa: E402
  # noqa: E402
        prompt = f"""You are evaluating if template responses are appropriate for text messages.  # noqa: E402
  # noqa: E402
Rate each response on a scale of 1-10:  # noqa: E402
- 9-10: Perfect natural response, sounds like a real person texting  # noqa: E402
- 7-8: Good response, appropriate for the context  # noqa: E402
- 5-6: Acceptable but could be better  # noqa: E402
- 3-4: Poor, somewhat awkward or inappropriate  # noqa: E402
- 1-2: Very bad, completely wrong tone or content  # noqa: E402
  # noqa: E402
{chr(10).join(eval_items)}  # noqa: E402
  # noqa: E402
Respond with ONLY a JSON array in this exact format:  # noqa: E402
[  # noqa: E402
  {{"score": <number>, "reasoning": "<brief explanation>", "better_alternative": "<suggested better response or null>"}},  # noqa: E402
  ... (one object for each evaluation)  # noqa: E402
]  # noqa: E402
"""  # noqa: E402
  # noqa: E402
        try:  # noqa: E402
            resp = client.chat.completions.create(  # noqa: E402
                model=JUDGE_MODEL,  # noqa: E402
                messages=[{"role": "user", "content": prompt}],  # noqa: E402
                temperature=0.0,  # noqa: E402
                max_tokens=1000,  # noqa: E402
            )  # noqa: E402
            content = resp.choices[0].message.content  # noqa: E402
  # noqa: E402
            # Extract JSON  # noqa: E402
            if "```json" in content:  # noqa: E402
                content = content.split("```json")[1].split("```")[0]  # noqa: E402
            elif "```" in content:  # noqa: E402
                content = content.split("```")[1].split("```")[0]  # noqa: E402
  # noqa: E402
            batch_results = json.loads(content.strip())  # noqa: E402
  # noqa: E402
            # Combine with original eval data  # noqa: E402
            for eval_item, result in zip(batch, batch_results):  # noqa: E402
                results.append(  # noqa: E402
                    {  # noqa: E402
                        **eval_item,  # noqa: E402
                        "score": result.get("score", 0),  # noqa: E402
                        "reasoning": result.get("reasoning", "No reasoning"),  # noqa: E402
                        "better_alternative": result.get("better_alternative"),  # noqa: E402
                    }  # noqa: E402
                )  # noqa: E402
  # noqa: E402
            print(  # noqa: E402
                f"  ✓ Batch {i // batch_size + 1}/{(len(evaluations) + batch_size - 1) // batch_size} complete"  # noqa: E402
            )  # noqa: E402
  # noqa: E402
            # Rate limit: 30 req/min = 2 sec between calls  # noqa: E402
            if i + batch_size < len(evaluations):  # noqa: E402
                time.sleep(2.1)  # noqa: E402
  # noqa: E402
        except Exception as e:  # noqa: E402
            print(f"  ⚠ Batch {i // batch_size + 1} failed: {e}")  # noqa: E402
            # Add failed items with 0 score  # noqa: E402
            for eval_item in batch:  # noqa: E402
                results.append(  # noqa: E402
                    {  # noqa: E402
                        **eval_item,  # noqa: E402
                        "score": 0,  # noqa: E402
                        "reasoning": f"API error: {e}",  # noqa: E402
                        "better_alternative": None,  # noqa: E402
                    }  # noqa: E402
                )  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
def find_template_matches(messages: list[dict], templates: list[ResponseTemplate]) -> dict:  # noqa: E402
    """Find which templates would match real messages."""  # noqa: E402
    matcher = TemplateMatcher(templates)  # noqa: E402
  # noqa: E402
    matches = []  # noqa: E402
    unmatched = []  # noqa: E402
  # noqa: E402
    for msg in messages:  # noqa: E402
        text = msg["text"]  # noqa: E402
        match = matcher.match(text, track_analytics=False)  # noqa: E402
  # noqa: E402
        if match:  # noqa: E402
            matches.append(  # noqa: E402
                {  # noqa: E402
                    "message": text,  # noqa: E402
                    "template_name": match.template.name,  # noqa: E402
                    "matched_pattern": match.matched_pattern,  # noqa: E402
                    "similarity": match.similarity,  # noqa: E402
                    "response": match.template.response,  # noqa: E402
                    "category": categorize_template(match.template),  # noqa: E402
                }  # noqa: E402
            )  # noqa: E402
        else:  # noqa: E402
            unmatched.append(text)  # noqa: E402
  # noqa: E402
    return {  # noqa: E402
        "matches": matches,  # noqa: E402
        "unmatched": unmatched,  # noqa: E402
        "hit_rate": len(matches) / len(messages) if messages else 0,  # noqa: E402
    }  # noqa: E402
  # noqa: E402
  # noqa: E402
def analyze_coverage(templates: list[ResponseTemplate]) -> dict:  # noqa: E402
    """Analyze template coverage and diversity."""  # noqa: E402
    categories = defaultdict(list)  # noqa: E402
    pattern_counts = defaultdict(int)  # noqa: E402
  # noqa: E402
    for template in templates:  # noqa: E402
        cat = categorize_template(template)  # noqa: E402
        categories[cat].append(template)  # noqa: E402
        pattern_counts[cat] += len(template.patterns)  # noqa: E402
  # noqa: E402
    return {  # noqa: E402
        "by_category": {  # noqa: E402
            cat: {  # noqa: E402
                "count": len(temps),  # noqa: E402
                "total_patterns": pattern_counts[cat],  # noqa: E402
                "avg_patterns_per_template": pattern_counts[cat] / len(temps) if temps else 0,  # noqa: E402
            }  # noqa: E402
            for cat, temps in categories.items()  # noqa: E402
        },  # noqa: E402
        "total_templates": len(templates),  # noqa: E402
        "total_patterns": sum(len(t.patterns) for t in templates),  # noqa: E402
    }  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_report(  # noqa: E402
    match_results: dict,  # noqa: E402
    judge_results: list[dict],  # noqa: E402
    coverage: dict,  # noqa: E402
    templates: list[ResponseTemplate],  # noqa: E402
) -> None:  # noqa: E402
    """Generate comprehensive evaluation report."""  # noqa: E402
    print("\n" + "=" * 80)  # noqa: E402
    print("SEMANTIC TEMPLATE EVALUATION REPORT")  # noqa: E402
    print("=" * 80)  # noqa: E402
  # noqa: E402
    # 1. Coverage Analysis  # noqa: E402
    print("\n📊 TEMPLATE COVERAGE")  # noqa: E402
    print("-" * 80)  # noqa: E402
    print(f"Total Templates: {coverage['total_templates']}")  # noqa: E402
    print(f"Total Patterns: {coverage['total_patterns']}")  # noqa: E402
    print("\nBy Category:")  # noqa: E402
    for cat, stats in sorted(coverage["by_category"].items()):  # noqa: E402
        print(  # noqa: E402
            f"  {cat:20}: {stats['count']:3} templates, {stats['total_patterns']:3} patterns "  # noqa: E402
            f"(avg {stats['avg_patterns_per_template']:.1f} patterns/template)"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # 2. Hit Rate Analysis  # noqa: E402
    print("\n🎯 MATCH PERFORMANCE")  # noqa: E402
    print("-" * 80)  # noqa: E402
    total_msgs = len(match_results["matches"]) + len(match_results["unmatched"])  # noqa: E402
    print(f"Messages Tested: {total_msgs}")  # noqa: E402
    print(f"Templates Matched: {len(match_results['matches'])}")  # noqa: E402
    print(f"Hit Rate: {match_results['hit_rate'] * 100:.1f}%")  # noqa: E402
  # noqa: E402
    if match_results["matches"]:  # noqa: E402
        similarities = [m["similarity"] for m in match_results["matches"]]  # noqa: E402
        print(f"Avg Similarity: {sum(similarities) / len(similarities):.3f}")  # noqa: E402
        print(f"Min Similarity: {min(similarities):.3f}")  # noqa: E402
        print(f"Max Similarity: {max(similarities):.3f}")  # noqa: E402
  # noqa: E402
    # 3. Category Performance  # noqa: E402
    print("\n📈 CATEGORY BREAKDOWN")  # noqa: E402
    print("-" * 80)  # noqa: E402
    category_matches = defaultdict(list)  # noqa: E402
    for match in match_results["matches"]:  # noqa: E402
        category_matches[match["category"]].append(match)  # noqa: E402
  # noqa: E402
    for cat, matches in sorted(category_matches.items()):  # noqa: E402
        cat_hit_rate = len(matches) / total_msgs  # noqa: E402
        avg_sim = sum(m["similarity"] for m in matches) / len(matches) if matches else 0  # noqa: E402
        print(  # noqa: E402
            f"  {cat:20}: {len(matches):3} hits, {cat_hit_rate * 100:5.1f}% hit rate, "  # noqa: E402
            f"{avg_sim:.3f} avg similarity"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # 4. Quality Judgment (Cerebras)  # noqa: E402
    if judge_results:  # noqa: E402
        print("\n⚖️  QUALITY ASSESSMENT (Cerebras Judge)")  # noqa: E402
        print("-" * 80)  # noqa: E402
        scores = [r["score"] for r in judge_results if r["score"] > 0]  # noqa: E402
        if scores:  # noqa: E402
            print(f"Evaluations: {len(scores)}")  # noqa: E402
            print(f"Average Score: {sum(scores) / len(scores):.1f}/10")  # noqa: E402
            print("Score Distribution:")  # noqa: E402
  # noqa: E402
            buckets = {  # noqa: E402
                "9-10 (Excellent)": 0,  # noqa: E402
                "7-8 (Good)": 0,  # noqa: E402
                "5-6 (Acceptable)": 0,  # noqa: E402
                "3-4 (Poor)": 0,  # noqa: E402
                "1-2 (Very Bad)": 0,  # noqa: E402
            }  # noqa: E402
            for score in scores:  # noqa: E402
                if score >= 9:  # noqa: E402
                    buckets["9-10 (Excellent)"] += 1  # noqa: E402
                elif score >= 7:  # noqa: E402
                    buckets["7-8 (Good)"] += 1  # noqa: E402
                elif score >= 5:  # noqa: E402
                    buckets["5-6 (Acceptable)"] += 1  # noqa: E402
                elif score >= 3:  # noqa: E402
                    buckets["3-4 (Poor)"] += 1  # noqa: E402
                else:  # noqa: E402
                    buckets["1-2 (Very Bad)"] += 1  # noqa: E402
  # noqa: E402
            for bucket, count in buckets.items():  # noqa: E402
                pct = count / len(scores) * 100  # noqa: E402
                print(f"    {bucket:20}: {count:3} ({pct:5.1f}%)")  # noqa: E402
  # noqa: E402
    # 5. Top Performers  # noqa: E402
    if match_results["matches"]:  # noqa: E402
        print("\n🏆 TOP MATCHING TEMPLATES")  # noqa: E402
        print("-" * 80)  # noqa: E402
        template_hit_counts = defaultdict(lambda: {"count": 0, "avg_sim": []})  # noqa: E402
        for match in match_results["matches"]:  # noqa: E402
            name = match["template_name"]  # noqa: E402
            template_hit_counts[name]["count"] += 1  # noqa: E402
            template_hit_counts[name]["avg_sim"].append(match["similarity"])  # noqa: E402
  # noqa: E402
        top_templates = sorted(  # noqa: E402
            template_hit_counts.items(), key=lambda x: x[1]["count"], reverse=True  # noqa: E402
        )[:10]  # noqa: E402
  # noqa: E402
        for name, data in top_templates:  # noqa: E402
            avg_sim = sum(data["avg_sim"]) / len(data["avg_sim"])  # noqa: E402
            print(f"  {name:40}: {data['count']:3} hits, {avg_sim:.3f} avg similarity")  # noqa: E402
  # noqa: E402
    # 6. Problem Areas  # noqa: E402
    if judge_results:  # noqa: E402
        print("\n⚠️  TEMPLATES NEEDING IMPROVEMENT (Score < 6)")  # noqa: E402
        print("-" * 80)  # noqa: E402
        poor_performers = [r for r in judge_results if r["score"] > 0 and r["score"] < 6]  # noqa: E402
        if poor_performers:  # noqa: E402
            for result in poor_performers[:10]:  # noqa: E402
                print(f"  Template: {result['template_name']}")  # noqa: E402
                print(f"  Message: '{result['message'][:50]}...'")  # noqa: E402
                print(f"  Score: {result['score']}/10 - {result['reasoning']}")  # noqa: E402
                if result.get("better_alternative"):  # noqa: E402
                    print(f"  Suggestion: '{result['better_alternative']}'")  # noqa: E402
                print()  # noqa: E402
        else:  # noqa: E402
            print("  ✓ All evaluated templates scored well!")  # noqa: E402
  # noqa: E402
    # 7. Unmatched Messages  # noqa: E402
    if match_results["unmatched"]:  # noqa: E402
        print("\n❌ UNMATCHED MESSAGES (Potential Gaps)")  # noqa: E402
        print("-" * 80)  # noqa: E402
        print(f"Count: {len(match_results['unmatched'])}")  # noqa: E402
        print("\nSample unmatched messages (potential new patterns):")  # noqa: E402
        for msg in match_results["unmatched"][:15]:  # noqa: E402
            print(f"  - '{msg[:60]}...'" if len(msg) > 60 else f"  - '{msg}'")  # noqa: E402
  # noqa: E402
    # 8. Recommendations  # noqa: E402
    print("\n💡 RECOMMENDATIONS")  # noqa: E402
    print("-" * 80)  # noqa: E402
  # noqa: E402
    recommendations = []  # noqa: E402
  # noqa: E402
    if match_results["hit_rate"] < 0.3:  # noqa: E402
        recommendations.append(  # noqa: E402
            "LOW HIT RATE: Consider adding more patterns to existing templates or "  # noqa: E402
            "creating new templates for common unmatched messages"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    if judge_results:  # noqa: E402
        avg_score = sum(r["score"] for r in judge_results if r["score"] > 0) / len(  # noqa: E402
            [r for r in judge_results if r["score"] > 0]  # noqa: E402
        )  # noqa: E402
        if avg_score < 7:  # noqa: E402
            recommendations.append(  # noqa: E402
                f"LOW QUALITY SCORE ({avg_score:.1f}/10): Many responses sound unnatural. "  # noqa: E402
                "Consider revising template responses to be more casual and brief."  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    # Check for category gaps  # noqa: E402
    if "group" not in category_matches and len(templates) > 50:  # noqa: E402
        recommendations.append(  # noqa: E402
            "NO GROUP MATCHES: Group chat templates aren't being triggered. "  # noqa: E402
            "Check group size detection or add more group-specific patterns."  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # Check pattern diversity  # noqa: E402
    for cat, stats in coverage["by_category"].items():  # noqa: E402
        if stats["avg_patterns_per_template"] < 3:  # noqa: E402
            recommendations.append(  # noqa: E402
                f"LOW PATTERN DIVERSITY in {cat}: Avg {stats['avg_patterns_per_template']:.1f} "  # noqa: E402
                "patterns per template. Add more variation to improve match rates."  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    if not recommendations:  # noqa: E402
        recommendations.append("✓ Templates are performing well overall!")  # noqa: E402
  # noqa: E402
    for i, rec in enumerate(recommendations, 1):  # noqa: E402
        print(f"{i}. {rec}")  # noqa: E402
  # noqa: E402
    # 9. Quick Wins  # noqa: E402
    print("\n🚀 QUICK WINS")  # noqa: E402
    print("-" * 80)  # noqa: E402
    print("To improve templates immediately:")  # noqa: E402
    print("  1. Add the unmatched messages above as patterns to existing templates")  # noqa: E402
    print("  2. Review poor-scoring templates and make responses more casual/brief")  # noqa: E402
    print("  3. Add emoji to templates where appropriate (👍, 😊, 🎉)")  # noqa: E402
    print("  4. Test with: uv run python evals/evaluate_semantic_templates.py --limit 200")  # noqa: E402
    print("\n" + "=" * 80)  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    parser = argparse.ArgumentParser(description="Evaluate semantic templates with Cerebras judge")  # noqa: E402
    parser.add_argument("--limit", type=int, default=100, help="Number of messages to test")  # noqa: E402
    parser.add_argument("--batch-size", type=int, default=10, help="API calls per batch")  # noqa: E402
    parser.add_argument("--skip-judge", action="store_true", help="Skip Cerebras judging")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    print("=" * 80)  # noqa: E402
    print("SEMANTIC TEMPLATE EVALUATION")  # noqa: E402
    print("=" * 80)  # noqa: E402
  # noqa: E402
    # 1. Load templates  # noqa: E402
    print("\n📦 Loading templates...")  # noqa: E402
    templates = get_all_templates()  # noqa: E402
    print(f"✓ Loaded {len(templates)} semantic templates")  # noqa: E402
  # noqa: E402
    # 2. Fetch messages  # noqa: E402
    print(f"\n📱 Fetching {args.limit} real messages from chat.db...")  # noqa: E402
    messages = fetch_real_messages(args.limit)  # noqa: E402
    if not messages:  # noqa: E402
        print("❌ No messages found")  # noqa: E402
        return  # noqa: E402
    print(f"✓ Fetched {len(messages)} messages")  # noqa: E402
  # noqa: E402
    # 3. Find matches  # noqa: E402
    print("\n🔍 Testing templates against messages...")  # noqa: E402
    match_results = find_template_matches(messages, templates)  # noqa: E402
    print(f"✓ Found {len(match_results['matches'])} template matches")  # noqa: E402
  # noqa: E402
    # 4. Analyze coverage  # noqa: E402
    print("\n📊 Analyzing template coverage...")  # noqa: E402
    coverage = analyze_coverage(templates)  # noqa: E402
  # noqa: E402
    # 5. Judge quality (batched)  # noqa: E402
    judge_results = []  # noqa: E402
    if not args.skip_judge:  # noqa: E402
        client = get_judge_client()  # noqa: E402
        if client:  # noqa: E402
            # Prepare evaluations for matched templates  # noqa: E402
            evaluations = []  # noqa: E402
            for match in match_results["matches"][:50]:  # Judge top 50 matches  # noqa: E402
                evaluations.append(  # noqa: E402
                    {  # noqa: E402
                        "message": match["message"],  # noqa: E402
                        "template_name": match["template_name"],  # noqa: E402
                        "response": match["response"],  # noqa: E402
                        "similarity": match["similarity"],  # noqa: E402
                    }  # noqa: E402
                )  # noqa: E402
  # noqa: E402
            if evaluations:  # noqa: E402
                print(f"\n⚖️  Judging {len(evaluations)} template responses with Cerebras...")  # noqa: E402
                print(  # noqa: E402
                    f"   (Batch size: {args.batch_size}, Estimated time: {len(evaluations) // args.batch_size * 2.1:.0f}s)"  # noqa: E402
                )  # noqa: E402
                judge_results = batch_judge_templates(evaluations, client, args.batch_size)  # noqa: E402
        else:  # noqa: E402
            print("\n⚠️  Cerebras client not configured (set CEREBRAS_API_KEY)")  # noqa: E402
  # noqa: E402
    # 6. Generate report  # noqa: E402
    generate_report(match_results, judge_results, coverage, templates)  # noqa: E402
  # noqa: E402
    # Save detailed results  # noqa: E402
    output_file = PROJECT_ROOT / "evals" / "template_evaluation_results.json"  # noqa: E402
    with open(output_file, "w") as f:  # noqa: E402
        json.dump(  # noqa: E402
            {  # noqa: E402
                "match_results": match_results,  # noqa: E402
                "judge_results": judge_results,  # noqa: E402
                "coverage": coverage,  # noqa: E402
                "config": {  # noqa: E402
                    "messages_tested": len(messages),  # noqa: E402
                    "templates_loaded": len(templates),  # noqa: E402
                    "batch_size": args.batch_size,  # noqa: E402
                },  # noqa: E402
            },  # noqa: E402
            f,  # noqa: E402
            indent=2,  # noqa: E402
            default=str,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    print(f"\n💾 Detailed results saved to: {output_file}")  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
