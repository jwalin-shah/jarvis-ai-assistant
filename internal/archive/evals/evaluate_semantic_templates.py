#!/usr/bin/env python3  # noqa: E501
"""Comprehensive batched evaluation of semantic templates using Cerebras judge.  # noqa: E501
  # noqa: E501
This script:  # noqa: E501
1. Fetches real messages from chat.db  # noqa: E501
2. Tests all 91 semantic templates against them  # noqa: E501
3. Batches Cerebras API calls (10 evaluations per call)  # noqa: E501
4. Analyzes hit rates, coverage, and gaps  # noqa: E501
5. Provides actionable recommendations  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/evaluate_semantic_templates.py [--limit 100] [--batch-size 10]  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sqlite3  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from collections import defaultdict  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E501

# noqa: E501
from models.template_defaults import get_minimal_fallback_templates  # noqa: E402  # noqa: E501
from models.templates import ResponseTemplate, TemplateMatcher  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
def fetch_real_messages(limit: int = 100) -> list[dict]:  # noqa: E501
    """Fetch diverse real messages from chat.db."""  # noqa: E501
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"  # noqa: E501
    if not chat_db.exists():  # noqa: E501
        print(f"❌ chat.db not found at {chat_db}")  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)  # noqa: E501
        cursor = conn.cursor()  # noqa: E501
  # noqa: E501
        # Get diverse messages of different lengths  # noqa: E501
        cursor.execute(  # noqa: E501
            """  # noqa: E501
            SELECT  # noqa: E501
                m.text,  # noqa: E501
                m.date,  # noqa: E501
                c.display_name,  # noqa: E501
                m.is_from_me,  # noqa: E501
                length(m.text) as msg_len  # noqa: E501
            FROM message m  # noqa: E501
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id  # noqa: E501
            JOIN chat c ON cmj.chat_id = c.ROWID  # noqa: E501
            WHERE m.text IS NOT NULL  # noqa: E501
              AND m.text NOT LIKE '%http%'  # noqa: E501
              AND m.text NOT LIKE '%http%'  # noqa: E501
              AND length(m.text) <= 100  # noqa: E501
              AND m.is_from_me = 0  # noqa: E501
            ORDER BY RANDOM()  # noqa: E501
            LIMIT ?  # noqa: E501
        """,  # noqa: E501
            (limit * 2,),  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        messages = []  # noqa: E501
        for row in cursor.fetchall():  # noqa: E501
            text, date, display_name, is_from_me, msg_len = row  # noqa: E501
            if text and len(messages) < limit:  # noqa: E501
                messages.append(  # noqa: E501
                    {  # noqa: E501
                        "text": text.strip(),  # noqa: E501
                        "contact": display_name or "Unknown",  # noqa: E501
                        "length": msg_len,  # noqa: E501
                    }  # noqa: E501
                )  # noqa: E501
  # noqa: E501
        conn.close()  # noqa: E501
        return messages  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"❌ Error reading chat.db: {e}")  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_all_templates() -> list[ResponseTemplate]:  # noqa: E501
    """Get all 91 semantic templates."""  # noqa: E501
    return get_minimal_fallback_templates()  # noqa: E501
  # noqa: E501
  # noqa: E501
def categorize_template(template: ResponseTemplate) -> str:  # noqa: E501
    """Categorize template by its purpose."""  # noqa: E501
    name = template.name.lower()  # noqa: E501
    if "group_" in name:  # noqa: E501
        return "group"  # noqa: E501
    elif any(x in name for x in ["summarize", "find_", "search", "show"]):  # noqa: E501
        return "assistant_query"  # noqa: E501
    elif any(x in name for x in ["quick_", "thank_you", "acknowledgment"]):  # noqa: E501
        return "quick_response"  # noqa: E501
    elif any(x in name for x in ["meeting", "schedule", "time", "plan"]):  # noqa: E501
        return "scheduling"  # noqa: E501
    else:  # noqa: E501
        return "general"  # noqa: E501
  # noqa: E501
  # noqa: E501
def batch_judge_templates(evaluations: list[dict], client, batch_size: int = 10) -> list[dict]:  # noqa: E501
    """Judge multiple template-message pairs in a single API call."""  # noqa: E501
    if not client:  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    # Process in batches  # noqa: E501
    for i in range(0, len(evaluations), batch_size):  # noqa: E501
        batch = evaluations[i : i + batch_size]  # noqa: E501
  # noqa: E501
        # Build batched prompt  # noqa: E501
        eval_items = []  # noqa: E501
        for idx, eval_item in enumerate(batch):  # noqa: E501
            eval_items.append(f"""  # noqa: E501
{idx + 1}. Message: "{eval_item["message"]}"  # noqa: E501
   Template Response: "{eval_item["response"]}"  # noqa: E501
   Template Name: {eval_item["template_name"]}""")  # noqa: E501
  # noqa: E501
        prompt = f"""You are evaluating if template responses are appropriate for text messages.  # noqa: E501
  # noqa: E501
Rate each response on a scale of 1-10:  # noqa: E501
- 9-10: Perfect natural response, sounds like a real person texting  # noqa: E501
- 7-8: Good response, appropriate for the context  # noqa: E501
- 5-6: Acceptable but could be better  # noqa: E501
- 3-4: Poor, somewhat awkward or inappropriate  # noqa: E501
- 1-2: Very bad, completely wrong tone or content  # noqa: E501
  # noqa: E501
{chr(10).join(eval_items)}  # noqa: E501
  # noqa: E501
Respond with ONLY a JSON array in this exact format:  # noqa: E501
[  # noqa: E501
  {{"score": <number>, "reasoning": "<brief explanation>", "better_alternative": "<suggested better response or null>"}},  # noqa: E501
  ... (one object for each evaluation)  # noqa: E501
]  # noqa: E501
"""  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            resp = client.chat.completions.create(  # noqa: E501
                model=JUDGE_MODEL,  # noqa: E501
                messages=[{"role": "user", "content": prompt}],  # noqa: E501
                temperature=0.0,  # noqa: E501
                max_tokens=1000,  # noqa: E501
            )  # noqa: E501
            content = resp.choices[0].message.content  # noqa: E501
  # noqa: E501
            # Extract JSON  # noqa: E501
            if "```json" in content:  # noqa: E501
                content = content.split("```json")[1].split("```")[0]  # noqa: E501
            elif "```" in content:  # noqa: E501
                content = content.split("```")[1].split("```")[0]  # noqa: E501
  # noqa: E501
            batch_results = json.loads(content.strip())  # noqa: E501
  # noqa: E501
            # Combine with original eval data  # noqa: E501
            for eval_item, result in zip(batch, batch_results):  # noqa: E501
                results.append(  # noqa: E501
                    {  # noqa: E501
                        **eval_item,  # noqa: E501
                        "score": result.get("score", 0),  # noqa: E501
                        "reasoning": result.get("reasoning", "No reasoning"),  # noqa: E501
                        "better_alternative": result.get("better_alternative"),  # noqa: E501
                    }  # noqa: E501
                )  # noqa: E501
  # noqa: E501
            print(  # noqa: E501
                f"  ✓ Batch {i // batch_size + 1}/{(len(evaluations) + batch_size - 1) // batch_size} complete"  # noqa: E501
            )  # noqa: E501
  # noqa: E501
            # Rate limit: 30 req/min = 2 sec between calls  # noqa: E501
            if i + batch_size < len(evaluations):  # noqa: E501
                time.sleep(2.1)  # noqa: E501
  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"  ⚠ Batch {i // batch_size + 1} failed: {e}")  # noqa: E501
            # Add failed items with 0 score  # noqa: E501
            for eval_item in batch:  # noqa: E501
                results.append(  # noqa: E501
                    {  # noqa: E501
                        **eval_item,  # noqa: E501
                        "score": 0,  # noqa: E501
                        "reasoning": f"API error: {e}",  # noqa: E501
                        "better_alternative": None,  # noqa: E501
                    }  # noqa: E501
                )  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def find_template_matches(messages: list[dict], templates: list[ResponseTemplate]) -> dict:  # noqa: E501
    """Find which templates would match real messages."""  # noqa: E501
    matcher = TemplateMatcher(templates)  # noqa: E501
  # noqa: E501
    matches = []  # noqa: E501
    unmatched = []  # noqa: E501
  # noqa: E501
    for msg in messages:  # noqa: E501
        text = msg["text"]  # noqa: E501
        match = matcher.match(text, track_analytics=False)  # noqa: E501
  # noqa: E501
        if match:  # noqa: E501
            matches.append(  # noqa: E501
                {  # noqa: E501
                    "message": text,  # noqa: E501
                    "template_name": match.template.name,  # noqa: E501
                    "matched_pattern": match.matched_pattern,  # noqa: E501
                    "similarity": match.similarity,  # noqa: E501
                    "response": match.template.response,  # noqa: E501
                    "category": categorize_template(match.template),  # noqa: E501
                }  # noqa: E501
            )  # noqa: E501
        else:  # noqa: E501
            unmatched.append(text)  # noqa: E501
  # noqa: E501
    return {  # noqa: E501
        "matches": matches,  # noqa: E501
        "unmatched": unmatched,  # noqa: E501
        "hit_rate": len(matches) / len(messages) if messages else 0,  # noqa: E501
    }  # noqa: E501
  # noqa: E501
  # noqa: E501
def analyze_coverage(templates: list[ResponseTemplate]) -> dict:  # noqa: E501
    """Analyze template coverage and diversity."""  # noqa: E501
    categories = defaultdict(list)  # noqa: E501
    pattern_counts = defaultdict(int)  # noqa: E501
  # noqa: E501
    for template in templates:  # noqa: E501
        cat = categorize_template(template)  # noqa: E501
        categories[cat].append(template)  # noqa: E501
        pattern_counts[cat] += len(template.patterns)  # noqa: E501
  # noqa: E501
    return {  # noqa: E501
        "by_category": {  # noqa: E501
            cat: {  # noqa: E501
                "count": len(temps),  # noqa: E501
                "total_patterns": pattern_counts[cat],  # noqa: E501
                "avg_patterns_per_template": pattern_counts[cat] / len(temps) if temps else 0,  # noqa: E501
            }  # noqa: E501
            for cat, temps in categories.items()  # noqa: E501
        },  # noqa: E501
        "total_templates": len(templates),  # noqa: E501
        "total_patterns": sum(len(t.patterns) for t in templates),  # noqa: E501
    }  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_report(  # noqa: E501
    match_results: dict,  # noqa: E501
    judge_results: list[dict],  # noqa: E501
    coverage: dict,  # noqa: E501
    templates: list[ResponseTemplate],  # noqa: E501
) -> None:  # noqa: E501
    """Generate comprehensive evaluation report."""  # noqa: E501
    print("\n" + "=" * 80)  # noqa: E501
    print("SEMANTIC TEMPLATE EVALUATION REPORT")  # noqa: E501
    print("=" * 80)  # noqa: E501
  # noqa: E501
    # 1. Coverage Analysis  # noqa: E501
    print("\n📊 TEMPLATE COVERAGE")  # noqa: E501
    print("-" * 80)  # noqa: E501
    print(f"Total Templates: {coverage['total_templates']}")  # noqa: E501
    print(f"Total Patterns: {coverage['total_patterns']}")  # noqa: E501
    print("\nBy Category:")  # noqa: E501
    for cat, stats in sorted(coverage["by_category"].items()):  # noqa: E501
        print(  # noqa: E501
            f"  {cat:20}: {stats['count']:3} templates, {stats['total_patterns']:3} patterns "  # noqa: E501
            f"(avg {stats['avg_patterns_per_template']:.1f} patterns/template)"  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # 2. Hit Rate Analysis  # noqa: E501
    print("\n🎯 MATCH PERFORMANCE")  # noqa: E501
    print("-" * 80)  # noqa: E501
    total_msgs = len(match_results["matches"]) + len(match_results["unmatched"])  # noqa: E501
    print(f"Messages Tested: {total_msgs}")  # noqa: E501
    print(f"Templates Matched: {len(match_results['matches'])}")  # noqa: E501
    print(f"Hit Rate: {match_results['hit_rate'] * 100:.1f}%")  # noqa: E501
  # noqa: E501
    if match_results["matches"]:  # noqa: E501
        similarities = [m["similarity"] for m in match_results["matches"]]  # noqa: E501
        print(f"Avg Similarity: {sum(similarities) / len(similarities):.3f}")  # noqa: E501
        print(f"Min Similarity: {min(similarities):.3f}")  # noqa: E501
        print(f"Max Similarity: {max(similarities):.3f}")  # noqa: E501
  # noqa: E501
    # 3. Category Performance  # noqa: E501
    print("\n📈 CATEGORY BREAKDOWN")  # noqa: E501
    print("-" * 80)  # noqa: E501
    category_matches = defaultdict(list)  # noqa: E501
    for match in match_results["matches"]:  # noqa: E501
        category_matches[match["category"]].append(match)  # noqa: E501
  # noqa: E501
    for cat, matches in sorted(category_matches.items()):  # noqa: E501
        cat_hit_rate = len(matches) / total_msgs  # noqa: E501
        avg_sim = sum(m["similarity"] for m in matches) / len(matches) if matches else 0  # noqa: E501
        print(  # noqa: E501
            f"  {cat:20}: {len(matches):3} hits, {cat_hit_rate * 100:5.1f}% hit rate, "  # noqa: E501
            f"{avg_sim:.3f} avg similarity"  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # 4. Quality Judgment (Cerebras)  # noqa: E501
    if judge_results:  # noqa: E501
        print("\n⚖️  QUALITY ASSESSMENT (Cerebras Judge)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        scores = [r["score"] for r in judge_results if r["score"] > 0]  # noqa: E501
        if scores:  # noqa: E501
            print(f"Evaluations: {len(scores)}")  # noqa: E501
            print(f"Average Score: {sum(scores) / len(scores):.1f}/10")  # noqa: E501
            print("Score Distribution:")  # noqa: E501
  # noqa: E501
            buckets = {  # noqa: E501
                "9-10 (Excellent)": 0,  # noqa: E501
                "7-8 (Good)": 0,  # noqa: E501
                "5-6 (Acceptable)": 0,  # noqa: E501
                "3-4 (Poor)": 0,  # noqa: E501
                "1-2 (Very Bad)": 0,  # noqa: E501
            }  # noqa: E501
            for score in scores:  # noqa: E501
                if score >= 9:  # noqa: E501
                    buckets["9-10 (Excellent)"] += 1  # noqa: E501
                elif score >= 7:  # noqa: E501
                    buckets["7-8 (Good)"] += 1  # noqa: E501
                elif score >= 5:  # noqa: E501
                    buckets["5-6 (Acceptable)"] += 1  # noqa: E501
                elif score >= 3:  # noqa: E501
                    buckets["3-4 (Poor)"] += 1  # noqa: E501
                else:  # noqa: E501
                    buckets["1-2 (Very Bad)"] += 1  # noqa: E501
  # noqa: E501
            for bucket, count in buckets.items():  # noqa: E501
                pct = count / len(scores) * 100  # noqa: E501
                print(f"    {bucket:20}: {count:3} ({pct:5.1f}%)")  # noqa: E501
  # noqa: E501
    # 5. Top Performers  # noqa: E501
    if match_results["matches"]:  # noqa: E501
        print("\n🏆 TOP MATCHING TEMPLATES")  # noqa: E501
        print("-" * 80)  # noqa: E501
        template_hit_counts = defaultdict(lambda: {"count": 0, "avg_sim": []})  # noqa: E501
        for match in match_results["matches"]:  # noqa: E501
            name = match["template_name"]  # noqa: E501
            template_hit_counts[name]["count"] += 1  # noqa: E501
            template_hit_counts[name]["avg_sim"].append(match["similarity"])  # noqa: E501
  # noqa: E501
        top_templates = sorted(  # noqa: E501
            template_hit_counts.items(), key=lambda x: x[1]["count"], reverse=True  # noqa: E501
        )[:10]  # noqa: E501
  # noqa: E501
        for name, data in top_templates:  # noqa: E501
            avg_sim = sum(data["avg_sim"]) / len(data["avg_sim"])  # noqa: E501
            print(f"  {name:40}: {data['count']:3} hits, {avg_sim:.3f} avg similarity")  # noqa: E501
  # noqa: E501
    # 6. Problem Areas  # noqa: E501
    if judge_results:  # noqa: E501
        print("\n⚠️  TEMPLATES NEEDING IMPROVEMENT (Score < 6)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        poor_performers = [r for r in judge_results if r["score"] > 0 and r["score"] < 6]  # noqa: E501
        if poor_performers:  # noqa: E501
            for result in poor_performers[:10]:  # noqa: E501
                print(f"  Template: {result['template_name']}")  # noqa: E501
                print(f"  Message: '{result['message'][:50]}...'")  # noqa: E501
                print(f"  Score: {result['score']}/10 - {result['reasoning']}")  # noqa: E501
                if result.get("better_alternative"):  # noqa: E501
                    print(f"  Suggestion: '{result['better_alternative']}'")  # noqa: E501
                print()  # noqa: E501
        else:  # noqa: E501
            print("  ✓ All evaluated templates scored well!")  # noqa: E501
  # noqa: E501
    # 7. Unmatched Messages  # noqa: E501
    if match_results["unmatched"]:  # noqa: E501
        print("\n❌ UNMATCHED MESSAGES (Potential Gaps)")  # noqa: E501
        print("-" * 80)  # noqa: E501
        print(f"Count: {len(match_results['unmatched'])}")  # noqa: E501
        print("\nSample unmatched messages (potential new patterns):")  # noqa: E501
        for msg in match_results["unmatched"][:15]:  # noqa: E501
            print(f"  - '{msg[:60]}...'" if len(msg) > 60 else f"  - '{msg}'")  # noqa: E501
  # noqa: E501
    # 8. Recommendations  # noqa: E501
    print("\n💡 RECOMMENDATIONS")  # noqa: E501
    print("-" * 80)  # noqa: E501
  # noqa: E501
    recommendations = []  # noqa: E501
  # noqa: E501
    if match_results["hit_rate"] < 0.3:  # noqa: E501
        recommendations.append(  # noqa: E501
            "LOW HIT RATE: Consider adding more patterns to existing templates or "  # noqa: E501
            "creating new templates for common unmatched messages"  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    if judge_results:  # noqa: E501
        avg_score = sum(r["score"] for r in judge_results if r["score"] > 0) / len(  # noqa: E501
            [r for r in judge_results if r["score"] > 0]  # noqa: E501
        )  # noqa: E501
        if avg_score < 7:  # noqa: E501
            recommendations.append(  # noqa: E501
                f"LOW QUALITY SCORE ({avg_score:.1f}/10): Many responses sound unnatural. "  # noqa: E501
                "Consider revising template responses to be more casual and brief."  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    # Check for category gaps  # noqa: E501
    if "group" not in category_matches and len(templates) > 50:  # noqa: E501
        recommendations.append(  # noqa: E501
            "NO GROUP MATCHES: Group chat templates aren't being triggered. "  # noqa: E501
            "Check group size detection or add more group-specific patterns."  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Check pattern diversity  # noqa: E501
    for cat, stats in coverage["by_category"].items():  # noqa: E501
        if stats["avg_patterns_per_template"] < 3:  # noqa: E501
            recommendations.append(  # noqa: E501
                f"LOW PATTERN DIVERSITY in {cat}: Avg {stats['avg_patterns_per_template']:.1f} "  # noqa: E501
                "patterns per template. Add more variation to improve match rates."  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    if not recommendations:  # noqa: E501
        recommendations.append("✓ Templates are performing well overall!")  # noqa: E501
  # noqa: E501
    for i, rec in enumerate(recommendations, 1):  # noqa: E501
        print(f"{i}. {rec}")  # noqa: E501
  # noqa: E501
    # 9. Quick Wins  # noqa: E501
    print("\n🚀 QUICK WINS")  # noqa: E501
    print("-" * 80)  # noqa: E501
    print("To improve templates immediately:")  # noqa: E501
    print("  1. Add the unmatched messages above as patterns to existing templates")  # noqa: E501
    print("  2. Review poor-scoring templates and make responses more casual/brief")  # noqa: E501
    print("  3. Add emoji to templates where appropriate (👍, 😊, 🎉)")  # noqa: E501
    print("  4. Test with: uv run python evals/evaluate_semantic_templates.py --limit 200")  # noqa: E501
    print("\n" + "=" * 80)  # noqa: E501
  # noqa: E501
  # noqa: E501
def main():  # noqa: E501
    parser = argparse.ArgumentParser(description="Evaluate semantic templates with Cerebras judge")  # noqa: E501
    parser.add_argument("--limit", type=int, default=100, help="Number of messages to test")  # noqa: E501
    parser.add_argument("--batch-size", type=int, default=10, help="API calls per batch")  # noqa: E501
    parser.add_argument("--skip-judge", action="store_true", help="Skip Cerebras judging")  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    print("=" * 80)  # noqa: E501
    print("SEMANTIC TEMPLATE EVALUATION")  # noqa: E501
    print("=" * 80)  # noqa: E501
  # noqa: E501
    # 1. Load templates  # noqa: E501
    print("\n📦 Loading templates...")  # noqa: E501
    templates = get_all_templates()  # noqa: E501
    print(f"✓ Loaded {len(templates)} semantic templates")  # noqa: E501
  # noqa: E501
    # 2. Fetch messages  # noqa: E501
    print(f"\n📱 Fetching {args.limit} real messages from chat.db...")  # noqa: E501
    messages = fetch_real_messages(args.limit)  # noqa: E501
    if not messages:  # noqa: E501
        print("❌ No messages found")  # noqa: E501
        return  # noqa: E501
    print(f"✓ Fetched {len(messages)} messages")  # noqa: E501
  # noqa: E501
    # 3. Find matches  # noqa: E501
    print("\n🔍 Testing templates against messages...")  # noqa: E501
    match_results = find_template_matches(messages, templates)  # noqa: E501
    print(f"✓ Found {len(match_results['matches'])} template matches")  # noqa: E501
  # noqa: E501
    # 4. Analyze coverage  # noqa: E501
    print("\n📊 Analyzing template coverage...")  # noqa: E501
    coverage = analyze_coverage(templates)  # noqa: E501
  # noqa: E501
    # 5. Judge quality (batched)  # noqa: E501
    judge_results = []  # noqa: E501
    if not args.skip_judge:  # noqa: E501
        client = get_judge_client()  # noqa: E501
        if client:  # noqa: E501
            # Prepare evaluations for matched templates  # noqa: E501
            evaluations = []  # noqa: E501
            for match in match_results["matches"][:50]:  # Judge top 50 matches  # noqa: E501
                evaluations.append(  # noqa: E501
                    {  # noqa: E501
                        "message": match["message"],  # noqa: E501
                        "template_name": match["template_name"],  # noqa: E501
                        "response": match["response"],  # noqa: E501
                        "similarity": match["similarity"],  # noqa: E501
                    }  # noqa: E501
                )  # noqa: E501
  # noqa: E501
            if evaluations:  # noqa: E501
                print(f"\n⚖️  Judging {len(evaluations)} template responses with Cerebras...")  # noqa: E501
                print(  # noqa: E501
                    f"   (Batch size: {args.batch_size}, Estimated time: {len(evaluations) // args.batch_size * 2.1:.0f}s)"  # noqa: E501
                )  # noqa: E501
                judge_results = batch_judge_templates(evaluations, client, args.batch_size)  # noqa: E501
        else:  # noqa: E501
            print("\n⚠️  Cerebras client not configured (set CEREBRAS_API_KEY)")  # noqa: E501
  # noqa: E501
    # 6. Generate report  # noqa: E501
    generate_report(match_results, judge_results, coverage, templates)  # noqa: E501
  # noqa: E501
    # Save detailed results  # noqa: E501
    output_file = PROJECT_ROOT / "evals" / "template_evaluation_results.json"  # noqa: E501
    with open(output_file, "w") as f:  # noqa: E501
        json.dump(  # noqa: E501
            {  # noqa: E501
                "match_results": match_results,  # noqa: E501
                "judge_results": judge_results,  # noqa: E501
                "coverage": coverage,  # noqa: E501
                "config": {  # noqa: E501
                    "messages_tested": len(messages),  # noqa: E501
                    "templates_loaded": len(templates),  # noqa: E501
                    "batch_size": args.batch_size,  # noqa: E501
                },  # noqa: E501
            },  # noqa: E501
            f,  # noqa: E501
            indent=2,  # noqa: E501
            default=str,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    print(f"\n💾 Detailed results saved to: {output_file}")  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
