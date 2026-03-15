#!/usr/bin/env python3  # noqa: E501
"""Ablation study: Does categorization actually help reply quality?  # noqa: E501
  # noqa: E501
Tests 3 variants:  # noqa: E501
1. categorized: Current system with category-specific instructions  # noqa: E501
2. universal: Single instruction for all messages  # noqa: E501
3. category_hint: Category mentioned but not prescriptive  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/ablation_categorization.py --judge  # noqa: E501
    uv run python evals/ablation_categorization.py --variant universal --judge  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass, field  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.eval_pipeline import (  # noqa: E402  # noqa: E501
    EVAL_DATASET_PATH,
    check_anti_ai,
    load_eval_dataset,
)
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class AblationResult:  # noqa: E501
    example_id: int  # noqa: E501
    variant: str  # "categorized", "universal", "category_hint"  # noqa: E501
    category: str  # noqa: E501
    generated_response: str  # noqa: E501
    latency_ms: float  # noqa: E501
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E501
    judge_score: float | None = None  # noqa: E501
    judge_reasoning: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
# Variant configurations  # noqa: E501
VARIANT_CONFIGS = {  # noqa: E501
    "categorized": {  # noqa: E501
        "description": "Category-specific system prompts (current system)",  # noqa: E501
        "system_prompt": None,  # Uses CATEGORY_CONFIGS  # noqa: E501
    },  # noqa: E501
    "universal": {  # noqa: E501
        "description": "Single universal instruction",  # noqa: E501
        "system_prompt": (  # noqa: E501
            "You are NOT an AI assistant. You are texting from your phone. "  # noqa: E501
            "Reply naturally, matching the conversation style. "  # noqa: E501
            "Be brief (1-2 sentences), casual, and sound like a real person."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    "category_hint": {  # noqa: E501
        "description": "Category as context, not instruction",  # noqa: E501
        "system_prompt": "hint",  # Special marker  # noqa: E501
    },  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_prompt_variant(  # noqa: E501
    context: list[str],  # noqa: E501
    last_message: str,  # noqa: E501
    category: str,  # noqa: E501
    variant: str,  # noqa: E501
    contact_style: str = "casual",  # noqa: E501
) -> str:  # noqa: E501
    """Build prompt for a specific variant."""  # noqa: E501
  # noqa: E501
    config = VARIANT_CONFIGS[variant]  # noqa: E501
  # noqa: E501
    # Format context with timestamps  # noqa: E501
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])  # noqa: E501
  # noqa: E501
    if variant == "categorized":  # noqa: E501
        # Use current category system  # noqa: E501
        from jarvis.prompts.constants import CATEGORY_CONFIGS  # noqa: E501
  # noqa: E501
        cat_config = CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])  # noqa: E501
        system = cat_config.system_prompt or (  # noqa: E501
            "You are NOT an AI assistant. You are texting from your phone. "  # noqa: E501
            "Reply naturally, matching the conversation style."  # noqa: E501
        )  # noqa: E501
    elif variant == "universal":  # noqa: E501
        system = config["system_prompt"]  # noqa: E501
    elif variant == "category_hint":  # noqa: E501
        system = (  # noqa: E501
            f"You are texting from your phone. The message appears to be a '{category}' type. "  # noqa: E501
            f"Reply naturally as yourself, matching the {contact_style} style of the conversation. "  # noqa: E501
            "Be brief and sound human."  # noqa: E501
        )  # noqa: E501
    else:  # noqa: E501
        raise ValueError(f"Unknown variant: {variant}")  # noqa: E501
  # noqa: E501
    # Use ChatML format  # noqa: E501
    prompt = (  # noqa: E501
        f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>user\n"  # noqa: E501
        f"Conversation:\n{context_str}\n\n"  # noqa: E501
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>assistant\n"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return prompt  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_with_variant(  # noqa: E501
    generator,  # noqa: E501
    context: list[str],  # noqa: E501
    last_message: str,  # noqa: E501
    category: str,  # noqa: E501
    variant: str,  # noqa: E501
    contact_style: str = "casual",  # noqa: E501
) -> tuple[str, float]:  # noqa: E501
    """Generate reply using specified variant. Returns (response, latency_ms)."""  # noqa: E501
    import time  # noqa: E501
  # noqa: E501
    prompt = build_prompt_variant(context, last_message, category, variant, contact_style)  # noqa: E501
  # noqa: E501
    start = time.perf_counter()  # noqa: E501
    result = generator.generate_sync(  # noqa: E501
        prompt=prompt,  # noqa: E501
        temperature=0.1,  # noqa: E501
        max_tokens=50,  # noqa: E501
        top_p=0.9,  # noqa: E501
        top_k=50,  # noqa: E501
        repetition_penalty=1.05,  # noqa: E501
    )  # noqa: E501
    latency_ms = (time.perf_counter() - start) * 1000  # noqa: E501
  # noqa: E501
    return result.text.strip(), latency_ms  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_ablation(  # noqa: E501
    variant: str,  # noqa: E501
    examples: list[Any],  # noqa: E501
    judge_client: Any | None = None,  # noqa: E501
) -> list[AblationResult]:  # noqa: E501
    """Run ablation for a single variant."""  # noqa: E501
  # noqa: E501
    from models.loader import get_model  # noqa: E501
  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        loader.load()  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print(f"Variant: {variant}")  # noqa: E501
    print(f"Description: {VARIANT_CONFIGS[variant]['description']}")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    for i, ex in enumerate(tqdm(examples, desc=f"Running {variant}"), 1):  # noqa: E501
        # Generate response  # noqa: E501
        response, latency = generate_with_variant(  # noqa: E501
            loader,  # noqa: E501
            ex.context,  # noqa: E501
            ex.last_message,  # noqa: E501
            ex.category,  # noqa: E501
            variant,  # noqa: E501
            ex.contact_style,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        # Check anti-AI  # noqa: E501
        anti_ai = check_anti_ai(response)  # noqa: E501
  # noqa: E501
        # Judge scoring  # noqa: E501
        judge_score = None  # noqa: E501
        judge_reasoning = ""  # noqa: E501
        if judge_client:  # noqa: E501
            try:  # noqa: E501
                prompt = (  # noqa: E501
                    "You are an expert evaluator for text message replies.\n\n"  # noqa: E501
                    f"Conversation:\n{chr(10).join(ex.context)}\n\n"  # noqa: E501
                    f"Message to reply to: {ex.last_message}\n\n"  # noqa: E501
                    f"Generated reply: {response}\n\n"  # noqa: E501
                    f"Ideal reply: {ex.ideal_response}\n\n"  # noqa: E501
                    f"Category: {ex.category}\n"  # noqa: E501
                    f"Notes: {ex.notes}\n\n"  # noqa: E501
                    "Score 0-10. Consider:\n"  # noqa: E501
                    "- Does it sound like a real text message (not AI)?\n"  # noqa: E501
                    "- Is it appropriate for the conversation?\n"  # noqa: E501
                    "- Does it match the ideal reply in intent/tone?\n\n"  # noqa: E501
                    'Respond: {"score": <0-10>, "reasoning": "<brief>"}'  # noqa: E501
                )  # noqa: E501
                resp = judge_client.chat.completions.create(  # noqa: E501
                    model=JUDGE_MODEL,  # noqa: E501
                    messages=[{"role": "user", "content": prompt}],  # noqa: E501
                    temperature=0.0,  # noqa: E501
                    max_tokens=150,  # noqa: E501
                )  # noqa: E501
                text = resp.choices[0].message.content.strip()  # noqa: E501
                if text.startswith("```"):  # noqa: E501
                    text = text.split("```")[1]  # noqa: E501
                    if text.startswith("json"):  # noqa: E501
                        text = text[4:]  # noqa: E501
                data = json.loads(text)  # noqa: E501
                judge_score = float(data["score"])  # noqa: E501
                judge_reasoning = data.get("reasoning", "")  # noqa: E501
            except Exception as e:  # noqa: E501
                judge_reasoning = f"judge error: {e}"  # noqa: E501
  # noqa: E501
        result = AblationResult(  # noqa: E501
            example_id=i,  # noqa: E501
            variant=variant,  # noqa: E501
            category=ex.category,  # noqa: E501
            generated_response=response,  # noqa: E501
            latency_ms=latency,  # noqa: E501
            anti_ai_violations=anti_ai,  # noqa: E501
            judge_score=judge_score,  # noqa: E501
            judge_reasoning=judge_reasoning,  # noqa: E501
        )  # noqa: E501
        results.append(result)  # noqa: E501
  # noqa: E501
        # Print progress  # noqa: E501
        status = "AI!" if anti_ai else "clean"  # noqa: E501
        judge_str = f" | Judge: {judge_score:.0f}/10" if judge_score else ""  # noqa: E501
        print(f"[{i:2d}] [{ex.category:12s}] {status}{judge_str} -> {response[:50]}")  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def analyze_results(results: list[AblationResult]) -> dict:  # noqa: E501
    """Analyze and compare results across variants."""  # noqa: E501
  # noqa: E501
    from collections import defaultdict  # noqa: E501
  # noqa: E501
    by_variant = defaultdict(list)  # noqa: E501
    for r in results:  # noqa: E501
        by_variant[r.variant].append(r)  # noqa: E501
  # noqa: E501
    analysis = {}  # noqa: E501
  # noqa: E501
    for variant, vresults in by_variant.items():  # noqa: E501
        scores = [r.judge_score for r in vresults if r.judge_score is not None]  # noqa: E501
        anti_ai_count = sum(1 for r in vresults if r.anti_ai_violations)  # noqa: E501
        latencies = [r.latency_ms for r in vresults]  # noqa: E501
  # noqa: E501
        # Category breakdown  # noqa: E501
        by_category = defaultdict(lambda: {"scores": [], "anti_ai": 0})  # noqa: E501
        for r in vresults:  # noqa: E501
            by_category[r.category]["scores"].append(r.judge_score or 0)  # noqa: E501
            if r.anti_ai_violations:  # noqa: E501
                by_category[r.category]["anti_ai"] += 1  # noqa: E501
  # noqa: E501
        analysis[variant] = {  # noqa: E501
            "total": len(vresults),  # noqa: E501
            "judge_avg": sum(scores) / len(scores) if scores else 0,  # noqa: E501
            "judge_median": sorted(scores)[len(scores) // 2] if scores else 0,  # noqa: E501
            "anti_ai_violations": anti_ai_count,  # noqa: E501
            "anti_ai_rate": anti_ai_count / len(vresults) if vresults else 0,  # noqa: E501
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,  # noqa: E501
            "by_category": {  # noqa: E501
                cat: {  # noqa: E501
                    "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,  # noqa: E501
                    "anti_ai": d["anti_ai"],  # noqa: E501
                }  # noqa: E501
                for cat, d in by_category.items()  # noqa: E501
            },  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    return analysis  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Categorization Ablation Study")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--variant",  # noqa: E501
        choices=["categorized", "universal", "category_hint", "all"],  # noqa: E501
        default="all",  # noqa: E501
        help="Which variant to run (default: all)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Load dataset  # noqa: E501
    if not EVAL_DATASET_PATH.exists():  # noqa: E501
        print(f"ERROR: Dataset not found at {EVAL_DATASET_PATH}")  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E501
    print(f"Loaded {len(examples)} examples from {EVAL_DATASET_PATH}")  # noqa: E501
  # noqa: E501
    # Initialize judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E501
        else:  # noqa: E501
            print("WARNING: Judge API key not set, skipping judge scoring")  # noqa: E501
  # noqa: E501
    # Determine variants to run  # noqa: E501
    variants = (  # noqa: E501
        ["categorized", "universal", "category_hint"] if args.variant == "all" else [args.variant]  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Run ablations  # noqa: E501
    all_results = []  # noqa: E501
    for variant in variants:  # noqa: E501
        results = run_ablation(variant, examples, judge_client)  # noqa: E501
        all_results.extend(results)  # noqa: E501
  # noqa: E501
    # Analyze  # noqa: E501
    analysis = analyze_results(all_results)  # noqa: E501
  # noqa: E501
    # Print summary  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("ABLATION RESULTS SUMMARY")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    for variant, stats in analysis.items():  # noqa: E501
        print(f"\n{variant.upper()}:")  # noqa: E501
        print(f"  Judge avg:     {stats['judge_avg']:.2f}/10")  # noqa: E501
        print(f"  Judge median:  {stats['judge_median']:.2f}/10")  # noqa: E501
        print(f"  Anti-AI rate:  {stats['anti_ai_rate']:.1%}")  # noqa: E501
        print(f"  Avg latency:   {stats['avg_latency_ms']:.0f}ms")  # noqa: E501
        print("  By category:")  # noqa: E501
        for cat, cat_stats in stats["by_category"].items():  # noqa: E501
            print(  # noqa: E501
                f"    {cat:12s}: avg={cat_stats['avg_score']:.1f}, anti_ai={cat_stats['anti_ai']}"  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "ablation_categorization.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "analysis": analysis,  # noqa: E501
        "raw_results": [  # noqa: E501
            {  # noqa: E501
                "example_id": r.example_id,  # noqa: E501
                "variant": r.variant,  # noqa: E501
                "category": r.category,  # noqa: E501
                "generated": r.generated_response,  # noqa: E501
                "latency_ms": round(r.latency_ms, 1),  # noqa: E501
                "anti_ai": r.anti_ai_violations,  # noqa: E501
                "judge_score": r.judge_score,  # noqa: E501
                "judge_reasoning": r.judge_reasoning,  # noqa: E501
            }  # noqa: E501
            for r in all_results  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    print(f"\nResults saved to: {output_path}")  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
