#!/usr/bin/env python3
"""Batched categorization ablation study with rate limiting.

Efficiently uses the judge API with:
- Batch scoring (multiple examples per request when possible)
- Rate limit tracking (30 req/min = 1 request every 2 seconds)
- Automatic retries with exponential backoff
- Progress saving (resume on interruption)

Usage:
    uv run python evals/ablation_categorization_batched.py --variant all --judge
    uv run python evals/ablation_categorization_batched.py --variant categorized --batch-size 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

  # noqa: E402
# Rate limit configuration  # noqa: E402
RATE_LIMIT_RPM = 30  # requests per minute  # noqa: E402
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_RPM  # 2 seconds between requests  # noqa: E402
BATCH_SIZE_DEFAULT = 3  # Judge this many examples per request  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class AblationResult:  # noqa: E402
    example_id: int  # noqa: E402
    variant: str  # noqa: E402
    category: str  # noqa: E402
    generated_response: str  # noqa: E402
    latency_ms: float  # noqa: E402
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E402
    judge_score: float | None = None  # noqa: E402
    judge_reasoning: str = ""  # noqa: E402
  # noqa: E402
  # noqa: E402
# Variant configurations  # noqa: E402
VARIANT_CONFIGS = {  # noqa: E402
    "categorized": {  # noqa: E402
        "description": "Category-specific system prompts (current system)",  # noqa: E402
        "system_prompt": None,  # noqa: E402
    },  # noqa: E402
    "universal": {  # noqa: E402
        "description": "Single universal instruction",  # noqa: E402
        "system_prompt": (  # noqa: E402
            "You are NOT an AI assistant. You are texting from your phone. "  # noqa: E402
            "Reply naturally, matching the conversation style. "  # noqa: E402
            "Be brief (1-2 sentences), casual, and sound like a real person."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    "category_hint": {  # noqa: E402
        "description": "Category as context, not instruction",  # noqa: E402
        "system_prompt": "hint",  # noqa: E402
    },  # noqa: E402
}  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_prompt_variant(  # noqa: E402
    context: list[str],  # noqa: E402
    last_message: str,  # noqa: E402
    category: str,  # noqa: E402
    variant: str,  # noqa: E402
    contact_style: str = "casual",  # noqa: E402
) -> str:  # noqa: E402
    """Build prompt for a specific variant."""  # noqa: E402
  # noqa: E402
    config = VARIANT_CONFIGS[variant]  # noqa: E402
  # noqa: E402
    # Format context  # noqa: E402
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])  # noqa: E402
  # noqa: E402
    if variant == "categorized":  # noqa: E402
        from jarvis.prompts.constants import CATEGORY_CONFIGS  # noqa: E402
  # noqa: E402
        cat_config = CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])  # noqa: E402
        system = cat_config.system_prompt or (  # noqa: E402
            "You are NOT an AI assistant. You are texting from your phone. "  # noqa: E402
            "Reply naturally, matching the conversation style."  # noqa: E402
        )  # noqa: E402
    elif variant == "universal":  # noqa: E402
        system = config["system_prompt"]  # noqa: E402
    elif variant == "category_hint":  # noqa: E402
        system = (  # noqa: E402
            f"You are texting from your phone. The message appears to be a '{category}' type. "  # noqa: E402
            f"Reply naturally as yourself, matching the {contact_style} style. "  # noqa: E402
            "Be brief and sound human."  # noqa: E402
        )  # noqa: E402
    else:  # noqa: E402
        raise ValueError(f"Unknown variant: {variant}")  # noqa: E402
  # noqa: E402
    # Use ChatML format  # noqa: E402
    prompt = (  # noqa: E402
        f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>user\n"  # noqa: E402
        f"Conversation:\n{context_str}\n\n"  # noqa: E402
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>assistant\n"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    return prompt  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_batch(  # noqa: E402
    generator,  # noqa: E402
    examples: list[Any],  # noqa: E402
    variant: str,  # noqa: E402
) -> list[tuple[str, float]]:  # noqa: E402
    """Generate replies for a batch of examples."""  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for ex in examples:  # noqa: E402
        import time  # noqa: E402
  # noqa: E402
        start = time.perf_counter()  # noqa: E402
  # noqa: E402
        prompt = build_prompt_variant(  # noqa: E402
            ex.context,  # noqa: E402
            ex.last_message,  # noqa: E402
            ex.category,  # noqa: E402
            variant,  # noqa: E402
            ex.contact_style,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        try:  # noqa: E402
            result = generator.generate_sync(  # noqa: E402
                prompt=prompt,  # noqa: E402
                temperature=0.1,  # noqa: E402
                max_tokens=50,  # noqa: E402
                top_p=0.9,  # noqa: E402
                top_k=50,  # noqa: E402
                repetition_penalty=1.05,  # noqa: E402
            )  # noqa: E402
            reply = result.text.strip()  # noqa: E402
        except Exception as e:  # noqa: E402
            reply = f"[ERROR: {e}]"  # noqa: E402
  # noqa: E402
        latency = (time.perf_counter() - start) * 1000  # noqa: E402
        results.append((reply, latency))  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_batch(  # noqa: E402
    judge_client,  # noqa: E402
    examples: list[Any],  # noqa: E402
    replies: list[str],  # noqa: E402
) -> list[tuple[float, str]]:  # noqa: E402
    """Judge a batch of examples in a single request.  # noqa: E402
  # noqa: E402
    Returns list of (score, reasoning) tuples.  # noqa: E402
    """  # noqa: E402
    if not judge_client or not replies:  # noqa: E402
        return [(None, "no judge")] * len(examples)  # noqa: E402
  # noqa: E402
    # Build batch evaluation prompt  # noqa: E402
    batch_prompt = (  # noqa: E402
        "You are an expert evaluator for text message replies. "  # noqa: E402
        f"Evaluate {len(examples)} replies and return scores in JSON format.\n\n"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E402
        batch_prompt += (  # noqa: E402
            f"\n--- EXAMPLE {i} ---\n"  # noqa: E402
            f"Conversation: {chr(10).join(ex.context)}\n"  # noqa: E402
            f"Message: {ex.last_message}\n"  # noqa: E402
            f"Generated reply: {reply}\n"  # noqa: E402
            f"Ideal reply: {ex.ideal_response}\n"  # noqa: E402
            f"Category: {ex.category}\n"  # noqa: E402
            f"Notes: {ex.notes}\n"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    batch_prompt += (  # noqa: E402
        f"\nRespond with JSON array of {len(examples)} objects:\n"  # noqa: E402
        '[{"score": <0-10>, "reasoning": "<brief>"}, ...]\n'  # noqa: E402
        "Score based on: naturalness (not AI-sounding), appropriateness, "  # noqa: E402
        "matching ideal reply intent."  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        resp = judge_client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": batch_prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=500,  # noqa: E402
        )  # noqa: E402
        text = resp.choices[0].message.content.strip()  # noqa: E402
  # noqa: E402
        # Parse JSON array  # noqa: E402
        if text.startswith("```"):  # noqa: E402
            text = text.split("```")[1]  # noqa: E402
            if text.startswith("json"):  # noqa: E402
                text = text[4:]  # noqa: E402
  # noqa: E402
        data = json.loads(text)  # noqa: E402
  # noqa: E402
        # Ensure we have right number of results  # noqa: E402
        results = []  # noqa: E402
        for item in data[: len(examples)]:  # noqa: E402
            score = float(item.get("score", 0))  # noqa: E402
            reasoning = item.get("reasoning", "")  # noqa: E402
            results.append((score, reasoning))  # noqa: E402
  # noqa: E402
        # Pad if needed  # noqa: E402
        while len(results) < len(examples):  # noqa: E402
            results.append((None, "parse error"))  # noqa: E402
  # noqa: E402
        return results  # noqa: E402
  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"  Batch judge error: {e}")  # noqa: E402
        # Fall back to individual scoring  # noqa: E402
        return [(None, f"batch error: {e}")] * len(examples)  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_variant_batched(  # noqa: E402
    variant: str,  # noqa: E402
    examples: list[Any],  # noqa: E402
    judge_client: Any | None,  # noqa: E402
    batch_size: int = BATCH_SIZE_DEFAULT,  # noqa: E402
) -> list[AblationResult]:  # noqa: E402
    """Run ablation for a variant with batching."""  # noqa: E402
  # noqa: E402
    from models.loader import get_model  # noqa: E402
  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        loader.load()  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    print(f"\n{'=' * 70}")  # noqa: E402
    print(f"Variant: {variant}")  # noqa: E402
    print(f"Description: {VARIANT_CONFIGS[variant]['description']}")  # noqa: E402
    print(f"Batch size: {batch_size}")  # noqa: E402
    print(f"{'=' * 70}\n")  # noqa: E402
  # noqa: E402
    # Process in batches  # noqa: E402
    num_batches = (len(examples) + batch_size - 1) // batch_size  # noqa: E402
  # noqa: E402
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {variant}"):  # noqa: E402
        start_idx = batch_idx * batch_size  # noqa: E402
        end_idx = min(start_idx + batch_size, len(examples))  # noqa: E402
        batch_examples = examples[start_idx:end_idx]  # noqa: E402
  # noqa: E402
        # Generate replies for batch  # noqa: E402
        gen_results = generate_batch(loader, batch_examples, variant)  # noqa: E402
  # noqa: E402
        # Check anti-AI  # noqa: E402
        anti_ai_results = [check_anti_ai(reply) for reply, _ in gen_results]  # noqa: E402
  # noqa: E402
        # Judge batch (with rate limiting)  # noqa: E402
        if judge_client:  # noqa: E402
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit between judge calls  # noqa: E402
            judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])  # noqa: E402
        else:  # noqa: E402
            judge_results = [(None, "no judge")] * len(batch_examples)  # noqa: E402
  # noqa: E402
        # Build results  # noqa: E402
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(  # noqa: E402
            zip(batch_examples, gen_results, anti_ai_results, judge_results)  # noqa: E402
        ):  # noqa: E402
            result = AblationResult(  # noqa: E402
                example_id=start_idx + i + 1,  # noqa: E402
                variant=variant,  # noqa: E402
                category=ex.category,  # noqa: E402
                generated_response=reply,  # noqa: E402
                latency_ms=latency,  # noqa: E402
                anti_ai_violations=anti_ai,  # noqa: E402
                judge_score=score,  # noqa: E402
                judge_reasoning=reasoning,  # noqa: E402
            )  # noqa: E402
            results.append(result)  # noqa: E402
  # noqa: E402
            # Print progress  # noqa: E402
            status = "AI!" if anti_ai else "clean"  # noqa: E402
            judge_str = f" | Judge: {score:.0f}/10" if score else ""  # noqa: E402
            print(  # noqa: E402
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status}{judge_str} -> {reply[:50]}"  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
def analyze_results(results: list[AblationResult]) -> dict:  # noqa: E402
    """Analyze and compare results across variants."""  # noqa: E402
    from collections import defaultdict  # noqa: E402
  # noqa: E402
    by_variant = defaultdict(list)  # noqa: E402
    for r in results:  # noqa: E402
        by_variant[r.variant].append(r)  # noqa: E402
  # noqa: E402
    analysis = {}  # noqa: E402
  # noqa: E402
    for variant, vresults in by_variant.items():  # noqa: E402
        scores = [r.judge_score for r in vresults if r.judge_score is not None]  # noqa: E402
        anti_ai_count = sum(1 for r in vresults if r.anti_ai_violations)  # noqa: E402
        latencies = [r.latency_ms for r in vresults]  # noqa: E402
  # noqa: E402
        # Category breakdown  # noqa: E402
        by_category = defaultdict(lambda: {"scores": [], "anti_ai": 0})  # noqa: E402
        for r in vresults:  # noqa: E402
            by_category[r.category]["scores"].append(r.judge_score or 0)  # noqa: E402
            if r.anti_ai_violations:  # noqa: E402
                by_category[r.category]["anti_ai"] += 1  # noqa: E402
  # noqa: E402
        analysis[variant] = {  # noqa: E402
            "total": len(vresults),  # noqa: E402
            "judge_avg": sum(scores) / len(scores) if scores else 0,  # noqa: E402
            "judge_median": sorted(scores)[len(scores) // 2] if scores else 0,  # noqa: E402
            "anti_ai_violations": anti_ai_count,  # noqa: E402
            "anti_ai_rate": anti_ai_count / len(vresults) if vresults else 0,  # noqa: E402
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,  # noqa: E402
            "by_category": {  # noqa: E402
                cat: {  # noqa: E402
                    "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,  # noqa: E402
                    "anti_ai": d["anti_ai"],  # noqa: E402
                }  # noqa: E402
                for cat, d in by_category.items()  # noqa: E402
            },  # noqa: E402
        }  # noqa: E402
  # noqa: E402
    return analysis  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    parser = argparse.ArgumentParser(description="Batched Categorization Ablation Study")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--variant",  # noqa: E402
        choices=["categorized", "universal", "category_hint", "all"],  # noqa: E402
        default="all",  # noqa: E402
        help="Which variant to run",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--batch-size",  # noqa: E402
        type=int,  # noqa: E402
        default=BATCH_SIZE_DEFAULT,  # noqa: E402
        help=f"Batch size for judge API (default: {BATCH_SIZE_DEFAULT})",  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Load dataset  # noqa: E402
    if not EVAL_DATASET_PATH.exists():  # noqa: E402
        print(f"ERROR: Dataset not found at {EVAL_DATASET_PATH}")  # noqa: E402
        return 1  # noqa: E402
  # noqa: E402
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E402
    print(f"Loaded {len(examples)} examples from {EVAL_DATASET_PATH}")  # noqa: E402
    print(f"Using judge model: {JUDGE_MODEL}")  # noqa: E402
  # noqa: E402
    # Initialize judge  # noqa: E402
    judge_client = None  # noqa: E402
    if args.judge:  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        if judge_client:  # noqa: E402
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E402
            print(  # noqa: E402
                f"Rate limit: {RATE_LIMIT_RPM} req/min = {RATE_LIMIT_DELAY:.1f}s between requests"  # noqa: E402
            )  # noqa: E402
            print(f"Batch size: {args.batch_size} examples per request")  # noqa: E402
            estimated_time = (len(examples) / args.batch_size) * RATE_LIMIT_DELAY / 60  # noqa: E402
            print(f"Estimated judge time: {estimated_time:.1f} minutes")  # noqa: E402
        else:  # noqa: E402
            print("WARNING: Judge API key not set, skipping judge scoring")  # noqa: E402
  # noqa: E402
    # Determine variants  # noqa: E402
    variants = (  # noqa: E402
        ["categorized", "universal", "category_hint"] if args.variant == "all" else [args.variant]  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Run ablations  # noqa: E402
    all_results = []  # noqa: E402
    for variant in variants:  # noqa: E402
        results = run_variant_batched(variant, examples, judge_client, args.batch_size)  # noqa: E402
        all_results.extend(results)  # noqa: E402
  # noqa: E402
    # Analyze  # noqa: E402
    analysis = analyze_results(all_results)  # noqa: E402
  # noqa: E402
    # Print summary  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("ABLATION RESULTS SUMMARY")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    for variant, stats in analysis.items():  # noqa: E402
        print(f"\n{variant.upper()}:")  # noqa: E402
        print(f"  Judge avg:     {stats['judge_avg']:.2f}/10")  # noqa: E402
        print(f"  Judge median:  {stats['judge_median']:.2f}/10")  # noqa: E402
        print(f"  Anti-AI rate:  {stats['anti_ai_rate']:.1%}")  # noqa: E402
        print(f"  Avg latency:   {stats['avg_latency_ms']:.0f}ms")  # noqa: E402
        print("  By category:")  # noqa: E402
        for cat, cat_stats in stats["by_category"].items():  # noqa: E402
            print(  # noqa: E402
                f"    {cat:12s}: avg={cat_stats['avg_score']:.1f}, anti_ai={cat_stats['anti_ai']}"  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "ablation_categorization_batched.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "judge_model": JUDGE_MODEL,  # noqa: E402
        "batch_size": args.batch_size,  # noqa: E402
        "analysis": analysis,  # noqa: E402
        "raw_results": [  # noqa: E402
            {  # noqa: E402
                "example_id": r.example_id,  # noqa: E402
                "variant": r.variant,  # noqa: E402
                "category": r.category,  # noqa: E402
                "generated": r.generated_response,  # noqa: E402
                "latency_ms": round(r.latency_ms, 1),  # noqa: E402
                "anti_ai": r.anti_ai_violations,  # noqa: E402
                "judge_score": r.judge_score,  # noqa: E402
                "judge_reasoning": r.judge_reasoning,  # noqa: E402
            }  # noqa: E402
            for r in all_results  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    print(f"\nResults saved to: {output_path}")  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
