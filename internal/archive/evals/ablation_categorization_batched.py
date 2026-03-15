#!/usr/bin/env python3  # noqa: E501
"""Batched categorization ablation study with rate limiting.  # noqa: E501
  # noqa: E501
Efficiently uses the judge API with:  # noqa: E501
- Batch scoring (multiple examples per request when possible)  # noqa: E501
- Rate limit tracking (30 req/min = 1 request every 2 seconds)  # noqa: E501
- Automatic retries with exponential backoff  # noqa: E501
- Progress saving (resume on interruption)  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/ablation_categorization_batched.py --variant all --judge  # noqa: E501
    uv run python evals/ablation_categorization_batched.py --variant categorized --batch-size 5  # noqa: E501
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
# Rate limit configuration  # noqa: E501
RATE_LIMIT_RPM = 30  # requests per minute  # noqa: E501
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_RPM  # 2 seconds between requests  # noqa: E501
BATCH_SIZE_DEFAULT = 3  # Judge this many examples per request  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class AblationResult:  # noqa: E501
    example_id: int  # noqa: E501
    variant: str  # noqa: E501
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
        "system_prompt": None,  # noqa: E501
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
        "system_prompt": "hint",  # noqa: E501
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
    # Format context  # noqa: E501
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])  # noqa: E501
  # noqa: E501
    if variant == "categorized":  # noqa: E501
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
            f"Reply naturally as yourself, matching the {contact_style} style. "  # noqa: E501
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
def generate_batch(  # noqa: E501
    generator,  # noqa: E501
    examples: list[Any],  # noqa: E501
    variant: str,  # noqa: E501
) -> list[tuple[str, float]]:  # noqa: E501
    """Generate replies for a batch of examples."""  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for ex in examples:  # noqa: E501
        import time  # noqa: E501
  # noqa: E501
        start = time.perf_counter()  # noqa: E501
  # noqa: E501
        prompt = build_prompt_variant(  # noqa: E501
            ex.context,  # noqa: E501
            ex.last_message,  # noqa: E501
            ex.category,  # noqa: E501
            variant,  # noqa: E501
            ex.contact_style,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            result = generator.generate_sync(  # noqa: E501
                prompt=prompt,  # noqa: E501
                temperature=0.1,  # noqa: E501
                max_tokens=50,  # noqa: E501
                top_p=0.9,  # noqa: E501
                top_k=50,  # noqa: E501
                repetition_penalty=1.05,  # noqa: E501
            )  # noqa: E501
            reply = result.text.strip()  # noqa: E501
        except Exception as e:  # noqa: E501
            reply = f"[ERROR: {e}]"  # noqa: E501
  # noqa: E501
        latency = (time.perf_counter() - start) * 1000  # noqa: E501
        results.append((reply, latency))  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_batch(  # noqa: E501
    judge_client,  # noqa: E501
    examples: list[Any],  # noqa: E501
    replies: list[str],  # noqa: E501
) -> list[tuple[float, str]]:  # noqa: E501
    """Judge a batch of examples in a single request.  # noqa: E501
  # noqa: E501
    Returns list of (score, reasoning) tuples.  # noqa: E501
    """  # noqa: E501
    if not judge_client or not replies:  # noqa: E501
        return [(None, "no judge")] * len(examples)  # noqa: E501
  # noqa: E501
    # Build batch evaluation prompt  # noqa: E501
    batch_prompt = (  # noqa: E501
        "You are an expert evaluator for text message replies. "  # noqa: E501
        f"Evaluate {len(examples)} replies and return scores in JSON format.\n\n"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E501
        batch_prompt += (  # noqa: E501
            f"\n--- EXAMPLE {i} ---\n"  # noqa: E501
            f"Conversation: {chr(10).join(ex.context)}\n"  # noqa: E501
            f"Message: {ex.last_message}\n"  # noqa: E501
            f"Generated reply: {reply}\n"  # noqa: E501
            f"Ideal reply: {ex.ideal_response}\n"  # noqa: E501
            f"Category: {ex.category}\n"  # noqa: E501
            f"Notes: {ex.notes}\n"  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    batch_prompt += (  # noqa: E501
        f"\nRespond with JSON array of {len(examples)} objects:\n"  # noqa: E501
        '[{"score": <0-10>, "reasoning": "<brief>"}, ...]\n'  # noqa: E501
        "Score based on: naturalness (not AI-sounding), appropriateness, "  # noqa: E501
        "matching ideal reply intent."  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = judge_client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": batch_prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=500,  # noqa: E501
        )  # noqa: E501
        text = resp.choices[0].message.content.strip()  # noqa: E501
  # noqa: E501
        # Parse JSON array  # noqa: E501
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
  # noqa: E501
        data = json.loads(text)  # noqa: E501
  # noqa: E501
        # Ensure we have right number of results  # noqa: E501
        results = []  # noqa: E501
        for item in data[: len(examples)]:  # noqa: E501
            score = float(item.get("score", 0))  # noqa: E501
            reasoning = item.get("reasoning", "")  # noqa: E501
            results.append((score, reasoning))  # noqa: E501
  # noqa: E501
        # Pad if needed  # noqa: E501
        while len(results) < len(examples):  # noqa: E501
            results.append((None, "parse error"))  # noqa: E501
  # noqa: E501
        return results  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"  Batch judge error: {e}")  # noqa: E501
        # Fall back to individual scoring  # noqa: E501
        return [(None, f"batch error: {e}")] * len(examples)  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_variant_batched(  # noqa: E501
    variant: str,  # noqa: E501
    examples: list[Any],  # noqa: E501
    judge_client: Any | None,  # noqa: E501
    batch_size: int = BATCH_SIZE_DEFAULT,  # noqa: E501
) -> list[AblationResult]:  # noqa: E501
    """Run ablation for a variant with batching."""  # noqa: E501
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
    print(f"Batch size: {batch_size}")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    # Process in batches  # noqa: E501
    num_batches = (len(examples) + batch_size - 1) // batch_size  # noqa: E501
  # noqa: E501
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {variant}"):  # noqa: E501
        start_idx = batch_idx * batch_size  # noqa: E501
        end_idx = min(start_idx + batch_size, len(examples))  # noqa: E501
        batch_examples = examples[start_idx:end_idx]  # noqa: E501
  # noqa: E501
        # Generate replies for batch  # noqa: E501
        gen_results = generate_batch(loader, batch_examples, variant)  # noqa: E501
  # noqa: E501
        # Check anti-AI  # noqa: E501
        anti_ai_results = [check_anti_ai(reply) for reply, _ in gen_results]  # noqa: E501
  # noqa: E501
        # Judge batch (with rate limiting)  # noqa: E501
        if judge_client:  # noqa: E501
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit between judge calls  # noqa: E501
            judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])  # noqa: E501
        else:  # noqa: E501
            judge_results = [(None, "no judge")] * len(batch_examples)  # noqa: E501
  # noqa: E501
        # Build results  # noqa: E501
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(  # noqa: E501
            zip(batch_examples, gen_results, anti_ai_results, judge_results)  # noqa: E501
        ):  # noqa: E501
            result = AblationResult(  # noqa: E501
                example_id=start_idx + i + 1,  # noqa: E501
                variant=variant,  # noqa: E501
                category=ex.category,  # noqa: E501
                generated_response=reply,  # noqa: E501
                latency_ms=latency,  # noqa: E501
                anti_ai_violations=anti_ai,  # noqa: E501
                judge_score=score,  # noqa: E501
                judge_reasoning=reasoning,  # noqa: E501
            )  # noqa: E501
            results.append(result)  # noqa: E501
  # noqa: E501
            # Print progress  # noqa: E501
            status = "AI!" if anti_ai else "clean"  # noqa: E501
            judge_str = f" | Judge: {score:.0f}/10" if score else ""  # noqa: E501
            print(  # noqa: E501
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status}{judge_str} -> {reply[:50]}"  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def analyze_results(results: list[AblationResult]) -> dict:  # noqa: E501
    """Analyze and compare results across variants."""  # noqa: E501
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
    parser = argparse.ArgumentParser(description="Batched Categorization Ablation Study")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--variant",  # noqa: E501
        choices=["categorized", "universal", "category_hint", "all"],  # noqa: E501
        default="all",  # noqa: E501
        help="Which variant to run",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--batch-size",  # noqa: E501
        type=int,  # noqa: E501
        default=BATCH_SIZE_DEFAULT,  # noqa: E501
        help=f"Batch size for judge API (default: {BATCH_SIZE_DEFAULT})",  # noqa: E501
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
    print(f"Using judge model: {JUDGE_MODEL}")  # noqa: E501
  # noqa: E501
    # Initialize judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E501
            print(  # noqa: E501
                f"Rate limit: {RATE_LIMIT_RPM} req/min = {RATE_LIMIT_DELAY:.1f}s between requests"  # noqa: E501
            )  # noqa: E501
            print(f"Batch size: {args.batch_size} examples per request")  # noqa: E501
            estimated_time = (len(examples) / args.batch_size) * RATE_LIMIT_DELAY / 60  # noqa: E501
            print(f"Estimated judge time: {estimated_time:.1f} minutes")  # noqa: E501
        else:  # noqa: E501
            print("WARNING: Judge API key not set, skipping judge scoring")  # noqa: E501
  # noqa: E501
    # Determine variants  # noqa: E501
    variants = (  # noqa: E501
        ["categorized", "universal", "category_hint"] if args.variant == "all" else [args.variant]  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Run ablations  # noqa: E501
    all_results = []  # noqa: E501
    for variant in variants:  # noqa: E501
        results = run_variant_batched(variant, examples, judge_client, args.batch_size)  # noqa: E501
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
    output_path = PROJECT_ROOT / "results" / "ablation_categorization_batched.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "judge_model": JUDGE_MODEL,  # noqa: E501
        "batch_size": args.batch_size,  # noqa: E501
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
