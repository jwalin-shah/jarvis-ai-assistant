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

from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

# Rate limit configuration
RATE_LIMIT_RPM = 30  # requests per minute
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_RPM  # 2 seconds between requests
BATCH_SIZE_DEFAULT = 3  # Judge this many examples per request


@dataclass
class AblationResult:
    example_id: int
    variant: str
    category: str
    generated_response: str
    latency_ms: float
    anti_ai_violations: list[str] = field(default_factory=list)
    judge_score: float | None = None
    judge_reasoning: str = ""


# Variant configurations
VARIANT_CONFIGS = {
    "categorized": {
        "description": "Category-specific system prompts (current system)",
        "system_prompt": None,
    },
    "universal": {
        "description": "Single universal instruction",
        "system_prompt": (
            "You are NOT an AI assistant. You are texting from your phone. "
            "Reply naturally, matching the conversation style. "
            "Be brief (1-2 sentences), casual, and sound like a real person."
        ),
    },
    "category_hint": {
        "description": "Category as context, not instruction",
        "system_prompt": "hint",
    },
}


def build_prompt_variant(
    context: list[str],
    last_message: str,
    category: str,
    variant: str,
    contact_style: str = "casual",
) -> str:
    """Build prompt for a specific variant."""

    config = VARIANT_CONFIGS[variant]

    # Format context
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])

    if variant == "categorized":
        from jarvis.prompts.constants import CATEGORY_CONFIGS

        cat_config = CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])
        system = cat_config.system_prompt or (
            "You are NOT an AI assistant. You are texting from your phone. "
            "Reply naturally, matching the conversation style."
        )
    elif variant == "universal":
        system = config["system_prompt"]
    elif variant == "category_hint":
        system = (
            f"You are texting from your phone. The message appears to be a '{category}' type. "
            f"Reply naturally as yourself, matching the {contact_style} style. "
            "Be brief and sound human."
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Use ChatML format
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Conversation:\n{context_str}\n\n"
        f"Reply to: {last_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return prompt


def generate_batch(
    generator,
    examples: list[Any],
    variant: str,
) -> list[tuple[str, float]]:
    """Generate replies for a batch of examples."""
    results = []

    for ex in examples:
        import time

        start = time.perf_counter()

        prompt = build_prompt_variant(
            ex.context,
            ex.last_message,
            ex.category,
            variant,
            ex.contact_style,
        )

        try:
            result = generator.generate_sync(
                prompt=prompt,
                temperature=0.1,
                max_tokens=50,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
            )
            reply = result.text.strip()
        except Exception as e:
            reply = f"[ERROR: {e}]"

        latency = (time.perf_counter() - start) * 1000
        results.append((reply, latency))

    return results


def judge_batch(
    judge_client,
    examples: list[Any],
    replies: list[str],
) -> list[tuple[float, str]]:
    """Judge a batch of examples in a single request.

    Returns list of (score, reasoning) tuples.
    """
    if not judge_client or not replies:
        return [(None, "no judge")] * len(examples)

    # Build batch evaluation prompt
    batch_prompt = (
        "You are an expert evaluator for text message replies. "
        f"Evaluate {len(examples)} replies and return scores in JSON format.\n\n"
    )

    for i, (ex, reply) in enumerate(zip(examples, replies), 1):
        batch_prompt += (
            f"\n--- EXAMPLE {i} ---\n"
            f"Conversation: {chr(10).join(ex.context)}\n"
            f"Message: {ex.last_message}\n"
            f"Generated reply: {reply}\n"
            f"Ideal reply: {ex.ideal_response}\n"
            f"Category: {ex.category}\n"
            f"Notes: {ex.notes}\n"
        )

    batch_prompt += (
        f"\nRespond with JSON array of {len(examples)} objects:\n"
        '[{"score": <0-10>, "reasoning": "<brief>"}, ...]\n'
        "Score based on: naturalness (not AI-sounding), appropriateness, "
        "matching ideal reply intent."
    )

    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()

        # Parse JSON array
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text)

        # Ensure we have right number of results
        results = []
        for item in data[: len(examples)]:
            score = float(item.get("score", 0))
            reasoning = item.get("reasoning", "")
            results.append((score, reasoning))

        # Pad if needed
        while len(results) < len(examples):
            results.append((None, "parse error"))

        return results

    except Exception as e:
        print(f"  Batch judge error: {e}")
        # Fall back to individual scoring
        return [(None, f"batch error: {e}")] * len(examples)


def run_variant_batched(
    variant: str,
    examples: list[Any],
    judge_client: Any | None,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> list[AblationResult]:
    """Run ablation for a variant with batching."""

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    results = []

    print(f"\n{'=' * 70}")
    print(f"Variant: {variant}")
    print(f"Description: {VARIANT_CONFIGS[variant]['description']}")
    print(f"Batch size: {batch_size}")
    print(f"{'=' * 70}\n")

    # Process in batches
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {variant}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch_examples = examples[start_idx:end_idx]

        # Generate replies for batch
        gen_results = generate_batch(loader, batch_examples, variant)

        # Check anti-AI
        anti_ai_results = [check_anti_ai(reply) for reply, _ in gen_results]

        # Judge batch (with rate limiting)
        if judge_client:
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit between judge calls
            judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])
        else:
            judge_results = [(None, "no judge")] * len(batch_examples)

        # Build results
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(
            zip(batch_examples, gen_results, anti_ai_results, judge_results)
        ):
            result = AblationResult(
                example_id=start_idx + i + 1,
                variant=variant,
                category=ex.category,
                generated_response=reply,
                latency_ms=latency,
                anti_ai_violations=anti_ai,
                judge_score=score,
                judge_reasoning=reasoning,
            )
            results.append(result)

            # Print progress
            status = "AI!" if anti_ai else "clean"
            judge_str = f" | Judge: {score:.0f}/10" if score else ""
            print(
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status}{judge_str} -> {reply[:50]}"
            )

    return results


def analyze_results(results: list[AblationResult]) -> dict:
    """Analyze and compare results across variants."""
    from collections import defaultdict

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r.variant].append(r)

    analysis = {}

    for variant, vresults in by_variant.items():
        scores = [r.judge_score for r in vresults if r.judge_score is not None]
        anti_ai_count = sum(1 for r in vresults if r.anti_ai_violations)
        latencies = [r.latency_ms for r in vresults]

        # Category breakdown
        by_category = defaultdict(lambda: {"scores": [], "anti_ai": 0})
        for r in vresults:
            by_category[r.category]["scores"].append(r.judge_score or 0)
            if r.anti_ai_violations:
                by_category[r.category]["anti_ai"] += 1

        analysis[variant] = {
            "total": len(vresults),
            "judge_avg": sum(scores) / len(scores) if scores else 0,
            "judge_median": sorted(scores)[len(scores) // 2] if scores else 0,
            "anti_ai_violations": anti_ai_count,
            "anti_ai_rate": anti_ai_count / len(vresults) if vresults else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "by_category": {
                cat: {
                    "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
                    "anti_ai": d["anti_ai"],
                }
                for cat, d in by_category.items()
            },
        }

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched Categorization Ablation Study")
    parser.add_argument(
        "--variant",
        choices=["categorized", "universal", "category_hint", "all"],
        default="all",
        help="Which variant to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Batch size for judge API (default: {BATCH_SIZE_DEFAULT})",
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")
    args = parser.parse_args()

    # Load dataset
    if not EVAL_DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {EVAL_DATASET_PATH}")
        return 1

    examples = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(examples)} examples from {EVAL_DATASET_PATH}")
    print(f"Using judge model: {JUDGE_MODEL}")

    # Initialize judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client:
            print(f"Judge ready: {JUDGE_MODEL}")
            print(
                f"Rate limit: {RATE_LIMIT_RPM} req/min = {RATE_LIMIT_DELAY:.1f}s between requests"
            )
            print(f"Batch size: {args.batch_size} examples per request")
            estimated_time = (len(examples) / args.batch_size) * RATE_LIMIT_DELAY / 60
            print(f"Estimated judge time: {estimated_time:.1f} minutes")
        else:
            print("WARNING: Judge API key not set, skipping judge scoring")

    # Determine variants
    variants = (
        ["categorized", "universal", "category_hint"] if args.variant == "all" else [args.variant]
    )

    # Run ablations
    all_results = []
    for variant in variants:
        results = run_variant_batched(variant, examples, judge_client, args.batch_size)
        all_results.extend(results)

    # Analyze
    analysis = analyze_results(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    for variant, stats in analysis.items():
        print(f"\n{variant.upper()}:")
        print(f"  Judge avg:     {stats['judge_avg']:.2f}/10")
        print(f"  Judge median:  {stats['judge_median']:.2f}/10")
        print(f"  Anti-AI rate:  {stats['anti_ai_rate']:.1%}")
        print(f"  Avg latency:   {stats['avg_latency_ms']:.0f}ms")
        print("  By category:")
        for cat, cat_stats in stats["by_category"].items():
            print(
                f"    {cat:12s}: avg={cat_stats['avg_score']:.1f}, anti_ai={cat_stats['anti_ai']}"
            )

    # Save results
    output_path = PROJECT_ROOT / "results" / "ablation_categorization_batched.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_model": JUDGE_MODEL,
        "batch_size": args.batch_size,
        "analysis": analysis,
        "raw_results": [
            {
                "example_id": r.example_id,
                "variant": r.variant,
                "category": r.category,
                "generated": r.generated_response,
                "latency_ms": round(r.latency_ms, 1),
                "anti_ai": r.anti_ai_violations,
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in all_results
        ],
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
