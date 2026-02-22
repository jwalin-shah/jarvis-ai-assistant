#!/usr/bin/env python3
"""Ablation study: Does categorization actually help reply quality?

Tests 3 variants:
1. categorized: Current system with category-specific instructions
2. universal: Single instruction for all messages
3. category_hint: Category mentioned but not prescriptive

Usage:
    uv run python evals/ablation_categorization.py --judge
    uv run python evals/ablation_categorization.py --variant universal --judge
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


@dataclass
class AblationResult:
    example_id: int
    variant: str  # "categorized", "universal", "category_hint"
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
        "system_prompt": None,  # Uses CATEGORY_CONFIGS
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
        "system_prompt": "hint",  # Special marker
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

    # Format context with timestamps
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])

    if variant == "categorized":
        # Use current category system
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
            f"Reply naturally as yourself, matching the {contact_style} style of the conversation. "
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


def generate_with_variant(
    generator,
    context: list[str],
    last_message: str,
    category: str,
    variant: str,
    contact_style: str = "casual",
) -> tuple[str, float]:
    """Generate reply using specified variant. Returns (response, latency_ms)."""
    import time

    prompt = build_prompt_variant(context, last_message, category, variant, contact_style)

    start = time.perf_counter()
    result = generator.generate_sync(
        prompt=prompt,
        temperature=0.1,
        max_tokens=50,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return result.text.strip(), latency_ms


def run_ablation(
    variant: str,
    examples: list[Any],
    judge_client: Any | None = None,
) -> list[AblationResult]:
    """Run ablation for a single variant."""

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    results = []

    print(f"\n{'=' * 70}")
    print(f"Variant: {variant}")
    print(f"Description: {VARIANT_CONFIGS[variant]['description']}")
    print(f"{'=' * 70}\n")

    for i, ex in enumerate(tqdm(examples, desc=f"Running {variant}"), 1):
        # Generate response
        response, latency = generate_with_variant(
            loader,
            ex.context,
            ex.last_message,
            ex.category,
            variant,
            ex.contact_style,
        )

        # Check anti-AI
        anti_ai = check_anti_ai(response)

        # Judge scoring
        judge_score = None
        judge_reasoning = ""
        if judge_client:
            try:
                prompt = (
                    "You are an expert evaluator for text message replies.\n\n"
                    f"Conversation:\n{chr(10).join(ex.context)}\n\n"
                    f"Message to reply to: {ex.last_message}\n\n"
                    f"Generated reply: {response}\n\n"
                    f"Ideal reply: {ex.ideal_response}\n\n"
                    f"Category: {ex.category}\n"
                    f"Notes: {ex.notes}\n\n"
                    "Score 0-10. Consider:\n"
                    "- Does it sound like a real text message (not AI)?\n"
                    "- Is it appropriate for the conversation?\n"
                    "- Does it match the ideal reply in intent/tone?\n\n"
                    'Respond: {"score": <0-10>, "reasoning": "<brief>"}'
                )
                resp = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                judge_score = float(data["score"])
                judge_reasoning = data.get("reasoning", "")
            except Exception as e:
                judge_reasoning = f"judge error: {e}"

        result = AblationResult(
            example_id=i,
            variant=variant,
            category=ex.category,
            generated_response=response,
            latency_ms=latency,
            anti_ai_violations=anti_ai,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
        )
        results.append(result)

        # Print progress
        status = "AI!" if anti_ai else "clean"
        judge_str = f" | Judge: {judge_score:.0f}/10" if judge_score else ""
        print(f"[{i:2d}] [{ex.category:12s}] {status}{judge_str} -> {response[:50]}")

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
    parser = argparse.ArgumentParser(description="Categorization Ablation Study")
    parser.add_argument(
        "--variant",
        choices=["categorized", "universal", "category_hint", "all"],
        default="all",
        help="Which variant to run (default: all)",
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")
    args = parser.parse_args()

    # Load dataset
    if not EVAL_DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {EVAL_DATASET_PATH}")
        return 1

    examples = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(examples)} examples from {EVAL_DATASET_PATH}")

    # Initialize judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client:
            print(f"Judge ready: {JUDGE_MODEL}")
        else:
            print("WARNING: Judge API key not set, skipping judge scoring")

    # Determine variants to run
    variants = (
        ["categorized", "universal", "category_hint"] if args.variant == "all" else [args.variant]
    )

    # Run ablations
    all_results = []
    for variant in variants:
        results = run_ablation(variant, examples, judge_client)
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
    output_path = PROJECT_ROOT / "results" / "ablation_categorization.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
