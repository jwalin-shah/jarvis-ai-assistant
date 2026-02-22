#!/usr/bin/env python3
"""Ablation study: Context depth and RAG impact on reply quality.

Tests:
1. Context depth: 0, 3, 5, 10 messages
2. RAG: with vs without similar examples
3. Combined: context + RAG vs baseline

Usage:
    uv run python evals/ablation_context_rag.py --judge
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

from evals.eval_pipeline import (  # noqa: E402
    EVAL_DATASET_PATH,
    check_anti_ai,
    load_eval_dataset,
)
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

BATCH_SIZE = 10
RATE_LIMIT_DELAY = 2.1


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


# The winning universal prompt from previous experiments
UNIVERSAL_PROMPT = """You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person."""


def build_prompt_with_context(
    system: str,
    context_messages: list[str],
    last_message: str,
    similar_examples: list[tuple[str, str]] | None = None,
) -> str:
    """Build prompt with variable context depth and optional RAG examples."""

    # Format context (variable depth)
    if context_messages:
        context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context_messages, 1)])
    else:
        context_str = "(No previous context)"

    # Build extra context with RAG examples if provided
    extra_parts = []

    if similar_examples:
        examples_str = "\n\n".join(
            [
                f"Example {i}:\nThey said: {ex[0][:100]}...\nYou replied: {ex[1]}"
                for i, ex in enumerate(similar_examples[:3], 1)
            ]
        )
        extra_parts.append(f"Your typical replies:\n{examples_str}")

    extra_context = "\n\n".join(extra_parts) if extra_parts else ""

    # Build full prompt
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"

    if extra_context:
        prompt += f"<|im_start|>user\n{extra_context}<|im_end|>\n"

    prompt += (
        f"<|im_start|>user\n"
        f"Conversation:\n{context_str}\n\n"
        f"Reply to: {last_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return prompt


def get_context_at_depth(full_context: list[str], depth: int) -> list[str]:
    """Get last N messages from context."""
    if depth == 0:
        return []
    return full_context[-depth:] if len(full_context) >= depth else full_context


def get_mock_rag_examples(category: str) -> list[tuple[str, str]]:
    """Generate mock RAG examples based on category."""
    examples = {
        "question": [
            ("What time is dinner?", "7pm"),
            ("Did you finish?", "yeah just did"),
        ],
        "request": [
            ("Can you pick me up?", "sure where?"),
            ("Send me the file", "on it"),
        ],
        "emotion": [
            ("I'm so stressed", "that sucks, wanna talk?"),
            ("I got the job!", "omg congrats!!"),
        ],
        "statement": [
            ("Just got home", "nice, how was it?"),
            ("Check out this photo", "looks great!"),
        ],
        "acknowledge": [
            ("Running late", "no worries"),
            ("Thanks!", "np"),
        ],
        "closing": [
            ("Talk later", "later!"),
            ("Goodnight", "night!"),
        ],
    }
    return examples.get(category, [])


def generate_batch(
    generator,
    examples: list[Any],
    variant_config: dict,
) -> list[tuple[str, float]]:
    """Generate replies for a batch."""
    results = []

    for ex in examples:
        import time

        start = time.perf_counter()

        # Get context at specified depth
        context = get_context_at_depth(ex.context, variant_config["context_depth"])

        # Get RAG examples if enabled
        rag_examples = None
        if variant_config.get("use_rag"):
            rag_examples = get_mock_rag_examples(ex.category)

        prompt = build_prompt_with_context(
            UNIVERSAL_PROMPT,
            context,
            ex.last_message,
            rag_examples,
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
    """Judge multiple examples in a single API call."""
    if not judge_client or not replies:
        return [(5.0, "no judge")] * len(examples)

    batch_text = f"""Evaluate {len(examples)} text message replies.
Score 0-10 based on naturalness and appropriateness.

"""

    for i, (ex, reply) in enumerate(zip(examples, replies), 1):
        batch_text += f"""
--- EXAMPLE {i} ---
Context: {" | ".join(ex.context[-3:])}
Message: {ex.last_message}
Generated: {reply}
Ideal: {ex.ideal_response}
"""

    batch_text += f"""
Respond with JSON array (no markdown):
[{{"score": 7, "reasoning": "brief"}}, ...]
Exactly {len(examples)} objects.
"""

    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": batch_text}],
            temperature=0.0,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        results = []
        for item in data[: len(examples)]:
            score = float(item.get("score", 5))
            reasoning = item.get("reasoning", "")
            results.append((score, reasoning))

        while len(results) < len(examples):
            results.append((5.0, "parse error"))

        return results

    except Exception as e:
        print(f"  Batch judge error: {e}")
        return [(5.0, f"error: {e}")] * len(examples)


def run_variant(
    name: str,
    config: dict,
    examples: list[Any],
    judge_client: Any | None,
) -> tuple[list[AblationResult], dict]:
    """Run ablation for a single variant."""

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    print(f"\n{'=' * 70}")
    print(f"Variant: {name}")
    print(f"Config: {config}")
    print(f"{'=' * 70}\n")

    results = []
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {name}"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(examples))
        batch_examples = examples[start_idx:end_idx]

        # Generate
        gen_results = generate_batch(loader, batch_examples, config)

        # Check anti-AI
        anti_ai_flags = [check_anti_ai(reply) for reply, _ in gen_results]

        # Judge with rate limiting
        if batch_idx > 0:
            time.sleep(RATE_LIMIT_DELAY)

        judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])

        # Build results
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(
            zip(batch_examples, gen_results, anti_ai_flags, judge_results)
        ):
            result = AblationResult(
                example_id=start_idx + i + 1,
                variant=name,
                category=ex.category,
                generated_response=reply,
                latency_ms=latency,
                anti_ai_violations=anti_ai,
                judge_score=score,
                judge_reasoning=reasoning,
            )
            results.append(result)

            status = "AI!" if anti_ai else "clean"
            print(
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | "
                f"{score:.0f}/10 | {reply[:50]}"
            )

    # Calculate summary
    scores = [r.judge_score for r in results if r.judge_score is not None]
    anti_ai_count = sum(1 for r in results if r.anti_ai_violations)
    latencies = [r.latency_ms for r in results]

    by_category: dict[str, list[float]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r.judge_score or 0)

    summary = {
        "name": name,
        "config": config,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "anti_ai_rate": anti_ai_count / len(results) if results else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "by_category": {cat: sum(s) / len(s) for cat, s in by_category.items()},
    }

    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Context & RAG Ablation Study")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")
    args = parser.parse_args()

    # Load dataset
    examples = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(examples)} examples")

    # Initialize judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client:
            print(f"Judge ready: {JUDGE_MODEL}")

    # Define variants to test
    variants = {
        "no_context": {"context_depth": 0, "use_rag": False},
        "context_3": {"context_depth": 3, "use_rag": False},
        "context_5": {"context_depth": 5, "use_rag": False},
        "context_10": {"context_depth": 10, "use_rag": False},
        "rag_only": {"context_depth": 0, "use_rag": True},
        "context_5_rag": {"context_depth": 5, "use_rag": True},
        "context_10_rag": {"context_depth": 10, "use_rag": True},
    }

    # Run all variants
    all_results = []
    all_summaries = []

    for name, config in variants.items():
        results, summary = run_variant(name, config, examples, judge_client)
        all_results.extend(results)
        all_summaries.append(summary)

    # Sort summaries by score
    all_summaries.sort(key=lambda s: s["avg_score"], reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print("CONTEXT & RAG ABLATION RESULTS")
    print("=" * 70)

    for s in all_summaries:
        print(f"\n{s['name'].upper()}")
        print(
            f"  Config: context_depth={s['config']['context_depth']}, "
            f"use_rag={s['config']['use_rag']}"
        )
        print(f"  Avg Score: {s['avg_score']:.2f}/10")
        print(f"  Anti-AI Rate: {s['anti_ai_rate']:.1%}")
        print(f"  Avg Latency: {s['avg_latency']:.0f}ms")
        print("  By Category:")
        for cat, score in sorted(s["by_category"].items()):
            print(f"    {cat:12s}: {score:.2f}")

    # Winner
    winner = all_summaries[0]
    print("\n" + "=" * 70)
    print(f"🏆 WINNER: {winner['name'].upper()}")
    print("=" * 70)
    print(f"Score: {winner['avg_score']:.2f}/10")
    print(
        f"Config: context_depth={winner['config']['context_depth']}, "
        f"use_rag={winner['config']['use_rag']}"
    )

    # Save results
    output_path = PROJECT_ROOT / "results" / "ablation_context_rag.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_model": JUDGE_MODEL,
        "summaries": all_summaries,
        "raw_results": [
            {
                "example_id": r.example_id,
                "variant": r.variant,
                "category": r.category,
                "generated": r.generated_response,
                "latency_ms": r.latency_ms,
                "anti_ai": r.anti_ai_violations,
                "judge_score": r.judge_score,
            }
            for r in all_results
        ],
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n📊 Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
