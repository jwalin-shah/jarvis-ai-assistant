#!/usr/bin/env python3
"""Batched universal prompt optimization.

Judges multiple examples in a single API call for efficiency.
With 60 examples and batch size of 10, only 6 judge calls instead of 360!

Usage:
    uv run python evals/optimize_universal_prompt_batched.py --judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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

# Test different universal prompt variations
PROMPT_VARIANTS = {
    "baseline": """You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.""",
    "minimal": """Text back naturally. Be brief, casual, human.""",
    "negative": """You are NOT an AI assistant. You are texting from your phone.
Rules:
- Be brief (1-2 sentences max)
- NO phrases like "I understand", "I'd be happy to", "Let me know"
- NO formal greetings or sign-offs
- Match their energy and style exactly
- Sound like a real person, not a bot""",
    "style_focused": """Reply to this text message as yourself.
Match their exact texting style (length, formality, punctuation, emoji).
Be brief and natural. No AI-sounding phrases.""",
    "persona": """You're a busy person texting from your iPhone.
Quick replies only. Match their vibe.
Don't overthink it - just text back like you normally would.""",
}

BATCH_SIZE = 10  # Judge 10 examples per API call
RATE_LIMIT_DELAY = 2.1  # seconds between judge calls (30 req/min)


@dataclass
class PromptResult:
    name: str
    prompt: str
    avg_judge_score: float
    anti_ai_rate: float
    avg_latency_ms: float
    per_category_scores: dict[str, float]


def build_chatml_prompt(system: str, context: list[str], last_message: str) -> str:
    """Build ChatML format prompt."""
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Conversation:\n{context_str}\n\n"
        f"Reply to: {last_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def judge_batch(
    judge_client,
    examples: list[Any],
    replies: list[str],
) -> list[tuple[float, str]]:
    """Judge multiple examples in a single API call.

    Returns list of (score, reasoning) tuples.
    """
    if not judge_client:
        return [(5.0, "no judge")] * len(examples)

    # Build batch evaluation prompt
    batch_text = f"""You are an expert evaluator for text message replies.
Evaluate {len(examples)} replies below and return ONLY a JSON array with scores.

Scoring criteria (0-10):
- 8-10: Natural, human-like, appropriate, matches ideal reply intent
- 5-7: Acceptable but could be better
- 0-4: AI-sounding, inappropriate, or misses the mark

Examples to evaluate:
"""

    for i, (ex, reply) in enumerate(zip(examples, replies), 1):
        batch_text += f"""
--- EXAMPLE {i} ---
Context: {" | ".join(ex.context[-3:])}
Message: {ex.last_message}
Generated: {reply}
Ideal: {ex.ideal_response}
Category: {ex.category}
"""

    batch_text += f"""
Respond with ONLY this JSON format (no markdown, no backticks):
[{{"score": 8, "reasoning": "brief reason"}}, {{"score": 5, "reasoning": "brief reason"}}, ...]
Must have exactly {len(examples)} objects in the array.
"""

    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": batch_text}],
            temperature=0.0,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()

        # Clean up response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # Parse JSON
        data = json.loads(text)

        # Extract scores
        results = []
        for item in data[: len(examples)]:
            score = float(item.get("score", 5))
            reasoning = item.get("reasoning", "")
            results.append((score, reasoning))

        # Pad if needed
        while len(results) < len(examples):
            results.append((5.0, "parse error"))

        return results

    except Exception as e:
        print(f"  Batch judge error: {e}")
        # Return default scores
        return [(5.0, f"error: {e}")] * len(examples)


def test_prompt_variant_batched(
    name: str,
    system_prompt: str,
    examples: list[Any],
    judge_client: Any | None,
) -> PromptResult:
    """Test a single prompt variant with batching."""

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    # Generate all replies first
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}\n")

    replies = []
    latencies = []
    anti_ai_flags = []

    for ex in tqdm(examples, desc="Generating"):
        prompt = build_chatml_prompt(system_prompt, ex.context, ex.last_message)

        start = time.perf_counter()
        try:
            result = loader.generate_sync(
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
        latencies.append(latency)
        replies.append(reply)
        anti_ai_flags.append(bool(check_anti_ai(reply)))

    # Judge in batches
    print(f"\nJudging in batches of {BATCH_SIZE}...")
    all_scores = []
    all_reasonings = []

    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(examples))

        batch_examples = examples[start_idx:end_idx]
        batch_replies = replies[start_idx:end_idx]

        # Rate limit delay
        if batch_idx > 0:
            time.sleep(RATE_LIMIT_DELAY)

        # Judge batch
        batch_results = judge_batch(judge_client, batch_examples, batch_replies)

        for score, reasoning in batch_results:
            all_scores.append(score)
            all_reasonings.append(reasoning)

        print(f"  Batch {batch_idx + 1}/{num_batches} complete")

    # Print results
    print("\nResults:")
    for i, (ex, reply, score, anti_ai) in enumerate(
        zip(examples, replies, all_scores, anti_ai_flags)
    ):
        status = "AI!" if anti_ai else "clean"
        print(f"[{ex.category:12s}] {status} | Judge: {score:.0f}/10 | {reply[:50]}")

    # Calculate metrics
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    anti_ai_rate = sum(anti_ai_flags) / len(examples) if examples else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Per category
    by_category: dict[str, list[float]] = {}
    for ex, score in zip(examples, all_scores):
        if ex.category not in by_category:
            by_category[ex.category] = []
        by_category[ex.category].append(score)

    per_category = {cat: sum(scores) / len(scores) for cat, scores in by_category.items()}

    return PromptResult(
        name=name,
        prompt=system_prompt,
        avg_judge_score=avg_score,
        anti_ai_rate=anti_ai_rate,
        avg_latency_ms=avg_latency,
        per_category_scores=per_category,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched Universal Prompt Optimization")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")
    args = parser.parse_args()

    # Load dataset
    examples = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(examples)} examples")
    print(
        f"Batch size: {BATCH_SIZE} (only {len(examples) // BATCH_SIZE + 1} judge calls per variant)"
    )

    # Initialize judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client:
            print(f"Judge ready: {JUDGE_MODEL}")
            print(f"Rate limit: 30 req/min, delay: {RATE_LIMIT_DELAY}s between calls")

    # Test each variant
    results = []
    for name, prompt in PROMPT_VARIANTS.items():
        result = test_prompt_variant_batched(name, prompt, examples, judge_client)
        results.append(result)

    # Sort by judge score
    results.sort(key=lambda r: r.avg_judge_score, reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print("PROMPT OPTIMIZATION RESULTS (BATCHED)")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.name.upper()}")
        print(f"   Judge Score: {r.avg_judge_score:.2f}/10")
        print(f"   Anti-AI Rate: {r.anti_ai_rate:.1%}")
        print(f"   Avg Latency: {r.avg_latency_ms:.0f}ms")
        print("   By Category:")
        for cat, score in sorted(r.per_category_scores.items()):
            print(f"      {cat:12s}: {score:.2f}")

    # Winner
    winner = results[0]
    print("\n" + "=" * 70)
    print(f"🏆 WINNER: {winner.name.upper()}")
    print("=" * 70)
    print(f"Score: {winner.avg_judge_score:.2f}/10")
    print(f"Anti-AI: {winner.anti_ai_rate:.1%}")
    print(f"\nFull Prompt:\n{winner.prompt}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "universal_prompt_optimization.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_model": JUDGE_MODEL,
        "winner": winner.name,
        "results": [
            {
                "name": r.name,
                "avg_score": r.avg_judge_score,
                "anti_ai_rate": r.anti_ai_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "per_category": r.per_category_scores,
                "prompt": r.prompt,
            }
            for r in results
        ],
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n📊 Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
