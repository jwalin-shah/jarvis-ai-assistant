#!/usr/bin/env python3
"""Direct prompt optimization without DSPy complexity.

Tests multiple universal prompt variations and picks the best one.
Much simpler than DSPy and works better with small models.

Usage:
    uv run python evals/optimize_universal_prompt.py --judge
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

from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset
from evals.judge_config import JUDGE_MODEL, get_judge_client

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
    "constraints": """Generate a casual text reply.
Constraints:
- Max 2 sentences
- Start lowercase if they did
- No "I would be happy to" or similar AI phrases
- Match their abbreviations and style
- Don't ask follow-up questions unless essential""",
}


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


def judge_single(
    judge_client,
    context: list[str],
    last_message: str,
    ideal_response: str,
    generated: str,
    category: str,
) -> tuple[float, str]:
    """Judge a single example."""
    if not judge_client:
        return 5.0, "no judge"

    prompt = (
        "You are an expert evaluator for text message replies.\n\n"
        f"Conversation: {chr(10).join(context)}\n"
        f"Message: {last_message}\n"
        f"Generated reply: {generated}\n"
        f"Ideal reply: {ideal_response}\n"
        f"Category: {category}\n\n"
        "Score 0-10 based on:\n"
        "- Does it sound like a real text (not AI)?\n"
        "- Is it appropriate for the conversation?\n"
        "- Does it match the ideal reply's intent?\n\n"
        'Respond: {"score": <0-10>, "reasoning": "<brief>"}'
    )

    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        text = resp.choices[0].message.content.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text)
        return float(data["score"]), data.get("reasoning", "")

    except Exception as e:
        return 0.0, f"judge error: {e}"


def test_prompt_variant(
    name: str,
    system_prompt: str,
    examples: list[Any],
    judge_client: Any | None,
) -> PromptResult:
    """Test a single prompt variant on all examples."""

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    scores = []
    anti_ai_count = 0
    latencies = []
    by_category: dict[str, list[float]] = {}

    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}\n")

    for ex in tqdm(examples, desc=f"Testing {name}"):
        # Build prompt
        prompt = build_chatml_prompt(system_prompt, ex.context, ex.last_message)

        # Generate
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

        # Check anti-AI
        if check_anti_ai(reply):
            anti_ai_count += 1

        # Judge
        score, reasoning = judge_single(
            judge_client,
            ex.context,
            ex.last_message,
            ex.ideal_response,
            reply,
            ex.category,
        )
        scores.append(score)

        # Track by category
        if ex.category not in by_category:
            by_category[ex.category] = []
        by_category[ex.category].append(score)

        # Print progress
        status = "AI!" if check_anti_ai(reply) else "clean"
        print(f"[{ex.category:12s}] {status} | Judge: {score:.0f}/10 | {reply[:50]}")

    # Calculate averages
    avg_score = sum(scores) / len(scores) if scores else 0
    anti_ai_rate = anti_ai_count / len(examples) if examples else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

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
    parser = argparse.ArgumentParser(description="Optimize Universal Prompt")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(PROMPT_VARIANTS.keys()) + ["all"],
        default=["all"],
        help="Which variants to test",
    )
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

    # Determine variants to test
    if "all" in args.variants:
        variants_to_test = PROMPT_VARIANTS
    else:
        variants_to_test = {k: v for k, v in PROMPT_VARIANTS.items() if k in args.variants}

    # Test each variant
    results = []
    for name, prompt in variants_to_test.items():
        result = test_prompt_variant(name, prompt, examples, judge_client)
        results.append(result)

    # Sort by judge score
    results.sort(key=lambda r: r.avg_judge_score, reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print("PROMPT OPTIMIZATION RESULTS")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.name.upper()}")
        print(f"   Judge Score: {r.avg_judge_score:.2f}/10")
        print(f"   Anti-AI Rate: {r.anti_ai_rate:.1%}")
        print(f"   Avg Latency: {r.avg_latency_ms:.0f}ms")
        print("   By Category:")
        for cat, score in sorted(r.per_category_scores.items()):
            print(f"      {cat:12s}: {score:.2f}")
        print(f"   Prompt (first 100 chars): {r.prompt[:100]}...")

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
