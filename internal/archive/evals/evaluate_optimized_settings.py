#!/usr/bin/env python3
"""Evaluate optimized generation settings with Cerebras judge."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.eval_pipeline import (  # noqa: E402  # noqa: E402
    EVAL_DATASET_PATH,
    EvalExample,
    load_eval_dataset,
)
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E402
from tqdm import tqdm  # noqa: E402  # noqa: E402

from models.loader import get_model  # noqa: E402  # noqa: E402

# Configuration
NUM_EXAMPLES = 20  # Small batch for quick eval

# Optimized settings (refined)
OPTIMIZED_CONFIG = {
    "temperature": 0.15,
    "repetition_penalty": 1.15,  # Higher = no echoing
    "max_tokens": 20,  # Shorter = more natural
    "top_p": 0.9,
}

# Baseline settings (old)
BASELINE_CONFIG = {
    "temperature": 0.1,
    "repetition_penalty": 1.05,
    "max_tokens": 50,
    "top_p": 0.9,
}

SYSTEM_PROMPT = """You are texting from your phone. Reply naturally, matching their style.
Be brief (1-2 sentences), casual, and sound like a real person."""


def build_prompt(context: str, last_message: str) -> str:
    """Build chat prompt."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context: {context}\n"
        f"Reply to: {last_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_reply(loader, prompt: str, config: dict) -> tuple[str, float]:
    """Generate reply with given config."""
    start = time.perf_counter()
    try:
        result = loader.generate_sync(
            prompt=prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            repetition_penalty=config["repetition_penalty"],
            top_p=config["top_p"],
        )
        latency = (time.perf_counter() - start) * 1000
        return result.text.strip(), latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return f"[ERROR: {e}]", latency


def judge_example(
    client, context: str, last_message: str, ideal: str, generated: str
) -> tuple[float, str]:
    """Judge a single example."""
    prompt = f"""You are an expert evaluator of text message replies.

Rate the generated reply on a scale of 1-10 based on how natural and appropriate it is.

Context: {context}
Message to reply to: {last_message}
Ideal reply: {ideal}
Generated reply: {generated}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 1-10>, "reasoning": "<brief explanation>"}}

Rating criteria:
- 9-10: Perfect natural reply, matches style
- 7-8: Good reply, minor issues
- 5-6: Acceptable but awkward or slightly off
- 3-4: Poor, unnatural or inappropriate
- 1-2: Very bad, completely wrong
"""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content.strip())
        return result.get("score", 0), result.get("reasoning", "")
    except Exception as e:
        return 0, f"Judge error: {e}"


def evaluate_config(loader, client, examples: list, config: dict, config_name: str) -> dict:
    """Evaluate a configuration on examples."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {config_name}")
    print(f"Config: {config}")
    print(f"{'=' * 70}")

    results = []
    scores = []

    for i, ex in enumerate(tqdm(examples, desc=config_name)):
        # Handle both dict and EvalExample
        if isinstance(ex, EvalExample):
            context = "\n".join(ex.context)
            last_message = ex.last_message
            ideal = ex.ideal_response
        else:
            context = ex.get("context", "")
            last_message = ex.get("last_message", "")
            ideal = ex.get("ideal_reply", "")

        prompt = build_prompt(context, last_message)
        reply, latency = generate_reply(loader, prompt, config)

        # Judge
        score, reasoning = judge_example(client, context, last_message, ideal, reply)
        scores.append(score)

        results.append(
            {
                "example_id": i,
                "last_message": last_message,
                "ideal": ideal,
                "generated": reply,
                "length": len(reply),
                "latency_ms": latency,
                "score": score,
                "reasoning": reasoning,
            }
        )

        time.sleep(2.1)  # Rate limit: 30 req/min

    avg_score = sum(scores) / len(scores) if scores else 0
    pass_rate = sum(1 for s in scores if s >= 6) / len(scores) if scores else 0
    avg_length = sum(r["length"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    return {
        "config_name": config_name,
        "config": config,
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "avg_length": avg_length,
        "avg_latency_ms": avg_latency,
        "results": results,
    }


def main():
    print("=" * 70)
    print("EVALUATING OPTIMIZED GENERATION SETTINGS")
    print("=" * 70)

    # Load model
    loader = get_model()
    if not loader.is_loaded():
        print("Loading model...")
        loader.load()

    # Load judge client
    client = get_judge_client()
    print(f"Judge: {JUDGE_MODEL}")

    # Load examples
    examples = load_eval_dataset(EVAL_DATASET_PATH)[:NUM_EXAMPLES]
    print(f"Loaded {len(examples)} examples")

    # Evaluate baseline
    baseline_results = evaluate_config(loader, client, examples, BASELINE_CONFIG, "Baseline (old)")

    # Evaluate optimized
    optimized_results = evaluate_config(
        loader, client, examples, OPTIMIZED_CONFIG, "Optimized (new)"
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print("\n📊 Baseline (rep=1.05, max_tokens=50):")
    print(f"   Average Score: {baseline_results['avg_score']:.2f}/10")
    print(f"   Pass Rate: {baseline_results['pass_rate'] * 100:.1f}%")
    print(f"   Avg Length: {baseline_results['avg_length']:.0f} chars")
    print(f"   Avg Latency: {baseline_results['avg_latency_ms']:.0f}ms")

    print("\n🚀 Optimized (rep=1.12, max_tokens=25):")
    print(f"   Average Score: {optimized_results['avg_score']:.2f}/10")
    print(f"   Pass Rate: {optimized_results['pass_rate'] * 100:.1f}%")
    print(f"   Avg Length: {optimized_results['avg_length']:.0f} chars")
    print(f"   Avg Latency: {optimized_results['avg_latency_ms']:.0f}ms")

    score_improvement = optimized_results["avg_score"] - baseline_results["avg_score"]
    length_reduction = baseline_results["avg_length"] - optimized_results["avg_length"]
    length_reduction_pct = length_reduction / baseline_results['avg_length'] * 100

    print("\n📈 Improvements:")
    print(f"   Score: {score_improvement:+.2f} points")
    print(f"   Length: -{length_reduction:.0f} chars ({length_reduction_pct:.0f}% shorter)")

    # Save results
    output = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvements": {
            "score": score_improvement,
            "length_reduction": length_reduction,
            "length_reduction_pct": length_reduction_pct,
        },
    }

    output_file = PROJECT_ROOT / "results" / "optimized_settings_eval.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

    return output


if __name__ == "__main__":
    main()
