#!/usr/bin/env python3
"""Pipeline Sweep: Systematically test context depths and optimized prompts.

Compares:
1. Baseline (context=3, baseline prompt)
2. Optimized Prompt (context=3, optimized prompt)
3. Extended Context (context=10, optimized prompt)
4. RAG-Enriched (context=3 + RAG, optimized prompt)

Evaluates 50 examples using Cerebras judge.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from evals.dspy_reply import TRAIN_EXAMPLES, clean_reply, judge_metric  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402
from models.loader import get_model  # noqa: E402

# Prompts
BASELINE_PROMPT = """You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person."""

OPTIMIZED_PROMPT = """You're a busy person texting from your iPhone.
Quick replies only. Match their vibe.
Don't overthink it - just text back like you normally would."""

# Configurations to test
CONFIGS = [
    {
        "name": "baseline",
        "system_prompt": BASELINE_PROMPT,
        "context_depth": 3,
        "use_rag": False,
    },
    {
        "name": "optimized_prompt",
        "system_prompt": OPTIMIZED_PROMPT,
        "context_depth": 3,
        "use_rag": False,
    },
    {
        "name": "extended_context",
        "system_prompt": OPTIMIZED_PROMPT,
        "context_depth": 10,
        "use_rag": False,
    },
    {
        "name": "rag_enriched",
        "system_prompt": OPTIMIZED_PROMPT,
        "context_depth": 3,
        "use_rag": True,
    },
]


def build_prompt(
    system_prompt: str, context: list[str], last_message: str, rag_context: str = ""
) -> str:
    """Build ChatML prompt with configurable context."""
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context, 1)])

    if rag_context:
        rag_section = f"\nRelevant Info:\n{rag_context}\n"
    else:
        rag_section = ""

    return (
        f"<|im_start|>system\n{system_prompt}{rag_section}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Conversation:\n{context_str}\n\n"
        f"Reply to: {last_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_sweep(judge_client):
    """Run evaluation sweep across configurations."""
    print(f"\n🚀 Starting Pipeline Sweep (Model: {JUDGE_MODEL})")
    print(f"Testing {len(CONFIGS)} configurations on {len(TRAIN_EXAMPLES)} examples...")

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    results = {}

    for config in CONFIGS:
        name = config["name"]
        print(f"\nTesting: {name.upper()}")
        print("-" * 40)

        scores = []
        latencies = []

        for ex in tqdm(TRAIN_EXAMPLES, desc=name):
            # 1. Prepare context based on depth
            context = ex.context
            if config["context_depth"] < len(context):
                context = context[-config["context_depth"] :]

            # 2. Simulate RAG (simple stub for now)
            rag_context = ""
            if config["use_rag"]:
                # In real app, this would query vector DB
                # Here we just use a placeholder to test prompt impact
                rag_context = "User prefers short replies. Avoid questions."

            # 3. Build prompt
            prompt = build_prompt(
                config["system_prompt"], context, ex.last_message, rag_context
            )

            # 4. Generate
            start = time.perf_counter()
            try:
                out = loader.generate_sync(
                    prompt,
                    max_tokens=50,
                    temperature=0.1,
                    repetition_penalty=1.05,
                )
                pred = clean_reply(out.text)
            except Exception as e:
                pred = f"[ERROR: {e}]"
            latencies.append((time.perf_counter() - start) * 1000)

            # 5. Judge
            score = judge_metric(
                ex.context,
                ex.last_message,
                pred,
                ex.ideal_response,
                client=judge_client,
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        avg_latency = sum(latencies) / len(latencies)
        results[name] = {"score": avg_score, "latency": avg_latency}

        print(f"  Score: {avg_score:.2f}/10")
        print(f"  Latency: {avg_latency:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<20} | {'Score':<10} | {'Latency':<10} | {'Improvement':<12}")
    print("-" * 60)

    baseline_score = results["baseline"]["score"]

    for name, metrics in results.items():
        score = metrics["score"]
        latency = metrics["latency"]
        diff = score - baseline_score
        print(f"{name:<20} | {score:<10.2f} | {latency:<10.0f} | {diff:+.2f}")

    # Save
    output_path = PROJECT_ROOT / "results" / "pipeline_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="Run with judge")
    args = parser.parse_args()

    if args.judge:
        client = get_judge_client()
        if not client:
            print("Judge client not available. Set env vars.")
            return 1
        run_sweep(client)
    else:
        print("Dry run complete. Use --judge to run evaluation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
