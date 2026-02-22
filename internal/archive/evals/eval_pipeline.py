#!/usr/bin/env python3
"""Eval pipeline for reply quality.

Usage:
    uv run python evals/eval_pipeline.py --judge --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    context: list[str]
    last_message: str
    ideal_response: str
    tone: str = "casual"
    user_style: str = "brief, lowercase"
    category: str = "general"
    rubric: str = ""


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> list[EvalExample]:
    """Load evaluation examples from JSONL."""
    examples = []
    if not path.exists():
        logger.warning(f"Dataset not found: {path}")
        return []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            examples.append(
                EvalExample(
                    context=data.get("context", []),
                    last_message=data.get("last_message", ""),
                    ideal_response=data.get("ideal_response", ""),
                    tone=data.get("tone", "casual"),
                    user_style=data.get("user_style", ""),
                    category=data.get("category", "general"),
                    rubric=data.get("rubric", ""),
                )
            )
    return examples


def check_anti_ai(text: str) -> str | None:
    """Check for common AI-isms."""
    ai_phrases = [
        "I understand",
        "I can help with that",
        "Is there anything else",
        "Let me know if",
        "Here is a",
        "Certainly",
        "As an AI",
    ]
    for phrase in ai_phrases:
        if phrase.lower() in text.lower():
            return phrase
    return None


def _judge_single_item(
    judge_client: object, judge_model: str, ex: EvalExample, generated: str
) -> tuple[float | None, str]:
    """Judge one item and return (score, reasoning)."""
    try:
        prompt = f"""Rate this reply 0-10 based on naturalness and intent match.
Context: {ex.context[-2:] if ex.context else []}
Last Message: {ex.last_message}
Ideal: {ex.ideal_response}
Generated: {generated}
Rubric: {ex.rubric}

Return JSON: {{"score": <float>, "reasoning": "<text>"}}"""

        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        data = json.loads(content)
        return float(data.get("score", 0)), data.get("reasoning", "")
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None, str(e)


def run_pipeline(limit: int = None, use_judge: bool = False):
    """Run full evaluation pipeline."""
    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    examples = load_eval_dataset()
    if limit:
        examples = examples[:limit]

    judge_client = None
    judge_model = ""
    if use_judge:
        from evals.judge_config import JUDGE_MODEL, get_judge_client

        judge_client = get_judge_client()
        judge_model = JUDGE_MODEL

    results = []
    total_latency = 0
    anti_ai_count = 0

    print(f"Running eval on {len(examples)} examples...")
    for ex in tqdm(examples):
        start = time.perf_counter()

        # Build prompt (simplified for brevity)
        prompt = (
            f"<|im_start|>system\nYou are a human. Reply briefly.<|im_end|>\n"
            f"<|im_start|>user\n{ex.last_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        response = loader.generate_sync(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3,
        )
        generated = response.text.strip()
        latency = (time.perf_counter() - start) * 1000
        total_latency += latency

        anti_ai = check_anti_ai(generated)
        if anti_ai:
            anti_ai_count += 1

        score, reasoning = None, ""
        if judge_client:
            score, reasoning = _judge_single_item(judge_client, judge_model, ex, generated)

        results.append(
            {
                "input": ex.last_message,
                "ideal": ex.ideal_response,
                "generated": generated,
                "latency_ms": latency,
                "anti_ai": anti_ai,
                "judge_score": score,
                "reasoning": reasoning,
            }
        )

    # Summary
    avg_latency = total_latency / len(examples) if examples else 0
    valid_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print("\nResults Summary:")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    print(f"  Anti-AI Rate: {anti_ai_count}/{len(examples)}")
    if use_judge:
        print(f"  Judge Score: {avg_score:.2f}/10")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_pipeline_baseline.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metrics": {
                    "latency_p95_ms": avg_latency,  # approximated
                    "anti_ai_clean_rate": 1.0 - (anti_ai_count / len(examples)),
                    "judge_avg": avg_score,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.limit, args.judge)
