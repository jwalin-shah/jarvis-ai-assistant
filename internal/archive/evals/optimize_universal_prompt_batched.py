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
  # noqa: E402
from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

  # noqa: E402
# Test different universal prompt variations  # noqa: E402
PROMPT_VARIANTS = {  # noqa: E402
    "baseline": """You are NOT an AI assistant. You are texting from your phone.  # noqa: E402
Reply naturally, matching the conversation style.  # noqa: E402
Be brief (1-2 sentences), casual, and sound like a real person.""",  # noqa: E402
    "minimal": """Text back naturally. Be brief, casual, human.""",  # noqa: E402
    "negative": """You are NOT an AI assistant. You are texting from your phone.  # noqa: E402
Rules:  # noqa: E402
- Be brief (1-2 sentences max)  # noqa: E402
- NO phrases like "I understand", "I'd be happy to", "Let me know"  # noqa: E402
- NO formal greetings or sign-offs  # noqa: E402
- Match their energy and style exactly  # noqa: E402
- Sound like a real person, not a bot""",  # noqa: E402
    "style_focused": """Reply to this text message as yourself.  # noqa: E402
Match their exact texting style (length, formality, punctuation, emoji).  # noqa: E402
Be brief and natural. No AI-sounding phrases.""",  # noqa: E402
    "persona": """You're a busy person texting from your iPhone.  # noqa: E402
Quick replies only. Match their vibe.  # noqa: E402
Don't overthink it - just text back like you normally would.""",  # noqa: E402
}  # noqa: E402
  # noqa: E402
BATCH_SIZE = 10  # Judge 10 examples per API call  # noqa: E402
RATE_LIMIT_DELAY = 2.1  # seconds between judge calls (30 req/min)  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class PromptResult:  # noqa: E402
    name: str  # noqa: E402
    prompt: str  # noqa: E402
    avg_judge_score: float  # noqa: E402
    anti_ai_rate: float  # noqa: E402
    avg_latency_ms: float  # noqa: E402
    per_category_scores: dict[str, float]  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_chatml_prompt(system: str, context: list[str], last_message: str) -> str:  # noqa: E402
    """Build ChatML format prompt."""  # noqa: E402
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])  # noqa: E402
  # noqa: E402
    return (  # noqa: E402
        f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>user\n"  # noqa: E402
        f"Conversation:\n{context_str}\n\n"  # noqa: E402
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>assistant\n"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_batch(  # noqa: E402
    judge_client,  # noqa: E402
    examples: list[Any],  # noqa: E402
    replies: list[str],  # noqa: E402
) -> list[tuple[float, str]]:  # noqa: E402
    """Judge multiple examples in a single API call.  # noqa: E402
  # noqa: E402
    Returns list of (score, reasoning) tuples.  # noqa: E402
    """  # noqa: E402
    if not judge_client:  # noqa: E402
        return [(5.0, "no judge")] * len(examples)  # noqa: E402
  # noqa: E402
    # Build batch evaluation prompt  # noqa: E402
    batch_text = f"""You are an expert evaluator for text message replies.  # noqa: E402
Evaluate {len(examples)} replies below and return ONLY a JSON array with scores.  # noqa: E402
  # noqa: E402
Scoring criteria (0-10):  # noqa: E402
- 8-10: Natural, human-like, appropriate, matches ideal reply intent  # noqa: E402
- 5-7: Acceptable but could be better  # noqa: E402
- 0-4: AI-sounding, inappropriate, or misses the mark  # noqa: E402
  # noqa: E402
Examples to evaluate:  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E402
        batch_text += f"""  # noqa: E402
--- EXAMPLE {i} ---  # noqa: E402
Context: {" | ".join(ex.context[-3:])}  # noqa: E402
Message: {ex.last_message}  # noqa: E402
Generated: {reply}  # noqa: E402
Ideal: {ex.ideal_response}  # noqa: E402
Category: {ex.category}  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    batch_text += f"""  # noqa: E402
Respond with ONLY this JSON format (no markdown, no backticks):  # noqa: E402
[{{"score": 8, "reasoning": "brief reason"}}, {{"score": 5, "reasoning": "brief reason"}}, ...]  # noqa: E402
Must have exactly {len(examples)} objects in the array.  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        resp = judge_client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": batch_text}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=500,  # noqa: E402
        )  # noqa: E402
        text = resp.choices[0].message.content.strip()  # noqa: E402
  # noqa: E402
        # Clean up response  # noqa: E402
        if text.startswith("```"):  # noqa: E402
            text = text.split("```")[1]  # noqa: E402
            if text.startswith("json"):  # noqa: E402
                text = text[4:]  # noqa: E402
        text = text.strip()  # noqa: E402
  # noqa: E402
        # Parse JSON  # noqa: E402
        data = json.loads(text)  # noqa: E402
  # noqa: E402
        # Extract scores  # noqa: E402
        results = []  # noqa: E402
        for item in data[: len(examples)]:  # noqa: E402
            score = float(item.get("score", 5))  # noqa: E402
            reasoning = item.get("reasoning", "")  # noqa: E402
            results.append((score, reasoning))  # noqa: E402
  # noqa: E402
        # Pad if needed  # noqa: E402
        while len(results) < len(examples):  # noqa: E402
            results.append((5.0, "parse error"))  # noqa: E402
  # noqa: E402
        return results  # noqa: E402
  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"  Batch judge error: {e}")  # noqa: E402
        # Return default scores  # noqa: E402
        return [(5.0, f"error: {e}")] * len(examples)  # noqa: E402
  # noqa: E402
  # noqa: E402
def test_prompt_variant_batched(  # noqa: E402
    name: str,  # noqa: E402
    system_prompt: str,  # noqa: E402
    examples: list[Any],  # noqa: E402
    judge_client: Any | None,  # noqa: E402
) -> PromptResult:  # noqa: E402
    """Test a single prompt variant with batching."""  # noqa: E402
  # noqa: E402
    from models.loader import get_model  # noqa: E402
  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        loader.load()  # noqa: E402
  # noqa: E402
    # Generate all replies first  # noqa: E402
    print(f"\n{'=' * 70}")  # noqa: E402
    print(f"Testing: {name}")  # noqa: E402
    print(f"{'=' * 70}\n")  # noqa: E402
  # noqa: E402
    replies = []  # noqa: E402
    latencies = []  # noqa: E402
    anti_ai_flags = []  # noqa: E402
  # noqa: E402
    for ex in tqdm(examples, desc="Generating"):  # noqa: E402
        prompt = build_chatml_prompt(system_prompt, ex.context, ex.last_message)  # noqa: E402
  # noqa: E402
        start = time.perf_counter()  # noqa: E402
        try:  # noqa: E402
            result = loader.generate_sync(  # noqa: E402
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
        latencies.append(latency)  # noqa: E402
        replies.append(reply)  # noqa: E402
        anti_ai_flags.append(bool(check_anti_ai(reply)))  # noqa: E402
  # noqa: E402
    # Judge in batches  # noqa: E402
    print(f"\nJudging in batches of {BATCH_SIZE}...")  # noqa: E402
    all_scores = []  # noqa: E402
    all_reasonings = []  # noqa: E402
  # noqa: E402
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE  # noqa: E402
    for batch_idx in range(num_batches):  # noqa: E402
        start_idx = batch_idx * BATCH_SIZE  # noqa: E402
        end_idx = min(start_idx + BATCH_SIZE, len(examples))  # noqa: E402
  # noqa: E402
        batch_examples = examples[start_idx:end_idx]  # noqa: E402
        batch_replies = replies[start_idx:end_idx]  # noqa: E402
  # noqa: E402
        # Rate limit delay  # noqa: E402
        if batch_idx > 0:  # noqa: E402
            time.sleep(RATE_LIMIT_DELAY)  # noqa: E402
  # noqa: E402
        # Judge batch  # noqa: E402
        batch_results = judge_batch(judge_client, batch_examples, batch_replies)  # noqa: E402
  # noqa: E402
        for score, reasoning in batch_results:  # noqa: E402
            all_scores.append(score)  # noqa: E402
            all_reasonings.append(reasoning)  # noqa: E402
  # noqa: E402
        print(f"  Batch {batch_idx + 1}/{num_batches} complete")  # noqa: E402
  # noqa: E402
    # Print results  # noqa: E402
    print("\nResults:")  # noqa: E402
    for i, (ex, reply, score, anti_ai) in enumerate(  # noqa: E402
        zip(examples, replies, all_scores, anti_ai_flags)  # noqa: E402
    ):  # noqa: E402
        status = "AI!" if anti_ai else "clean"  # noqa: E402
        print(f"[{ex.category:12s}] {status} | Judge: {score:.0f}/10 | {reply[:50]}")  # noqa: E402
  # noqa: E402
    # Calculate metrics  # noqa: E402
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0  # noqa: E402
    anti_ai_rate = sum(anti_ai_flags) / len(examples) if examples else 0  # noqa: E402
    avg_latency = sum(latencies) / len(latencies) if latencies else 0  # noqa: E402
  # noqa: E402
    # Per category  # noqa: E402
    by_category: dict[str, list[float]] = {}  # noqa: E402
    for ex, score in zip(examples, all_scores):  # noqa: E402
        if ex.category not in by_category:  # noqa: E402
            by_category[ex.category] = []  # noqa: E402
        by_category[ex.category].append(score)  # noqa: E402
  # noqa: E402
    per_category = {cat: sum(scores) / len(scores) for cat, scores in by_category.items()}  # noqa: E402
  # noqa: E402
    return PromptResult(  # noqa: E402
        name=name,  # noqa: E402
        prompt=system_prompt,  # noqa: E402
        avg_judge_score=avg_score,  # noqa: E402
        anti_ai_rate=anti_ai_rate,  # noqa: E402
        avg_latency_ms=avg_latency,  # noqa: E402
        per_category_scores=per_category,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    parser = argparse.ArgumentParser(description="Batched Universal Prompt Optimization")  # noqa: E402
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Load dataset  # noqa: E402
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E402
    print(f"Loaded {len(examples)} examples")  # noqa: E402
    print(  # noqa: E402
        f"Batch size: {BATCH_SIZE} (only {len(examples) // BATCH_SIZE + 1} judge calls per variant)"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Initialize judge  # noqa: E402
    judge_client = None  # noqa: E402
    if args.judge:  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        if judge_client:  # noqa: E402
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E402
            print(f"Rate limit: 30 req/min, delay: {RATE_LIMIT_DELAY}s between calls")  # noqa: E402
  # noqa: E402
    # Test each variant  # noqa: E402
    results = []  # noqa: E402
    for name, prompt in PROMPT_VARIANTS.items():  # noqa: E402
        result = test_prompt_variant_batched(name, prompt, examples, judge_client)  # noqa: E402
        results.append(result)  # noqa: E402
  # noqa: E402
    # Sort by judge score  # noqa: E402
    results.sort(key=lambda r: r.avg_judge_score, reverse=True)  # noqa: E402
  # noqa: E402
    # Print summary  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("PROMPT OPTIMIZATION RESULTS (BATCHED)")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    for i, r in enumerate(results, 1):  # noqa: E402
        print(f"\n{i}. {r.name.upper()}")  # noqa: E402
        print(f"   Judge Score: {r.avg_judge_score:.2f}/10")  # noqa: E402
        print(f"   Anti-AI Rate: {r.anti_ai_rate:.1%}")  # noqa: E402
        print(f"   Avg Latency: {r.avg_latency_ms:.0f}ms")  # noqa: E402
        print("   By Category:")  # noqa: E402
        for cat, score in sorted(r.per_category_scores.items()):  # noqa: E402
            print(f"      {cat:12s}: {score:.2f}")  # noqa: E402
  # noqa: E402
    # Winner  # noqa: E402
    winner = results[0]  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print(f"🏆 WINNER: {winner.name.upper()}")  # noqa: E402
    print("=" * 70)  # noqa: E402
    print(f"Score: {winner.avg_judge_score:.2f}/10")  # noqa: E402
    print(f"Anti-AI: {winner.anti_ai_rate:.1%}")  # noqa: E402
    print(f"\nFull Prompt:\n{winner.prompt}")  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "universal_prompt_optimization.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "judge_model": JUDGE_MODEL,  # noqa: E402
        "winner": winner.name,  # noqa: E402
        "results": [  # noqa: E402
            {  # noqa: E402
                "name": r.name,  # noqa: E402
                "avg_score": r.avg_judge_score,  # noqa: E402
                "anti_ai_rate": r.anti_ai_rate,  # noqa: E402
                "avg_latency_ms": r.avg_latency_ms,  # noqa: E402
                "per_category": r.per_category_scores,  # noqa: E402
                "prompt": r.prompt,  # noqa: E402
            }  # noqa: E402
            for r in results  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    print(f"\n📊 Results saved to: {output_path}")  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
