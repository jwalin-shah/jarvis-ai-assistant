#!/usr/bin/env python3  # noqa: E501
"""Batched universal prompt optimization.  # noqa: E501
  # noqa: E501
Judges multiple examples in a single API call for efficiency.  # noqa: E501
With 60 examples and batch size of 10, only 6 judge calls instead of 360!  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/optimize_universal_prompt_batched.py --judge  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
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
# Test different universal prompt variations  # noqa: E501
PROMPT_VARIANTS = {  # noqa: E501
    "baseline": """You are NOT an AI assistant. You are texting from your phone.  # noqa: E501
Reply naturally, matching the conversation style.  # noqa: E501
Be brief (1-2 sentences), casual, and sound like a real person.""",  # noqa: E501
    "minimal": """Text back naturally. Be brief, casual, human.""",  # noqa: E501
    "negative": """You are NOT an AI assistant. You are texting from your phone.  # noqa: E501
Rules:  # noqa: E501
- Be brief (1-2 sentences max)  # noqa: E501
- NO phrases like "I understand", "I'd be happy to", "Let me know"  # noqa: E501
- NO formal greetings or sign-offs  # noqa: E501
- Match their energy and style exactly  # noqa: E501
- Sound like a real person, not a bot""",  # noqa: E501
    "style_focused": """Reply to this text message as yourself.  # noqa: E501
Match their exact texting style (length, formality, punctuation, emoji).  # noqa: E501
Be brief and natural. No AI-sounding phrases.""",  # noqa: E501
    "persona": """You're a busy person texting from your iPhone.  # noqa: E501
Quick replies only. Match their vibe.  # noqa: E501
Don't overthink it - just text back like you normally would.""",  # noqa: E501
}  # noqa: E501
  # noqa: E501
BATCH_SIZE = 10  # Judge 10 examples per API call  # noqa: E501
RATE_LIMIT_DELAY = 2.1  # seconds between judge calls (30 req/min)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class PromptResult:  # noqa: E501
    name: str  # noqa: E501
    prompt: str  # noqa: E501
    avg_judge_score: float  # noqa: E501
    anti_ai_rate: float  # noqa: E501
    avg_latency_ms: float  # noqa: E501
    per_category_scores: dict[str, float]  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_chatml_prompt(system: str, context: list[str], last_message: str) -> str:  # noqa: E501
    """Build ChatML format prompt."""  # noqa: E501
    context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context[-10:], 1)])  # noqa: E501
  # noqa: E501
    return (  # noqa: E501
        f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>user\n"  # noqa: E501
        f"Conversation:\n{context_str}\n\n"  # noqa: E501
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>assistant\n"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_batch(  # noqa: E501
    judge_client,  # noqa: E501
    examples: list[Any],  # noqa: E501
    replies: list[str],  # noqa: E501
) -> list[tuple[float, str]]:  # noqa: E501
    """Judge multiple examples in a single API call.  # noqa: E501
  # noqa: E501
    Returns list of (score, reasoning) tuples.  # noqa: E501
    """  # noqa: E501
    if not judge_client:  # noqa: E501
        return [(5.0, "no judge")] * len(examples)  # noqa: E501
  # noqa: E501
    # Build batch evaluation prompt  # noqa: E501
    batch_text = f"""You are an expert evaluator for text message replies.  # noqa: E501
Evaluate {len(examples)} replies below and return ONLY a JSON array with scores.  # noqa: E501
  # noqa: E501
Scoring criteria (0-10):  # noqa: E501
- 8-10: Natural, human-like, appropriate, matches ideal reply intent  # noqa: E501
- 5-7: Acceptable but could be better  # noqa: E501
- 0-4: AI-sounding, inappropriate, or misses the mark  # noqa: E501
  # noqa: E501
Examples to evaluate:  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E501
        batch_text += f"""  # noqa: E501
--- EXAMPLE {i} ---  # noqa: E501
Context: {" | ".join(ex.context[-3:])}  # noqa: E501
Message: {ex.last_message}  # noqa: E501
Generated: {reply}  # noqa: E501
Ideal: {ex.ideal_response}  # noqa: E501
Category: {ex.category}  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    batch_text += f"""  # noqa: E501
Respond with ONLY this JSON format (no markdown, no backticks):  # noqa: E501
[{{"score": 8, "reasoning": "brief reason"}}, {{"score": 5, "reasoning": "brief reason"}}, ...]  # noqa: E501
Must have exactly {len(examples)} objects in the array.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = judge_client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": batch_text}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=500,  # noqa: E501
        )  # noqa: E501
        text = resp.choices[0].message.content.strip()  # noqa: E501
  # noqa: E501
        # Clean up response  # noqa: E501
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
        text = text.strip()  # noqa: E501
  # noqa: E501
        # Parse JSON  # noqa: E501
        data = json.loads(text)  # noqa: E501
  # noqa: E501
        # Extract scores  # noqa: E501
        results = []  # noqa: E501
        for item in data[: len(examples)]:  # noqa: E501
            score = float(item.get("score", 5))  # noqa: E501
            reasoning = item.get("reasoning", "")  # noqa: E501
            results.append((score, reasoning))  # noqa: E501
  # noqa: E501
        # Pad if needed  # noqa: E501
        while len(results) < len(examples):  # noqa: E501
            results.append((5.0, "parse error"))  # noqa: E501
  # noqa: E501
        return results  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"  Batch judge error: {e}")  # noqa: E501
        # Return default scores  # noqa: E501
        return [(5.0, f"error: {e}")] * len(examples)  # noqa: E501
  # noqa: E501
  # noqa: E501
def test_prompt_variant_batched(  # noqa: E501
    name: str,  # noqa: E501
    system_prompt: str,  # noqa: E501
    examples: list[Any],  # noqa: E501
    judge_client: Any | None,  # noqa: E501
) -> PromptResult:  # noqa: E501
    """Test a single prompt variant with batching."""  # noqa: E501
  # noqa: E501
    from models.loader import get_model  # noqa: E501
  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        loader.load()  # noqa: E501
  # noqa: E501
    # Generate all replies first  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print(f"Testing: {name}")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    replies = []  # noqa: E501
    latencies = []  # noqa: E501
    anti_ai_flags = []  # noqa: E501
  # noqa: E501
    for ex in tqdm(examples, desc="Generating"):  # noqa: E501
        prompt = build_chatml_prompt(system_prompt, ex.context, ex.last_message)  # noqa: E501
  # noqa: E501
        start = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            result = loader.generate_sync(  # noqa: E501
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
        latencies.append(latency)  # noqa: E501
        replies.append(reply)  # noqa: E501
        anti_ai_flags.append(bool(check_anti_ai(reply)))  # noqa: E501
  # noqa: E501
    # Judge in batches  # noqa: E501
    print(f"\nJudging in batches of {BATCH_SIZE}...")  # noqa: E501
    all_scores = []  # noqa: E501
    all_reasonings = []  # noqa: E501
  # noqa: E501
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE  # noqa: E501
    for batch_idx in range(num_batches):  # noqa: E501
        start_idx = batch_idx * BATCH_SIZE  # noqa: E501
        end_idx = min(start_idx + BATCH_SIZE, len(examples))  # noqa: E501
  # noqa: E501
        batch_examples = examples[start_idx:end_idx]  # noqa: E501
        batch_replies = replies[start_idx:end_idx]  # noqa: E501
  # noqa: E501
        # Rate limit delay  # noqa: E501
        if batch_idx > 0:  # noqa: E501
            time.sleep(RATE_LIMIT_DELAY)  # noqa: E501
  # noqa: E501
        # Judge batch  # noqa: E501
        batch_results = judge_batch(judge_client, batch_examples, batch_replies)  # noqa: E501
  # noqa: E501
        for score, reasoning in batch_results:  # noqa: E501
            all_scores.append(score)  # noqa: E501
            all_reasonings.append(reasoning)  # noqa: E501
  # noqa: E501
        print(f"  Batch {batch_idx + 1}/{num_batches} complete")  # noqa: E501
  # noqa: E501
    # Print results  # noqa: E501
    print("\nResults:")  # noqa: E501
    for i, (ex, reply, score, anti_ai) in enumerate(  # noqa: E501
        zip(examples, replies, all_scores, anti_ai_flags)  # noqa: E501
    ):  # noqa: E501
        status = "AI!" if anti_ai else "clean"  # noqa: E501
        print(f"[{ex.category:12s}] {status} | Judge: {score:.0f}/10 | {reply[:50]}")  # noqa: E501
  # noqa: E501
    # Calculate metrics  # noqa: E501
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0  # noqa: E501
    anti_ai_rate = sum(anti_ai_flags) / len(examples) if examples else 0  # noqa: E501
    avg_latency = sum(latencies) / len(latencies) if latencies else 0  # noqa: E501
  # noqa: E501
    # Per category  # noqa: E501
    by_category: dict[str, list[float]] = {}  # noqa: E501
    for ex, score in zip(examples, all_scores):  # noqa: E501
        if ex.category not in by_category:  # noqa: E501
            by_category[ex.category] = []  # noqa: E501
        by_category[ex.category].append(score)  # noqa: E501
  # noqa: E501
    per_category = {cat: sum(scores) / len(scores) for cat, scores in by_category.items()}  # noqa: E501
  # noqa: E501
    return PromptResult(  # noqa: E501
        name=name,  # noqa: E501
        prompt=system_prompt,  # noqa: E501
        avg_judge_score=avg_score,  # noqa: E501
        anti_ai_rate=anti_ai_rate,  # noqa: E501
        avg_latency_ms=avg_latency,  # noqa: E501
        per_category_scores=per_category,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Batched Universal Prompt Optimization")  # noqa: E501
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Load dataset  # noqa: E501
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E501
    print(f"Loaded {len(examples)} examples")  # noqa: E501
    print(  # noqa: E501
        f"Batch size: {BATCH_SIZE} (only {len(examples) // BATCH_SIZE + 1} judge calls per variant)"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Initialize judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E501
            print(f"Rate limit: 30 req/min, delay: {RATE_LIMIT_DELAY}s between calls")  # noqa: E501
  # noqa: E501
    # Test each variant  # noqa: E501
    results = []  # noqa: E501
    for name, prompt in PROMPT_VARIANTS.items():  # noqa: E501
        result = test_prompt_variant_batched(name, prompt, examples, judge_client)  # noqa: E501
        results.append(result)  # noqa: E501
  # noqa: E501
    # Sort by judge score  # noqa: E501
    results.sort(key=lambda r: r.avg_judge_score, reverse=True)  # noqa: E501
  # noqa: E501
    # Print summary  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("PROMPT OPTIMIZATION RESULTS (BATCHED)")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    for i, r in enumerate(results, 1):  # noqa: E501
        print(f"\n{i}. {r.name.upper()}")  # noqa: E501
        print(f"   Judge Score: {r.avg_judge_score:.2f}/10")  # noqa: E501
        print(f"   Anti-AI Rate: {r.anti_ai_rate:.1%}")  # noqa: E501
        print(f"   Avg Latency: {r.avg_latency_ms:.0f}ms")  # noqa: E501
        print("   By Category:")  # noqa: E501
        for cat, score in sorted(r.per_category_scores.items()):  # noqa: E501
            print(f"      {cat:12s}: {score:.2f}")  # noqa: E501
  # noqa: E501
    # Winner  # noqa: E501
    winner = results[0]  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print(f"🏆 WINNER: {winner.name.upper()}")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print(f"Score: {winner.avg_judge_score:.2f}/10")  # noqa: E501
    print(f"Anti-AI: {winner.anti_ai_rate:.1%}")  # noqa: E501
    print(f"\nFull Prompt:\n{winner.prompt}")  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "universal_prompt_optimization.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "judge_model": JUDGE_MODEL,  # noqa: E501
        "winner": winner.name,  # noqa: E501
        "results": [  # noqa: E501
            {  # noqa: E501
                "name": r.name,  # noqa: E501
                "avg_score": r.avg_judge_score,  # noqa: E501
                "anti_ai_rate": r.anti_ai_rate,  # noqa: E501
                "avg_latency_ms": r.avg_latency_ms,  # noqa: E501
                "per_category": r.per_category_scores,  # noqa: E501
                "prompt": r.prompt,  # noqa: E501
            }  # noqa: E501
            for r in results  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    print(f"\n📊 Results saved to: {output_path}")  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
