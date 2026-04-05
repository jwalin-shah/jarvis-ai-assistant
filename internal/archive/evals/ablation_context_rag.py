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
  # noqa: E402
from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

  # noqa: E402
BATCH_SIZE = 10  # noqa: E402
RATE_LIMIT_DELAY = 2.1  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class AblationResult:  # noqa: E402
    example_id: int  # noqa: E402
    variant: str  # noqa: E402
    category: str  # noqa: E402
    generated_response: str  # noqa: E402
    latency_ms: float  # noqa: E402
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E402
    judge_score: float | None = None  # noqa: E402
    judge_reasoning: str = ""  # noqa: E402
  # noqa: E402
  # noqa: E402
# The winning universal prompt from previous experiments  # noqa: E402
UNIVERSAL_PROMPT = """You are NOT an AI assistant. You are texting from your phone.  # noqa: E402
Reply naturally, matching the conversation style.  # noqa: E402
Be brief (1-2 sentences), casual, and sound like a real person."""  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_prompt_with_context(  # noqa: E402
    system: str,  # noqa: E402
    context_messages: list[str],  # noqa: E402
    last_message: str,  # noqa: E402
    similar_examples: list[tuple[str, str]] | None = None,  # noqa: E402
) -> str:  # noqa: E402
    """Build prompt with variable context depth and optional RAG examples."""  # noqa: E402
  # noqa: E402
    # Format context (variable depth)  # noqa: E402
    if context_messages:  # noqa: E402
        context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context_messages, 1)])  # noqa: E402
    else:  # noqa: E402
        context_str = "(No previous context)"  # noqa: E402
  # noqa: E402
    # Build extra context with RAG examples if provided  # noqa: E402
    extra_parts = []  # noqa: E402
  # noqa: E402
    if similar_examples:  # noqa: E402
        examples_str = "\n\n".join(  # noqa: E402
            [  # noqa: E402
                f"Example {i}:\nThey said: {ex[0][:100]}...\nYou replied: {ex[1]}"  # noqa: E402
                for i, ex in enumerate(similar_examples[:3], 1)  # noqa: E402
            ]  # noqa: E402
        )  # noqa: E402
        extra_parts.append(f"Your typical replies:\n{examples_str}")  # noqa: E402
  # noqa: E402
    extra_context = "\n\n".join(extra_parts) if extra_parts else ""  # noqa: E402
  # noqa: E402
    # Build full prompt  # noqa: E402
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E402
  # noqa: E402
    if extra_context:  # noqa: E402
        prompt += f"<|im_start|>user\n{extra_context}<|im_end|>\n"  # noqa: E402
  # noqa: E402
    prompt += (  # noqa: E402
        f"<|im_start|>user\n"  # noqa: E402
        f"Conversation:\n{context_str}\n\n"  # noqa: E402
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>assistant\n"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    return prompt  # noqa: E402
  # noqa: E402
  # noqa: E402
def get_context_at_depth(full_context: list[str], depth: int) -> list[str]:  # noqa: E402
    """Get last N messages from context."""  # noqa: E402
    if depth == 0:  # noqa: E402
        return []  # noqa: E402
    return full_context[-depth:] if len(full_context) >= depth else full_context  # noqa: E402
  # noqa: E402
  # noqa: E402
def get_mock_rag_examples(category: str) -> list[tuple[str, str]]:  # noqa: E402
    """Generate mock RAG examples based on category."""  # noqa: E402
    examples = {  # noqa: E402
        "question": [  # noqa: E402
            ("What time is dinner?", "7pm"),  # noqa: E402
            ("Did you finish?", "yeah just did"),  # noqa: E402
        ],  # noqa: E402
        "request": [  # noqa: E402
            ("Can you pick me up?", "sure where?"),  # noqa: E402
            ("Send me the file", "on it"),  # noqa: E402
        ],  # noqa: E402
        "emotion": [  # noqa: E402
            ("I'm so stressed", "that sucks, wanna talk?"),  # noqa: E402
            ("I got the job!", "omg congrats!!"),  # noqa: E402
        ],  # noqa: E402
        "statement": [  # noqa: E402
            ("Just got home", "nice, how was it?"),  # noqa: E402
            ("Check out this photo", "looks great!"),  # noqa: E402
        ],  # noqa: E402
        "acknowledge": [  # noqa: E402
            ("Running late", "no worries"),  # noqa: E402
            ("Thanks!", "np"),  # noqa: E402
        ],  # noqa: E402
        "closing": [  # noqa: E402
            ("Talk later", "later!"),  # noqa: E402
            ("Goodnight", "night!"),  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
    return examples.get(category, [])  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_batch(  # noqa: E402
    generator,  # noqa: E402
    examples: list[Any],  # noqa: E402
    variant_config: dict,  # noqa: E402
) -> list[tuple[str, float]]:  # noqa: E402
    """Generate replies for a batch."""  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for ex in examples:  # noqa: E402
        import time  # noqa: E402
  # noqa: E402
        start = time.perf_counter()  # noqa: E402
  # noqa: E402
        # Get context at specified depth  # noqa: E402
        context = get_context_at_depth(ex.context, variant_config["context_depth"])  # noqa: E402
  # noqa: E402
        # Get RAG examples if enabled  # noqa: E402
        rag_examples = None  # noqa: E402
        if variant_config.get("use_rag"):  # noqa: E402
            rag_examples = get_mock_rag_examples(ex.category)  # noqa: E402
  # noqa: E402
        prompt = build_prompt_with_context(  # noqa: E402
            UNIVERSAL_PROMPT,  # noqa: E402
            context,  # noqa: E402
            ex.last_message,  # noqa: E402
            rag_examples,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        try:  # noqa: E402
            result = generator.generate_sync(  # noqa: E402
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
        results.append((reply, latency))  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_batch(  # noqa: E402
    judge_client,  # noqa: E402
    examples: list[Any],  # noqa: E402
    replies: list[str],  # noqa: E402
) -> list[tuple[float, str]]:  # noqa: E402
    """Judge multiple examples in a single API call."""  # noqa: E402
    if not judge_client or not replies:  # noqa: E402
        return [(5.0, "no judge")] * len(examples)  # noqa: E402
  # noqa: E402
    batch_text = f"""Evaluate {len(examples)} text message replies.  # noqa: E402
Score 0-10 based on naturalness and appropriateness.  # noqa: E402
  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E402
        batch_text += f"""  # noqa: E402
--- EXAMPLE {i} ---  # noqa: E402
Context: {" | ".join(ex.context[-3:])}  # noqa: E402
Message: {ex.last_message}  # noqa: E402
Generated: {reply}  # noqa: E402
Ideal: {ex.ideal_response}  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    batch_text += f"""  # noqa: E402
Respond with JSON array (no markdown):  # noqa: E402
[{{"score": 7, "reasoning": "brief"}}, ...]  # noqa: E402
Exactly {len(examples)} objects.  # noqa: E402
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
        if text.startswith("```"):  # noqa: E402
            text = text.split("```")[1]  # noqa: E402
            if text.startswith("json"):  # noqa: E402
                text = text[4:]  # noqa: E402
        text = text.strip()  # noqa: E402
  # noqa: E402
        data = json.loads(text)  # noqa: E402
        results = []  # noqa: E402
        for item in data[: len(examples)]:  # noqa: E402
            score = float(item.get("score", 5))  # noqa: E402
            reasoning = item.get("reasoning", "")  # noqa: E402
            results.append((score, reasoning))  # noqa: E402
  # noqa: E402
        while len(results) < len(examples):  # noqa: E402
            results.append((5.0, "parse error"))  # noqa: E402
  # noqa: E402
        return results  # noqa: E402
  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"  Batch judge error: {e}")  # noqa: E402
        return [(5.0, f"error: {e}")] * len(examples)  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_variant(  # noqa: E402
    name: str,  # noqa: E402
    config: dict,  # noqa: E402
    examples: list[Any],  # noqa: E402
    judge_client: Any | None,  # noqa: E402
) -> tuple[list[AblationResult], dict]:  # noqa: E402
    """Run ablation for a single variant."""  # noqa: E402
  # noqa: E402
    from models.loader import get_model  # noqa: E402
  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        loader.load()  # noqa: E402
  # noqa: E402
    print(f"\n{'=' * 70}")  # noqa: E402
    print(f"Variant: {name}")  # noqa: E402
    print(f"Config: {config}")  # noqa: E402
    print(f"{'=' * 70}\n")  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE  # noqa: E402
  # noqa: E402
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {name}"):  # noqa: E402
        start_idx = batch_idx * BATCH_SIZE  # noqa: E402
        end_idx = min(start_idx + BATCH_SIZE, len(examples))  # noqa: E402
        batch_examples = examples[start_idx:end_idx]  # noqa: E402
  # noqa: E402
        # Generate  # noqa: E402
        gen_results = generate_batch(loader, batch_examples, config)  # noqa: E402
  # noqa: E402
        # Check anti-AI  # noqa: E402
        anti_ai_flags = [check_anti_ai(reply) for reply, _ in gen_results]  # noqa: E402
  # noqa: E402
        # Judge with rate limiting  # noqa: E402
        if batch_idx > 0:  # noqa: E402
            time.sleep(RATE_LIMIT_DELAY)  # noqa: E402
  # noqa: E402
        judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])  # noqa: E402
  # noqa: E402
        # Build results  # noqa: E402
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(  # noqa: E402
            zip(batch_examples, gen_results, anti_ai_flags, judge_results)  # noqa: E402
        ):  # noqa: E402
            result = AblationResult(  # noqa: E402
                example_id=start_idx + i + 1,  # noqa: E402
                variant=name,  # noqa: E402
                category=ex.category,  # noqa: E402
                generated_response=reply,  # noqa: E402
                latency_ms=latency,  # noqa: E402
                anti_ai_violations=anti_ai,  # noqa: E402
                judge_score=score,  # noqa: E402
                judge_reasoning=reasoning,  # noqa: E402
            )  # noqa: E402
            results.append(result)  # noqa: E402
  # noqa: E402
            status = "AI!" if anti_ai else "clean"  # noqa: E402
            print(  # noqa: E402
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | {score:.0f}/10 | {reply[:50]}"  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    # Calculate summary  # noqa: E402
    scores = [r.judge_score for r in results if r.judge_score is not None]  # noqa: E402
    anti_ai_count = sum(1 for r in results if r.anti_ai_violations)  # noqa: E402
    latencies = [r.latency_ms for r in results]  # noqa: E402
  # noqa: E402
    by_category: dict[str, list[float]] = {}  # noqa: E402
    for r in results:  # noqa: E402
        if r.category not in by_category:  # noqa: E402
            by_category[r.category] = []  # noqa: E402
        by_category[r.category].append(r.judge_score or 0)  # noqa: E402
  # noqa: E402
    summary = {  # noqa: E402
        "name": name,  # noqa: E402
        "config": config,  # noqa: E402
        "avg_score": sum(scores) / len(scores) if scores else 0,  # noqa: E402
        "anti_ai_rate": anti_ai_count / len(results) if results else 0,  # noqa: E402
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,  # noqa: E402
        "by_category": {cat: sum(s) / len(s) for cat, s in by_category.items()},  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    return results, summary  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    parser = argparse.ArgumentParser(description="Context & RAG Ablation Study")  # noqa: E402
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Load dataset  # noqa: E402
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E402
    print(f"Loaded {len(examples)} examples")  # noqa: E402
  # noqa: E402
    # Initialize judge  # noqa: E402
    judge_client = None  # noqa: E402
    if args.judge:  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        if judge_client:  # noqa: E402
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E402
  # noqa: E402
    # Define variants to test  # noqa: E402
    variants = {  # noqa: E402
        "no_context": {"context_depth": 0, "use_rag": False},  # noqa: E402
        "context_3": {"context_depth": 3, "use_rag": False},  # noqa: E402
        "context_5": {"context_depth": 5, "use_rag": False},  # noqa: E402
        "context_10": {"context_depth": 10, "use_rag": False},  # noqa: E402
        "rag_only": {"context_depth": 0, "use_rag": True},  # noqa: E402
        "context_5_rag": {"context_depth": 5, "use_rag": True},  # noqa: E402
        "context_10_rag": {"context_depth": 10, "use_rag": True},  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    # Run all variants  # noqa: E402
    all_results = []  # noqa: E402
    all_summaries = []  # noqa: E402
  # noqa: E402
    for name, config in variants.items():  # noqa: E402
        results, summary = run_variant(name, config, examples, judge_client)  # noqa: E402
        all_results.extend(results)  # noqa: E402
        all_summaries.append(summary)  # noqa: E402
  # noqa: E402
    # Sort summaries by score  # noqa: E402
    all_summaries.sort(key=lambda s: s["avg_score"], reverse=True)  # noqa: E402
  # noqa: E402
    # Print summary  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("CONTEXT & RAG ABLATION RESULTS")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    for s in all_summaries:  # noqa: E402
        print(f"\n{s['name'].upper()}")  # noqa: E402
        print(  # noqa: E402
            f"  Config: context_depth={s['config']['context_depth']}, use_rag={s['config']['use_rag']}"  # noqa: E402
        )  # noqa: E402
        print(f"  Avg Score: {s['avg_score']:.2f}/10")  # noqa: E402
        print(f"  Anti-AI Rate: {s['anti_ai_rate']:.1%}")  # noqa: E402
        print(f"  Avg Latency: {s['avg_latency']:.0f}ms")  # noqa: E402
        print("  By Category:")  # noqa: E402
        for cat, score in sorted(s["by_category"].items()):  # noqa: E402
            print(f"    {cat:12s}: {score:.2f}")  # noqa: E402
  # noqa: E402
    # Winner  # noqa: E402
    winner = all_summaries[0]  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print(f"🏆 WINNER: {winner['name'].upper()}")  # noqa: E402
    print("=" * 70)  # noqa: E402
    print(f"Score: {winner['avg_score']:.2f}/10")  # noqa: E402
    print(  # noqa: E402
        f"Config: context_depth={winner['config']['context_depth']}, use_rag={winner['config']['use_rag']}"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "ablation_context_rag.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "judge_model": JUDGE_MODEL,  # noqa: E402
        "summaries": all_summaries,  # noqa: E402
        "raw_results": [  # noqa: E402
            {  # noqa: E402
                "example_id": r.example_id,  # noqa: E402
                "variant": r.variant,  # noqa: E402
                "category": r.category,  # noqa: E402
                "generated": r.generated_response,  # noqa: E402
                "latency_ms": r.latency_ms,  # noqa: E402
                "anti_ai": r.anti_ai_violations,  # noqa: E402
                "judge_score": r.judge_score,  # noqa: E402
            }  # noqa: E402
            for r in all_results  # noqa: E402
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
