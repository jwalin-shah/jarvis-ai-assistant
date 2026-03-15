#!/usr/bin/env python3  # noqa: E501
"""Ablation study: Context depth and RAG impact on reply quality.  # noqa: E501
  # noqa: E501
Tests:  # noqa: E501
1. Context depth: 0, 3, 5, 10 messages  # noqa: E501
2. RAG: with vs without similar examples  # noqa: E501
3. Combined: context + RAG vs baseline  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/ablation_context_rag.py --judge  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass, field  # noqa: E402  # noqa: E501
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
BATCH_SIZE = 10  # noqa: E501
RATE_LIMIT_DELAY = 2.1  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class AblationResult:  # noqa: E501
    example_id: int  # noqa: E501
    variant: str  # noqa: E501
    category: str  # noqa: E501
    generated_response: str  # noqa: E501
    latency_ms: float  # noqa: E501
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E501
    judge_score: float | None = None  # noqa: E501
    judge_reasoning: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
# The winning universal prompt from previous experiments  # noqa: E501
UNIVERSAL_PROMPT = """You are NOT an AI assistant. You are texting from your phone.  # noqa: E501
Reply naturally, matching the conversation style.  # noqa: E501
Be brief (1-2 sentences), casual, and sound like a real person."""  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_prompt_with_context(  # noqa: E501
    system: str,  # noqa: E501
    context_messages: list[str],  # noqa: E501
    last_message: str,  # noqa: E501
    similar_examples: list[tuple[str, str]] | None = None,  # noqa: E501
) -> str:  # noqa: E501
    """Build prompt with variable context depth and optional RAG examples."""  # noqa: E501
  # noqa: E501
    # Format context (variable depth)  # noqa: E501
    if context_messages:  # noqa: E501
        context_str = "\n".join([f"[{i}] {msg}" for i, msg in enumerate(context_messages, 1)])  # noqa: E501
    else:  # noqa: E501
        context_str = "(No previous context)"  # noqa: E501
  # noqa: E501
    # Build extra context with RAG examples if provided  # noqa: E501
    extra_parts = []  # noqa: E501
  # noqa: E501
    if similar_examples:  # noqa: E501
        examples_str = "\n\n".join(  # noqa: E501
            [  # noqa: E501
                f"Example {i}:\nThey said: {ex[0][:100]}...\nYou replied: {ex[1]}"  # noqa: E501
                for i, ex in enumerate(similar_examples[:3], 1)  # noqa: E501
            ]  # noqa: E501
        )  # noqa: E501
        extra_parts.append(f"Your typical replies:\n{examples_str}")  # noqa: E501
  # noqa: E501
    extra_context = "\n\n".join(extra_parts) if extra_parts else ""  # noqa: E501
  # noqa: E501
    # Build full prompt  # noqa: E501
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"  # noqa: E501
  # noqa: E501
    if extra_context:  # noqa: E501
        prompt += f"<|im_start|>user\n{extra_context}<|im_end|>\n"  # noqa: E501
  # noqa: E501
    prompt += (  # noqa: E501
        f"<|im_start|>user\n"  # noqa: E501
        f"Conversation:\n{context_str}\n\n"  # noqa: E501
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>assistant\n"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return prompt  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_context_at_depth(full_context: list[str], depth: int) -> list[str]:  # noqa: E501
    """Get last N messages from context."""  # noqa: E501
    if depth == 0:  # noqa: E501
        return []  # noqa: E501
    return full_context[-depth:] if len(full_context) >= depth else full_context  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_mock_rag_examples(category: str) -> list[tuple[str, str]]:  # noqa: E501
    """Generate mock RAG examples based on category."""  # noqa: E501
    examples = {  # noqa: E501
        "question": [  # noqa: E501
            ("What time is dinner?", "7pm"),  # noqa: E501
            ("Did you finish?", "yeah just did"),  # noqa: E501
        ],  # noqa: E501
        "request": [  # noqa: E501
            ("Can you pick me up?", "sure where?"),  # noqa: E501
            ("Send me the file", "on it"),  # noqa: E501
        ],  # noqa: E501
        "emotion": [  # noqa: E501
            ("I'm so stressed", "that sucks, wanna talk?"),  # noqa: E501
            ("I got the job!", "omg congrats!!"),  # noqa: E501
        ],  # noqa: E501
        "statement": [  # noqa: E501
            ("Just got home", "nice, how was it?"),  # noqa: E501
            ("Check out this photo", "looks great!"),  # noqa: E501
        ],  # noqa: E501
        "acknowledge": [  # noqa: E501
            ("Running late", "no worries"),  # noqa: E501
            ("Thanks!", "np"),  # noqa: E501
        ],  # noqa: E501
        "closing": [  # noqa: E501
            ("Talk later", "later!"),  # noqa: E501
            ("Goodnight", "night!"),  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
    return examples.get(category, [])  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_batch(  # noqa: E501
    generator,  # noqa: E501
    examples: list[Any],  # noqa: E501
    variant_config: dict,  # noqa: E501
) -> list[tuple[str, float]]:  # noqa: E501
    """Generate replies for a batch."""  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for ex in examples:  # noqa: E501
        import time  # noqa: E501
  # noqa: E501
        start = time.perf_counter()  # noqa: E501
  # noqa: E501
        # Get context at specified depth  # noqa: E501
        context = get_context_at_depth(ex.context, variant_config["context_depth"])  # noqa: E501
  # noqa: E501
        # Get RAG examples if enabled  # noqa: E501
        rag_examples = None  # noqa: E501
        if variant_config.get("use_rag"):  # noqa: E501
            rag_examples = get_mock_rag_examples(ex.category)  # noqa: E501
  # noqa: E501
        prompt = build_prompt_with_context(  # noqa: E501
            UNIVERSAL_PROMPT,  # noqa: E501
            context,  # noqa: E501
            ex.last_message,  # noqa: E501
            rag_examples,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            result = generator.generate_sync(  # noqa: E501
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
        results.append((reply, latency))  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_batch(  # noqa: E501
    judge_client,  # noqa: E501
    examples: list[Any],  # noqa: E501
    replies: list[str],  # noqa: E501
) -> list[tuple[float, str]]:  # noqa: E501
    """Judge multiple examples in a single API call."""  # noqa: E501
    if not judge_client or not replies:  # noqa: E501
        return [(5.0, "no judge")] * len(examples)  # noqa: E501
  # noqa: E501
    batch_text = f"""Evaluate {len(examples)} text message replies.  # noqa: E501
Score 0-10 based on naturalness and appropriateness.  # noqa: E501
  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    for i, (ex, reply) in enumerate(zip(examples, replies), 1):  # noqa: E501
        batch_text += f"""  # noqa: E501
--- EXAMPLE {i} ---  # noqa: E501
Context: {" | ".join(ex.context[-3:])}  # noqa: E501
Message: {ex.last_message}  # noqa: E501
Generated: {reply}  # noqa: E501
Ideal: {ex.ideal_response}  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    batch_text += f"""  # noqa: E501
Respond with JSON array (no markdown):  # noqa: E501
[{{"score": 7, "reasoning": "brief"}}, ...]  # noqa: E501
Exactly {len(examples)} objects.  # noqa: E501
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
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
        text = text.strip()  # noqa: E501
  # noqa: E501
        data = json.loads(text)  # noqa: E501
        results = []  # noqa: E501
        for item in data[: len(examples)]:  # noqa: E501
            score = float(item.get("score", 5))  # noqa: E501
            reasoning = item.get("reasoning", "")  # noqa: E501
            results.append((score, reasoning))  # noqa: E501
  # noqa: E501
        while len(results) < len(examples):  # noqa: E501
            results.append((5.0, "parse error"))  # noqa: E501
  # noqa: E501
        return results  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"  Batch judge error: {e}")  # noqa: E501
        return [(5.0, f"error: {e}")] * len(examples)  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_variant(  # noqa: E501
    name: str,  # noqa: E501
    config: dict,  # noqa: E501
    examples: list[Any],  # noqa: E501
    judge_client: Any | None,  # noqa: E501
) -> tuple[list[AblationResult], dict]:  # noqa: E501
    """Run ablation for a single variant."""  # noqa: E501
  # noqa: E501
    from models.loader import get_model  # noqa: E501
  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        loader.load()  # noqa: E501
  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print(f"Variant: {name}")  # noqa: E501
    print(f"Config: {config}")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE  # noqa: E501
  # noqa: E501
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {name}"):  # noqa: E501
        start_idx = batch_idx * BATCH_SIZE  # noqa: E501
        end_idx = min(start_idx + BATCH_SIZE, len(examples))  # noqa: E501
        batch_examples = examples[start_idx:end_idx]  # noqa: E501
  # noqa: E501
        # Generate  # noqa: E501
        gen_results = generate_batch(loader, batch_examples, config)  # noqa: E501
  # noqa: E501
        # Check anti-AI  # noqa: E501
        anti_ai_flags = [check_anti_ai(reply) for reply, _ in gen_results]  # noqa: E501
  # noqa: E501
        # Judge with rate limiting  # noqa: E501
        if batch_idx > 0:  # noqa: E501
            time.sleep(RATE_LIMIT_DELAY)  # noqa: E501
  # noqa: E501
        judge_results = judge_batch(judge_client, batch_examples, [r[0] for r in gen_results])  # noqa: E501
  # noqa: E501
        # Build results  # noqa: E501
        for i, (ex, (reply, latency), anti_ai, (score, reasoning)) in enumerate(  # noqa: E501
            zip(batch_examples, gen_results, anti_ai_flags, judge_results)  # noqa: E501
        ):  # noqa: E501
            result = AblationResult(  # noqa: E501
                example_id=start_idx + i + 1,  # noqa: E501
                variant=name,  # noqa: E501
                category=ex.category,  # noqa: E501
                generated_response=reply,  # noqa: E501
                latency_ms=latency,  # noqa: E501
                anti_ai_violations=anti_ai,  # noqa: E501
                judge_score=score,  # noqa: E501
                judge_reasoning=reasoning,  # noqa: E501
            )  # noqa: E501
            results.append(result)  # noqa: E501
  # noqa: E501
            status = "AI!" if anti_ai else "clean"  # noqa: E501
            print(  # noqa: E501
                f"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | {score:.0f}/10 | {reply[:50]}"  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    # Calculate summary  # noqa: E501
    scores = [r.judge_score for r in results if r.judge_score is not None]  # noqa: E501
    anti_ai_count = sum(1 for r in results if r.anti_ai_violations)  # noqa: E501
    latencies = [r.latency_ms for r in results]  # noqa: E501
  # noqa: E501
    by_category: dict[str, list[float]] = {}  # noqa: E501
    for r in results:  # noqa: E501
        if r.category not in by_category:  # noqa: E501
            by_category[r.category] = []  # noqa: E501
        by_category[r.category].append(r.judge_score or 0)  # noqa: E501
  # noqa: E501
    summary = {  # noqa: E501
        "name": name,  # noqa: E501
        "config": config,  # noqa: E501
        "avg_score": sum(scores) / len(scores) if scores else 0,  # noqa: E501
        "anti_ai_rate": anti_ai_count / len(results) if results else 0,  # noqa: E501
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,  # noqa: E501
        "by_category": {cat: sum(s) / len(s) for cat, s in by_category.items()},  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    return results, summary  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Context & RAG Ablation Study")  # noqa: E501
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge")  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Load dataset  # noqa: E501
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E501
    print(f"Loaded {len(examples)} examples")  # noqa: E501
  # noqa: E501
    # Initialize judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL}")  # noqa: E501
  # noqa: E501
    # Define variants to test  # noqa: E501
    variants = {  # noqa: E501
        "no_context": {"context_depth": 0, "use_rag": False},  # noqa: E501
        "context_3": {"context_depth": 3, "use_rag": False},  # noqa: E501
        "context_5": {"context_depth": 5, "use_rag": False},  # noqa: E501
        "context_10": {"context_depth": 10, "use_rag": False},  # noqa: E501
        "rag_only": {"context_depth": 0, "use_rag": True},  # noqa: E501
        "context_5_rag": {"context_depth": 5, "use_rag": True},  # noqa: E501
        "context_10_rag": {"context_depth": 10, "use_rag": True},  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    # Run all variants  # noqa: E501
    all_results = []  # noqa: E501
    all_summaries = []  # noqa: E501
  # noqa: E501
    for name, config in variants.items():  # noqa: E501
        results, summary = run_variant(name, config, examples, judge_client)  # noqa: E501
        all_results.extend(results)  # noqa: E501
        all_summaries.append(summary)  # noqa: E501
  # noqa: E501
    # Sort summaries by score  # noqa: E501
    all_summaries.sort(key=lambda s: s["avg_score"], reverse=True)  # noqa: E501
  # noqa: E501
    # Print summary  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("CONTEXT & RAG ABLATION RESULTS")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    for s in all_summaries:  # noqa: E501
        print(f"\n{s['name'].upper()}")  # noqa: E501
        print(  # noqa: E501
            f"  Config: context_depth={s['config']['context_depth']}, use_rag={s['config']['use_rag']}"  # noqa: E501
        )  # noqa: E501
        print(f"  Avg Score: {s['avg_score']:.2f}/10")  # noqa: E501
        print(f"  Anti-AI Rate: {s['anti_ai_rate']:.1%}")  # noqa: E501
        print(f"  Avg Latency: {s['avg_latency']:.0f}ms")  # noqa: E501
        print("  By Category:")  # noqa: E501
        for cat, score in sorted(s["by_category"].items()):  # noqa: E501
            print(f"    {cat:12s}: {score:.2f}")  # noqa: E501
  # noqa: E501
    # Winner  # noqa: E501
    winner = all_summaries[0]  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print(f"🏆 WINNER: {winner['name'].upper()}")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print(f"Score: {winner['avg_score']:.2f}/10")  # noqa: E501
    print(  # noqa: E501
        f"Config: context_depth={winner['config']['context_depth']}, use_rag={winner['config']['use_rag']}"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "ablation_context_rag.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "judge_model": JUDGE_MODEL,  # noqa: E501
        "summaries": all_summaries,  # noqa: E501
        "raw_results": [  # noqa: E501
            {  # noqa: E501
                "example_id": r.example_id,  # noqa: E501
                "variant": r.variant,  # noqa: E501
                "category": r.category,  # noqa: E501
                "generated": r.generated_response,  # noqa: E501
                "latency_ms": r.latency_ms,  # noqa: E501
                "anti_ai": r.anti_ai_violations,  # noqa: E501
                "judge_score": r.judge_score,  # noqa: E501
            }  # noqa: E501
            for r in all_results  # noqa: E501
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
