#!/usr/bin/env python3
"""Evaluate fine-tuned adapters against test set with LLM judge.

Usage:
    uv run python scripts/training/eval_adapters.py --adapter 350m --judge
    uv run python scripts/training/eval_adapters.py --adapter 700m --judge
    uv run python scripts/training/eval_adapters.py --adapter 1.2b --judge
    uv run python scripts/training/eval_adapters.py --all --judge
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            import os
            os.environ.setdefault(key.strip(), val.strip())

from evals.judge_config import JUDGE_MODEL, get_judge_client


ADAPTER_CONFIGS = {
    "350m": {
        "base_model": "mlx-community/LFM2-350M-4bit",
        "adapter_path": "adapters/personal/350m-dora-v2",
        "name": "350M DoRA v2",
    },
    "700m": {
        "base_model": "mlx-community/LFM2-700M-4bit",
        "adapter_path": "adapters/personal/700m-dora-v2",
        "name": "700M DoRA v2",
    },
    "1.2b": {
        "base_model": "mlx-community/LFM2-1.2B-4bit",
        "adapter_path": "adapters/personal/1.2b-dora-v2",
        "name": "1.2B DoRA v2",
    },
    "0.7b-lora-conservative": {
        "base_model": "lmstudio-community/LFM2-700M-MLX-8bit",
        "adapter_path": "adapters/personal/0.7b-lora-conservative",
        "name": "0.7B LoRA Conservative (200 iter)",
    },
}


@dataclass
class EvalResult:
    prompt: str
    expected: str
    generated: str
    latency_ms: float
    judge_score: float | None = None
    judge_reasoning: str = ""


def load_test_data(test_path: Path, max_examples: int | None = None) -> list[dict]:
    """Load test examples from JSONL."""
    examples = []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data = json.loads(line)
            # Extract the conversation and expected response
            messages = data.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[-2]  # Second to last is the user input
                assistant_msg = messages[-1]  # Last is the expected response
                examples.append({
                    "prompt": user_msg.get("content", ""),
                    "expected": assistant_msg.get("content", ""),
                })
    return examples


def _extract_json(text: str) -> dict | None:
    """Extract JSON from text, handling markdown fences and partial JSON."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
        if text.startswith("json"):
            text = text[4:].strip()
    
    # Try to parse as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        pass
    
    return None


def judge_response(
    judge_client: Any, 
    prompt: str, 
    expected: str, 
    generated: str
) -> tuple[float | None, str]:
    """Score a response using the LLM judge."""
    # Escape any JSON-sensitive characters in the content
    def safe_json_str(s: str) -> str:
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
    
    judge_prompt = (
        "You are an expert evaluator for a text message reply generator.\n\n"
        f"CONVERSATION CONTEXT:\n{safe_json_str(prompt)}\n\n"
        f"IDEAL RESPONSE:\n{safe_json_str(expected)}\n\n"
        f"GENERATED REPLY:\n{safe_json_str(generated)}\n\n"
        "Score the generated reply from 0-10. Consider:\n"
        "- Does it match the tone and intent of the ideal response?\n"
        "- Does it sound like a real person texting (not an AI)?\n"
        "- Is the length appropriate?\n\n"
        'Respond in JSON: {"score": <0-10>, "reasoning": "<brief 1 sentence>"}'
    )
    
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        data = _extract_json(text)
        if data is None:
            return None, f"json parse error: {text[:100]}"
        score = float(data.get("score", 0))
        if score < 0:
            score = 0.0
        if score > 10:
            score = 10.0
        return score, data.get("reasoning", "")
    except Exception as e:
        return None, f"judge error: {e}"


def evaluate_adapter(
    adapter_key: str,
    test_data: list[dict],
    judge_client: Any | None = None,
    max_examples: int | None = None,
) -> dict:
    """Evaluate an adapter against test data."""
    config = ADAPTER_CONFIGS[adapter_key]
    adapter_path = PROJECT_ROOT / config["adapter_path"]
    base_model = config["base_model"]
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {config['name']}")
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"{'='*70}")
    
    # Import mlx_lm for loading
    from mlx_lm import load
    
    print("Loading model with adapter...")
    load_start = time.perf_counter()
    model, tokenizer = load(base_model, adapter_path=str(adapter_path))
    load_time = time.perf_counter() - load_start
    print(f"Loaded in {load_time:.1f}s")
    
    # Limit examples if specified
    if max_examples:
        test_data = test_data[:max_examples]
    
    results: list[EvalResult] = []
    total_start = time.perf_counter()
    
    for i, example in enumerate(tqdm(test_data, desc="Generating"), 1):
        prompt = example["prompt"]
        expected = example["expected"]
        
        gen_start = time.perf_counter()
        try:
            # Build chat messages and apply chat template
            # MUST match the system prompt used in training data
            system_prompt = (
                "You are Jwalin. Reply to text messages in your natural texting style.\n"
                "Rules:\n"
                "- Match your typical reply length (9 words avg)\n"
                "- Use your abbreviations naturally: wanna, bc, gonna, kinda, btw\n"
                "- No emoji usage\n"
                "- Never sound like an AI assistant\n"
                "- No formal greetings or sign-offs\n"
                "- Just text back like you normally would"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            # Generate response using mlx_lm's generate
            from mlx_lm import generate as mlx_generate
            output = mlx_generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=100,
            )
            latency_ms = (time.perf_counter() - gen_start) * 1000
            
            # Judge if enabled
            judge_score = None
            judge_reasoning = ""
            if judge_client:
                judge_score, judge_reasoning = judge_response(
                    judge_client, prompt, expected, output
                )
                # Small delay to avoid rate limits
                time.sleep(0.5)
            
            result = EvalResult(
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                expected=expected,
                generated=output,
                latency_ms=latency_ms,
                judge_score=judge_score,
                judge_reasoning=judge_reasoning,
            )
            results.append(result)
            
            # Print progress
            judge_str = f" | Judge: {judge_score:.0f}/10" if judge_score else ""
            print(f"[{i}/{len(test_data)}] {latency_ms:.0f}ms{judge_str}")
            print(f'  Expected: "{expected[:60]}..."' if len(expected) > 60 else f'  Expected: "{expected}"')
            print(f'  Generated: "{output[:60]}..."' if len(output) > 60 else f'  Generated: "{output}"')
            if judge_reasoning:
                print(f"  Reasoning: {judge_reasoning}")
            print()
            
        except Exception as e:
            print(f"Error generating: {e}")
            results.append(EvalResult(
                prompt=prompt[:100],
                expected=expected,
                generated=f"[ERROR: {e}]",
                latency_ms=(time.perf_counter() - gen_start) * 1000,
            ))
    
    total_time = time.perf_counter() - total_start
    
    # Calculate metrics
    latencies = [r.latency_ms for r in results if not r.generated.startswith("[ERROR")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    scored_results = [r for r in results if r.judge_score is not None]
    if scored_results:
        scores = [r.judge_score for r in scored_results]
        avg_score = sum(scores) / len(scores)
        pass_7 = sum(1 for s in scores if s >= 7)
        pass_5 = sum(1 for s in scores if s >= 5)
    else:
        avg_score = None
        pass_7 = 0
        pass_5 = 0
        scores = []
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {config['name']}")
    print(f"{'='*70}")
    print(f"Examples evaluated: {len(results)}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    if avg_score is not None:
        print(f"Judge avg score: {avg_score:.2f}/10")
        print(f"Pass rate (>=7): {pass_7}/{len(scored_results)} ({pass_7/len(scored_results)*100:.0f}%)")
        print(f"Pass rate (>=5): {pass_5}/{len(scored_results)} ({pass_5/len(scored_results)*100:.0f}%)")
    print(f"Total time: {total_time:.1f}s")
    
    return {
        "adapter": adapter_key,
        "name": config["name"],
        "base_model": base_model,
        "num_examples": len(results),
        "avg_latency_ms": avg_latency,
        "judge_avg": avg_score,
        "judge_pass_7": pass_7,
        "judge_pass_7_rate": pass_7 / len(scored_results) if scored_results else 0,
        "judge_pass_5": pass_5,
        "judge_pass_5_rate": pass_5 / len(scored_results) if scored_results else 0,
        "scores": scores,
        "results": [
            {
                "prompt": r.prompt,
                "expected": r.expected,
                "generated": r.generated,
                "latency_ms": r.latency_ms,
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned adapters")
    parser.add_argument("--adapter", choices=["350m", "700m", "1.2b", "0.7b-lora-conservative"], help="Adapter to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all adapters")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge scoring")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--test-path", type=str, default="data/personal/raw_style_variable/test.jsonl",
                        help="Path to test data")
    args = parser.parse_args()
    
    if not args.adapter and not args.all:
        print("Error: specify --adapter or --all")
        return 1
    
    # Load test data
    test_path = PROJECT_ROOT / args.test_path
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return 1
    
    print(f"Loading test data from {test_path}...")
    test_data = load_test_data(test_path)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client is None:
            print("WARNING: Judge API key not set, skipping judge evaluation")
            print("Set CEREBRAS_API_KEY in .env to enable judging")
        else:
            print(f"Judge enabled: {JUDGE_MODEL}")
    
    # Determine which adapters to evaluate
    adapters_to_eval = ["350m", "700m", "1.2b"] if args.all else [args.adapter]
    
    all_results = []
    for adapter_key in adapters_to_eval:
        if adapter_key not in ADAPTER_CONFIGS:
            print(f"Unknown adapter: {adapter_key}")
            continue
        
        result = evaluate_adapter(
            adapter_key=adapter_key,
            test_data=test_data,
            judge_client=judge_client,
            max_examples=args.max_examples,
        )
        all_results.append(result)
    
    # Summary comparison if multiple adapters
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        for r in all_results:
            judge_str = f"{r['judge_avg']:.2f}/10" if r['judge_avg'] else "N/A"
            pass_str = f"{r['judge_pass_7_rate']*100:.0f}%" if r['judge_avg'] else "N/A"
            print(f"{r['name']:15s} | Judge: {judge_str:8s} | Pass: {pass_str:4s} | Latency: {r['avg_latency_ms']:.0f}ms")
    
    # Save results
    output_path = PROJECT_ROOT / "results" / "adapter_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "judge_model": JUDGE_MODEL if judge_client else None,
            "adapters": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
