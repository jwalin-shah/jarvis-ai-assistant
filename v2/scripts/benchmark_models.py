#!/usr/bin/env python3
"""Benchmark different MLX models for text reply generation.

Tests speed and quality of various small LLMs for JARVIS text replies.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --models qwen3-1.7b llama-3.2-1b
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

# Models to benchmark
MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-MLX-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-MLX-4bit",
    "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "lfm2.5-1.2b": "lmstudio-community/LFM2.5-1.2B-Instruct-MLX-4bit",
}

# Test prompts (text reply scenarios)
TEST_PROMPTS = [
    {
        "name": "dinner_invite",
        "prompt": '{"task":"reply","you":"Jwalin","messages":[">hey you free for dinner tomorrow?"],"reply_to":"hey you free for dinner tomorrow?","style":"brief, casual, no emoji"}\nOutput: {"reply":',
        "expected_style": "short acceptance/decline",
    },
    {
        "name": "reaction_reply",
        "prompt": '{"task":"reply","you":"Jwalin","messages":["I got the painting for you",">oh nice thanks!",">when can i pick it up"],"reply_to":"when can i pick it up","style":"brief, casual"}\nOutput: {"reply":',
        "expected_style": "time/logistics",
    },
    {
        "name": "question_answer",
        "prompt": '{"task":"reply","you":"Jwalin","messages":[">what time works for you?"],"reply_to":"what time works for you?","style":"brief, casual, abbreviations ok"}\nOutput: {"reply":',
        "expected_style": "time suggestion",
    },
    {
        "name": "acknowledgment",
        "prompt": '{"task":"reply","you":"Jwalin","messages":[">sounds good see you then"],"reply_to":"sounds good see you then","style":"brief, casual"}\nOutput: {"reply":',
        "expected_style": "short acknowledgment",
    },
    {
        "name": "group_chat",
        "prompt": '{"task":"reply","you":"Jwalin","messages":[">who is bringing the drinks?",">i can bring snacks",">jwalin what about you?"],"reply_to":"jwalin what about you?","style":"brief, casual"}\nOutput: {"reply":',
        "expected_style": "offer to bring something",
    },
]


@dataclass
class BenchmarkResult:
    model_name: str
    model_path: str
    load_time_s: float
    avg_tokens_per_sec: float
    avg_generation_time_ms: float
    responses: list[dict]
    memory_gb: float = 0.0


def get_memory_usage() -> float:
    """Get current memory usage in GB (MLX Metal memory)."""
    try:
        import mlx.core as mx
        # Get Metal memory usage (use mx.device_info, not deprecated mx.metal.device_info)
        info = mx.device_info()
        # Peak memory in bytes
        return info.get("memory_size", 0) / (1024 ** 3)
    except Exception:
        pass

    # Fallback to process memory
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0


def get_model_size_gb(model_path: str) -> float:
    """Estimate model size from HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if model_path.replace("/", "--") in str(repo.repo_path):
                return repo.size_on_disk / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def benchmark_model(model_name: str, model_path: str, prompts: list[dict]) -> BenchmarkResult:
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({model_path})")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    load_start = time.time()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(model_path)
        load_time = time.time() - load_start
        print(f"Loaded in {load_time:.1f}s")

        memory_after_load = get_memory_usage()
        print(f"Memory usage: {memory_after_load:.2f} GB")

    except Exception as e:
        print(f"Failed to load: {e}")
        return BenchmarkResult(
            model_name=model_name,
            model_path=model_path,
            load_time_s=0,
            avg_tokens_per_sec=0,
            avg_generation_time_ms=0,
            responses=[{"error": str(e)}],
        )

    # Create sampler
    sampler = make_sampler(temp=0.7, top_p=0.8)

    # Run benchmarks
    results = []
    total_tokens = 0
    total_time = 0

    for i, test in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {test['name']}...")

        # Apply chat template if available
        prompt = test["prompt"]
        if tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                pass  # Use raw prompt

        # Generate
        start = time.time()
        try:
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=30,
                sampler=sampler,
            )
            gen_time = (time.time() - start) * 1000

            # Extract reply from JSON output
            reply = output.strip()
            if reply.startswith('"'):
                # Parse JSON string
                try:
                    import re
                    match = re.match(r'^"((?:[^"\\]|\\.)*)"', reply)
                    if match:
                        reply = match.group(1)
                except Exception:
                    pass

            # Estimate tokens
            tokens = len(output.split())
            tokens_per_sec = tokens / (gen_time / 1000) if gen_time > 0 else 0

            total_tokens += tokens
            total_time += gen_time

            print(f"  Output: {reply[:50]}...")
            print(f"  Time: {gen_time:.0f}ms, {tokens_per_sec:.1f} tok/s")

            results.append({
                "test": test["name"],
                "output": reply,
                "time_ms": gen_time,
                "tokens": tokens,
                "tokens_per_sec": tokens_per_sec,
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "test": test["name"],
                "error": str(e),
            })

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass

    # Calculate averages
    valid_results = [r for r in results if "tokens_per_sec" in r]
    avg_tps = sum(r["tokens_per_sec"] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_time = sum(r["time_ms"] for r in valid_results) / len(valid_results) if valid_results else 0

    return BenchmarkResult(
        model_name=model_name,
        model_path=model_path,
        load_time_s=load_time,
        avg_tokens_per_sec=avg_tps,
        avg_generation_time_ms=avg_time,
        responses=results,
        memory_gb=memory_after_load,
    )


def print_summary(results: list[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Sort by speed
    results = sorted(results, key=lambda r: r.avg_tokens_per_sec, reverse=True)

    print(f"\n{'Model':<20} {'Load(s)':<10} {'Gen(ms)':<10} {'Tok/s':<10} {'Memory(GB)':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r.model_name:<20} {r.load_time_s:<10.1f} {r.avg_generation_time_ms:<10.0f} {r.avg_tokens_per_sec:<10.1f} {r.memory_gb:<12.2f}")

    print("\n" + "-" * 70)
    print("\nSample outputs:")

    for r in results[:3]:  # Top 3
        print(f"\n{r.model_name}:")
        for resp in r.responses[:2]:
            if "output" in resp:
                print(f"  [{resp['test']}] {resp['output'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX models for text replies")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help=f"Models to test. Available: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("JARVIS Model Benchmark - Text Reply Generation")
    print("=" * 80)
    print(f"\nTesting {len(args.models)} models on {len(TEST_PROMPTS)} prompts\n")

    results = []
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        result = benchmark_model(model_name, MODELS[model_name], TEST_PROMPTS)
        results.append(result)

    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                [
                    {
                        "model": r.model_name,
                        "path": r.model_path,
                        "load_time_s": r.load_time_s,
                        "avg_tokens_per_sec": r.avg_tokens_per_sec,
                        "avg_generation_time_ms": r.avg_generation_time_ms,
                        "memory_gb": r.memory_gb,
                        "responses": r.responses,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )
        print(f"\nResults saved to {output_path}")

    # Recommendation
    if results:
        best = max(results, key=lambda r: r.avg_tokens_per_sec)
        print(f"\n🏆 Fastest: {best.model_name} ({best.avg_tokens_per_sec:.1f} tok/s)")

        # Quality pick (larger model)
        quality = max(results, key=lambda r: r.memory_gb)
        if quality != best:
            print(f"📊 Best quality (largest): {quality.model_name}")


if __name__ == "__main__":
    main()
