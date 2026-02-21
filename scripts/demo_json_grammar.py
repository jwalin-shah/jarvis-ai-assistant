#!/usr/bin/env python3
"""Demo script comparing JSON generation with and without grammar constraints.

Shows how small models can produce invalid JSON, and how grammar masking
guarantees valid output.

Usage:
    python scripts/demo_json_grammar.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from models.loader import MLXModelLoader, ModelConfig

JSON_PROMPT = """Return a JSON object with the following fields:
- name: a person's name
- age: their age as a number
- city: where they live

Return ONLY valid JSON, no other text."""


def validate_json(text: str) -> tuple[bool, str]:
    """Check if text is valid JSON.

    Returns:
        Tuple of (is_valid, error_message).
    """
    text = text.strip()
    if not text:
        return False, "Empty output"

    if not text.startswith("{") and not text.startswith("["):
        return False, f"Does not start with {{ or [: {text[:50]}..."

    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return True, "Valid JSON"
        return False, f"JSON but not object/array: {type(parsed)}"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"


def run_comparison(model_id: str = "lfm-0.7b", num_runs: int = 3):
    """Compare JSON generation with and without grammar constraints."""
    print("=" * 60)
    print("JSON Grammar Constraint Demo")
    print("=" * 60)
    print(f"\nLoading model: {model_id}")

    config = ModelConfig(model_id=model_id)
    loader = MLXModelLoader(config)

    print("Loading model weights...")
    loader.load()
    print(f"Model loaded: {loader.config.display_name}\n")

    tokenizer = loader._tokenizer

    from models.json_grammar import JSONGrammarProcessor, SimpleJSONGrammarProcessor

    grammar_processor = JSONGrammarProcessor(tokenizer)
    simple_processor = SimpleJSONGrammarProcessor(tokenizer)

    messages = [{"role": "user", "content": JSON_PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Prompt: {prompt[:100]}...")

    print("-" * 60)
    print("TEST 1: Without grammar constraints (baseline)")
    print("-" * 60)

    results_no_grammar = []
    times_no_grammar = []

    for i in range(num_runs):
        start = time.perf_counter()
        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            pre_formatted=True,
        )
        elapsed = time.perf_counter() - start
        times_no_grammar.append(elapsed)

        is_valid, msg = validate_json(result.text)
        results_no_grammar.append((result.text[:200], is_valid, msg))

        status = "VALID" if is_valid else "INVALID"
        print(f"\nRun {i + 1}: [{status}] ({elapsed:.2f}s)")
        print(f"  Output: {result.text[:150]}...")
        print(f"  Reason: {msg}")

    print("\n" + "-" * 60)
    print("TEST 2: With Simple JSON grammar (brace/bracket/quote balance)")
    print("-" * 60)

    simple_processor.reset()
    results_simple = []
    times_simple = []

    for i in range(num_runs):
        simple_processor.reset()
        start = time.perf_counter()

        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            pre_formatted=True,
            extra_logits_processors=[simple_processor],
        )
        elapsed = time.perf_counter() - start
        times_simple.append(elapsed)

        is_valid, msg = validate_json(result.text)
        results_simple.append((result.text[:200], is_valid, msg))

        status = "VALID" if is_valid else "INVALID"
        print(f"\nRun {i + 1}: [{status}] ({elapsed:.2f}s)")
        print(f"  Output: {result.text[:150]}...")
        print(f"  Reason: {msg}")

    print("\n" + "-" * 60)
    print("TEST 3: With Full JSON grammar (structural validation)")
    print("-" * 60)

    results_full = []
    times_full = []

    for i in range(num_runs):
        grammar_processor.reset()
        start = time.perf_counter()

        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            pre_formatted=True,
            extra_logits_processors=[grammar_processor],
        )
        elapsed = time.perf_counter() - start
        times_full.append(elapsed)

        is_valid, msg = validate_json(result.text)
        results_full.append((result.text[:200], is_valid, msg))

        status = "VALID" if is_valid else "INVALID"
        print(f"\nRun {i + 1}: [{status}] ({elapsed:.2f}s)")
        print(f"  Output: {result.text[:150]}...")
        print(f"  Reason: {msg}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_no_grammar = sum(1 for _, v, _ in results_no_grammar if v)
    valid_simple = sum(1 for _, v, _ in results_simple if v)
    valid_full = sum(1 for _, v, _ in results_full if v)

    avg_time_no_grammar = sum(times_no_grammar) / len(times_no_grammar)
    avg_time_simple = sum(times_simple) / len(times_simple)
    avg_time_full = sum(times_full) / len(times_full)

    print("\nValid JSON rate:")
    print(
        f"  No grammar:    {valid_no_grammar}/{num_runs} ({100 * valid_no_grammar / num_runs:.0f}%)"
    )
    print(f"  Simple grammar: {valid_simple}/{num_runs} ({100 * valid_simple / num_runs:.0f}%)")
    print(f"  Full grammar:   {valid_full}/{num_runs} ({100 * valid_full / num_runs:.0f}%)")

    print("\nAverage latency:")
    print(f"  No grammar:     {avg_time_no_grammar:.2f}s")
    print(
        f"  Simple grammar: {avg_time_simple:.2f}s ({100 * (avg_time_simple / avg_time_no_grammar - 1):+.1f}% overhead)"
    )
    print(
        f"  Full grammar:   {avg_time_full:.2f}s ({100 * (avg_time_full / avg_time_no_grammar - 1):+.1f}% overhead)"
    )

    print("\n" + "-" * 60)
    print("Conclusion:")
    if valid_full > valid_no_grammar:
        print("  Grammar constraints IMPROVED JSON validity!")
    else:
        print("  Model already produces valid JSON (grammar not needed for this model)")
    print("-" * 60)

    loader.unload()


def quick_test():
    """Quick single-run test for development."""
    print("Quick JSON Grammar Test\n")

    config = ModelConfig(model_id="lfm-0.7b")
    loader = MLXModelLoader(config)
    loader.load()

    tokenizer = loader._tokenizer
    from models.json_grammar import SimpleJSONGrammarProcessor

    processor = SimpleJSONGrammarProcessor(tokenizer)

    messages = [{"role": "user", "content": JSON_PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Prompt: {prompt[:100]}...")

    print("Generating without constraint...")
    result1 = loader.generate_sync(
        prompt=prompt, max_tokens=80, temperature=0.7, pre_formatted=True
    )
    valid1, msg1 = validate_json(result1.text)
    print(f"  Output: {result1.text[:100]}")
    print(f"  Valid: {valid1} ({msg1})")

    print("\nGenerating with constraint...")
    processor.reset()
    result2 = loader.generate_sync(
        prompt=prompt,
        max_tokens=80,
        temperature=0.7,
        pre_formatted=True,
        extra_logits_processors=[processor],
    )
    valid2, msg2 = validate_json(result2.text)
    print(f"  Output: {result2.text[:100]}")
    print(f"  Valid: {valid2} ({msg2})")

    loader.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo JSON grammar constraints")
    parser.add_argument("--quick", action="store_true", help="Quick single-run test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--model", default="lfm-0.7b", help="Model ID to use")

    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        run_comparison(model_id=args.model, num_runs=args.runs)
