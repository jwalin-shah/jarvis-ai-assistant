"""Evaluate personal fine-tuned models.

Loads each fine-tuned model (or adapter), generates replies on the test set,
computes metrics, and produces a leaderboard.

Output: evals/results/personal-ft-comparison.json

Usage:
    uv run python scripts/evaluate_personal_ft.py
    uv run python scripts/evaluate_personal_ft.py --report-only
    uv run python scripts/evaluate_personal_ft.py --config ft_configs/personal_1.2b_lora_cataware.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_PATH = Path("evals/results/personal-ft-comparison.json")
LOG_PATH = Path("evaluate_personal.log")

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with both file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

# Emoji pattern for style matching
EMOJI_PAT = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+"
)


def load_test_data(data_dir: str) -> list[dict]:
    """Load test examples from a data directory."""
    test_path = Path(data_dir) / "test.jsonl"
    if not test_path.exists():
        log.warning("No test.jsonl in %s", data_dir)
        return []
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def compute_metrics(generated: list[str], references: list[str]) -> dict[str, float]:
    """Compute evaluation metrics between generated and reference replies."""
    if not generated or not references:
        return {}

    # Length ratio
    gen_lens = [len(g) for g in generated]
    ref_lens = [len(r) for r in references]
    avg_gen_len = np.mean(gen_lens)
    avg_ref_len = np.mean(ref_lens)
    length_ratio = avg_gen_len / max(avg_ref_len, 1)

    # Emoji match rate
    gen_has_emoji = [1 if EMOJI_PAT.search(g) else 0 for g in generated]
    ref_has_emoji = [1 if EMOJI_PAT.search(r) else 0 for r in references]
    emoji_match = sum(1 for g, r in zip(gen_has_emoji, ref_has_emoji) if g == r) / len(generated)

    # Case match (lowercase tendency)
    gen_lower = [1 if g and g[0].islower() else 0 for g in generated]
    ref_lower = [1 if r and r[0].islower() else 0 for r in references]
    case_match = sum(1 for g, r in zip(gen_lower, ref_lower) if g == r) / len(generated)

    # Simple word overlap (pseudo-BLEU)
    overlaps = []
    for gen, ref in zip(generated, references):
        gen_words = set(gen.lower().split())
        ref_words = set(ref.lower().split())
        if ref_words:
            overlaps.append(len(gen_words & ref_words) / len(ref_words))
        else:
            overlaps.append(0.0)
    word_overlap = np.mean(overlaps)

    # Clean output rate (no AI-isms)
    ai_phrases = [
        "i understand",
        "let me know",
        "i hope this helps",
        "is there anything",
        "happy to help",
        "feel free",
        "i'm here to",
        "as an ai",
    ]
    clean_count = 0
    for g in generated:
        g_lower = g.lower()
        if not any(phrase in g_lower for phrase in ai_phrases):
            clean_count += 1
    clean_rate = clean_count / len(generated)

    return {
        "avg_gen_length": round(avg_gen_len, 1),
        "avg_ref_length": round(avg_ref_len, 1),
        "length_ratio": round(length_ratio, 3),
        "emoji_match_rate": round(emoji_match, 3),
        "case_match_rate": round(case_match, 3),
        "word_overlap": round(word_overlap, 3),
        "clean_output_rate": round(clean_rate, 3),
        "n_examples": len(generated),
    }


def evaluate_config(config_path: Path, max_examples: int = 50) -> dict | None:
    """Evaluate a single fine-tuned model from its config.

    Loads the model, generates replies on test set, computes metrics.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    adapter_path = Path(config["adapter_path"])
    data_dir = config["data"]
    model_path = config["model"]
    config_name = config_path.stem

    log.info("Evaluating %s...", config_name)

    # Check if adapter exists
    if not adapter_path.exists():
        log.warning("  Adapter not found: %s (skipping)", adapter_path)
        return None

    # Load test data
    test_examples = load_test_data(data_dir)
    if not test_examples:
        log.warning("  No test data for %s", config_name)
        return None

    test_examples = test_examples[:max_examples]
    log.info("  Using %d test examples", len(test_examples))

    # Load model with adapter
    try:
        import mlx.core as mx
        from mlx_lm import generate, load

        mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB

        log.info("  Loading model %s with adapter %s...", model_path, adapter_path)
        model, tokenizer = load(model_path, adapter_path=str(adapter_path))

        generated = []
        references = []
        total_time = 0.0

        for i, example in enumerate(test_examples):
            if (i + 1) % 10 == 0:
                log.info("  Generating %d/%d...", i + 1, len(test_examples))

            messages = example["messages"]
            reference = messages[-1]["content"]  # assistant message
            # Build prompt from system + user messages
            prompt_messages = messages[:-1]

            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join(m["content"] for m in prompt_messages)

            start = time.time()
            result = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
            elapsed = time.time() - start
            total_time += elapsed

            generated.append(result.strip())
            references.append(reference)

        # Compute metrics
        metrics = compute_metrics(generated, references)
        metrics["total_generation_time"] = round(total_time, 1)
        metrics["avg_time_per_example"] = round(total_time / len(test_examples), 3)
        metrics["config"] = config_name
        metrics["model"] = model_path
        metrics["adapter"] = str(adapter_path)

        log.info("  Results: %s", json.dumps(metrics, indent=2))

        # Cleanup
        del model, tokenizer
        mx.clear_cache()

        return metrics

    except Exception as e:
        log.error("  Failed to evaluate %s: %s", config_name, e)
        return None


def print_leaderboard(results: list[dict]) -> None:
    """Print a formatted leaderboard."""
    if not results:
        print("No results to display.", flush=True)
        return

    # Sort by composite score: word_overlap * clean_rate * (1/abs(1-length_ratio))
    def score(r: dict) -> float:
        wo = r.get("word_overlap", 0)
        cr = r.get("clean_output_rate", 0)
        lr = r.get("length_ratio", 1)
        length_penalty = 1.0 / (1.0 + abs(1.0 - lr))
        return wo * cr * length_penalty

    results.sort(key=score, reverse=True)

    print("\n" + "=" * 90, flush=True)
    print("PERSONAL FINE-TUNING LEADERBOARD", flush=True)
    print("=" * 90, flush=True)
    print(
        f"{'Rank':<5} {'Config':<40} {'WordOvlp':<10} {'Clean%':<8} "
        f"{'LenRatio':<10} {'Emoji%':<8} {'Time/ex':<8}",
        flush=True,
    )
    print("-" * 90, flush=True)

    for i, r in enumerate(results):
        print(
            f"{i + 1:<5} {r['config']:<40} {r.get('word_overlap', 0):<10.3f} "
            f"{r.get('clean_output_rate', 0):<8.3f} {r.get('length_ratio', 0):<10.3f} "
            f"{r.get('emoji_match_rate', 0):<8.3f} {r.get('avg_time_per_example', 0):<8.3f}s",
            flush=True,
        )

    print("=" * 90, flush=True)
    if results:
        winner = results[0]
        print(f"\nWinner: {winner['config']}", flush=True)
        print(f"  Model: {winner['model']}", flush=True)
        print(f"  Adapter: {winner['adapter']}", flush=True)


def main() -> None:
    _setup_logging()
    log.info("Starting evaluate_personal_ft.py")
    parser = argparse.ArgumentParser(description="Evaluate personal fine-tuned models")
    parser.add_argument("--report-only", action="store_true", help="Just print leaderboard")
    parser.add_argument("--config", help="Evaluate a single config file")
    parser.add_argument("--max-examples", type=int, default=50, help="Max test examples")
    args = parser.parse_args()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        if RESULTS_PATH.exists():
            results = json.loads(RESULTS_PATH.read_text())
            print_leaderboard(results)
        else:
            print(f"No results file found at {RESULTS_PATH}", flush=True)
        return

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            log.error("Config not found: %s", config_path)
            sys.exit(1)
        result = evaluate_config(config_path, max_examples=args.max_examples)
        if result:
            # Append to or create results
            results = []
            if RESULTS_PATH.exists():
                results = json.loads(RESULTS_PATH.read_text())
            # Replace existing entry for this config
            results = [r for r in results if r.get("config") != result["config"]]
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
            print_leaderboard(results)
        return

    # Evaluate all configs
    config_dir = Path("ft_configs")
    configs = sorted(config_dir.glob("personal_*.yaml"))

    if not configs:
        log.error("No config files found in ft_configs/")
        sys.exit(1)

    log.info("Found %d configs to evaluate", len(configs))

    results = []
    for config_path in configs:
        result = evaluate_config(config_path, max_examples=args.max_examples)
        if result:
            results.append(result)
            # Save incrementally
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print_leaderboard(results)
    log.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
