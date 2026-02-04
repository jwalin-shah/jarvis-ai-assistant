#!/usr/bin/env python3
"""Experiment 1: Test LLM labeling accuracy against human labels.

This tests whether LFM 1.2B (or other small LLMs) can reliably label
messages for the cold-start scenario where users have no labeled data.

Usage:
    uv run python -m scripts.experiment_llm_labeling
    uv run python -m scripts.experiment_llm_labeling --model Qwen/Qwen2.5-0.5B-Instruct-MLX-4bit
    uv run python -m scripts.experiment_llm_labeling --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MLX imports
try:
    from mlx_lm import generate, load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class LabelingResult:
    """Result of LLM labeling a single message."""

    text: str
    human_label: str
    llm_label: str
    llm_raw_output: str
    correct: bool
    latency_ms: float


@dataclass
class ExperimentResults:
    """Aggregate results from the experiment."""

    model_name: str
    task: str  # "trigger" or "response"
    n_samples: int
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    avg_latency_ms: float
    total_time_s: float
    results: list[LabelingResult]


# Label mappings for normalization
TRIGGER_LABELS = ["commitment", "question", "reaction", "social", "statement"]
RESPONSE_LABELS = ["agree", "decline", "defer", "other", "question", "reaction"]


def load_labeled_data(path: Path, label_field: str = "label") -> list[dict]:
    """Load labeled data from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("response", "")
            label = row.get(label_field, "").lower()
            if text and label:
                data.append({"text": text.strip(), "label": label})
    return data


def sample_balanced(data: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Sample balanced data across classes."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict]] = {}
    for item in data:
        label = item["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item)

    sampled = []
    for label, items in by_label.items():
        if len(items) <= n_per_class:
            sampled.extend(items)
        else:
            indices = rng.choice(len(items), size=n_per_class, replace=False)
            sampled.extend([items[i] for i in indices])

    rng.shuffle(sampled)
    return sampled


def create_trigger_prompt(messages: list[str]) -> str:
    """Create prompt for trigger classification."""
    prompt = """Classify each message into ONE category. Output ONLY the number and label, nothing else.

Categories:
- commitment: Requests, invitations, asks for action (e.g., "Can you pick me up?", "Want to hang out?")
- question: Information-seeking questions (e.g., "What time is it?", "How was your day?")
- reaction: Emotional responses, opinions (e.g., "That's crazy!", "I love it")
- social: Greetings, small talk (e.g., "Hey!", "What's up")
- statement: Neutral information sharing (e.g., "I'm at home", "The meeting is at 3")

Messages:
"""
    for i, msg in enumerate(messages, 1):
        prompt += f'{i}. "{msg}"\n'

    prompt += "\nOutput (format: 1:label, 2:label, ...):\n"
    return prompt


def create_response_prompt(messages: list[str]) -> str:
    """Create prompt for response classification."""
    prompt = """Classify each message into ONE category. Output ONLY the number and label, nothing else.

Categories:
- agree: Acceptance, confirmation (e.g., "Sure", "Okay", "Yes", "Sounds good")
- decline: Rejection, refusal (e.g., "No", "I can't", "Sorry, not today")
- defer: Postponing, uncertainty (e.g., "Maybe later", "Let me check", "I'll see")
- question: Follow-up questions (e.g., "What time?", "Where?")
- reaction: Emotional response (e.g., "Haha", "Wow", "That's cool")
- other: Everything else (statements, info sharing)

Messages:
"""
    for i, msg in enumerate(messages, 1):
        prompt += f'{i}. "{msg}"\n'

    prompt += "\nOutput (format: 1:label, 2:label, ...):\n"
    return prompt


def parse_llm_output(output: str, n_messages: int, valid_labels: list[str]) -> list[str]:
    """Parse LLM output into labels."""
    labels = ["unknown"] * n_messages

    # Try to find patterns like "1:label" or "1. label"
    import re

    # Match patterns: "1:label", "1. label", "1 - label", "1) label"
    pattern = r"(\d+)[:\.\-\)]\s*(\w+)"
    matches = re.findall(pattern, output.lower())

    for num_str, label in matches:
        try:
            idx = int(num_str) - 1  # Convert to 0-indexed
            if 0 <= idx < n_messages:
                # Normalize label
                label = label.strip().lower()
                # Find closest match in valid labels
                for valid in valid_labels:
                    if label.startswith(valid[:3]) or valid.startswith(label[:3]):
                        labels[idx] = valid
                        break
                else:
                    # Try exact match
                    if label in valid_labels:
                        labels[idx] = label
        except ValueError:
            continue

    return labels


def run_experiment(
    model_name: str,
    task: str,
    data: list[dict],
    batch_size: int = 10,
) -> ExperimentResults:
    """Run the LLM labeling experiment."""
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available. Install with: pip install mlx-lm")

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded!")

    valid_labels = TRIGGER_LABELS if task == "trigger" else RESPONSE_LABELS
    create_prompt = create_trigger_prompt if task == "trigger" else create_response_prompt

    results: list[LabelingResult] = []
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start : batch_start + batch_size]
        texts = [item["text"] for item in batch]
        human_labels = [item["label"] for item in batch]

        prompt = create_prompt(texts)

        # Generate
        start_time = time.time()

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        output = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=200,
            verbose=False,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Parse output
        llm_labels = parse_llm_output(output, len(texts), valid_labels)

        # Record results
        per_msg_latency = latency_ms / len(texts)
        for text, human_label, llm_label in zip(texts, human_labels, llm_labels):
            results.append(
                LabelingResult(
                    text=text,
                    human_label=human_label,
                    llm_label=llm_label,
                    llm_raw_output=output,
                    correct=(human_label == llm_label),
                    latency_ms=per_msg_latency,
                )
            )

        # Progress
        done = min(batch_start + batch_size, len(data))
        correct_so_far = sum(1 for r in results if r.correct)
        print(
            f"  Processed {done}/{len(data)} - "
            f"Accuracy so far: {correct_so_far / len(results):.1%} - "
            f"Batch latency: {latency_ms:.0f}ms"
        )

    total_time = time.time() - total_start

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results)

    # Per-class accuracy
    per_class_correct: dict[str, int] = Counter()
    per_class_total: dict[str, int] = Counter()
    for r in results:
        per_class_total[r.human_label] += 1
        if r.correct:
            per_class_correct[r.human_label] += 1

    per_class_accuracy = {
        label: per_class_correct[label] / per_class_total[label]
        if per_class_total[label] > 0
        else 0
        for label in valid_labels
    }

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {label: Counter() for label in valid_labels}
    for r in results:
        if r.human_label in confusion:
            confusion[r.human_label][r.llm_label] += 1

    avg_latency = sum(r.latency_ms for r in results) / len(results)

    return ExperimentResults(
        model_name=model_name,
        task=task,
        n_samples=len(results),
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        avg_latency_ms=avg_latency,
        total_time_s=total_time,
        results=results,
    )


def print_results(results: ExperimentResults) -> None:
    """Print experiment results."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT RESULTS: {results.task.upper()} CLASSIFICATION")
    print("=" * 70)
    print(f"Model: {results.model_name}")
    print(f"Samples: {results.n_samples}")
    print(f"Total time: {results.total_time_s:.1f}s")
    print(f"Avg latency per message: {results.avg_latency_ms:.1f}ms")

    print(f"\nOverall Accuracy: {results.accuracy:.1%}")

    print("\nPer-Class Accuracy:")
    print(f"  {'Class':<15} {'Accuracy':>10} {'Samples':>10}")
    print("  " + "-" * 35)
    for label in sorted(results.per_class_accuracy.keys()):
        acc = results.per_class_accuracy[label]
        total = sum(results.confusion_matrix.get(label, {}).values())
        print(f"  {label:<15} {acc:>10.1%} {total:>10}")

    print("\nConfusion Matrix (rows=human, cols=predicted):")
    labels = sorted(results.per_class_accuracy.keys())
    header = "  " + " " * 12 + "".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for human_label in labels:
        row = f"  {human_label:<12}"
        for pred_label in labels:
            count = results.confusion_matrix.get(human_label, {}).get(pred_label, 0)
            row += f"{count:>10}"
        print(row)

    # Show some errors
    errors = [r for r in results.results if not r.correct][:5]
    if errors:
        print("\nSample Errors:")
        for e in errors:
            print(f'  Text: "{e.text[:50]}..."')
            print(f"  Human: {e.human_label}, LLM: {e.llm_label}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Test LLM labeling accuracy")
    parser.add_argument(
        "--model",
        default="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        help="MLX model to use",
    )
    parser.add_argument(
        "--task",
        choices=["trigger", "response", "both"],
        default="both",
        help="Which task to test",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to test (balanced across classes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Messages per LLM call",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llm_labeling_experiment.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    all_results = {}

    if args.task in ["trigger", "both"]:
        print("\n" + "=" * 70)
        print("TRIGGER CLASSIFICATION")
        print("=" * 70)

        # Load and sample data
        trigger_data = load_labeled_data(Path("data/trigger_labeling.jsonl"))
        n_per_class = args.n_samples // len(TRIGGER_LABELS)
        trigger_sample = sample_balanced(trigger_data, n_per_class)
        print(f"Sampled {len(trigger_sample)} trigger messages ({n_per_class} per class)")
        print(f"Distribution: {Counter(d['label'] for d in trigger_sample)}")

        trigger_results = run_experiment(
            model_name=args.model,
            task="trigger",
            data=trigger_sample,
            batch_size=args.batch_size,
        )
        print_results(trigger_results)
        all_results["trigger"] = {
            "accuracy": trigger_results.accuracy,
            "per_class_accuracy": trigger_results.per_class_accuracy,
            "confusion_matrix": trigger_results.confusion_matrix,
            "avg_latency_ms": trigger_results.avg_latency_ms,
            "total_time_s": trigger_results.total_time_s,
            "n_samples": trigger_results.n_samples,
        }

    if args.task in ["response", "both"]:
        print("\n" + "=" * 70)
        print("RESPONSE CLASSIFICATION")
        print("=" * 70)

        # Load and sample data
        response_data = load_labeled_data(Path("data/response_labeling.jsonl"), label_field="label")
        # Normalize response labels
        for item in response_data:
            item["label"] = item["label"].lower()

        n_per_class = args.n_samples // len(RESPONSE_LABELS)
        response_sample = sample_balanced(response_data, n_per_class)
        print(f"Sampled {len(response_sample)} response messages ({n_per_class} per class)")
        print(f"Distribution: {Counter(d['label'] for d in response_sample)}")

        response_results = run_experiment(
            model_name=args.model,
            task="response",
            data=response_sample,
            batch_size=args.batch_size,
        )
        print_results(response_results)
        all_results["response"] = {
            "accuracy": response_results.accuracy,
            "per_class_accuracy": response_results.per_class_accuracy,
            "confusion_matrix": response_results.confusion_matrix,
            "avg_latency_ms": response_results.avg_latency_ms,
            "total_time_s": response_results.total_time_s,
            "n_samples": response_results.n_samples,
        }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": args.model,
        "n_samples_requested": args.n_samples,
        "batch_size": args.batch_size,
        "results": all_results,
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {args.output}")

    # Update experiment doc
    print("\n" + "=" * 70)
    print("SUMMARY FOR COLD_START_EXPERIMENT.md")
    print("=" * 70)
    for task, res in all_results.items():
        print(f"\n{task.upper()}:")
        print(f"  Accuracy: {res['accuracy']:.1%}")
        print(f"  Avg latency: {res['avg_latency_ms']:.1f}ms per message")
        print(f"  Per-class: {res['per_class_accuracy']}")


if __name__ == "__main__":
    main()
