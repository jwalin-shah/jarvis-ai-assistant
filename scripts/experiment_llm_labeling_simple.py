#!/usr/bin/env python3
"""Experiment 1b: Test LLM labeling with SIMPLIFIED categories.

Hypothesis: Fine-grained categories (5-6 classes) are too hard for small LLMs.
Simplified categories might achieve much higher accuracy.

Simplified trigger categories:
- needs_action: Requests, questions, invitations that need a response
- casual: Reactions, statements, social chat - no action needed

Simplified response categories:
- positive: Agrees, confirms, accepts
- negative: Declines, defers, rejects
- neutral: Questions, reactions, other

Usage:
    uv run python -m scripts.experiment_llm_labeling_simple
"""

from __future__ import annotations

import argparse
import json
import re
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


# Mapping from fine-grained to simplified labels
TRIGGER_SIMPLE_MAP = {
    "commitment": "needs_action",
    "question": "needs_action",
    "reaction": "casual",
    "social": "casual",
    "statement": "casual",
}

RESPONSE_SIMPLE_MAP = {
    "agree": "positive",
    "decline": "negative",
    "defer": "negative",
    "other": "neutral",
    "question": "neutral",
    "reaction": "neutral",
}

TRIGGER_SIMPLE_LABELS = ["needs_action", "casual"]
RESPONSE_SIMPLE_LABELS = ["positive", "negative", "neutral"]


@dataclass
class LabelingResult:
    """Result of LLM labeling a single message."""

    text: str
    human_label: str
    llm_label: str
    llm_raw_output: str
    correct: bool
    latency_ms: float


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


def create_trigger_prompt_simple(messages: list[str]) -> str:
    """Create prompt for simplified trigger classification."""
    prompt = """Classify each message. Does it NEED A RESPONSE/ACTION or is it just casual chat?

Categories:
- needs_action: Questions, requests, invitations that need a reply (e.g., "Can you help me?", "Want to hang out?", "What time?")
- casual: Statements, reactions, small talk - no response needed (e.g., "That's cool", "I'm home", "Haha", "Hey!")

Messages:
"""
    for i, msg in enumerate(messages, 1):
        prompt += f'{i}. "{msg}"\n'

    prompt += "\nOutput each as: number:label (e.g., 1:needs_action, 2:casual)\nOutput:\n"
    return prompt


def create_response_prompt_simple(messages: list[str]) -> str:
    """Create prompt for simplified response classification."""
    prompt = """Classify each message as a type of response.

Categories:
- positive: Agrees, accepts, confirms (e.g., "Sure", "Yes", "Okay", "Sounds good", "I'm down")
- negative: Declines, refuses, can't do it (e.g., "No", "I can't", "Sorry", "Maybe later", "Not today")
- neutral: Everything else - questions, reactions, statements (e.g., "What time?", "Haha", "Cool", "I don't know")

Messages:
"""
    for i, msg in enumerate(messages, 1):
        prompt += f'{i}. "{msg}"\n'

    prompt += "\nOutput each as: number:label (e.g., 1:positive, 2:negative, 3:neutral)\nOutput:\n"
    return prompt


def parse_llm_output(output: str, n_messages: int, valid_labels: list[str]) -> list[str]:
    """Parse LLM output into labels."""
    labels = ["unknown"] * n_messages

    # Match patterns: "1:label", "1. label", "1 - label", "1) label"
    pattern = r"(\d+)[:\.\-\)]\s*(\w+)"
    matches = re.findall(pattern, output.lower())

    for num_str, label in matches:
        try:
            idx = int(num_str) - 1
            if 0 <= idx < n_messages:
                label = label.strip().lower()
                # Match label
                for valid in valid_labels:
                    if label.startswith(valid[:4]) or valid.startswith(label[:4]):
                        labels[idx] = valid
                        break
        except ValueError:
            continue

    return labels


def run_experiment(
    model,
    tokenizer,
    task: str,
    data: list[dict],
    label_map: dict[str, str],
    valid_labels: list[str],
    batch_size: int = 10,
) -> dict:
    """Run the LLM labeling experiment with simplified labels."""
    create_prompt = (
        create_trigger_prompt_simple if task == "trigger" else create_response_prompt_simple
    )

    results: list[LabelingResult] = []
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start : batch_start + batch_size]
        texts = [item["text"] for item in batch]
        # Map fine-grained labels to simplified
        human_labels = [label_map.get(item["label"], "unknown") for item in batch]

        prompt = create_prompt(texts)

        start_time = time.time()

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
            max_tokens=100,
            verbose=False,
        )
        latency_ms = (time.time() - start_time) * 1000

        llm_labels = parse_llm_output(output, len(texts), valid_labels)

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

        done = min(batch_start + batch_size, len(data))
        correct_so_far = sum(1 for r in results if r.correct)
        print(
            f"  [{done}/{len(data)}] Accuracy: {correct_so_far / len(results):.1%} | "
            f"Batch: {latency_ms:.0f}ms"
        )

    total_time = time.time() - total_start

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results)

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

    return {
        "task": task,
        "n_samples": len(results),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        "total_time_s": total_time,
        "errors": [
            {"text": r.text[:80], "human": r.human_label, "llm": r.llm_label}
            for r in results
            if not r.correct
        ][:10],
    }


def print_results(res: dict) -> None:
    """Print experiment results."""
    print(f"\n{'=' * 60}")
    print(f"{res['task'].upper()} - SIMPLIFIED CATEGORIES")
    print(f"{'=' * 60}")
    print(f"Samples: {res['n_samples']}")
    print(f"Overall Accuracy: {res['accuracy']:.1%}")
    print(f"Latency: {res['avg_latency_ms']:.1f}ms/msg, Total: {res['total_time_s']:.1f}s")

    print("\nPer-Class:")
    for label, acc in sorted(res["per_class_accuracy"].items()):
        total = sum(res["confusion_matrix"].get(label, {}).values())
        print(f"  {label:<15} {acc:.1%} ({total} samples)")

    print("\nConfusion Matrix:")
    labels = sorted(res["per_class_accuracy"].keys())
    header = " " * 15 + "".join(f"{l[:10]:>12}" for l in labels)
    print(header)
    for human in labels:
        row = f"{human:<15}"
        for pred in labels:
            count = res["confusion_matrix"].get(human, {}).get(pred, 0)
            row += f"{count:>12}"
        print(row)

    if res["errors"]:
        print("\nSample Errors:")
        for e in res["errors"][:5]:
            print(f'  "{e["text"][:50]}..." → human:{e["human"]}, llm:{e["llm"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit", help="MLX model"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/llm_labeling_simple.json"))
    args = parser.parse_args()

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded!\n")

    all_results = {}

    # Trigger
    print("=" * 60)
    print("TRIGGER CLASSIFICATION (simplified)")
    print("=" * 60)
    trigger_data = load_labeled_data(Path("data/trigger_labeling.jsonl"))
    n_per_class = args.n_samples // 5  # Original has 5 classes
    trigger_sample = sample_balanced(trigger_data, n_per_class)
    print(f"Sampled {len(trigger_sample)} messages")
    orig_dist = Counter(d["label"] for d in trigger_sample)
    simple_dist = Counter(TRIGGER_SIMPLE_MAP[d["label"]] for d in trigger_sample)
    print(f"Original distribution: {dict(orig_dist)}")
    print(f"Simplified distribution: {dict(simple_dist)}")

    trigger_res = run_experiment(
        model,
        tokenizer,
        task="trigger",
        data=trigger_sample,
        label_map=TRIGGER_SIMPLE_MAP,
        valid_labels=TRIGGER_SIMPLE_LABELS,
        batch_size=args.batch_size,
    )
    print_results(trigger_res)
    all_results["trigger"] = trigger_res

    # Response
    print("\n" + "=" * 60)
    print("RESPONSE CLASSIFICATION (simplified)")
    print("=" * 60)
    response_data = load_labeled_data(Path("data/response_labeling.jsonl"))
    for item in response_data:
        item["label"] = item["label"].lower()
    n_per_class = args.n_samples // 6
    response_sample = sample_balanced(response_data, n_per_class)
    print(f"Sampled {len(response_sample)} messages")
    orig_dist = Counter(d["label"] for d in response_sample)
    simple_dist = Counter(RESPONSE_SIMPLE_MAP.get(d["label"], "neutral") for d in response_sample)
    print(f"Original distribution: {dict(orig_dist)}")
    print(f"Simplified distribution: {dict(simple_dist)}")

    response_res = run_experiment(
        model,
        tokenizer,
        task="response",
        data=response_sample,
        label_map=RESPONSE_SIMPLE_MAP,
        valid_labels=RESPONSE_SIMPLE_LABELS,
        batch_size=args.batch_size,
    )
    print_results(response_res)
    all_results["response"] = response_res

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model, "results": all_results}, indent=2))
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON: Fine-grained vs Simplified")
    print("=" * 60)
    print(f"Trigger: 38% (5 classes) → {trigger_res['accuracy']:.1%} (2 classes)")
    print(f"Response: 43% (6 classes) → {response_res['accuracy']:.1%} (3 classes)")


if __name__ == "__main__":
    main()
