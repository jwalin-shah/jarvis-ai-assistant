#!/usr/bin/env python3
"""Benchmark small NLI models for trigger classification.

Tests lightweight NLI models as alternatives to heavy BART-MNLI:
- prajjwal1/bert-tiny-mnli (~17MB, 60% MNLI)
- MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli (~100MB, 78% MNLI)

Usage:
    uv run python -m scripts.benchmark_small_nli
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Label hypotheses for NLI zero-shot classification
LABEL_HYPOTHESES = {
    "commitment": "an invitation, request, or plan to do something",
    "question": "asking a question that needs an answer",
    "reaction": "an emotional reaction expressing feelings",
    "social": "a greeting, acknowledgment, or casual response",
    "statement": "sharing information or a neutral statement",
}

LABELS = list(LABEL_HYPOTHESES.keys())


def load_labeled_data(path: Path) -> tuple[list[str], list[str]]:
    """Load labeled trigger data."""
    texts = []
    labels = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            label = row.get("label")
            if text and label:
                labels.append(label.lower())
                texts.append(text)
    return texts, labels


class NLIClassifier:
    """Zero-shot classifier using NLI model."""

    def __init__(self, model_name: str, device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Get label mapping - NLI models output [contradiction, neutral, entailment]
        # or [entailment, neutral, contradiction] depending on model
        self.id2label = self.model.config.id2label
        self.entailment_id = None
        for idx, label in self.id2label.items():
            if "entail" in label.lower():
                self.entailment_id = idx
                break
        if self.entailment_id is None:
            # Default to last class (often entailment)
            self.entailment_id = len(self.id2label) - 1

    def classify(self, text: str, labels: list[str], hypotheses: dict[str, str]) -> str:
        """Classify text using NLI entailment scores."""
        scores = {}

        for label in labels:
            hypothesis = hypotheses[label]
            # NLI format: premise [SEP] hypothesis
            inputs = self.tokenizer(
                text,
                f"This is {hypothesis}.",
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # Get entailment probability
                entail_score = probs[0][self.entailment_id].item()
                scores[label] = entail_score

        return max(scores, key=scores.get)

    def classify_batch(
        self, texts: list[str], labels: list[str], hypotheses: dict[str, str], batch_size: int = 16
    ) -> list[str]:
        """Classify multiple texts efficiently."""
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_preds = []

            for text in batch_texts:
                pred = self.classify(text, labels, hypotheses)
                batch_preds.append(pred)

            predictions.extend(batch_preds)

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
                print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")

        return predictions


def benchmark_nli_model(
    model_name: str,
    texts: list[str],
    labels: list[str],
) -> dict:
    """Benchmark an NLI model on trigger classification."""
    print(f"\nLoading {model_name}...")
    start_load = time.perf_counter()

    try:
        classifier = NLIClassifier(model_name, device="mps")
    except Exception as e:
        print(f"  Failed to load on MPS, trying CPU: {e}")
        classifier = NLIClassifier(model_name, device="cpu")

    load_time = time.perf_counter() - start_load
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Entailment ID: {classifier.entailment_id}")
    print(f"  Labels: {classifier.id2label}")

    # Classify all texts
    print(f"  Classifying {len(texts)} messages...")
    start_infer = time.perf_counter()

    predictions = classifier.classify_batch(texts, LABELS, LABEL_HYPOTHESES)

    infer_time = time.perf_counter() - start_infer
    avg_latency = (infer_time / len(texts)) * 1000  # ms per message

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    print(f"\n  Results for {model_name}:")
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Macro F1: {macro_f1:.1%}")
    print(f"    Avg latency: {avg_latency:.1f}ms/msg")

    # Per-class report
    print(classification_report(labels, predictions, zero_division=0))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "load_time_s": load_time,
        "total_infer_time_s": infer_time,
    }


def benchmark_v3(texts: list[str], labels: list[str]) -> dict:
    """Benchmark V3 classifier for comparison."""
    from jarvis.classifiers.trigger_classifier import classify_trigger

    print("\nBenchmarking V3 (current baseline)...")

    # Warm up
    classify_trigger("hello")

    start = time.perf_counter()
    predictions = []
    methods = Counter()

    for text in texts:
        result = classify_trigger(text)
        predictions.append(result.trigger_type.value)
        methods[result.method] += 1

    infer_time = time.perf_counter() - start
    avg_latency = (infer_time / len(texts)) * 1000

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    print("\n  V3 Results:")
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Macro F1: {macro_f1:.1%}")
    print(f"    Avg latency: {avg_latency:.1f}ms/msg")
    print("\n  Method distribution:")
    for method, count in methods.most_common():
        print(f"    {method}: {count} ({count / len(texts):.1%})")

    print(classification_report(labels, predictions, zero_division=0))

    return {
        "model": "V3 (hybrid)",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "method_distribution": dict(methods),
    }


def main():
    data_path = Path("data/trigger_labeling.jsonl")

    print("=" * 60)
    print("SMALL NLI MODEL BENCHMARK")
    print("=" * 60)

    # Load data
    texts, labels = load_labeled_data(data_path)
    print(f"\nLoaded {len(texts)} labeled examples")
    print(f"Distribution: {dict(Counter(labels).most_common())}")

    results = []

    # 1. Benchmark V3 (baseline)
    v3_result = benchmark_v3(texts, labels)
    results.append(v3_result)

    # 2. Benchmark tiny BERT
    try:
        tiny_result = benchmark_nli_model(
            "prajjwal1/bert-tiny-mnli",
            texts,
            labels,
        )
        results.append(tiny_result)
    except Exception as e:
        print(f"bert-tiny-mnli failed: {e}")
        import traceback

        traceback.print_exc()

    # 3. Benchmark MiniLM
    try:
        minilm_result = benchmark_nli_model(
            "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            texts,
            labels,
        )
        results.append(minilm_result)
    except Exception as e:
        print(f"MiniLM failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<50} {'Accuracy':>10} {'Macro F1':>10} {'Latency':>12}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['model']:<50} {r['accuracy']:>10.1%} {r['macro_f1']:>10.1%} {r['avg_latency_ms']:>10.1f}ms"
        )

    # Save results
    output_path = Path("results/small_nli_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
