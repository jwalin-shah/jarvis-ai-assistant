#!/usr/bin/env python3
"""Experiment: Zero-shot classification using embeddings + cosine similarity.

Instead of using an LLM to classify, we:
1. Create text descriptions for each category
2. Embed the descriptions
3. Embed the message
4. Classify by finding the most similar category description

This uses the existing bge-small encoder and requires NO training data.

Usage:
    uv run python -m scripts.experiment_embedding_zeroshot
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jarvis.embedding_adapter import get_embedder

# Category descriptions - the key to zero-shot classification
# Multiple descriptions per category to capture different phrasings
TRIGGER_CATEGORIES = {
    "needs_action": [
        "This message is asking for something or needs a response",
        "A request, question, or invitation that requires action",
        "Someone is asking me to do something or answer a question",
        "This is a question that needs an answer",
        "Can you help me with something",
        "Are you available to do this",
        "Would you like to join me",
    ],
    "casual": [
        "This message is just casual chat, no response needed",
        "A statement, reaction, or small talk",
        "Someone sharing information or reacting to something",
        "This is just a comment or observation",
        "That's cool, nice, interesting",
        "I'm just letting you know",
        "Haha, lol, funny",
    ],
}

RESPONSE_CATEGORIES = {
    "positive": [
        "This is an agreement or acceptance",
        "Yes, sure, okay, I'll do it",
        "Sounds good, I'm in, let's do it",
        "Confirming or accepting a request",
        "A positive response agreeing to something",
    ],
    "negative": [
        "This is a decline or refusal",
        "No, I can't, sorry, not today",
        "Declining or refusing a request",
        "A negative response rejecting something",
        "Maybe later, I'll pass, not right now",
    ],
    "neutral": [
        "This is neither yes nor no",
        "A question, reaction, or general statement",
        "Asking for more information",
        "Just a comment or observation",
        "What time, where, how",
        "Haha, cool, nice",
    ],
}


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


class EmbeddingZeroShotClassifier:
    """Zero-shot classifier using embedding similarity."""

    def __init__(self, categories: dict[str, list[str]]):
        """
        Args:
            categories: Dict mapping label -> list of description texts
        """
        self.categories = categories
        self.embedder = get_embedder()
        self.category_embeddings: dict[str, np.ndarray] = {}
        self._compute_category_embeddings()

    def _compute_category_embeddings(self):
        """Pre-compute embeddings for all category descriptions."""
        print("Computing category embeddings...")
        for label, descriptions in self.categories.items():
            # Embed all descriptions for this category
            embeddings = self.embedder.encode(descriptions, normalize=True)
            # Average them to get a single prototype
            prototype = np.mean(embeddings, axis=0)
            # Re-normalize
            prototype = prototype / np.linalg.norm(prototype)
            self.category_embeddings[label] = prototype
            print(f"  {label}: {len(descriptions)} descriptions -> prototype")

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        """Classify a single text.

        Returns:
            (predicted_label, scores_dict)
        """
        # Embed the text
        text_emb = self.embedder.encode([text], normalize=True)[0]

        # Compute similarity to each category
        scores = {}
        for label, prototype in self.category_embeddings.items():
            scores[label] = float(np.dot(text_emb, prototype))

        # Return highest scoring category
        predicted = max(scores, key=lambda k: scores[k])
        return predicted, scores

    def classify_batch(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """Classify multiple texts efficiently."""
        # Embed all texts at once
        text_embs = self.embedder.encode(texts, normalize=True)

        results = []
        for text_emb in text_embs:
            scores = {}
            for label, prototype in self.category_embeddings.items():
                scores[label] = float(np.dot(text_emb, prototype))
            predicted = max(scores, key=lambda k: scores[k])
            results.append((predicted, scores))

        return results


def run_experiment(
    classifier: EmbeddingZeroShotClassifier,
    data: list[dict],
    label_map: dict[str, str],
    task: str,
) -> dict:
    """Run the classification experiment."""
    texts = [item["text"] for item in data]
    human_labels = [label_map.get(item["label"], "unknown") for item in data]

    start_time = time.time()
    results = classifier.classify_batch(texts)
    total_time = time.time() - start_time

    predictions = [r[0] for r in results]
    scores_list = [r[1] for r in results]

    # Compute metrics
    correct = sum(1 for h, p in zip(human_labels, predictions) if h == p)
    accuracy = correct / len(human_labels)

    # Per-class
    per_class_correct: dict[str, int] = Counter()
    per_class_total: dict[str, int] = Counter()
    for h, p in zip(human_labels, predictions):
        per_class_total[h] += 1
        if h == p:
            per_class_correct[h] += 1

    valid_labels = list(classifier.category_embeddings.keys())
    per_class_accuracy = {
        label: per_class_correct[label] / per_class_total[label]
        if per_class_total[label] > 0
        else 0
        for label in valid_labels
    }

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {label: Counter() for label in valid_labels}
    for h, p in zip(human_labels, predictions):
        if h in confusion:
            confusion[h][p] += 1

    # Collect errors
    errors = []
    for i, (h, p) in enumerate(zip(human_labels, predictions)):
        if h != p:
            errors.append(
                {
                    "text": texts[i][:80],
                    "human": h,
                    "predicted": p,
                    "scores": scores_list[i],
                }
            )

    return {
        "task": task,
        "n_samples": len(data),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "total_time_s": total_time,
        "avg_latency_ms": total_time * 1000 / len(data),
        "errors": errors[:10],
    }


def print_results(res: dict) -> None:
    """Print experiment results."""
    print(f"\n{'=' * 60}")
    print(f"{res['task'].upper()} - EMBEDDING ZERO-SHOT")
    print(f"{'=' * 60}")
    print(f"Samples: {res['n_samples']}")
    print(f"Overall Accuracy: {res['accuracy']:.1%}")
    print(f"Time: {res['total_time_s']:.2f}s ({res['avg_latency_ms']:.1f}ms/msg)")

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
            scores_str = ", ".join(f"{k}:{v:.2f}" for k, v in e["scores"].items())
            print(f'  "{e["text"][:50]}..."')
            print(f"    human:{e['human']}, pred:{e['predicted']} | {scores_str}")


def main():
    print("=" * 60)
    print("EMBEDDING ZERO-SHOT CLASSIFICATION")
    print("Using bge-small-en-v1.5 with category prototypes")
    print("=" * 60)

    # Create classifiers
    trigger_clf = EmbeddingZeroShotClassifier(TRIGGER_CATEGORIES)
    response_clf = EmbeddingZeroShotClassifier(RESPONSE_CATEGORIES)

    # Load data
    trigger_data = load_labeled_data(Path("data/trigger_labeling.jsonl"))
    trigger_sample = sample_balanced(trigger_data, 20)  # 20 per class, 100 total
    print(f"\nTrigger: {len(trigger_sample)} samples")
    print(f"Distribution: {Counter(d['label'] for d in trigger_sample)}")

    response_data = load_labeled_data(Path("data/response_labeling.jsonl"))
    for item in response_data:
        item["label"] = item["label"].lower()
    response_sample = sample_balanced(response_data, 16)  # 16 per class, 96 total
    print(f"\nResponse: {len(response_sample)} samples")
    print(f"Distribution: {Counter(d['label'] for d in response_sample)}")

    # Run experiments
    trigger_res = run_experiment(trigger_clf, trigger_sample, TRIGGER_SIMPLE_MAP, "trigger")
    print_results(trigger_res)

    response_res = run_experiment(response_clf, response_sample, RESPONSE_SIMPLE_MAP, "response")
    print_results(response_res)

    # Save results
    output_path = Path("results/embedding_zeroshot_experiment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "method": "embedding_similarity_zeroshot",
                "embedding_model": "bge-small-en-v1.5",
                "trigger": trigger_res,
                "response": response_res,
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {output_path}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: LLM vs Embedding Zero-Shot")
    print("=" * 60)
    print(f"{'Method':<25} {'Trigger':>12} {'Response':>12}")
    print("-" * 50)
    print(f"{'LLM 1.2B (fine-grained)':<25} {'38%':>12} {'43%':>12}")
    print(f"{'LLM 1.2B (simplified)':<25} {'56%':>12} {'30%':>12}")
    print(
        f"{'Embedding zero-shot':<25} {trigger_res['accuracy']:>11.0%} {response_res['accuracy']:>11.0%}"
    )


if __name__ == "__main__":
    main()
