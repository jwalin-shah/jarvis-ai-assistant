#!/usr/bin/env python3
"""Experiment: Zero-shot classification with BETTER response categories.

The key insight: What does the USER care about?
- Did they say YES? (actionable - follow up)
- Did they say NO? (closed - try something else)
- Did they say MAYBE? (pending - wait/check back)
- Something else? (just info, no action needed)

Usage:
    uv run python -m scripts.experiment_embedding_zeroshot_v2
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jarvis.embedding_adapter import get_embedder

# =============================================================================
# TRIGGER: Same as before (worked well at 84%)
# =============================================================================
TRIGGER_CATEGORIES = {
    "needs_action": [
        "This message is asking for something or needs a response",
        "A request, question, or invitation that requires action",
        "Someone is asking me to do something or answer a question",
        "Can you help me with something",
        "Are you available",
        "Would you like to",
        "What do you think",
    ],
    "casual": [
        "This message is just casual chat, no response needed",
        "A statement, reaction, or small talk",
        "Someone sharing information or reacting",
        "Just letting you know",
        "That's cool",
        "Haha funny",
        "I'm at home",
    ],
}

TRIGGER_MAP = {
    "commitment": "needs_action",
    "question": "needs_action",
    "reaction": "casual",
    "social": "casual",
    "statement": "casual",
}

# =============================================================================
# RESPONSE: New 4-category system
# =============================================================================
RESPONSE_CATEGORIES_V2 = {
    "yes": [
        "This is a yes, agreement, or acceptance",
        "I agree, I'm in, I'll do it",
        "Sure, okay, sounds good, yes",
        "Confirming or accepting",
        "I'm down, let's do it",
        "Yee, yep, ok, fs",
    ],
    "no": [
        "This is a no, decline, or refusal",
        "I can't, I won't, no thanks",
        "Declining or refusing",
        "Not going to happen",
        "Nah, no, I'm not doing that",
        "I would never, not interested",
    ],
    "maybe": [
        "This is uncertain, a maybe, or needs more info",
        "Let me check, I'll see, not sure yet",
        "Deferring the decision",
        "We'll see what happens",
        "I need to think about it",
        "Maybe, possibly, depends",
        "I don't know yet, hmm",
    ],
    "other": [
        "This is just information, a reaction, or a question",
        "Sharing a link, statement, or random info",
        "A reaction like haha, wow, nice",
        "Asking a follow-up question",
        "Not answering yes, no, or maybe",
        "Just chatting or commenting",
    ],
}

RESPONSE_MAP_V2 = {
    "agree": "yes",
    "decline": "no",
    "defer": "maybe",
    "other": "other",
    "question": "other",
    "reaction": "other",
}


# =============================================================================
# Also try: 3-category "answer type" framing
# =============================================================================
RESPONSE_CATEGORIES_V3 = {
    "answered_yes": [
        "They said yes or agreed to something",
        "Acceptance, confirmation, I'm in",
        "Sure, okay, sounds good, yes, yep",
        "They will do it, they're down",
    ],
    "answered_no": [
        "They said no or declined something",
        "Refusal, rejection, can't do it",
        "No, nah, I won't, not happening",
        "They're not going to do it",
    ],
    "no_answer": [
        "They didn't give a yes or no answer",
        "Uncertain, maybe, we'll see",
        "A question, reaction, or just info",
        "Not a direct response to a request",
        "Haha, cool, what time, I'm home",
    ],
}

RESPONSE_MAP_V3 = {
    "agree": "answered_yes",
    "decline": "answered_no",
    "defer": "no_answer",  # Key change: defer is NOT a "no"
    "other": "no_answer",
    "question": "no_answer",
    "reaction": "no_answer",
}


def load_labeled_data(path: Path) -> list[dict]:
    """Load labeled data from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("response", "")
            label = row.get("label", "").lower()
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
        n = min(n_per_class, len(items))
        indices = rng.choice(len(items), size=n, replace=False)
        sampled.extend([items[i] for i in indices])

    rng.shuffle(sampled)
    return sampled


class EmbeddingClassifier:
    """Zero-shot classifier using embedding similarity."""

    def __init__(self, categories: dict[str, list[str]], name: str = ""):
        self.categories = categories
        self.name = name
        self.embedder = get_embedder()
        self.prototypes: dict[str, np.ndarray] = {}
        self._compute_prototypes()

    def _compute_prototypes(self):
        """Compute prototype embeddings for each category."""
        for label, descriptions in self.categories.items():
            embeddings = self.embedder.encode(descriptions, normalize=True)
            prototype = np.mean(embeddings, axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            self.prototypes[label] = prototype

    def classify_batch(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """Classify texts by similarity to prototypes."""
        embeddings = self.embedder.encode(texts, normalize=True)
        results = []
        for emb in embeddings:
            scores = {label: float(np.dot(emb, proto)) for label, proto in self.prototypes.items()}
            predicted = max(scores, key=lambda k: scores[k])
            results.append((predicted, scores))
        return results


def evaluate(
    clf: EmbeddingClassifier,
    data: list[dict],
    label_map: dict[str, str],
) -> dict:
    """Evaluate classifier on data."""
    texts = [d["text"] for d in data]
    human_labels = [label_map.get(d["label"], "unknown") for d in data]

    start = time.time()
    results = clf.classify_batch(texts)
    elapsed = time.time() - start

    predictions = [r[0] for r in results]

    # Accuracy
    correct = sum(1 for h, p in zip(human_labels, predictions) if h == p)
    accuracy = correct / len(human_labels)

    # Per-class
    valid_labels = list(clf.prototypes.keys())
    per_class = {}
    confusion = {l: Counter() for l in valid_labels}

    for h, p in zip(human_labels, predictions):
        if h in confusion:
            confusion[h][p] += 1

    for label in valid_labels:
        total = sum(confusion[label].values())
        correct = confusion[label][label]
        per_class[label] = correct / total if total > 0 else 0

    # Errors
    errors = []
    for i, (h, p) in enumerate(zip(human_labels, predictions)):
        if h != p:
            errors.append(
                {
                    "text": texts[i][:60],
                    "human": h,
                    "pred": p,
                    "scores": results[i][1],
                }
            )

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "time_s": elapsed,
        "errors": errors[:10],
    }


def print_results(name: str, res: dict):
    """Print results."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {res['accuracy']:.1%}")
    print("\nPer-class:")
    for label, acc in sorted(res["per_class"].items()):
        total = sum(res["confusion"].get(label, {}).values())
        print(f"  {label:<15} {acc:.1%} ({total} samples)")

    print("\nConfusion matrix:")
    labels = sorted(res["per_class"].keys())
    print(" " * 16 + "".join(f"{l[:8]:>10}" for l in labels))
    for h in labels:
        row = f"{h:<16}"
        for p in labels:
            count = res["confusion"].get(h, {}).get(p, 0)
            row += f"{count:>10}"
        print(row)

    if res["errors"]:
        print("\nErrors:")
        for e in res["errors"][:5]:
            print(f'  "{e["text"]}..."')
            print(f"    {e['human']} -> {e['pred']}")


def main():
    print("=" * 60)
    print("RESPONSE CLASSIFICATION - COMPARING CATEGORY SCHEMES")
    print("=" * 60)

    # Load response data
    response_data = load_labeled_data(Path("data/response_labeling.jsonl"))
    response_sample = sample_balanced(response_data, 30)  # 30 per class = 180 total
    print(f"\nSampled {len(response_sample)} responses")
    print(f"Distribution: {Counter(d['label'] for d in response_sample)}")

    # Test different category schemes
    results = {}

    # V2: 4 categories (yes/no/maybe/other)
    print("\n" + "-" * 60)
    print("Testing V2: yes / no / maybe / other")
    clf_v2 = EmbeddingClassifier(RESPONSE_CATEGORIES_V2, "v2")
    res_v2 = evaluate(clf_v2, response_sample, RESPONSE_MAP_V2)
    print_results("V2: yes/no/maybe/other", res_v2)
    results["v2_4cat"] = res_v2

    # V3: 3 categories (answered_yes/answered_no/no_answer)
    print("\n" + "-" * 60)
    print("Testing V3: answered_yes / answered_no / no_answer")
    clf_v3 = EmbeddingClassifier(RESPONSE_CATEGORIES_V3, "v3")
    res_v3 = evaluate(clf_v3, response_sample, RESPONSE_MAP_V3)
    print_results("V3: answered_yes/answered_no/no_answer", res_v3)
    results["v3_3cat"] = res_v3

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scheme':<35} {'Accuracy':>10}")
    print("-" * 45)
    print(f"{'Previous (positive/negative/neutral)':<35} {'62.5%':>10}")
    print(f"{'V2 (yes/no/maybe/other)':<35} {res_v2['accuracy']:>10.1%}")
    print(f"{'V3 (answered_yes/no/no_answer)':<35} {res_v3['accuracy']:>10.1%}")

    # Save
    output = Path("results/response_category_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
