#!/usr/bin/env python3
"""Test all 4 encoder models for zero-shot classification.

Models:
- bge-small (384 dim, ~100-150ms, MTEB ~62)
- gte-tiny (384 dim, ~50-70ms, MTEB ~57)
- minilm-l6 (384 dim, ~50-70ms, MTEB ~56)
- bge-micro (384 dim, ~30-40ms, MTEB ~54)

Usage:
    uv run python -m scripts.experiment_all_encoders
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import numpy as np

# =============================================================================
# Categories (best performing from v2 experiment)
# =============================================================================

TRIGGER_CATEGORIES = {
    "needs_action": [
        "This message is asking for something or needs a response",
        "A request, question, or invitation that requires action",
        "Someone is asking me to do something or answer a question",
        "Can you help me with something",
        "Are you available",
        "Would you like to",
    ],
    "casual": [
        "This message is just casual chat, no response needed",
        "A statement, reaction, or small talk",
        "Someone sharing information or reacting",
        "Just letting you know",
        "That's cool, haha, nice",
    ],
}

TRIGGER_MAP = {
    "commitment": "needs_action",
    "question": "needs_action",
    "reaction": "casual",
    "social": "casual",
    "statement": "casual",
}

# V3 categories - best performing
RESPONSE_CATEGORIES = {
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
    ],
    "no_answer": [
        "They didn't give a yes or no answer",
        "Uncertain, maybe, we'll see, let me check",
        "A question, reaction, or just info",
        "Haha, cool, what time, I'm home",
    ],
}

RESPONSE_MAP = {
    "agree": "answered_yes",
    "decline": "answered_no",
    "defer": "no_answer",
    "other": "no_answer",
    "question": "no_answer",
    "reaction": "no_answer",
}


# =============================================================================
# MLX Service Client (direct socket, supports model switching)
# =============================================================================


class MLXClient:
    """Direct client to MLX embedding service."""

    def __init__(self, socket_path: str = "/tmp/jarvis-embed.sock"):
        self.socket_path = socket_path
        self.current_model = None

    def _request(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        """Send JSON-RPC request."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self.socket_path)
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": 1,
            }
            sock.sendall(json.dumps(request).encode() + b"\n")

            response = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                response += chunk
                if b"\n" in chunk:
                    break

            result = json.loads(response.decode())
            if "error" in result:
                raise RuntimeError(result["error"])
            return result.get("result", {})
        finally:
            sock.close()

    def encode(self, texts: list[str], normalize: bool = True, model: str = None) -> np.ndarray:
        """Encode texts to embeddings, optionally specifying model."""
        params = {"texts": texts, "normalize": normalize}
        if model:
            params["model"] = model
        result = self._request("embed", params, timeout=60)
        self.current_model = result.get("model")
        return np.array(result["embeddings"])

    def health(self) -> dict:
        """Check service health."""
        return self._request("health")


# =============================================================================
# Classifier
# =============================================================================


class ZeroShotClassifier:
    """Zero-shot classifier using embedding prototypes."""

    def __init__(self, client: MLXClient, categories: dict[str, list[str]], model: str = None):
        self.client = client
        self.categories = categories
        self.model = model
        self.prototypes: dict[str, np.ndarray] = {}
        self._compute_prototypes()

    def _compute_prototypes(self):
        """Compute prototype for each category."""
        for label, descriptions in self.categories.items():
            embeddings = self.client.encode(descriptions, normalize=True, model=self.model)
            prototype = np.mean(embeddings, axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            self.prototypes[label] = prototype

    def classify(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """Classify texts."""
        embeddings = self.client.encode(texts, normalize=True, model=self.model)
        results = []
        for emb in embeddings:
            scores = {l: float(np.dot(emb, p)) for l, p in self.prototypes.items()}
            pred = max(scores, key=lambda k: scores[k])
            results.append((pred, scores))
        return results


def load_data(path: Path) -> list[dict]:
    """Load JSONL data."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = row.get("text") or row.get("response", "")
                label = row.get("label", "").lower()
                if text and label:
                    data.append({"text": text, "label": label})
    return data


def sample_balanced(data: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Sample balanced across classes."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for d in data:
        by_label.setdefault(d["label"], []).append(d)

    sampled = []
    for items in by_label.values():
        n = min(n_per_class, len(items))
        idx = rng.choice(len(items), size=n, replace=False)
        sampled.extend([items[i] for i in idx])
    rng.shuffle(sampled)
    return sampled


def evaluate(clf: ZeroShotClassifier, data: list[dict], label_map: dict) -> dict:
    """Evaluate classifier."""
    texts = [d["text"] for d in data]
    human = [label_map.get(d["label"], "unknown") for d in data]

    start = time.time()
    results = clf.classify(texts)
    elapsed = time.time() - start

    preds = [r[0] for r in results]
    correct = sum(1 for h, p in zip(human, preds) if h == p)

    return {
        "accuracy": correct / len(human),
        "time_s": elapsed,
        "n_samples": len(data),
    }


def main():
    print("=" * 70)
    print("COMPARING ALL 4 ENCODER MODELS")
    print("=" * 70)

    models = ["bge-small", "gte-tiny", "minilm-l6", "bge-micro"]
    client = MLXClient()

    # Check service
    health = client.health()
    print(f"Service status: {health['status']}, current model: {health['model_name']}")

    # Load data
    trigger_data = load_data(Path("data/trigger_labeling.jsonl"))
    trigger_sample = sample_balanced(trigger_data, 20)  # 100 total

    response_data = load_data(Path("data/response_labeling.jsonl"))
    response_sample = sample_balanced(response_data, 25)  # 150 total

    print(f"\nTrigger: {len(trigger_sample)} samples")
    print(f"Response: {len(response_sample)} samples")

    results = {}

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        # Create classifiers with this model (will auto-load on first encode)
        print(f"Loading {model} and computing prototypes...")
        load_start = time.time()
        trigger_clf = ZeroShotClassifier(client, TRIGGER_CATEGORIES, model=model)
        response_clf = ZeroShotClassifier(client, RESPONSE_CATEGORIES, model=model)
        print(f"  Ready in {time.time() - load_start:.1f}s")

        # Evaluate trigger
        trigger_res = evaluate(trigger_clf, trigger_sample, TRIGGER_MAP)
        print("\nTrigger (needs_action vs casual):")
        print(f"  Accuracy: {trigger_res['accuracy']:.1%}")
        trigger_ms = trigger_res["time_s"] * 1000 / trigger_res["n_samples"]
        print(f"  Time: {trigger_res['time_s']:.2f}s ({trigger_ms:.1f}ms/msg)")

        # Evaluate response
        response_res = evaluate(response_clf, response_sample, RESPONSE_MAP)
        print("\nResponse (yes/no/no_answer):")
        print(f"  Accuracy: {response_res['accuracy']:.1%}")
        response_ms = response_res["time_s"] * 1000 / response_res["n_samples"]
        print(f"  Time: {response_res['time_s']:.2f}s ({response_ms:.1f}ms/msg)")

        results[model] = {
            "trigger_accuracy": trigger_res["accuracy"],
            "trigger_time_ms": trigger_res["time_s"] * 1000 / trigger_res["n_samples"],
            "response_accuracy": response_res["accuracy"],
            "response_time_ms": response_res["time_s"] * 1000 / response_res["n_samples"],
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Trigger':>12} {'Response':>12} {'Avg Speed':>12}")
    print("-" * 55)
    for model, res in results.items():
        avg_time = (res["trigger_time_ms"] + res["response_time_ms"]) / 2
        t_acc = res["trigger_accuracy"]
        r_acc = res["response_accuracy"]
        print(f"{model:<15} {t_acc:>11.1%} {r_acc:>11.1%} {avg_time:>10.1f}ms")

    # Save results
    output = Path("results/encoder_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_trigger = max(results.items(), key=lambda x: x[1]["trigger_accuracy"])
    best_response = max(results.items(), key=lambda x: x[1]["response_accuracy"])
    fastest = min(results.items(), key=lambda x: x[1]["trigger_time_ms"])

    print(f"Best trigger accuracy:  {best_trigger[0]} ({best_trigger[1]['trigger_accuracy']:.1%})")
    resp_acc = best_response[1]["response_accuracy"]
    print(f"Best response accuracy: {best_response[0]} ({resp_acc:.1%})")
    print(f"Fastest:                {fastest[0]} ({fastest[1]['trigger_time_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
