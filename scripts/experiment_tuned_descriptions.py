#!/usr/bin/env python3
"""Test tuned category descriptions across all models + RAM usage.

Usage:
    uv run python -m scripts.experiment_tuned_descriptions
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import psutil

# =============================================================================
# TUNED Category Descriptions
# =============================================================================

# TRIGGER: More specific, using actual message patterns
TRIGGER_TUNED = {
    "needs_action": [
        # Direct questions
        "Are you free?",
        "What time?",
        "Can you help me?",
        "Do you want to?",
        "Where are you?",
        # Requests
        "Can you send me",
        "Pick me up",
        "Let me know",
        "Call me",
        # Invitations
        "Want to hang out?",
        "You down?",
        "Wanna come?",
    ],
    "casual": [
        # Reactions
        "Haha that's funny",
        "lol",
        "That's crazy",
        "Nice",
        "omg",
        # Statements
        "I'm at home",
        "Just got here",
        "On my way",
        "I'll be there soon",
        # Social
        "Hey!",
        "What's up",
        "Good morning",
    ],
}

# RESPONSE: Tuned based on actual data patterns
RESPONSE_TUNED_V1 = {
    "answered_yes": [
        # Direct yes
        "Yes",
        "Yeah",
        "Yep",
        "Yee",
        "Ya",
        # Agreement
        "Ok",
        "Okay",
        "Sure",
        "Sounds good",
        "I'm down",
        "Bet",
        "Fs",  # "for sure"
        "Down",
    ],
    "answered_no": [
        # Direct no
        "No",
        "Nah",
        "Nope",
        # Decline phrases
        "I can't",
        "I won't",
        "Not gonna",
        "I'm not going",
        "Can't do it",
        "Not today",
        "I'm good",  # often means "no thanks"
    ],
    "no_answer": [
        # Uncertain/defer
        "Maybe",
        "We'll see",
        "Let me check",
        "I'll let you know",
        "Not sure yet",
        "Depends",
        "idk",
        # Questions
        "What time?",
        "Where?",
        "How much?",
        # Reactions
        "Haha",
        "lol",
        "Nice",
        "That's cool",
        # Other info
        "I'm home",
        "Just got here",
    ],
}

# Alternative: More descriptive (semantic) style
RESPONSE_TUNED_V2 = {
    "answered_yes": [
        "I agree and will do it",
        "Yes I'm in",
        "Confirming acceptance",
        "Positive response agreeing to participate",
        "Saying yes to the request",
        "ok sure sounds good yes",
    ],
    "answered_no": [
        "I decline and won't do it",
        "No I'm not going to",
        "Refusing the request",
        "Negative response declining to participate",
        "Saying no to the request",
        "nah no can't won't not gonna",
    ],
    "no_answer": [
        "Not giving a yes or no answer",
        "Uncertain or asking for more info",
        "Just a reaction or comment",
        "Sharing information without answering",
        "maybe we'll see let me check idk",
        "haha lol nice cool what time where",
    ],
}

# Label mappings
TRIGGER_MAP = {
    "commitment": "needs_action",
    "question": "needs_action",
    "reaction": "casual",
    "social": "casual",
    "statement": "casual",
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
# MLX Client
# =============================================================================


class MLXClient:
    def __init__(self, socket_path: str = "/tmp/jarvis-embed.sock"):
        self.socket_path = socket_path

    def _request(self, method: str, params: dict = None, timeout: float = 60) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self.socket_path)
            request = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
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

    def encode(self, texts: list[str], model: str = None) -> np.ndarray:
        params = {"texts": texts, "normalize": True}
        if model:
            params["model"] = model
        result = self._request("embed", params)
        return np.array(result["embeddings"])

    def health(self) -> dict:
        return self._request("health")

    def unload(self) -> dict:
        return self._request("unload")


# =============================================================================
# Classifier
# =============================================================================


class ZeroShotClassifier:
    def __init__(self, client: MLXClient, categories: dict[str, list[str]], model: str):
        self.client = client
        self.model = model
        self.prototypes = {}
        for label, descs in categories.items():
            embs = client.encode(descs, model=model)
            proto = np.mean(embs, axis=0)
            proto = proto / np.linalg.norm(proto)
            self.prototypes[label] = proto

    def classify(self, texts: list[str]) -> list[str]:
        embs = self.client.encode(texts, model=self.model)
        preds = []
        for emb in embs:
            scores = {l: float(np.dot(emb, p)) for l, p in self.prototypes.items()}
            preds.append(max(scores, key=lambda k: scores[k]))
        return preds


# =============================================================================
# Data & Evaluation
# =============================================================================


def load_data(path: Path) -> list[dict]:
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


def evaluate(clf: ZeroShotClassifier, data: list[dict], label_map: dict) -> float:
    texts = [d["text"] for d in data]
    human = [label_map.get(d["label"], "unknown") for d in data]
    preds = clf.classify(texts)
    correct = sum(1 for h, p in zip(human, preds) if h == p)
    return correct / len(human)


def get_ram_mb() -> float:
    """Get current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_data_quality(data: list[dict], label_map: dict, task: str):
    """Check for potential mislabels in test data."""
    print(f"\n=== {task.upper()} DATA QUALITY CHECK ===")

    # Look for suspicious patterns
    suspicious = []

    for d in data:
        text = d["text"].lower()
        label = d["label"]
        mapped = label_map.get(label, "unknown")

        # Check for mismatches
        if task == "response":
            # "yes" words in non-yes labels
            if mapped != "answered_yes" and any(
                w in text for w in ["ok", "okay", "sure", "yes", "yep", "down", "bet"]
            ):
                if not any(w in text for w in ["?", "not", "don't", "idk"]):
                    suspicious.append((text[:50], label, mapped, "has yes-words"))

            # "no" words in non-no labels
            if mapped != "answered_no" and any(
                w in text for w in ["no", "nah", "can't", "won't", "not gonna"]
            ):
                if "know" not in text:  # exclude "idk", "don't know"
                    suspicious.append((text[:50], label, mapped, "has no-words"))

    if suspicious:
        print(f"Found {len(suspicious)} potentially mislabeled examples:")
        for text, orig, mapped, reason in suspicious[:10]:
            print(f"  '{text}' | {orig}->{mapped} | {reason}")
    else:
        print("No obvious mislabels detected")

    return len(suspicious)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("TUNED DESCRIPTIONS + RAM USAGE TEST")
    print("=" * 70)

    models = ["bge-small", "gte-tiny", "minilm-l6", "bge-micro"]
    client = MLXClient()

    # Load data
    trigger_data = load_data(Path("data/trigger_labeling.jsonl"))
    trigger_sample = sample_balanced(trigger_data, 25)  # 125 total

    response_data = load_data(Path("data/response_labeling.jsonl"))
    response_sample = sample_balanced(response_data, 30)  # 180 total

    print(f"Trigger: {len(trigger_sample)} samples")
    print(f"Response: {len(response_sample)} samples")

    # Check data quality
    check_data_quality(response_sample, RESPONSE_MAP, "response")

    # Unload to get baseline RAM
    client.unload()
    time.sleep(1)
    baseline_ram = get_ram_mb()
    print(f"\nBaseline RAM (no model): {baseline_ram:.0f} MB")

    results = {}

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        # Measure RAM after loading
        _ = client.encode(["test"], model=model)  # Force load
        time.sleep(0.5)
        model_ram = get_ram_mb()
        ram_delta = model_ram - baseline_ram
        print(f"RAM usage: {model_ram:.0f} MB (+{ram_delta:.0f} MB for model)")

        # Test original descriptions
        trigger_clf_orig = ZeroShotClassifier(
            client,
            {
                "needs_action": [
                    "This message is asking for something or needs a response",
                    "A request, question, or invitation that requires action",
                    "Can you help me with something",
                ],
                "casual": [
                    "This message is just casual chat, no response needed",
                    "A statement, reaction, or small talk",
                    "Just letting you know, that's cool",
                ],
            },
            model,
        )
        trigger_acc_orig = evaluate(trigger_clf_orig, trigger_sample, TRIGGER_MAP)

        response_clf_orig = ZeroShotClassifier(
            client,
            {
                "answered_yes": ["They said yes or agreed", "Sure okay yes yep"],
                "answered_no": ["They said no or declined", "No nah can't won't"],
                "no_answer": ["Not a yes or no", "Maybe, question, reaction"],
            },
            model,
        )
        response_acc_orig = evaluate(response_clf_orig, response_sample, RESPONSE_MAP)

        # Test tuned descriptions
        trigger_clf_tuned = ZeroShotClassifier(client, TRIGGER_TUNED, model)
        trigger_acc_tuned = evaluate(trigger_clf_tuned, trigger_sample, TRIGGER_MAP)

        response_clf_v1 = ZeroShotClassifier(client, RESPONSE_TUNED_V1, model)
        response_acc_v1 = evaluate(response_clf_v1, response_sample, RESPONSE_MAP)

        response_clf_v2 = ZeroShotClassifier(client, RESPONSE_TUNED_V2, model)
        response_acc_v2 = evaluate(response_clf_v2, response_sample, RESPONSE_MAP)

        print("\nTrigger:")
        print(f"  Original:  {trigger_acc_orig:.1%}")
        print(
            f"  Tuned:     {trigger_acc_tuned:.1%} ({'+' if trigger_acc_tuned > trigger_acc_orig else ''}{(trigger_acc_tuned - trigger_acc_orig) * 100:.1f}%)"
        )

        print("\nResponse:")
        print(f"  Original:  {response_acc_orig:.1%}")
        print(f"  Tuned V1:  {response_acc_v1:.1%} (example-based)")
        print(f"  Tuned V2:  {response_acc_v2:.1%} (semantic)")

        results[model] = {
            "ram_mb": model_ram,
            "ram_delta_mb": ram_delta,
            "trigger_orig": trigger_acc_orig,
            "trigger_tuned": trigger_acc_tuned,
            "response_orig": response_acc_orig,
            "response_v1": response_acc_v1,
            "response_v2": response_acc_v2,
        }

        # Unload before next model
        client.unload()
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<12} {'RAM':>8} {'Trig Orig':>10} {'Trig Tune':>10} {'Resp Orig':>10} {'Resp V1':>10} {'Resp V2':>10}"
    )
    print("-" * 75)
    for model, res in results.items():
        print(
            f"{model:<12} {res['ram_delta_mb']:>6.0f}MB {res['trigger_orig']:>10.1%} {res['trigger_tuned']:>10.1%} "
            f"{res['response_orig']:>10.1%} {res['response_v1']:>10.1%} {res['response_v2']:>10.1%}"
        )

    # Best configs
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    best_trigger = max(results.items(), key=lambda x: x[1]["trigger_tuned"])
    best_response = max(
        results.items(), key=lambda x: max(x[1]["response_v1"], x[1]["response_v2"])
    )
    smallest_ram = min(results.items(), key=lambda x: x[1]["ram_delta_mb"])

    print(f"Best trigger:  {best_trigger[0]} @ {best_trigger[1]['trigger_tuned']:.1%}")
    print(
        f"Best response: {best_response[0]} @ {max(best_response[1]['response_v1'], best_response[1]['response_v2']):.1%}"
    )
    print(f"Smallest RAM:  {smallest_ram[0]} @ {smallest_ram[1]['ram_delta_mb']:.0f}MB")

    # Save
    output = Path("results/tuned_descriptions_comparison.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
