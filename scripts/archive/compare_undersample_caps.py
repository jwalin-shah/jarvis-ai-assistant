#!/usr/bin/env python3
"""Compare classifier performance across different undersample caps."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from jarvis.embedding_adapter import get_embedder


def load_data(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row.get("text", ""))
            labels.append(row.get("label", ""))
    return texts, labels


def train_and_eval(texts: list[str], labels: list[str], embedder, seed: int = 42) -> dict:
    embeddings = embedder.encode(texts, normalize=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.arange(len(labels)), labels))

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
              probability=True, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def main():
    caps = [128, 200, 300]
    files = {
        128: Path("data/trigger_7label_cap128.jsonl"),
        200: Path("data/trigger_7label_cap200.jsonl"),
        300: Path("data/trigger_7label_cap300.jsonl"),
    }

    print("Loading embedder...")
    embedder = get_embedder()

    results = {}
    for cap in caps:
        print(f"\n=== Training with cap={cap} ===")
        texts, labels = load_data(files[cap])
        print(f"Samples: {len(texts)}")
        print(f"Distribution: {Counter(labels)}")

        result = train_and_eval(texts, labels, embedder)
        results[cap] = result

        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Macro F1: {result['macro_f1']:.3f}")
        print(f"Weighted F1: {result['weighted_f1']:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Cap':<10} {'Samples':<10} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 60)
    for cap in caps:
        r = results[cap]
        print(f"{cap:<10} {r['train_size'] + r['test_size']:<10} {r['accuracy']:<12.3f} {r['macro_f1']:<12.3f} {r['weighted_f1']:<12.3f}")


if __name__ == "__main__":
    main()
