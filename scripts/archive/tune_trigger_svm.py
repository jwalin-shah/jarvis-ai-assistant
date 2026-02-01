#!/usr/bin/env python3
"""Tune SVM hyperparameters for trigger classification.

Usage:
    uv run python -m scripts.tune_trigger_svm \
        --input ~/.jarvis/gold_trigger_labels_500.jsonl \
        --output results/trigger_svm_tuning.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from jarvis.embedding_adapter import get_embedder


DEFAULT_MERGE_MAP = {
    "info_question": "question",
    "yn_question": "question",
    "invitation": "action",
    "request": "action",
    "good_news": "emotional",
    "bad_news": "emotional",
    "reaction": "emotional",
    "ack": "acknowledgment",
    "greeting": "acknowledgment",
    "statement": "statement",
}


def _load_merge_map(config_path: Path | None) -> dict[str, str]:
    if config_path is None or not config_path.exists():
        return DEFAULT_MERGE_MAP
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return payload.get("merge_map", DEFAULT_MERGE_MAP)
    except Exception:
        return DEFAULT_MERGE_MAP


def _load_rows(input_path: Path, merge_map: dict[str, str]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = (row.get("label") or "").strip().lower()
            text = (row.get("trigger_text") or "").strip()
            if not label or not text:
                continue
            merged = merge_map.get(label)
            if not merged:
                continue
            texts.append(text)
            labels.append(merged)
    return texts, labels


def _embed(texts: list[str]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts, normalize=True)


def _tune(
    embeddings: np.ndarray,
    labels: list[str],
    c_values: list[float],
    gamma_values: list[float | str],
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    indices = list(range(len(labels)))
    train_idx, test_idx = next(splitter.split(indices, labels))

    x_train = embeddings[train_idx]
    x_test = embeddings[test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    results = []
    for c in c_values:
        for gamma in gamma_values:
            clf = SVC(
                kernel="rbf",
                C=c,
                gamma=gamma,
                class_weight="balanced",
                probability=True,
                random_state=seed,
            )
            clf.fit(x_train, y_train)
            preds = clf.predict(x_test)
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            results.append(
                {
                    "C": c,
                    "gamma": gamma,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                }
            )

    results.sort(key=lambda item: (item["accuracy"], item["weighted_f1"]), reverse=True)
    return {
        "counts": Counter(labels),
        "best": results[0] if results else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune SVM hyperparameters for triggers")
    parser.add_argument("--input", type=Path, required=True, help="Gold JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".jarvis" / "trigger_classifier_model" / "config.json",
        help="Path to config with merge map",
    )
    parser.add_argument(
        "--c",
        type=str,
        default="0.5,1,2,5,10",
        help="Comma-separated C values",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale,auto,0.1,0.5",
        help="Comma-separated gamma values",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    c_values = [float(v.strip()) for v in args.c.split(",") if v.strip()]
    gamma_values: list[float | str] = []
    for value in args.gamma.split(","):
        value = value.strip()
        if not value:
            continue
        if value in {"scale", "auto"}:
            gamma_values.append(value)
        else:
            gamma_values.append(float(value))

    merge_map = _load_merge_map(args.config)
    texts, labels = _load_rows(args.input, merge_map)
    embeddings = _embed(texts)
    report = _tune(embeddings, labels, c_values, gamma_values, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = report.get("best")
    if best:
        print(
            f"Best: C={best['C']} gamma={best['gamma']} acc={best['accuracy']:.3f} "
            f"macro_f1={best['macro_f1']:.3f} weighted_f1={best['weighted_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
