#!/usr/bin/env python3
"""Train a lightweight message-level keep/discard classifier.

This script consumes JSONL rows from merge_goldsets.py:
- train_message_gate.jsonl
- dev_message_gate.jsonl

Schema (required):
    {
      "text": "...",
      "label": 0 or 1,
      "is_from_me": bool,          # optional, defaults False
      "bucket": "random|likely|negative"  # optional
    }

Model:
- TF-IDF text features (word ngrams)
- lightweight numeric features
- Logistic Regression or LinearSVC
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.utils.logging import setup_script_logging

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """In-memory dataset for message gate training."""

    texts: list[str]
    labels: list[int]
    is_from_me: list[bool]
    buckets: list[str]


class MessageGateFeatures:
    """Extract lightweight numeric features from messages."""

    PREF_WORDS = {
        "love",
        "like",
        "hate",
        "prefer",
        "obsessed",
        "favorite",
        "enjoy",
        "allergic",
    }
    LOCATION_WORDS = {
        "live",
        "living",
        "moving",
        "moved",
        "from",
        "to",
        "based",
        "relocating",
    }
    REL_WORDS = {
        "my",
        "mom",
        "dad",
        "sister",
        "brother",
        "wife",
        "husband",
        "girlfriend",
        "boyfriend",
        "partner",
        "friend",
    }
    HEALTH_WORDS = {
        "pain",
        "hospital",
        "injury",
        "allergic",
        "anxious",
        "depressed",
        "headache",
    }
    BOT_PATTERNS = {
        "cvs pharmacy",
        "prescription is ready",
        "unsubscribe",
        "check out this job",
        "apply now",
    }

    def __init__(self) -> None:
        self._feature_names = [
            "char_len",
            "word_len",
            "upper_ratio",
            "digit_ratio",
            "has_question",
            "has_exclaim",
            "first_person",
            "pref_marker",
            "location_marker",
            "relationship_marker",
            "health_marker",
            "likely_bot",
            "is_short_msg",
            "is_from_me",
            "bucket_random",
            "bucket_likely",
            "bucket_negative",
            "bucket_other",
        ]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def transform(self, texts: list[str], is_from_me: list[bool], buckets: list[str]) -> np.ndarray:
        """Convert rows to numeric feature matrix."""
        out: list[list[float]] = []
        total = len(texts)
        for i, (text, from_me, bucket) in enumerate(zip(texts, is_from_me, buckets)):
            if (i + 1) % 1000 == 0 or i + 1 == total:
                print(f"Extracting numeric features {i + 1}/{total}...", flush=True)
            t = (text or "").strip()
            lower = t.lower()
            words = lower.split()

            char_len = len(t)
            word_len = len(words)
            upper_count = sum(1 for c in t if c.isupper())
            digit_count = sum(1 for c in t if c.isdigit())

            row = [
                float(char_len),
                float(word_len),
                (upper_count / char_len) if char_len else 0.0,
                (digit_count / char_len) if char_len else 0.0,
                1.0 if "?" in t else 0.0,
                1.0 if "!" in t else 0.0,
                1.0 if any(w in {"i", "i'm", "my", "me"} for w in words[:5]) else 0.0,
                1.0 if any(w in lower for w in self.PREF_WORDS) else 0.0,
                1.0 if any(w in lower for w in self.LOCATION_WORDS) else 0.0,
                1.0 if any(w in lower for w in self.REL_WORDS) else 0.0,
                1.0 if any(w in lower for w in self.HEALTH_WORDS) else 0.0,
                1.0 if any(p in lower for p in self.BOT_PATTERNS) else 0.0,
                1.0 if word_len <= 3 else 0.0,
                1.0 if from_me else 0.0,
                1.0 if bucket == "random" else 0.0,
                1.0 if bucket == "likely" else 0.0,
                1.0 if bucket == "negative" else 0.0,
                1.0 if bucket not in {"random", "likely", "negative"} else 0.0,
            ]
            out.append(row)

        return np.asarray(out, dtype=np.float32)


@dataclass
class MessageGateModel:
    """Bundle for trained gate model."""

    model_type: str
    model: Any
    vectorizer: Any
    scaler: Any
    num_features: MessageGateFeatures
    threshold: float

    def _build_matrix(self, data: Dataset) -> Any:
        from scipy.sparse import csr_matrix, hstack

        x_text = self.vectorizer.transform(data.texts)
        x_num_arr = self.num_features.transform(data.texts, data.is_from_me, data.buckets)
        x_num = csr_matrix(self.scaler.transform(x_num_arr))
        return hstack([x_text, x_num], format="csr")

    def predict_scores(self, data: Dataset) -> np.ndarray:
        x = self._build_matrix(data)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        decision = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))

    def predict(self, data: Dataset, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        scores = self.predict_scores(data)
        return (scores >= thr).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model_type": self.model_type,
                    "model": self.model,
                    "vectorizer": self.vectorizer,
                    "scaler": self.scaler,
                    "threshold": self.threshold,
                    "feature_names": self.num_features.feature_names,
                },
                f,
            )


def load_dataset(path: Path) -> Dataset:
    """Load JSONL dataset with message-gate schema."""
    texts: list[str] = []
    labels: list[int] = []
    is_from_me: list[bool] = []
    buckets: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            text = str(row.get("text", ""))
            if not text:
                continue

            if "label" in row:
                label_raw = row["label"]
            elif "gold_keep" in row:
                label_raw = row["gold_keep"]
            else:
                raise ValueError(f"Missing label in {path}:{idx}")

            label = int(label_raw)
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label} in {path}:{idx}")

            texts.append(text)
            labels.append(label)
            is_from_me.append(bool(row.get("is_from_me", False)))
            buckets.append(str(row.get("bucket", "other")))

    if not texts:
        raise ValueError(f"No usable rows loaded from {path}")

    return Dataset(texts=texts, labels=labels, is_from_me=is_from_me, buckets=buckets)


def fit_model(
    train: Dataset,
    model_type: str,
    max_features: int,
    min_df: int,
    c_value: float,
    threshold: float,
) -> MessageGateModel:
    """Train message gate model."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    x_text = vectorizer.fit_transform(train.texts)

    num_features = MessageGateFeatures()
    x_num_arr = num_features.transform(train.texts, train.is_from_me, train.buckets)

    scaler = StandardScaler(with_mean=False)
    x_num = csr_matrix(scaler.fit_transform(x_num_arr))

    x = hstack([x_text, x_num], format="csr")
    y = np.asarray(train.labels, dtype=np.int32)

    if model_type == "logistic":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=c_value,
        )
    else:
        model = LinearSVC(
            class_weight="balanced",
            max_iter=3000,
            C=c_value,
        )

    model.fit(x, y)

    return MessageGateModel(
        model_type=model_type,
        model=model,
        vectorizer=vectorizer,
        scaler=scaler,
        num_features=num_features,
        threshold=threshold,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC AUC requires both classes present.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def evaluate(model: MessageGateModel, data: Dataset, split_name: str) -> dict[str, float]:
    """Evaluate trained model on a dataset split."""
    y_true = np.asarray(data.labels, dtype=np.int32)
    y_score = model.predict_scores(data)
    y_pred = model.predict(data)

    metrics = compute_metrics(y_true, y_pred, y_score)

    logger.info("%s metrics:", split_name)
    for key, value in metrics.items():
        logger.info("  %s: %.4f", key, value)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    logger.info("  support: pos=%d, neg=%d", pos, neg)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train message-level keep/discard classifier")
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to train_message_gate.jsonl",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        help="Path to dev_message_gate.jsonl",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "svm"],
        default="logistic",
        help="Model family",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Max TF-IDF features",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Regularization strength inverse (C)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on positive score",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/message_gate.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON path for metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_script_logging("train_message_gate")

    logger.info("Loading train data from %s", args.train)
    train = load_dataset(args.train)

    logger.info(
        "Training %s model on %d rows (pos=%d neg=%d)",
        args.model_type,
        len(train.labels),
        sum(train.labels),
        len(train.labels) - sum(train.labels),
    )

    model = fit_model(
        train=train,
        model_type=args.model_type,
        max_features=args.max_features,
        min_df=args.min_df,
        c_value=args.c_value,
        threshold=args.threshold,
    )

    train_metrics = evaluate(model, train, "Train")

    dev_metrics: dict[str, float] | None = None
    if args.dev:
        logger.info("Loading dev data from %s", args.dev)
        dev = load_dataset(args.dev)
        dev_metrics = evaluate(model, dev, "Dev")

    model.save(args.output)
    logger.info("Saved model to %s", args.output)

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "train": train_metrics,
            "dev": dev_metrics,
            "model_type": args.model_type,
            "train_size": len(train.labels),
            "dev_size": len(dev.labels) if args.dev else None,
            "threshold": args.threshold,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "c_value": args.c_value,
            "train_positive_rate": float(sum(train.labels) / len(train.labels)),
        }
        with args.metrics_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        logger.info("Saved metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
