#!/usr/bin/env python3
"""Comprehensive classifier training across all embedding models.

Trains trigger and response classifiers for all embedding models using:
- Multiple classifier types (SVM, LogisticRegression, RandomForest, XGBoost)
- Bayesian hyperparameter optimization (Optuna)
- Cross-validation for robust evaluation

Usage:
    uv run python -m scripts.train_all_classifiers
    uv run python -m scripts.train_all_classifiers --save-best
    uv run python -m scripts.train_all_classifiers --models bge-small gte-tiny
    uv run python -m scripts.train_all_classifiers --classifiers trigger response
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Suppress optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

from jarvis.config import (
    get_config,
    reset_config,
    save_config,
)
from jarvis.embedding_adapter import (
    EMBEDDING_MODEL_REGISTRY,
    reset_embedder,
)
from jarvis.text_normalizer import normalize_text

logger = logging.getLogger(__name__)

# Try importing XGBoost (optional - needs libomp on macOS)
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # XGBoostError, ImportError, OSError, etc.
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore

# Try importing LightGBM (optional - needs libomp on macOS)
try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:  # ImportError, OSError, etc.
    HAS_LIGHTGBM = False
    LGBMClassifier = None  # type: ignore


@dataclass
class ClassifierResult:
    """Result from training a classifier."""

    embedding_model: str
    classifier_type: str  # "trigger" or "response"
    model_class: str  # "SVM", "LogisticRegression", etc.
    sampling_strategy: str  # "natural", "balanced", "ratio_3x", etc.
    dataset_name: str  # "base", "large", "xl", etc.
    params: dict[str, Any]
    cv_f1_mean: float
    cv_f1_std: float
    test_f1: float
    test_accuracy: float
    per_class: dict[str, dict[str, float]]
    train_size: int
    test_size: int


@dataclass
class TrainingConfig:
    """Configuration for training run."""

    embedding_models: list[str] = field(
        default_factory=lambda: list(EMBEDDING_MODEL_REGISTRY.keys())
    )
    classifier_types: list[str] = field(default_factory=lambda: ["trigger", "response"])
    n_trials: int = 50  # Optuna trials per classifier type
    cv_folds: int = 5
    # Parallelize CV across all cores - embeddings are numpy arrays at this point,
    # no torch/MLX involved in sklearn classifiers
    n_jobs: int = -1
    save_best: bool = False
    output_dir: Path = field(default_factory=lambda: Path("results/classifier_training"))
    fixed_params: bool = False  # Use known-good params instead of Optuna search


# Known-good hyperparameters from existing trained classifiers
FIXED_PARAMS = {
    "SVM": {"C": 10.0, "gamma": "scale"},
    "LogisticRegression": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "LightGBM": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


@dataclass
class DatasetConfig:
    """Configuration for dataset size."""

    name: str
    trigger_files: list[str]
    min_confidence: float  # Default minimum confidence for auto-labeled data
    description: str
    # Per-class confidence thresholds (higher for majority classes)
    # If None, uses min_confidence for all classes
    per_class_confidence: dict[str, float] | None = None


# Per-class confidence thresholds:
# - Majority classes (statement, reaction): require 90% confidence
# - Minority classes (commitment, question, social): require 80% confidence
PER_CLASS_CONFIDENCE_STRICT = {
    "statement": 0.90,
    "reaction": 0.90,
    "commitment": 0.80,
    "question": 0.80,
    "social": 0.80,
}

PER_CLASS_CONFIDENCE_MODERATE = {
    "statement": 0.85,
    "reaction": 0.85,
    "commitment": 0.75,
    "question": 0.75,
    "social": 0.75,
}

# Dataset size configurations for trigger classifier
# Uses high confidence thresholds for auto-labeled data to ensure quality
# Per-class confidence thresholds for RESPONSE:
# - Majority classes (OTHER, REACTION, QUESTION): require 90% confidence
# - Minority classes (AGREE, DECLINE, DEFER): require 80% confidence
RESPONSE_PER_CLASS_CONFIDENCE = {
    "OTHER": 0.90,
    "REACTION": 0.90,
    "QUESTION": 0.90,
    "AGREE": 0.80,
    "DECLINE": 0.80,
    "DEFER": 0.80,
}

# Response dataset configurations
RESPONSE_DATASET_CONFIGS = [
    DatasetConfig(
        name="base",
        trigger_files=["data/response_labeling.jsonl"],  # Reusing field name
        min_confidence=0.0,
        description="Base human-labeled data (~4.8k)",
    ),
    DatasetConfig(
        name="medium",
        trigger_files=[
            "data/response_labeling.jsonl",
            "experiments/data/auto_labeled_90pct.jsonl",
        ],
        min_confidence=0.90,
        description="Human + auto-labeled 90% conf (~10k)",
        per_class_confidence=RESPONSE_PER_CLASS_CONFIDENCE,
    ),
    DatasetConfig(
        name="large",
        trigger_files=[
            "data/response_labeling.jsonl",
            "experiments/data/auto_labeled_90pct.jsonl",
        ],
        min_confidence=0.85,
        description="Human + auto-labeled 85% conf (~20k)",
        per_class_confidence={
            "OTHER": 0.85,
            "REACTION": 0.85,
            "QUESTION": 0.85,
            "AGREE": 0.75,
            "DECLINE": 0.75,
            "DEFER": 0.75,
        },
    ),
    DatasetConfig(
        name="xl",
        trigger_files=[
            "data/response_labeling.jsonl",
            "experiments/data/auto_labeled_90pct.jsonl",
        ],
        min_confidence=0.80,
        description="Human + all auto-labeled (~34k)",
        per_class_confidence={
            "OTHER": 0.80,
            "REACTION": 0.80,
            "QUESTION": 0.80,
            "AGREE": 0.70,
            "DECLINE": 0.70,
            "DEFER": 0.70,
        },
    ),
]

TRIGGER_DATASET_CONFIGS = [
    DatasetConfig(
        name="base",
        trigger_files=["data/trigger_labeling.jsonl"],
        min_confidence=0.0,  # Human-labeled, no filtering
        description="Base human-labeled data (~4.8k)",
    ),
    DatasetConfig(
        name="base_plus",
        trigger_files=[
            "data/trigger_labeling.jsonl",
            "data/trigger_auto_labeled.jsonl",
            "data/trigger_commitment_corrected.jsonl",
        ],
        min_confidence=0.8,  # Fallback, per-class used
        description="Base + high-conf auto-labeled (~5.5k)",
        per_class_confidence=PER_CLASS_CONFIDENCE_STRICT,
    ),
    DatasetConfig(
        name="large_strict",
        trigger_files=[
            "data/trigger_labeling.jsonl",
            "data/trigger_auto_labeled.jsonl",
            "data/trigger_commitment_corrected.jsonl",
            "data/trigger_new_batch_3000.jsonl",
        ],
        min_confidence=0.85,  # Fallback
        description="Large dataset, strict per-class conf (~7k)",
        per_class_confidence=PER_CLASS_CONFIDENCE_STRICT,
    ),
    DatasetConfig(
        name="large",
        trigger_files=[
            "data/trigger_labeling.jsonl",
            "data/trigger_auto_labeled.jsonl",
            "data/trigger_commitment_corrected.jsonl",
            "data/trigger_new_batch_3000.jsonl",
        ],
        min_confidence=0.75,  # Fallback
        description="Large dataset, moderate per-class conf (~8k)",
        per_class_confidence=PER_CLASS_CONFIDENCE_MODERATE,
    ),
    DatasetConfig(
        name="xl",
        trigger_files=[
            "data/trigger_labeling.jsonl",
            "data/trigger_auto_labeled.jsonl",
            "data/trigger_commitment_corrected.jsonl",
            "data/trigger_new_batch_3000.jsonl",
            "data/trigger_needs_review.jsonl",
        ],
        min_confidence=0.75,  # Fallback
        description="XL dataset, moderate per-class conf (~10k+)",
        per_class_confidence=PER_CLASS_CONFIDENCE_MODERATE,
    ),
]


def load_trigger_data_from_files(
    files: list[str],
    apply_normalization: bool = True,
    min_confidence: float = 0.0,
    per_class_confidence: dict[str, float] | None = None,
) -> tuple[list[str], list[str]]:
    """Load labeled trigger data from multiple files.

    Args:
        files: List of JSONL file paths to load.
        apply_normalization: If True, apply text normalization to match inference.
        min_confidence: Default minimum confidence threshold for auto-labeled data.
            Only applies to samples that have a "confidence" field.
            Human-labeled data (no confidence field) is always included.
        per_class_confidence: Optional per-class confidence thresholds.
            Use higher thresholds for majority classes (statement, reaction)
            and lower thresholds for minority classes (commitment, question, social).
    """
    texts = []
    labels = []
    skipped_norm = 0
    skipped_conf = 0
    seen_texts: set[str] = set()  # Deduplicate

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"    Warning: {file_path} not found, skipping", flush=True)
            continue

        file_count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                # Support both "label" and "auto_label" fields
                label = row.get("label") or row.get("auto_label")
                confidence = row.get("confidence")

                if text and label:
                    label_lower = label.lower()

                    # Filter by confidence for auto-labeled data
                    if confidence is not None:
                        # Use per-class threshold if available, otherwise use default
                        if per_class_confidence and label_lower in per_class_confidence:
                            threshold = per_class_confidence[label_lower]
                        else:
                            threshold = min_confidence

                        if confidence < threshold:
                            skipped_conf += 1
                            continue

                    # Apply normalization to match inference behavior
                    if apply_normalization:
                        text = normalize_text(text)
                        if not text:  # Skip reactions/empty after normalization
                            skipped_norm += 1
                            continue

                    # Deduplicate
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)

                    labels.append(label_lower)
                    texts.append(text)
                    file_count += 1

        print(f"    {path.name}: {file_count} samples", flush=True)

    if skipped_norm > 0:
        print(f"    Skipped {skipped_norm} samples (normalization)", flush=True)
    if skipped_conf > 0:
        threshold_info = "per-class" if per_class_confidence else f">= {min_confidence}"
        print(f"    Skipped {skipped_conf} samples (confidence {threshold_info})", flush=True)

    return texts, labels


def load_trigger_data(path: Path, apply_normalization: bool = True) -> tuple[list[str], list[str]]:
    """Load labeled trigger data from a single file.

    Args:
        path: Path to JSONL file with labeled triggers.
        apply_normalization: If True, apply text normalization to match inference.
    """
    return load_trigger_data_from_files([str(path)], apply_normalization)


def load_response_data_from_files(
    files: list[str],
    apply_normalization: bool = True,
    min_confidence: float = 0.0,
    per_class_confidence: dict[str, float] | None = None,
) -> tuple[list[str], list[str]]:
    """Load labeled response data from multiple files.

    Args:
        files: List of JSONL file paths to load.
        apply_normalization: If True, apply text normalization to match inference.
        min_confidence: Default minimum confidence threshold for auto-labeled data.
        per_class_confidence: Optional per-class confidence thresholds.
    """
    texts = []
    labels = []
    skipped_norm = 0
    skipped_conf = 0
    seen_texts: set[str] = set()  # Deduplicate

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"    Warning: {file_path} not found, skipping", flush=True)
            continue

        file_count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Support both "response" field (human-labeled) and "text" field (auto-labeled)
                text = row.get("response", "").strip() or row.get("text", "").strip()
                label = row.get("label")
                confidence = row.get("confidence")

                if text and label:
                    label_upper = label.upper()

                    # Filter by confidence for auto-labeled data
                    if confidence is not None:
                        if per_class_confidence and label_upper in per_class_confidence:
                            threshold = per_class_confidence[label_upper]
                        else:
                            threshold = min_confidence

                        if confidence < threshold:
                            skipped_conf += 1
                            continue

                    # Apply normalization to match inference behavior
                    if apply_normalization:
                        text = normalize_text(text)
                        if not text:
                            skipped_norm += 1
                            continue

                    # Deduplicate
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)

                    labels.append(label_upper)
                    texts.append(text)
                    file_count += 1

        print(f"    {path.name}: {file_count} samples", flush=True)

    if skipped_norm > 0:
        print(f"    Skipped {skipped_norm} samples (normalization)", flush=True)
    if skipped_conf > 0:
        threshold_info = "per-class" if per_class_confidence else f">= {min_confidence}"
        print(f"    Skipped {skipped_conf} samples (confidence {threshold_info})", flush=True)

    return texts, labels


def load_response_data(path: Path, apply_normalization: bool = True) -> tuple[list[str], list[str]]:
    """Load labeled response data from a single file.

    Args:
        path: Path to JSONL file with labeled responses.
        apply_normalization: If True, apply text normalization to match inference.
    """
    return load_response_data_from_files([str(path)], apply_normalization)


@dataclass
class SamplingStrategy:
    """A sampling strategy for training data."""

    name: str
    max_per_class: int | None = None
    target_ratio: float | None = None
    # For "protect" strategies: target total samples, keep minority intact
    protect_target: int | None = None
    # Classes to protect (keep all samples) - defaults set per classifier type
    protect_classes: list[str] | None = None

    def __str__(self) -> str:
        return self.name


# Default sampling strategies to try
DEFAULT_SAMPLING_STRATEGIES = [
    SamplingStrategy("natural", None, None),  # No sampling
    SamplingStrategy("balanced", None, 1.0),  # Equal per class
    SamplingStrategy("ratio_2x", None, 2.0),  # Max 2x minority
    SamplingStrategy("ratio_3x", None, 3.0),  # Max 3x minority
    SamplingStrategy("cap_500", 500, None),  # Cap at 500 per class
    SamplingStrategy("cap_300", 300, None),  # Cap at 300 per class
    # Protect minority classes, only downsample majority (STATEMENT for triggers)
    SamplingStrategy("protect_2500", None, None, protect_target=2500),
    SamplingStrategy("protect_2000", None, None, protect_target=2000),
    SamplingStrategy("protect_1500", None, None, protect_target=1500),
]


def apply_sampling(
    texts: list[str],
    labels: list[str],
    strategy: SamplingStrategy,
    classifier_type: str = "trigger",
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[str]]:
    """Apply a sampling strategy to training data.

    Args:
        texts: Input texts
        labels: Input labels
        strategy: Sampling strategy to apply
        classifier_type: "trigger" or "response" - affects which classes to protect
        rng: Random generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Natural = no sampling
    if (
        strategy.max_per_class is None
        and strategy.target_ratio is None
        and strategy.protect_target is None
    ):
        return texts, labels

    counts = Counter(labels)
    by_label: dict[str, list[str]] = {lbl: [] for lbl in set(labels)}
    for text, label in zip(texts, labels):
        by_label[label].append(text)

    # Handle "protect" strategies - keep minority classes intact, downsample majority
    if strategy.protect_target is not None:
        # Determine which classes to protect based on classifier type
        if strategy.protect_classes is not None:
            minority_classes = set(strategy.protect_classes)
        elif classifier_type == "trigger":
            # For triggers: protect all except STATEMENT (the majority)
            minority_classes = {"commitment", "question", "reaction", "social"}
        else:
            # For response: protect AGREE, DEFER, DECLINE (commitment responses)
            minority_classes = {"AGREE", "DEFER", "DECLINE"}

        # Calculate how much to keep from majority classes
        minority_total = sum(counts[lbl] for lbl in minority_classes if lbl in counts)
        majority_classes = [lbl for lbl in counts if lbl not in minority_classes]
        majority_total = sum(counts[lbl] for lbl in majority_classes)

        majority_budget = strategy.protect_target - minority_total
        if majority_budget <= 0:
            # Not enough budget, just return as-is
            return texts, labels

        new_texts = []
        new_labels = []

        # Keep ALL minority class samples
        for label in minority_classes:
            if label in by_label:
                new_texts.extend(by_label[label])
                new_labels.extend([label] * len(by_label[label]))

        # Proportionally reduce majority classes
        for label in majority_classes:
            if majority_total > 0:
                target_count = int(counts[label] / majority_total * majority_budget)
            else:
                target_count = 0
            label_texts = by_label[label]

            if len(label_texts) <= target_count:
                sampled = label_texts
            else:
                indices = rng.choice(len(label_texts), size=max(1, target_count), replace=False)
                sampled = [label_texts[i] for i in indices]

            new_texts.extend(sampled)
            new_labels.extend([label] * len(sampled))

        return new_texts, new_labels

    # Handle ratio/cap strategies
    min_count = min(counts.values())

    # Determine cap
    if strategy.max_per_class is not None:
        cap = strategy.max_per_class
    elif strategy.target_ratio is not None:
        cap = int(min_count * strategy.target_ratio)
    else:
        cap = min_count

    new_texts = []
    new_labels = []

    for label, label_texts in by_label.items():
        if len(label_texts) <= cap:
            sampled = label_texts
        else:
            indices = rng.choice(len(label_texts), size=cap, replace=False)
            sampled = [label_texts[i] for i in indices]
        new_texts.extend(sampled)
        new_labels.extend([label] * len(sampled))

    return new_texts, new_labels


def get_embeddings(texts: list[str], model_name: str, batch_size: int = 500) -> np.ndarray:
    """Get embeddings for texts using specified model.

    Uses the currently configured embedding model (set before calling this function).
    Processes in batches for memory efficiency and progress reporting.

    Args:
        texts: List of texts to embed.
        model_name: Expected model name (for logging only).
        batch_size: Batch size for progress reporting (MLX service handles internal batching).
    """
    from jarvis.embedding_adapter import get_embedder

    # Reset embedder to pick up new config
    reset_embedder()

    # Get embedder (will use current config)
    embedder = get_embedder()
    print(f"    Model: {embedder.model_name}", flush=True)
    print(f"    Texts: {len(texts)}", flush=True)

    # Process all at once - MLXEmbedder handles internal batching at 100 texts
    # The embedder's internal batching is optimized for the MLX service
    embeddings = embedder.encode(texts, normalize=True)
    return embeddings


def create_svm_objective(X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int) -> Any:
    """Create Optuna objective for SVM."""

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        gamma_float = trial.suggest_float("gamma_float", 0.0001, 1.0, log=True)

        # Use gamma_float only if gamma is not scale/auto
        use_gamma = trial.suggest_categorical("use_gamma_float", [True, False])
        actual_gamma: str | float = gamma_float if use_gamma else gamma

        clf = SVC(
            kernel="rbf",
            C=C,
            gamma=actual_gamma,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=n_jobs)
        return scores.mean()

    return objective


def create_logreg_objective(X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int) -> Any:
    """Create Optuna objective for LogisticRegression."""

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 2000)

        clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=n_jobs)
        return scores.mean()

    return objective


def create_rf_objective(X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int) -> Any:
    """Create Optuna objective for RandomForest."""

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=n_jobs)
        return scores.mean()

    return objective


def create_xgb_objective(
    X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int, num_classes: int
) -> Any:
    """Create Optuna objective for XGBoost."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed")

    # Encode string labels to integers (XGBoost requires this)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }

        if num_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_classes
        else:
            params["objective"] = "binary:logistic"

        clf = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="f1_macro", n_jobs=1)
        return scores.mean()

    return objective


def create_lgbm_objective(
    X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int, num_classes: int
) -> Any:
    """Create Optuna objective for LightGBM."""
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed")

    # Encode string labels to integers (LightGBM requires this)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbose": -1,
        }

        if num_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = num_classes
        else:
            params["objective"] = "binary"

        clf = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="f1_macro", n_jobs=1)
        return scores.mean()

    return objective


def train_classifier_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: str,
    config: TrainingConfig,
    embedding_model: str,
    classifier_type: str,
    sampling_strategy: str,
    dataset_name: str,
) -> ClassifierResult | None:
    """Train a classifier using Optuna for hyperparameter optimization."""
    num_classes = len(set(y_train))

    # Create objective function
    try:
        if model_class == "SVM":
            objective = create_svm_objective(X_train, y_train, config.cv_folds, config.n_jobs)
        elif model_class == "LogisticRegression":
            objective = create_logreg_objective(X_train, y_train, config.cv_folds, config.n_jobs)
        elif model_class == "RandomForest":
            objective = create_rf_objective(X_train, y_train, config.cv_folds, config.n_jobs)
        elif model_class == "XGBoost":
            if not HAS_XGBOOST:
                print("    Skipping XGBoost (not installed)", flush=True)
                return None
            objective = create_xgb_objective(
                X_train, y_train, config.cv_folds, config.n_jobs, num_classes
            )
        elif model_class == "LightGBM":
            if not HAS_LIGHTGBM:
                print("    Skipping LightGBM (not installed)", flush=True)
                return None
            objective = create_lgbm_objective(
                X_train, y_train, config.cv_folds, config.n_jobs, num_classes
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    except ImportError as e:
        print(f"    Skipping {model_class}: {e}", flush=True)
        return None

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    best_params = study.best_params
    study.best_value

    # Get CV std from best trial
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    # Train final model with best params
    if model_class == "SVM":
        gamma = (
            best_params.get("gamma_float")
            if best_params.get("use_gamma_float")
            else best_params.get("gamma")
        )
        final_clf = SVC(
            kernel="rbf",
            C=best_params["C"],
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    elif model_class == "LogisticRegression":
        final_clf = LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            max_iter=best_params["max_iter"],
            class_weight="balanced",
            random_state=42,
        )
    elif model_class == "RandomForest":
        final_clf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=config.n_jobs,
        )
    elif model_class == "XGBoost":
        xgb_params = {k: v for k, v in best_params.items()}
        xgb_params["random_state"] = 42
        xgb_params["n_jobs"] = config.n_jobs
        xgb_params["verbosity"] = 0
        if num_classes > 2:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = num_classes
        final_clf = XGBClassifier(**xgb_params)
    elif model_class == "LightGBM":
        lgbm_params = {k: v for k, v in best_params.items()}
        lgbm_params["random_state"] = 42
        lgbm_params["n_jobs"] = config.n_jobs
        lgbm_params["verbose"] = -1
        if num_classes > 2:
            lgbm_params["objective"] = "multiclass"
            lgbm_params["num_class"] = num_classes
        final_clf = LGBMClassifier(**lgbm_params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # For XGBoost/LightGBM, encode string labels to integers
    label_encoder = None
    y_train_final = y_train
    if model_class in ("XGBoost", "LightGBM"):
        label_encoder = LabelEncoder()
        y_train_final = label_encoder.fit_transform(y_train)
        label_encoder.transform(y_test)

    # Get CV scores for std (avoid nested parallelism for boosting models)
    cv_n_jobs = 1 if model_class in ("XGBoost", "LightGBM") else config.n_jobs
    cv_scores = cross_val_score(
        final_clf, X_train, y_train_final, cv=cv, scoring="f1_macro", n_jobs=cv_n_jobs
    )

    # Train on full training set
    final_clf.fit(X_train, y_train_final)

    # Evaluate on test set
    y_pred_raw = final_clf.predict(X_test)

    # Decode predictions back to original labels for metrics
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_raw)
    else:
        y_pred = y_pred_raw

    y_test_arr = np.asarray(y_test)  # Use original labels for metrics
    test_f1 = f1_score(y_test_arr, y_pred, average="macro", zero_division=0)
    test_accuracy = (y_pred == y_test_arr).mean()

    # Get per-class metrics
    report = classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
    per_class = {
        k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    return ClassifierResult(
        embedding_model=embedding_model,
        classifier_type=classifier_type,
        model_class=model_class,
        sampling_strategy=sampling_strategy,
        dataset_name=dataset_name,
        params=best_params,
        cv_f1_mean=cv_scores.mean(),
        cv_f1_std=cv_scores.std(),
        test_f1=test_f1,
        test_accuracy=test_accuracy,
        per_class=per_class,
        train_size=len(y_train),
        test_size=len(y_test),
    )


def train_classifier_with_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: str,
    config: TrainingConfig,
    embedding_model: str,
    classifier_type: str,
    sampling_strategy: str,
    dataset_name: str,
) -> ClassifierResult | None:
    """Train a classifier using fixed known-good hyperparameters (no Optuna)."""
    num_classes = len(set(y_train))
    params = FIXED_PARAMS.get(model_class, {}).copy()

    # For XGBoost/LightGBM, encode string labels to integers
    label_encoder = None
    y_train_encoded = y_train
    if model_class in ("XGBoost", "LightGBM"):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        label_encoder.transform(y_test)

    # Create classifier with fixed params
    if model_class == "SVM":
        clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    elif model_class == "LogisticRegression":
        clf = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            max_iter=params["max_iter"],
            class_weight="balanced",
            random_state=42,
        )
    elif model_class == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=config.n_jobs,
        )
    elif model_class == "XGBoost":
        if not HAS_XGBOOST:
            print("    Skipping XGBoost (not installed)", flush=True)
            return None
        xgb_params = params.copy()
        xgb_params["random_state"] = 42
        xgb_params["n_jobs"] = config.n_jobs
        xgb_params["verbosity"] = 0
        if num_classes > 2:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = num_classes
        clf = XGBClassifier(**xgb_params)
    elif model_class == "LightGBM":
        if not HAS_LIGHTGBM:
            print("    Skipping LightGBM (not installed)", flush=True)
            return None
        lgbm_params = params.copy()
        lgbm_params["random_state"] = 42
        lgbm_params["n_jobs"] = config.n_jobs
        lgbm_params["verbose"] = -1
        if num_classes > 2:
            lgbm_params["objective"] = "multiclass"
            lgbm_params["num_class"] = num_classes
        clf = LGBMClassifier(**lgbm_params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Get CV scores
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
    cv_n_jobs = 1 if model_class in ("XGBoost", "LightGBM") else config.n_jobs
    cv_scores = cross_val_score(
        clf, X_train, y_train_encoded, cv=cv, scoring="f1_macro", n_jobs=cv_n_jobs
    )

    # Train on full training set
    clf.fit(X_train, y_train_encoded)

    # Evaluate on test set
    y_pred_encoded = clf.predict(X_test)

    # Decode predictions back to original labels for metrics
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test_for_metrics = y_test
    else:
        y_pred = y_pred_encoded
        y_test_for_metrics = y_test

    y_test_arr = np.asarray(y_test_for_metrics)
    test_f1 = f1_score(y_test_arr, y_pred, average="macro", zero_division=0)
    test_accuracy = (y_pred == y_test_arr).mean()

    # Get per-class metrics
    report = classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
    per_class = {
        k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    return ClassifierResult(
        embedding_model=embedding_model,
        classifier_type=classifier_type,
        model_class=model_class,
        sampling_strategy=sampling_strategy,
        dataset_name=dataset_name,
        params=params,
        cv_f1_mean=cv_scores.mean(),
        cv_f1_std=cv_scores.std(),
        test_f1=test_f1,
        test_accuracy=test_accuracy,
        per_class=per_class,
        train_size=len(y_train),
        test_size=len(y_test),
    )


def save_best_model(
    result: ClassifierResult,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str,
) -> None:
    """Save the best model to disk."""
    from jarvis.config import get_response_classifier_path, get_trigger_classifier_path

    if classifier_type == "trigger":
        model_path = get_trigger_classifier_path()
    else:
        model_path = get_response_classifier_path()

    model_path.mkdir(parents=True, exist_ok=True)

    # Recreate and train the final model
    if result.model_class == "SVM":
        gamma = (
            result.params.get("gamma_float")
            if result.params.get("use_gamma_float")
            else result.params.get("gamma")
        )
        clf = SVC(
            kernel="rbf",
            C=result.params["C"],
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    elif result.model_class == "LogisticRegression":
        clf = LogisticRegression(
            C=result.params["C"],
            solver=result.params["solver"],
            max_iter=result.params["max_iter"],
            class_weight="balanced",
            random_state=42,
        )
    elif result.model_class == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=result.params["n_estimators"],
            max_depth=result.params["max_depth"],
            min_samples_split=result.params["min_samples_split"],
            min_samples_leaf=result.params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif result.model_class == "XGBoost":
        num_classes = len(set(y_train))
        xgb_params = {k: v for k, v in result.params.items()}
        xgb_params["random_state"] = 42
        xgb_params["n_jobs"] = -1
        xgb_params["verbosity"] = 0
        if num_classes > 2:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = num_classes
        clf = XGBClassifier(**xgb_params)
    elif result.model_class == "LightGBM":
        num_classes = len(set(y_train))
        lgbm_params = {k: v for k, v in result.params.items()}
        lgbm_params["random_state"] = 42
        lgbm_params["n_jobs"] = -1
        lgbm_params["verbose"] = -1
        if num_classes > 2:
            lgbm_params["objective"] = "multiclass"
            lgbm_params["num_class"] = num_classes
        clf = LGBMClassifier(**lgbm_params)
    else:
        raise ValueError(f"Unknown model class: {result.model_class}")

    clf.fit(X_train, y_train)

    # Save model
    with open(model_path / "svm.pkl", "wb") as f:  # Keep filename for compatibility
        pickle.dump(clf, f)

    # Save config
    config = {
        "labels": sorted(set(y_train)),
        "model_class": result.model_class,
        "params": result.params,
        "cv_f1_mean": result.cv_f1_mean,
        "cv_f1_std": result.cv_f1_std,
        "test_f1": result.test_f1,
        "test_accuracy": result.test_accuracy,
        "train_size": result.train_size,
    }
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Compute and save centroids
    unique_labels = sorted(set(y_train))
    centroids = {}
    for label in unique_labels:
        mask = [i for i, lbl in enumerate(y_train) if lbl == label]
        if mask:
            class_embeddings = X_train[mask]
            centroid = np.mean(class_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[label] = centroid.tolist()

    np.save(model_path / "centroids.npy", centroids)

    print(f"  Saved {result.model_class} model to {model_path}", flush=True)


def save_results_incremental(
    results: list[ClassifierResult],
    output_dir: Path,
    embedding_model: str,
) -> None:
    """Save results incrementally to avoid data loss."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"results_{embedding_model}.json"
    results_data = [
        {
            "embedding_model": r.embedding_model,
            "classifier_type": r.classifier_type,
            "model_class": r.model_class,
            "sampling_strategy": r.sampling_strategy,
            "params": r.params,
            "cv_f1_mean": r.cv_f1_mean,
            "cv_f1_std": r.cv_f1_std,
            "test_f1": r.test_f1,
            "test_accuracy": r.test_accuracy,
            "per_class": r.per_class,
            "train_size": r.train_size,
            "test_size": r.test_size,
        }
        for r in results
    ]
    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"    [Saved {len(results)} results to {results_file}]", flush=True)


def train_for_embedding_model(
    embedding_model: str,
    config: TrainingConfig,
    trigger_texts: list[str],
    trigger_labels: list[str],
    response_texts: list[str],
    response_labels: list[str],
    dataset_name: str = "base",
    classifier_types_override: list[str] | None = None,
) -> list[ClassifierResult]:
    """Train all classifiers for a single embedding model."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"EMBEDDING MODEL: {embedding_model}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Update config to use this embedding model
    jarvis_config = get_config()
    jarvis_config.embedding.model_name = embedding_model
    save_config(jarvis_config)
    reset_config()
    reset_embedder()

    results: list[ClassifierResult] = []
    model_classes = ["SVM", "LogisticRegression"]  # RF removed - consistently underperforms
    if HAS_XGBOOST:
        model_classes.append("XGBoost")
    if HAS_LIGHTGBM:
        model_classes.append("LightGBM")

    # Use override if provided, otherwise use config
    classifier_types = classifier_types_override or config.classifier_types

    for classifier_type in classifier_types:
        print(f"\n--- Training {classifier_type.upper()} classifiers ---", flush=True)

        # Get raw data (before sampling)
        if classifier_type == "trigger":
            raw_texts, raw_labels = trigger_texts.copy(), trigger_labels.copy()
        else:
            raw_texts, raw_labels = response_texts.copy(), response_labels.copy()

        print(f"  Raw data: {len(raw_texts)} samples", flush=True)

        # Compute embeddings for ALL texts ONCE (before sampling)
        # This allows reusing embeddings across sampling strategies
        print("  Computing embeddings for all texts...", flush=True)
        all_embeddings = get_embeddings(raw_texts, embedding_model)
        print(f"  Embeddings shape: {all_embeddings.shape}", flush=True)

        # Create text-to-embedding index for efficient lookup
        text_to_idx = {t: i for i, t in enumerate(raw_texts)}

        # Try each sampling strategy
        for strategy in DEFAULT_SAMPLING_STRATEGIES:
            print(f"\n  [Sampling: {strategy.name}]", flush=True)

            # Apply sampling
            sampled_texts, sampled_labels = apply_sampling(
                raw_texts, raw_labels, strategy, classifier_type=classifier_type
            )
            print(
                f"    Samples: {len(sampled_texts)}, Dist: {dict(Counter(sampled_labels))}",
                flush=True,
            )

            # Get embeddings for sampled texts (lookup, not recompute)
            sampled_indices = [text_to_idx[t] for t in sampled_texts]
            embeddings = all_embeddings[sampled_indices]

            # Split train/test (convert labels to numpy for consistent handling)
            from sklearn.model_selection import train_test_split

            labels_arr = np.asarray(sampled_labels)
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings,
                labels_arr,
                test_size=0.2,
                random_state=42,
                stratify=labels_arr,
            )

            # Train each model class with this sampling strategy
            for model_class in model_classes:
                print(f"    Training {model_class}...", flush=True)

                # Choose training function based on config
                train_fn = (
                    train_classifier_with_fixed_params
                    if config.fixed_params
                    else train_classifier_with_optuna
                )
                result = train_fn(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_class=model_class,
                    config=config,
                    embedding_model=embedding_model,
                    classifier_type=classifier_type,
                    sampling_strategy=strategy.name,
                    dataset_name=dataset_name,
                )

                if result is None:
                    continue

                results.append(result)
                print(
                    f"      CV F1: {result.cv_f1_mean:.3f} ± {result.cv_f1_std:.3f}, "
                    f"Test F1: {result.test_f1:.3f}",
                    flush=True,
                )

                # Save incrementally after each experiment
                save_results_incremental(results, config.output_dir, embedding_model)

        # Find best result for this classifier type (across all sampling strategies)
        type_results = [r for r in results if r.classifier_type == classifier_type]
        if config.save_best and type_results:
            best_result = max(type_results, key=lambda r: r.test_f1)
            print(
                f"\n  Best {classifier_type}: {best_result.model_class} ({best_result.sampling_strategy}) F1={best_result.test_f1:.3f}"
            )
            # Need to recreate the train data for saving
            best_strategy = next(
                s for s in DEFAULT_SAMPLING_STRATEGIES if s.name == best_result.sampling_strategy
            )
            best_texts, best_labels = apply_sampling(
                raw_texts, raw_labels, best_strategy, classifier_type=classifier_type
            )
            best_indices = [text_to_idx[t] for t in best_texts]
            best_embeddings = all_embeddings[best_indices]
            best_labels_arr = np.asarray(best_labels)
            X_train_best, _, y_train_best, _ = train_test_split(
                best_embeddings,
                best_labels_arr,
                test_size=0.2,
                random_state=42,
                stratify=best_labels_arr,
            )
            save_best_model(best_result, X_train_best, y_train_best, classifier_type)

    return results


def print_summary(all_results: list[ClassifierResult]) -> None:
    """Print summary of all results."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    # Group by embedding model and classifier type
    by_embedding: dict[str, dict[str, list[ClassifierResult]]] = {}
    for r in all_results:
        if r.embedding_model not in by_embedding:
            by_embedding[r.embedding_model] = {}
        if r.classifier_type not in by_embedding[r.embedding_model]:
            by_embedding[r.embedding_model][r.classifier_type] = []
        by_embedding[r.embedding_model][r.classifier_type].append(r)

    # Print table
    print(
        f"\n{'Embedding':<12} {'Type':<8} {'Sampling':<10} {'Model':<15} {'CV F1':>12} {'Test F1':>8} {'Train':>6}"
    )
    print("-" * 85)

    for emb_model in sorted(by_embedding.keys()):
        for clf_type in sorted(by_embedding[emb_model].keys()):
            results = by_embedding[emb_model][clf_type]
            # Sort by test F1
            results.sort(key=lambda x: x.test_f1, reverse=True)
            # Only show top 5 per embedding/type combo to avoid overwhelming output
            for r in results[:5]:
                cv_str = f"{r.cv_f1_mean:.3f}±{r.cv_f1_std:.3f}"
                print(
                    f"{r.embedding_model:<12} {r.classifier_type:<8} {r.sampling_strategy:<10} "
                    f"{r.model_class:<15} {cv_str:>12} {r.test_f1:>8.3f} {r.train_size:>6}"
                )

    # Find best overall per classifier type
    print("\n" + "=" * 80)
    print("BEST MODELS PER CLASSIFIER TYPE")
    print("=" * 80)

    for clf_type in ["trigger", "response"]:
        type_results = [r for r in all_results if r.classifier_type == clf_type]
        if not type_results:
            continue
        best = max(type_results, key=lambda x: x.test_f1)
        print(f"\n{clf_type.upper()}:")
        print(f"  Embedding: {best.embedding_model}")
        print(f"  Model: {best.model_class}")
        print(f"  Sampling: {best.sampling_strategy}")
        print(f"  Test F1: {best.test_f1:.3f}")
        print(f"  CV F1: {best.cv_f1_mean:.3f} ± {best.cv_f1_std:.3f}")
        print(f"  Train size: {best.train_size}")
        print(f"  Params: {best.params}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifiers for all embedding models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(EMBEDDING_MODEL_REGISTRY.keys()),
        choices=list(EMBEDDING_MODEL_REGISTRY.keys()),
        help="Embedding models to train on",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["trigger", "response"],
        choices=["trigger", "response"],
        help="Classifier types to train",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best model for each embedding model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/classifier_training"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--trigger-data",
        type=Path,
        default=Path("data/trigger_labeling.jsonl"),
        help="Path to trigger labeling data (used if --dataset is not specified)",
    )
    parser.add_argument(
        "--response-data",
        type=Path,
        default=Path("data/response_labeling.jsonl"),
        help="Path to response labeling data",
    )
    parser.add_argument(
        "--trigger-dataset",
        choices=[c.name for c in TRIGGER_DATASET_CONFIGS],
        default=None,
        help="Trigger dataset: base, base_plus, large_strict, large, xl",
    )
    parser.add_argument(
        "--response-dataset",
        choices=[c.name for c in RESPONSE_DATASET_CONFIGS],
        default=None,
        help="Response dataset: base, medium, large, xl",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Train on ALL dataset sizes (adds dataset as another dimension)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization (not recommended)",
    )
    parser.add_argument(
        "--fixed-params",
        action="store_true",
        help="Use known-good hyperparameters instead of Optuna search (much faster)",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        embedding_models=args.models,
        classifier_types=args.classifiers,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        save_best=args.save_best,
        output_dir=args.output,
        fixed_params=args.fixed_params,
    )

    print("=" * 70)
    print("COMPREHENSIVE CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Embedding models: {config.embedding_models}")
    print(f"Classifier types: {config.classifier_types}")
    if config.fixed_params:
        print("Mode: FIXED PARAMS (no Optuna search)")
    else:
        print(f"Optuna trials: {config.n_trials}")
    print(f"CV folds: {config.cv_folds}")
    print(f"Save best: {config.save_best}")

    # Check for optional dependencies
    print("\nOptional dependencies:")
    print(f"  XGBoost: {'available' if HAS_XGBOOST else 'not installed'}")
    print(f"  LightGBM: {'available' if HAS_LIGHTGBM else 'not installed'}")

    # Architecture summary
    print("\nTraining architecture:")
    print("  - Processing embedding models SEQUENTIALLY (one at a time)")
    print(
        f"  - CV uses n_jobs={config.n_jobs} (parallel for SVM/LR/RF, n_jobs=1 for XGBoost/LightGBM)"
    )
    print("  - Classifiers (RF, XGBoost, LightGBM) use internal parallelism")
    print("  - MLX embedder batches internally at 100 texts")
    print("  - Embeddings computed once per classifier type, reused for all model classes")
    print("  - Results saved incrementally after each experiment (no data loss)")

    # Determine which dataset configs to use
    apply_norm = not args.no_normalize

    # Trigger dataset configs
    if args.all_datasets:
        trigger_dataset_configs = TRIGGER_DATASET_CONFIGS
        print(f"\nTrigger: Training on ALL {len(trigger_dataset_configs)} dataset sizes")
    elif args.trigger_dataset:
        trigger_dataset_configs = [
            c for c in TRIGGER_DATASET_CONFIGS if c.name == args.trigger_dataset
        ]
        print(f"\nTrigger dataset: {args.trigger_dataset}")
    else:
        trigger_dataset_configs = [
            DatasetConfig(
                name="base",
                trigger_files=[str(args.trigger_data)],
                min_confidence=0.0,
                description="Base human-labeled",
            )
        ]

    # Response dataset configs
    if args.all_datasets:
        response_dataset_configs = RESPONSE_DATASET_CONFIGS
        print(f"Response: Training on ALL {len(response_dataset_configs)} dataset sizes")
    elif args.response_dataset:
        response_dataset_configs = [
            c for c in RESPONSE_DATASET_CONFIGS if c.name == args.response_dataset
        ]
        print(f"Response dataset: {args.response_dataset}")
    else:
        response_dataset_configs = [
            DatasetConfig(
                name="base",
                trigger_files=[str(args.response_data)],
                min_confidence=0.0,
                description="Base human-labeled",
            )
        ]

    # Train for each embedding model and dataset config
    all_results: list[ClassifierResult] = []
    original_model = get_config().embedding.model_name

    try:
        for embedding_model in config.embedding_models:
            # Train TRIGGER classifiers with trigger dataset configs
            if "trigger" in config.classifier_types:
                for dataset_config in trigger_dataset_configs:
                    print(f"\n{'#' * 70}", flush=True)
                    print(
                        f"# TRIGGER Dataset: {dataset_config.name} - {dataset_config.description}",
                        flush=True,
                    )
                    print(f"{'#' * 70}", flush=True)

                    conf_info = (
                        "per-class"
                        if dataset_config.per_class_confidence
                        else f">= {dataset_config.min_confidence}"
                    )
                    print(f"  Loading trigger data (confidence: {conf_info})...", flush=True)
                    trigger_texts, trigger_labels = load_trigger_data_from_files(
                        dataset_config.trigger_files,
                        apply_normalization=apply_norm,
                        min_confidence=dataset_config.min_confidence,
                        per_class_confidence=dataset_config.per_class_confidence,
                    )
                    print(
                        f"  Trigger: {len(trigger_texts)} samples, {len(set(trigger_labels))} classes"
                    )
                    print(f"  Distribution: {dict(Counter(trigger_labels))}")

                    results = train_for_embedding_model(
                        embedding_model=embedding_model,
                        config=config,
                        trigger_texts=trigger_texts,
                        trigger_labels=trigger_labels,
                        response_texts=[],  # Not used for trigger
                        response_labels=[],
                        dataset_name=dataset_config.name,
                        classifier_types_override=["trigger"],
                    )
                    all_results.extend(results)

            # Train RESPONSE classifiers with response dataset configs
            if "response" in config.classifier_types:
                for dataset_config in response_dataset_configs:
                    print(f"\n{'#' * 70}", flush=True)
                    print(
                        f"# RESPONSE Dataset: {dataset_config.name} - {dataset_config.description}",
                        flush=True,
                    )
                    print(f"{'#' * 70}", flush=True)

                    conf_info = (
                        "per-class"
                        if dataset_config.per_class_confidence
                        else f">= {dataset_config.min_confidence}"
                    )
                    print(f"  Loading response data (confidence: {conf_info})...", flush=True)
                    response_texts, response_labels = load_response_data_from_files(
                        dataset_config.trigger_files,  # Reusing field name
                        apply_normalization=apply_norm,
                        min_confidence=dataset_config.min_confidence,
                        per_class_confidence=dataset_config.per_class_confidence,
                    )
                    print(
                        f"  Response: {len(response_texts)} samples, {len(set(response_labels))} classes"
                    )
                    print(f"  Distribution: {dict(Counter(response_labels))}")

                    results = train_for_embedding_model(
                        embedding_model=embedding_model,
                        config=config,
                        trigger_texts=[],  # Not used for response
                        trigger_labels=[],
                        response_texts=response_texts,
                        response_labels=response_labels,
                        dataset_name=dataset_config.name,
                        classifier_types_override=["response"],
                    )
                    all_results.extend(results)
    finally:
        # Restore original embedding model
        jarvis_config = get_config()
        jarvis_config.embedding.model_name = original_model
        save_config(jarvis_config)
        reset_config()
        reset_embedder()

    # Print summary
    print_summary(all_results)

    # Save results to JSON
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = config.output_dir / "training_results.json"
    results_data = [
        {
            "embedding_model": r.embedding_model,
            "classifier_type": r.classifier_type,
            "model_class": r.model_class,
            "sampling_strategy": r.sampling_strategy,
            "params": r.params,
            "cv_f1_mean": r.cv_f1_mean,
            "cv_f1_std": r.cv_f1_std,
            "test_f1": r.test_f1,
            "test_accuracy": r.test_accuracy,
            "per_class": r.per_class,
            "train_size": r.train_size,
            "test_size": r.test_size,
        }
        for r in all_results
    ]
    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
