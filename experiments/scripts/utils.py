"""Shared utilities for experiment scripts.

Provides:
- Data loading and saving
- SVM training with cross-validation
- Metrics computation
- Embedding computation and caching
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Experiment paths
EXPERIMENTS_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"


@dataclass
class LabeledExample:
    """A labeled response example."""

    text: str
    label: str
    source: str = "human"  # 'human' or 'auto'
    confidence: float = 1.0


@dataclass
class CVResult:
    """Result from cross-validation."""

    mean_f1: float
    std_f1: float
    fold_scores: list[float]
    per_class_f1: dict[str, float]


@dataclass
class SearchResult:
    """Result from a hyperparameter search."""

    size: int
    C: float
    gamma: str
    cv_mean: float
    cv_std: float
    per_class_f1: dict[str, float]


def load_labeled_data(path: Path) -> list[LabeledExample]:
    """Load labeled data from JSONL file.

    Expected format: {"response": "text", "label": "LABEL", ...}
    """
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("response", "").strip()
            label = row.get("label", "").upper()
            if text and label:
                examples.append(
                    LabeledExample(
                        text=text,
                        label=label,
                        source=row.get("source", "human"),
                        confidence=row.get("confidence", 1.0),
                    )
                )
    return examples


def save_labeled_data(examples: list[LabeledExample], path: Path) -> None:
    """Save labeled data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            row = {
                "response": ex.text,
                "label": ex.label,
                "source": ex.source,
                "confidence": ex.confidence,
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Saved %d examples to %s", len(examples), path)


def compute_embeddings(
    texts: list[str],
    cache_path: Path | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute embeddings with optional caching.

    Args:
        texts: List of texts to embed
        cache_path: If provided, save/load from this npz file
        batch_size: Batch size for embedding computation

    Returns:
        Embeddings array of shape (len(texts), embedding_dim)
    """
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    logger.info("Computing embeddings for %d texts...", len(texts))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_emb = embedder.encode(batch, normalize=True)
        all_embeddings.append(batch_emb)
        if len(texts) > 1000 and (i + batch_size) % 1000 == 0:
            logger.info("  Embedded %d/%d texts", min(i + batch_size, len(texts)), len(texts))

    embeddings = np.vstack(all_embeddings)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        logger.info("Cached embeddings to %s", cache_path)

    return embeddings


def load_cached_embeddings(cache_path: Path) -> np.ndarray | None:
    """Load cached embeddings if available."""
    if cache_path.exists():
        data = np.load(cache_path)
        logger.info("Loaded cached embeddings from %s", cache_path)
        return data["embeddings"]
    return None


def train_svm(
    X_train: np.ndarray,  # noqa: N803
    y_train: list[str],
    C: float = 1.0,  # noqa: N803
    gamma: str = "scale",
) -> SVC:
    """Train an SVM classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        C: Regularization parameter
        gamma: Kernel coefficient ('scale', 'auto', or float)

    Returns:
        Trained SVC model
    """
    clf = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def cross_validate(
    X: np.ndarray,  # noqa: N803
    y: list[str],
    C: float = 1.0,  # noqa: N803
    gamma: str = "scale",
    n_folds: int = 5,
    n_jobs: int = -1,
) -> CVResult:
    """Run k-fold cross-validation with parallel fold execution.

    Uses cross_val_predict to get predictions in ONE pass (not two),
    then computes metrics from those predictions.

    Args:
        X: Feature matrix
        y: Labels
        C: SVM C parameter
        gamma: SVM gamma parameter
        n_folds: Number of folds
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        CVResult with mean/std F1 and per-class metrics
    """
    y_array = np.array(y)

    clf = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        probability=False,  # Faster without probability
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Get all predictions in ONE pass using cross_val_predict
    # This trains n_folds models in parallel and returns out-of-fold predictions
    y_pred = cross_val_predict(clf, X, y_array, cv=skf, n_jobs=n_jobs)

    # Compute fold scores from predictions
    fold_scores = []
    for train_idx, val_idx in skf.split(X, y_array):
        fold_f1 = f1_score(
            y_array[val_idx],
            y_pred[val_idx],
            average="macro",
            zero_division=0,
        )
        fold_scores.append(fold_f1)

    # Compute per-class F1 from all predictions
    report = classification_report(y_array, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {
        label: metrics["f1-score"]
        for label, metrics in report.items()
        if isinstance(metrics, dict) and label not in ["accuracy", "macro avg", "weighted avg"]
    }

    return CVResult(
        mean_f1=float(np.mean(fold_scores)),
        std_f1=float(np.std(fold_scores)),
        fold_scores=fold_scores,
        per_class_f1=per_class_f1,
    )


def grid_search_cv(
    X: np.ndarray,  # noqa: N803
    y: list[str],
    c_values: list[float],
    gamma_values: list[str],
    n_folds: int = 5,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run grid search with cross-validation using sklearn's GridSearchCV.

    This is more efficient than manual looping because:
    - Parallelizes across parameter combinations AND folds
    - Uses efficient memory management
    - Caches repeated computations

    Args:
        X: Feature matrix
        y: Labels
        c_values: List of C values to try
        gamma_values: List of gamma values to try
        n_folds: Number of CV folds
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Dict with 'best_params', 'best_score', 'cv_results'
    """
    y_array = np.array(y)

    clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=42,
    )

    param_grid = {
        "C": c_values,
        "gamma": gamma_values,
    }

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=n_jobs,
        return_train_score=False,
        verbose=0,
    )

    grid_search.fit(X, y_array)

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": {
            "params": grid_search.cv_results_["params"],
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }


def get_label_distribution(labels: list[str]) -> dict[str, int]:
    """Get label distribution as dict."""
    return dict(Counter(labels))


def stratified_split(
    examples: list[LabeledExample],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Stratified train/test split.

    Args:
        examples: List of labeled examples
        test_ratio: Fraction for test set
        seed: Random seed

    Returns:
        (train_examples, test_examples)
    """
    rng = np.random.default_rng(seed)

    # Group by label
    by_label: dict[str, list[LabeledExample]] = {}
    for ex in examples:
        if ex.label not in by_label:
            by_label[ex.label] = []
        by_label[ex.label].append(ex)

    train_examples = []
    test_examples = []

    for label, label_examples in by_label.items():
        # Shuffle within label
        indices = rng.permutation(len(label_examples))
        shuffled = [label_examples[i] for i in indices]

        # Split
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_examples.extend(shuffled[:n_test])
        train_examples.extend(shuffled[n_test:])

    return train_examples, test_examples


def save_results(results: dict[str, Any], path: Path) -> None:
    """Save results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", path)


def load_results(path: Path) -> dict[str, Any] | None:
    """Load results from JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
