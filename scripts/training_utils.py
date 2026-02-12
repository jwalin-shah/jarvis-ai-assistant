"""Shared evaluation helpers for training scripts.

Provides common binary classification metrics computation and logging
to avoid duplication across train_fact_filter.py, train_message_gate.py, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_score: Optional continuous scores for ROC AUC.

    Returns:
        Dict with accuracy, precision, recall, f1, and optionally roc_auc.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def log_metrics(
    metrics: dict[str, float],
    split_name: str,
    logger: logging.Logger | None = None,
) -> None:
    """Log metrics dict with a split name prefix.

    Args:
        metrics: Dict of metric_name -> value.
        split_name: Label for the split (e.g. "Train", "Dev", "Test").
        logger: Logger instance. Falls back to module logger if None.
    """
    log = logger or globals()["logger"]
    log.info("%s metrics:", split_name)
    for key, value in metrics.items():
        log.info("  %s: %.4f", key, value)
