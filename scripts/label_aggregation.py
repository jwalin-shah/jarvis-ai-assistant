"""Label Aggregation for Weak Supervision - Aggregate noisy LF votes into clean labels.

Two methods:
1. Weighted Majority Vote (default) - fast, simple, good for 25+ diverse LFs
2. Dawid-Skene EM (optional) - learns per-LF accuracies, better quality

Usage:
    from scripts.label_aggregation import aggregate_labels
    from scripts.labeling_functions import get_registry

    registry = get_registry()
    examples = [
        {"text": "ok", "context": [], "last_message": "", "metadata": None},
        {"text": "What time?", "context": [], "last_message": "", "metadata": None},
    ]

    labels, confidences = aggregate_labels(examples, registry, method="majority")
"""

from __future__ import annotations

import numpy as np

from scripts.labeling_functions import ABSTAIN, LabelingFunctionRegistry

# ---------------------------------------------------------------------------
# Weighted Majority Vote
# ---------------------------------------------------------------------------


def _majority_vote(
    label_matrix: np.ndarray,
    weights: list[float],
    categories: list[str],
) -> tuple[list[str], list[float]]:
    """Aggregate labels using weighted majority vote.

    Args:
        label_matrix: (n_examples, n_lfs) array of label indices (-1 = abstain).
        weights: Weight for each LF (length n_lfs).
        categories: List of category names.

    Returns:
        Tuple of (labels, confidences) for each example.
    """
    n_examples, n_lfs = label_matrix.shape
    n_categories = len(categories)
    weights_arr = np.array(weights, dtype=np.float32)

    labels = []
    confidences = []

    for i in range(n_examples):
        votes = label_matrix[i]  # (n_lfs,)

        # Count weighted votes per category
        category_weights = np.zeros(n_categories, dtype=np.float32)
        for lf_idx in range(n_lfs):
            label_idx = votes[lf_idx]
            if label_idx >= 0:  # Not abstain
                category_weights[label_idx] += weights_arr[lf_idx]

        total_weight = category_weights.sum()

        # If no votes, default to "social"
        if total_weight == 0:
            labels.append("social")
            confidences.append(0.0)
            continue

        # Winner = highest weighted sum
        winner_idx = int(category_weights.argmax())
        winner_weight = category_weights[winner_idx]
        confidence = float(winner_weight / total_weight)

        labels.append(categories[winner_idx])
        confidences.append(confidence)

    return labels, confidences


# ---------------------------------------------------------------------------
# Dawid-Skene EM Algorithm
# ---------------------------------------------------------------------------


def _initialize_confusion_matrices(
    label_matrix: np.ndarray, n_categories: int, n_lfs: int
) -> np.ndarray:
    """Initialize confusion matrices for each LF.

    Args:
        label_matrix: (n_examples, n_lfs) array of label indices.
        n_categories: Number of categories.
        n_lfs: Number of labeling functions.

    Returns:
        (n_lfs, n_categories, n_categories) array of confusion matrices.
        confusion[lf, i, j] = P(LF predicts j | true label is i)
    """
    confusion = np.zeros((n_lfs, n_categories, n_categories), dtype=np.float32)

    # Initialize with identity + small uniform noise
    for lf_idx in range(n_lfs):
        for i in range(n_categories):
            confusion[lf_idx, i, i] = 0.7  # Diagonal (correct)
            for j in range(n_categories):
                if i != j:
                    confusion[lf_idx, i, j] = 0.3 / (n_categories - 1)  # Off-diagonal

    return confusion


def _e_step(
    label_matrix: np.ndarray,
    confusion: np.ndarray,
    class_priors: np.ndarray,
) -> np.ndarray:
    """E-step: Compute P(true_label | votes).

    Args:
        label_matrix: (n_examples, n_lfs) array of label indices.
        confusion: (n_lfs, n_categories, n_categories) confusion matrices.
        class_priors: (n_categories,) prior probabilities for each category.

    Returns:
        (n_examples, n_categories) posterior probabilities.
    """
    n_examples, n_lfs = label_matrix.shape
    n_categories = len(class_priors)

    posteriors = np.zeros((n_examples, n_categories), dtype=np.float32)

    for i in range(n_examples):
        votes = label_matrix[i]

        # For each possible true label, compute likelihood
        for true_label in range(n_categories):
            log_prob = np.log(class_priors[true_label] + 1e-10)

            # Multiply likelihoods from each LF
            for lf_idx in range(n_lfs):
                predicted_label = votes[lf_idx]
                if predicted_label >= 0:  # Not abstain
                    prob = confusion[lf_idx, true_label, predicted_label]
                    log_prob += np.log(prob + 1e-10)

            posteriors[i, true_label] = np.exp(log_prob)

        # Normalize
        row_sum = posteriors[i].sum()
        if row_sum > 0:
            posteriors[i] /= row_sum
        else:
            posteriors[i] = 1.0 / n_categories  # Uniform if all zero

    return posteriors


def _m_step(
    label_matrix: np.ndarray,
    posteriors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """M-step: Update confusion matrices and class priors.

    Args:
        label_matrix: (n_examples, n_lfs) array of label indices.
        posteriors: (n_examples, n_categories) posterior probabilities.

    Returns:
        Tuple of (confusion, class_priors).
    """
    n_examples, n_lfs = label_matrix.shape
    n_categories = posteriors.shape[1]

    confusion = np.zeros((n_lfs, n_categories, n_categories), dtype=np.float32)

    # Update confusion matrices
    for lf_idx in range(n_lfs):
        for i in range(n_categories):  # True label
            for j in range(n_categories):  # Predicted label
                # Count expected occurrences
                count = 0.0
                for example_idx in range(n_examples):
                    if label_matrix[example_idx, lf_idx] == j:
                        count += posteriors[example_idx, i]
                confusion[lf_idx, i, j] = count

        # Normalize rows
        for i in range(n_categories):
            row_sum = confusion[lf_idx, i].sum()
            if row_sum > 0:
                confusion[lf_idx, i] /= row_sum
            else:
                # If no votes for this true label, use uniform
                confusion[lf_idx, i] = 1.0 / n_categories

    # Update class priors
    class_priors = posteriors.mean(axis=0)
    class_priors /= class_priors.sum()

    return confusion, class_priors


def _dawid_skene_em(
    label_matrix: np.ndarray,
    categories: list[str],
    max_iter: int = 20,
    tol: float = 1e-4,
) -> tuple[list[str], list[float]]:
    """Dawid-Skene EM algorithm for label aggregation.

    Args:
        label_matrix: (n_examples, n_lfs) array of label indices (-1 = abstain).
        categories: List of category names.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.

    Returns:
        Tuple of (labels, confidences) for each example.
    """
    n_examples, n_lfs = label_matrix.shape
    n_categories = len(categories)

    # Initialize
    class_priors = np.ones(n_categories, dtype=np.float32) / n_categories
    confusion = _initialize_confusion_matrices(label_matrix, n_categories, n_lfs)

    prev_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step
        posteriors = _e_step(label_matrix, confusion, class_priors)

        # M-step
        confusion, class_priors = _m_step(label_matrix, posteriors)

        # Check convergence (log-likelihood)
        log_likelihood = 0.0
        for i in range(n_examples):
            log_likelihood += np.log(posteriors[i].max() + 1e-10)

        if abs(log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = log_likelihood

    # Final predictions
    labels = []
    confidences = []

    for i in range(n_examples):
        winner_idx = int(posteriors[i].argmax())
        confidence = float(posteriors[i, winner_idx])

        labels.append(categories[winner_idx])
        confidences.append(confidence)

    return labels, confidences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_labels(
    examples: list[dict],
    registry: LabelingFunctionRegistry,
    method: str = "majority",
) -> tuple[list[str], list[float]]:
    """Aggregate labels from labeling functions.

    Args:
        examples: List of examples, each with keys: text, context, last_message, metadata.
        registry: LabelingFunctionRegistry with all LFs.
        method: "majority" or "dawid_skene".

    Returns:
        Tuple of (labels, confidences) for each example.
    """
    # Define category order
    categories = ["ack", "info", "emotional", "social", "clarify"]
    category_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Apply all LFs to all examples
    n_examples = len(examples)
    n_lfs = len(registry.lfs)

    label_matrix = np.full((n_examples, n_lfs), -1, dtype=np.int8)  # -1 = abstain

    for i, example in enumerate(examples):
        text = example["text"]
        context = example.get("context", [])
        last_message = example.get("last_message", "")
        metadata = example.get("metadata")

        lf_labels = registry.apply_all(text, context, last_message, metadata)

        for j, label in enumerate(lf_labels):
            if label != ABSTAIN:
                label_matrix[i, j] = category_to_idx[label]

    # Aggregate
    weights = registry.get_weights()

    if method == "majority":
        return _majority_vote(label_matrix, weights, categories)
    elif method == "dawid_skene":
        return _dawid_skene_em(label_matrix, categories)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


__all__ = ["aggregate_labels"]
