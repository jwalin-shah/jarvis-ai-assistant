"""
Adaptive threshold logic for single vs multi-label prediction.

Predicts label count based on:
1. Model confidence (high confidence → single label)
2. Probability gap (large gap → single label)
3. Message features (short/simple → single label)
"""

from __future__ import annotations

import numpy as np


def predict_with_adaptive_thresholds(
    probabilities: np.ndarray,
    text: str | None = None,
    categories: list[str] | None = None,
) -> list[str]:
    """
    Predict labels using adaptive thresholds based on confidence.

    Args:
        probabilities: Array of class probabilities [0-1]
        text: Optional message text for feature-based adjustment
        categories: List of category names

    Returns:
        List of predicted categories
    """
    if categories is None:
        categories = ["acknowledge", "closing", "emotion", "question", "request", "statement"]

    # Sort probabilities
    sorted_probs = sorted(probabilities, reverse=True)
    top_prob = sorted_probs[0]
    second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
    confidence_gap = top_prob - second_prob

    # Feature-based adjustment (if text provided)
    likely_single_label = False
    if text:
        words = text.split()
        num_words = len(words)
        has_conjunction = any(word in text.lower() for word in ['but', 'and', 'also', 'though', 'however'])
        num_punctuation_types = sum([
            '?' in text,
            '!' in text,
            '.' in text and not text.strip().endswith('...'),
        ])

        # Indicators of single-label message
        if num_words < 5:  # Very short
            likely_single_label = True
        elif num_words < 10 and not has_conjunction and num_punctuation_types <= 1:
            likely_single_label = True

    # Decision logic
    if likely_single_label or (top_prob > 0.7 and confidence_gap > 0.3):
        # VERY confident → single label
        threshold = 0.7
        predicted = [cat for cat, prob in zip(categories, probabilities) if prob >= threshold]
        if len(predicted) == 0:  # Fallback
            predicted = [categories[np.argmax(probabilities)]]

    elif top_prob > 0.55 and second_prob > 0.35 and confidence_gap < 0.25:
        # Two strong candidates → likely multi-label
        threshold = 0.35
        predicted = [cat for cat, prob in zip(categories, probabilities) if prob >= threshold]

    elif top_prob < 0.45:
        # Very uncertain → allow multiple labels
        threshold = 0.3
        predicted = [cat for cat, prob in zip(categories, probabilities) if prob >= threshold]

    else:
        # Default: moderate confidence
        threshold = 0.45
        predicted = [cat for cat, prob in zip(categories, probabilities) if prob >= threshold]

    # Always return at least one label
    if len(predicted) == 0:
        predicted = [categories[np.argmax(probabilities)]]

    return predicted


def evaluate_adaptive_strategy(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    texts: list[str],
    mlb,
) -> dict:
    """
    Evaluate adaptive threshold strategy.

    Args:
        y_proba: Probability predictions (N, num_classes)
        y_true: True labels (N, num_classes)
        texts: Message texts
        mlb: MultiLabelBinarizer

    Returns:
        Metrics dictionary
    """
    from sklearn.metrics import f1_score, hamming_loss

    # Predict with adaptive thresholds
    y_pred = np.zeros_like(y_true)
    for i, (probs, text) in enumerate(zip(y_proba, texts)):
        predicted_cats = predict_with_adaptive_thresholds(probs, text, mlb.classes_.tolist())
        for cat in predicted_cats:
            cat_idx = list(mlb.classes_).index(cat)
            y_pred[i, cat_idx] = 1

    # Metrics
    f1_samples = f1_score(y_true, y_pred, average="samples")
    hamming = hamming_loss(y_true, y_pred)

    # Count predictions per example
    pred_counts = y_pred.sum(axis=1)
    true_counts = y_true.sum(axis=1)

    return {
        "f1_samples": f1_samples,
        "hamming_loss": hamming,
        "avg_predicted_labels": pred_counts.mean(),
        "avg_true_labels": true_counts.mean(),
        "single_label_accuracy": (pred_counts == 1).sum() / len(pred_counts),
    }
