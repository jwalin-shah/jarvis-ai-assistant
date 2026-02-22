from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def softmax(x: Sequence[float]) -> list[float]:
    arr = np.array(x, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp)
    if denom <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (exp / denom).tolist()


def soft_best_of_n(
    candidates: list[str],
    scores: list[float],
    temperature: float = 2.0,
    min_threshold: float = 0.6,
) -> tuple[str, float, bool]:
    if not candidates:
        raise ValueError("No candidates provided")
    if len(candidates) != len(scores):
        raise ValueError("Candidates/scores length mismatch")

    valid = [(c, s) for c, s in zip(candidates, scores) if s >= min_threshold]
    fallback_used = False
    if not valid:
        idx = int(np.argmax(scores))
        return candidates[idx], scores[idx], True

    weights = softmax([score / temperature for _, score in valid])
    selected = random.choices(valid, weights=weights, k=1)[0]
    return selected[0], selected[1], fallback_used
