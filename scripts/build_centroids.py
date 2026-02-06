#!/usr/bin/env python3
"""Centroid Generation and Pattern Mining.

Computes semantic centroids for response and trigger classifiers using
the configured embedding model. These centroids are used for:
1. Verifying structural pattern matches (high precision)
2. Fallback classification when patterns don't match (high coverage)

This replaces the old SVM training pipeline with a lightweight,
embedding-native approach that aligns with the LFM architecture.

Usage:
    uv run python -m scripts.train_all_classifiers
    uv run python -m scripts.train_all_classifiers --models bge-small
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jarvis.config import (
    get_config,
    get_response_classifier_path,
    get_trigger_classifier_path,
    reset_config,
    save_config,
)
from jarvis.embedding_adapter import (
    EMBEDDING_MODEL_REGISTRY,
    reset_embedder,
)
from jarvis.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class CentroidResult:
    """Result from centroid computation."""

    embedding_model: str
    classifier_type: str  # "trigger" or "response"
    dataset_name: str
    num_classes: int
    num_samples: int
    centroids: dict[str, list[float]]


@dataclass
class DatasetConfig:
    """Configuration for dataset."""

    name: str
    files: list[str]
    min_confidence: float
    description: str


# Dataset configurations
RESPONSE_DATASET_CONFIGS = [
    DatasetConfig(
        name="standard",
        files=[
            "data/response_labeling.jsonl",
            "experiments/data/auto_labeled_90pct.jsonl",
        ],
        min_confidence=0.85,
        description="Standard training set (Human + High Conf Auto)",
    ),
]

TRIGGER_DATASET_CONFIGS = [
    DatasetConfig(
        name="standard",
        files=[
            "data/trigger_labeling.jsonl",
            "data/trigger_auto_labeled.jsonl",
            "data/trigger_commitment_corrected.jsonl",
        ],
        min_confidence=0.80,
        description="Standard training set (Human + High Conf Auto)",
    ),
]


def load_data_from_files(
    files: list[str],
    field_name: str = "text",
    label_field: str = "label",
    min_confidence: float = 0.0,
    normalization: bool = True,
) -> tuple[list[str], list[str]]:
    """Load labeled data from multiple files.

    Args:
        files: List of JSONL file paths.
        field_name: JSON field for text ("text" or "response").
        label_field: JSON field for label.
        min_confidence: Minimum confidence for auto-labeled data.
        normalization: Whether to normalize text.
    """
    texts = []
    labels = []
    seen = set()

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"    Warning: {file_path} not found, skipping", flush=True)
            continue

        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                # Handle field variations
                text = row.get(field_name) or row.get("text") or row.get("response")
                if not text:
                    continue

                text = text.strip()
                label = row.get(label_field) or row.get("auto_label")
                confidence = row.get("confidence")

                if text and label:
                    # Filter by confidence
                    if confidence is not None and confidence < min_confidence:
                        continue

                    # Normalize
                    if normalization:
                        text = normalize_text(text)
                        if not text:
                            continue

                    # Deduplicate
                    if text in seen:
                        continue
                    seen.add(text)

                    texts.append(text)
                    labels.append(label)
                    count += 1

        print(f"    Loaded {count} samples from {path.name}", flush=True)

    return texts, labels


def compute_centroids(
    texts: list[str],
    labels: list[str],
    embedding_model: str,
) -> dict[str, list[float]]:
    """Compute centroids for each class using the embedding model."""
    from jarvis.embedding_adapter import get_embedder

    print(f"  Computing embeddings for {len(texts)} texts...", flush=True)

    # Get embedder
    embedder = get_embedder()

    # Compute embeddings in one batch
    embeddings = embedder.encode(texts, normalize=True)

    # Group by label
    label_to_embeddings: dict[str, list[np.ndarray]] = {}
    for i, label in enumerate(labels):
        if label not in label_to_embeddings:
            label_to_embeddings[label] = []
        label_to_embeddings[label].append(embeddings[i])

    # Compute centroids
    centroids = {}
    for label, vectors in label_to_embeddings.items():
        # Mean vector
        mean_vec = np.mean(vectors, axis=0)
        # L2 Normalize
        normalized = mean_vec / np.linalg.norm(mean_vec)
        centroids[label] = normalized.tolist()

    return centroids


def save_centroids(
    centroids: dict[str, list[float]],
    classifier_type: str,
    output_dir: Path | None = None,
) -> None:
    """Save centroids to disk."""
    if output_dir:
        save_path = output_dir
    else:
        if classifier_type == "trigger":
            save_path = get_trigger_classifier_path()
        else:
            save_path = get_response_classifier_path()

    save_path.mkdir(parents=True, exist_ok=True)

    # Save as numpy archive (npz) without pickle for security
    # Convert lists to arrays for npz format
    centroids_arrays = {label: np.array(centroid) for label, centroid in centroids.items()}
    np.savez(save_path / "centroids.npz", **centroids_arrays)  # type: ignore[arg-type]

    # Also save as JSON for inspection
    with open(save_path / "centroids.json", "w") as f:
        json.dump(centroids, f, indent=2)

    print(f"  Saved {len(centroids)} centroids to {save_path}", flush=True)


def process_embedding_model(
    embedding_model: str,
    classifier_types: list[str],
) -> None:
    """Process a single embedding model."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL: {embedding_model}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Set config
    jarvis_config = get_config()
    original_model = jarvis_config.embedding.model_name
    jarvis_config.embedding.model_name = embedding_model
    save_config(jarvis_config)
    reset_config()
    reset_embedder()

    try:
        # Trigger Classifiers
        if "trigger" in classifier_types:
            print("\n--- Processing Trigger Centroids ---", flush=True)
            config = TRIGGER_DATASET_CONFIGS[0]  # Use standard set

            texts, labels = load_data_from_files(
                config.files, label_field="label", min_confidence=config.min_confidence
            )

            # Normalize labels (lowercase for triggers)
            labels = [l.lower() for l in labels]
            print(f"  Distribution: {dict(Counter(labels))}")

            centroids = compute_centroids(texts, labels, embedding_model)
            save_centroids(centroids, "trigger")

        # Response Classifiers
        if "response" in classifier_types:
            print("\n--- Processing Response Centroids ---", flush=True)
            config = RESPONSE_DATASET_CONFIGS[0]

            texts, labels = load_data_from_files(
                config.files,
                field_name="response",
                label_field="label",
                min_confidence=config.min_confidence,
            )

            # Normalize labels (uppercase for responses)
            labels = [l.upper() for l in labels]
            print(f"  Distribution: {dict(Counter(labels))}")

            centroids = compute_centroids(texts, labels, embedding_model)
            save_centroids(centroids, "response")

    finally:
        # Restore config
        jarvis_config = get_config()
        jarvis_config.embedding.model_name = original_model
        save_config(jarvis_config)
        reset_config()
        reset_embedder()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate centroids for classifiers")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(EMBEDDING_MODEL_REGISTRY.keys()),
        choices=list(EMBEDDING_MODEL_REGISTRY.keys()),
        help="Embedding models to process",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["trigger", "response"],
        choices=["trigger", "response"],
        help="Classifier types to process",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CENTROID GENERATION")
    print("=" * 60)

    for model in args.models:
        process_embedding_model(model, args.classifiers)


if __name__ == "__main__":
    main()
