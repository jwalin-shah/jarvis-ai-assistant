#!/usr/bin/env python3
"""Prepare trigger classifier data for optimization experiments.

This script:
1. Loads human-labeled data from data/trigger_labeling.jsonl
2. Creates stratified 80/20 split (train_seed + test_human)
3. Loads auto-labeled data from data/trigger_new_batch_3000.jsonl
4. Computes and caches embeddings

Usage:
    uv run python -m experiments.trigger.prepare_data
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PROJECT_DATA = Path("data")

# Trigger labels (5 classes)
LABELS = ["commitment", "question", "reaction", "social", "statement"]


@dataclass
class LabeledExample:
    text: str
    label: str
    source: str = "human"  # "human" or "auto"
    confidence: float = 1.0


def load_human_labeled() -> list[LabeledExample]:
    """Load human-labeled trigger data."""
    path = PROJECT_DATA / "trigger_labeling.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                examples.append(
                    LabeledExample(
                        text=d["text"],
                        label=d["label"].lower(),
                        source="human",
                    )
                )
    return examples


def load_auto_labeled(min_confidence: float = 0.0) -> list[LabeledExample]:
    """Load auto-labeled trigger data."""
    path = PROJECT_DATA / "trigger_new_batch_3000.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                conf = d.get("confidence", 1.0)
                if conf >= min_confidence:
                    examples.append(
                        LabeledExample(
                            text=d["text"],
                            label=d["auto_label"].lower(),
                            source="auto",
                            confidence=conf,
                        )
                    )
    return examples


def get_distribution(examples: list[LabeledExample]) -> dict[str, int]:
    """Get label distribution."""
    return dict(Counter(e.label for e in examples))


def stratified_split(
    examples: list[LabeledExample],
    test_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Stratified train/test split."""
    texts = [e.text for e in examples]
    labels = [e.label for e in examples]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Reconstruct examples
    text_to_example = {e.text: e for e in examples}
    train = [text_to_example[t] for t in train_texts]
    test = [text_to_example[t] for t in test_texts]

    return train, test


def save_examples(examples: list[LabeledExample], path: Path) -> None:
    """Save examples to JSONL."""
    with open(path, "w") as f:
        for e in examples:
            json.dump(
                {
                    "text": e.text,
                    "label": e.label,
                    "source": e.source,
                    "confidence": e.confidence,
                },
                f,
            )
            f.write("\n")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and split human-labeled data
    logger.info("=" * 70)
    logger.info("STEP 1: Load and split human-labeled data")
    logger.info("=" * 70)

    human = load_human_labeled()
    logger.info("Loaded %d human-labeled examples", len(human))
    logger.info("Distribution: %s", get_distribution(human))

    train_seed, test_human = stratified_split(human, test_ratio=0.20, seed=42)
    logger.info("Train seed: %d", len(train_seed))
    logger.info("Test (LOCKED): %d", len(test_human))

    save_examples(train_seed, DATA_DIR / "train_seed.jsonl")
    save_examples(test_human, DATA_DIR / "test_human.jsonl")

    # Step 2: Load auto-labeled data
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Load auto-labeled data")
    logger.info("=" * 70)

    auto = load_auto_labeled(min_confidence=0.0)
    logger.info("Loaded %d auto-labeled examples", len(auto))
    logger.info("Distribution: %s", get_distribution(auto))

    save_examples(auto, DATA_DIR / "auto_labeled.jsonl")

    # Step 3: Compute embeddings
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Compute embeddings")
    logger.info("=" * 70)

    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()

    # Train embeddings (seed + auto)
    all_train = train_seed + auto
    train_texts = [e.text for e in all_train]
    logger.info("Computing embeddings for %d training texts...", len(train_texts))
    train_embeddings = embedder.encode(train_texts, normalize=True)

    np.savez_compressed(
        DATA_DIR / "embeddings_cache.npz",
        embeddings=train_embeddings,
        n_seed=len(train_seed),
        n_auto=len(auto),
    )
    logger.info("Saved train embeddings: shape=%s", train_embeddings.shape)

    # Test embeddings (separate)
    test_texts = [e.text for e in test_human]
    logger.info("Computing embeddings for %d test texts...", len(test_texts))
    test_embeddings = embedder.encode(test_texts, normalize=True)

    np.savez_compressed(
        DATA_DIR / "test_embeddings.npz",
        embeddings=test_embeddings,
    )
    logger.info("Saved test embeddings: shape=%s", test_embeddings.shape)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Train seed: %d", len(train_seed))
    logger.info("Auto-labeled: %d", len(auto))
    logger.info("Test (LOCKED): %d", len(test_human))
    logger.info("Total available for training: %d", len(train_seed) + len(auto))


if __name__ == "__main__":
    main()
