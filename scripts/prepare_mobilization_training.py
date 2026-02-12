#!/usr/bin/env python3
"""Prepare mobilization-only training data from Gemini labels.

Simple features only (no BERT for speed):
- Message length, word count, punctuation
- Question marks, exclamation marks
- Emojis, abbreviations
- Professional keywords
- etc.

Usage:
    uv run python scripts/prepare_mobilization_training.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "evals" / "data" / "pipeline_eval_labeled.jsonl"


def setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("prepare_mobilization_training.log")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=EVAL_PATH,
        help="Path to labeled evaluation JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "mobilization_gemini",
        help="Directory to write mobilization training artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data in test split (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_gemini_examples(eval_path: Path, logger: logging.Logger) -> list[dict]:
    """Load Gemini-labeled examples with mobilization labels.

    Uses both Gemini and auto-labeled examples for better class balance.
    (Gemini-only was too imbalanced: only 1 HIGH example)
    """
    examples = []
    try:
        with eval_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    # Use both Gemini and auto labels for better class distribution
                    if ex.get("mobilization") and ex.get("label_confidence") in ("gemini", "auto"):
                        examples.append(ex)
    except OSError as exc:
        logger.error("Failed to read eval data %s: %s", eval_path, exc)
        raise SystemExit(1) from exc

    logger.info(f"Loaded {len(examples)} Gemini+auto examples with mobilization")
    return examples


def extract_simple_features(text: str) -> np.ndarray:
    """Extract simple hand-crafted features (no embeddings)."""
    features = []

    # Basic structure
    features.append(float(len(text)))  # message length
    features.append(float(len(text.split())))  # word count
    features.append(float(text.count("?")))  # question marks
    features.append(float(text.count("!")))  # exclamation marks

    # Punctuation density
    punct_count = sum(1 for c in text if c in "?!.,;:")
    features.append(float(punct_count) / max(len(text), 1))

    # Interrogative markers
    text_lower = text.lower()
    wh_words = {"what", "where", "when", "who", "why", "how"}
    has_wh = 1.0 if any(text_lower.startswith(w) for w in wh_words) else 0.0
    features.append(has_wh)

    # Auxiliary verbs (can, could, will, would, etc.)
    aux_verbs = {"can", "could", "will", "would", "should", "do", "does", "did"}
    first_word = text_lower.split()[0] if text_lower.split() else ""
    has_aux = 1.0 if first_word in aux_verbs else 0.0
    features.append(has_aux)

    # Request patterns
    request_patterns = {"can you", "could you", "would you", "will you", "can i", "let me", "let's"}
    has_request = 1.0 if any(p in text_lower for p in request_patterns) else 0.0
    features.append(has_request)

    # Closing patterns
    closing_patterns = {"bye", "goodbye", "later", "see you", "talk soon", "gotta go", "ttyl"}
    has_closing = 1.0 if any(p in text_lower for p in closing_patterns) else 0.0
    features.append(has_closing)

    # Emotional markers
    emotional_words = {"lol", "lmao", "haha", "omg", "wow", "amazing", "horrible", "terrible"}
    emotional_count = sum(1 for word in text_lower.split() if word in emotional_words)
    features.append(float(emotional_count))

    # Capitalization
    caps_count = sum(1 for c in text if c.isupper())
    features.append(float(caps_count) / max(len(text), 1))

    # Emojis (basic count)
    emoji_count = len([c for c in text if ord(c) > 127])
    features.append(float(emoji_count))

    # Short messages (acks, backchannels)
    is_short = 1.0 if len(text.split()) <= 3 else 0.0
    features.append(is_short)

    # Negations
    negations = {"no", "not", "don't", "didn't", "won't", "can't"}
    has_negation = 1.0 if any(n in text_lower for n in negations) else 0.0
    features.append(has_negation)

    return np.array(features, dtype=np.float32)


def extract_all_features(
    examples: list[dict], logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features for all examples."""
    from tqdm import tqdm

    logger.info(f"Extracting features for {len(examples)} examples...")

    features_list = []
    labels = []
    ids = []

    for i, ex in enumerate(tqdm(examples, desc="Extracting features", unit="ex")):
        features = extract_simple_features(ex["text"])
        features_list.append(features)
        labels.append(ex["mobilization"].upper())
        ids.append(ex["id"])

        if (i + 1) % 500 == 0:
            logger.info(f"  {i + 1}/{len(examples)} features extracted")

    X = np.array(features_list)  # noqa: N806
    y = np.array(labels)

    logger.info(f"\nExtracted {X.shape[0]} examples with {X.shape[1]} features")
    logger.info(f"Feature dimensions: {X.shape[1]}")

    # Label distribution
    from collections import Counter

    label_dist = Counter(y)
    logger.info("Label distribution:")
    for label, count in sorted(label_dist.items()):
        logger.info(f"  {label:10s}: {count:4d}")

    return X, y, ids


def create_splits(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    ids: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> tuple:
    """Create stratified train/test split."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(  # noqa: N806
        X, y, ids, test_size=test_size, stratify=y, random_state=seed
    )

    if logger:
        logger.info("\nTrain/test split:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Test: {X_test.shape}")

        from collections import Counter

        logger.info("\nTrain distribution:")
        for label, count in sorted(Counter(y_train).items()):
            logger.info(f"  {label:10s}: {count:4d}")

        logger.info("\nTest distribution:")
        for label, count in sorted(Counter(y_test).items()):
            logger.info(f"  {label:10s}: {count:4d}")

    return X_train, X_test, y_train, y_test, ids_train, ids_test


def save_training_data(
    output_dir: Path,
    X_train: np.ndarray,  # noqa: N803
    X_test: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> None:
    """Save training data."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directory %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    try:
        np.savez(output_dir / "train.npz", X=X_train, y=y_train)
        np.savez(output_dir / "test.npz", X=X_test, y=y_test)
    except OSError as exc:
        logger.error("Failed to write NPZ training data in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    labels = sorted(set(y_train) | set(y_test))
    metadata = {
        "source": "gemini_mobilization",
        "labels": labels,
        "label_map": {label: i for i, label in enumerate(labels)},
        "feature_dims": int(X_train.shape[1]),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": [
            "message_length",
            "word_count",
            "question_marks",
            "exclamation_marks",
            "punct_density",
            "has_wh",
            "has_aux",
            "has_request",
            "has_closing",
            "emotional_count",
            "caps_density",
            "emoji_count",
            "is_short",
            "has_negation",
        ],
    }

    try:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    except OSError as exc:
        logger.error("Failed to write metadata in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    logger.info(f"\n✓ Training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")
    logger.info("  - metadata.json")


def main(argv: Sequence[str] | None = None) -> None:
    logger = setup_logging()
    args = parse_args(argv)
    examples = load_gemini_examples(args.eval_path, logger)
    X, y, ids = extract_all_features(examples, logger)  # noqa: N806
    X_train, X_test, y_train, y_test, ids_train, ids_test = create_splits(  # noqa: N806
        X,
        y,
        ids,
        test_size=args.test_size,
        seed=args.seed,
        logger=logger,
    )

    save_training_data(args.output_dir, X_train, X_test, y_train, y_test, logger)

    logger.info("\n" + "=" * 70)
    logger.info("READY TO TRAIN")
    logger.info("=" * 70)
    logger.info("Next:")
    logger.info("  1. uv run python scripts/train_mobilization_logistic.py")
    logger.info("  2. uv run python scripts/train_mobilization_lightgbm.py")
    logger.info("  3. Compare results")


if __name__ == "__main__":
    main()
