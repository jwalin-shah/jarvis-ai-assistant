#!/usr/bin/env python3
"""Phase 1: Prepare data for classifier optimization experiments.

This script:
1. Loads human-labeled data from data/response_labeling.jsonl
2. Creates stratified 80/20 split:
   - test_human.jsonl (20%, ~973 examples) - GOLD STANDARD, locked
   - train_seed.jsonl (80%, ~3892 examples) - Base for training
3. Auto-labels unlabeled responses from the database
4. Filters to high-confidence (>=90%) auto-labels
5. Computes and caches embeddings for train and test sets SEPARATELY

Usage:
    uv run python -m experiments.scripts.prepare_data
    uv run python -m experiments.scripts.prepare_data --confidence-threshold 0.85
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from experiments.scripts.utils import (
    DATA_DIR,
    LabeledExample,
    get_label_distribution,
    load_labeled_data,
    save_labeled_data,
    stratified_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_human_labeled_data() -> list[LabeledExample]:
    """Load human-labeled data from the main labeling file."""
    input_path = Path("data/response_labeling.jsonl")
    if not input_path.exists():
        logger.error("Human-labeled data not found at %s", input_path)
        sys.exit(1)

    examples = load_labeled_data(input_path)
    logger.info("Loaded %d human-labeled examples", len(examples))
    logger.info("Distribution: %s", get_label_distribution([e.label for e in examples]))
    return examples


def auto_label_with_embeddings(
    confidence_threshold: float = 0.80,
    minority_threshold: float = 0.80,
    batch_size: int = 512,
) -> tuple[list[LabeledExample], list[str], np.ndarray]:
    """Auto-label unlabeled responses, returning embeddings for reuse.

    This is more efficient than auto_label_from_db because it:
    1. Computes embeddings ONCE for all candidates
    2. Uses the SVM directly with pre-computed embeddings
    3. Returns embeddings so they don't need to be recomputed

    Uses class-specific thresholds: lower for minority classes (AGREE, DECLINE, DEFER)
    to capture more examples of rare classes.

    Args:
        confidence_threshold: Minimum confidence for majority classes (default 0.90)
        minority_threshold: Minimum confidence for minority classes (default 0.50)
        batch_size: Batch size for embedding computation

    Returns:
        Tuple of (high_conf_examples, all_texts, all_embeddings)
        - high_conf_examples: List of auto-labeled examples passing threshold
        - all_texts: All candidate texts (for index alignment)
        - all_embeddings: Embeddings for all candidates
    """
    MINORITY_CLASSES = {"AGREE", "DECLINE", "DEFER"}
    from jarvis.classifiers.response_classifier import get_response_classifier

    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder

    logger.info("Loading pairs from database...")
    db = get_db()
    pairs = db.get_all_pairs()
    logger.info("Loaded %d pairs from database", len(pairs))

    # Extract unique response texts
    response_texts = list(set(p.response_text for p in pairs if p.response_text))
    logger.info("Found %d unique response texts", len(response_texts))

    # Load existing human-labeled to exclude
    human_path = Path("data/response_labeling.jsonl")
    if human_path.exists():
        human_examples = load_labeled_data(human_path)
        human_texts = {e.text for e in human_examples}
        response_texts = [t for t in response_texts if t not in human_texts]
        logger.info("After excluding human-labeled: %d texts", len(response_texts))

    if not response_texts:
        logger.warning("No texts to auto-label!")
        return [], [], np.array([])

    # Step 1: Compute embeddings ONCE for all candidates
    logger.info(
        "Computing embeddings for %d candidates (batch_size=%d)...", len(response_texts), batch_size
    )
    embedder = get_embedder()

    all_embeddings = []
    for i in range(0, len(response_texts), batch_size):
        batch_end = min(i + batch_size, len(response_texts))
        batch_texts = response_texts[i:batch_end]
        batch_emb = embedder.encode(batch_texts, normalize=True)
        all_embeddings.append(batch_emb)

        if len(response_texts) > 5000 and (i + batch_size) % 5000 == 0:
            logger.info(
                "  Embedded %d/%d texts (%.1f%%)",
                batch_end,
                len(response_texts),
                100 * batch_end / len(response_texts),
            )

    embeddings = np.vstack(all_embeddings)
    logger.info("Computed embeddings: shape=%s", embeddings.shape)

    # Step 2: Classify using pre-computed embeddings
    logger.info("Classifying with pre-computed embeddings...")
    classifier = get_response_classifier()

    # Use the SVM directly with pre-computed embeddings (much faster)
    if classifier.svm is not None and classifier._svm_labels:
        logger.info("Using SVM with pre-computed embeddings (fast path)")

        # Batch predict with SVM
        probs = classifier.svm.predict_proba(embeddings)
        pred_indices = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(probs)), pred_indices]

        high_conf_examples = []
        high_conf_indices = []

        for i, (text, pred_idx, conf) in enumerate(zip(response_texts, pred_indices, confidences)):
            # Use raw SVM label (6-label scheme: AGREE, DECLINE, DEFER, OTHER, QUESTION, REACTION)
            label = classifier._svm_labels[pred_idx]

            # Class-specific threshold: lower for minority classes
            threshold = minority_threshold if label in MINORITY_CLASSES else confidence_threshold

            if conf >= threshold:
                high_conf_examples.append(
                    LabeledExample(
                        text=text,
                        label=label,
                        source="auto",
                        confidence=float(conf),
                    )
                )
                high_conf_indices.append(i)
    else:
        # Fallback: use classify_batch (slower, recomputes embeddings)
        logger.warning("SVM not available, falling back to classify_batch")
        results = classifier.classify_batch(response_texts, batch_size=batch_size)

        # Map ResponseType back to 6-label scheme for consistency
        RESPONSE_TO_SVM = {
            "REACT_POSITIVE": "REACTION",
            "REACT_SYMPATHY": "REACTION",
            "ANSWER": "OTHER",
            "ACKNOWLEDGE": "OTHER",
            "GREETING": "OTHER",
            "STATEMENT": "OTHER",
        }

        high_conf_examples = []
        high_conf_indices = []

        for i, (text, result) in enumerate(zip(response_texts, results)):
            label = result.label.value
            label = RESPONSE_TO_SVM.get(label, label)  # Normalize to 6-label

            # Class-specific threshold: lower for minority classes
            threshold = minority_threshold if label in MINORITY_CLASSES else confidence_threshold

            if result.confidence >= threshold:
                high_conf_examples.append(
                    LabeledExample(
                        text=text,
                        label=label,
                        source="auto",
                        confidence=result.confidence,
                    )
                )
                high_conf_indices.append(i)

    logger.info(
        "Found %d examples (majority >= %.0f%%, minority >= %.0f%%)",
        len(high_conf_examples),
        confidence_threshold * 100,
        minority_threshold * 100,
    )
    auto_dist = get_label_distribution([e.label for e in high_conf_examples])
    logger.info("Auto-label distribution: %s", auto_dist)

    return high_conf_examples, response_texts, embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare data for experiments")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help="Minimum confidence for majority classes (default: 0.80)",
    )
    parser.add_argument(
        "--minority-threshold",
        type=float,
        default=0.80,
        help="Minimum confidence for minority classes AGREE/DECLINE/DEFER (default: 0.80)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Fraction of human-labeled data for test set (default: 0.20)",
    )
    parser.add_argument(
        "--skip-auto-label",
        action="store_true",
        help="Skip auto-labeling step (use existing auto_labeled file)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation (use existing cache)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for embedding computation (default: 512)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load and split human-labeled data
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: Split human-labeled data")
    logger.info("=" * 70)

    human_examples = load_human_labeled_data()

    train_seed, test_human = stratified_split(
        human_examples,
        test_ratio=args.test_ratio,
        seed=42,
    )

    logger.info("Train seed: %d examples", len(train_seed))
    logger.info("Train distribution: %s", get_label_distribution([e.label for e in train_seed]))
    logger.info("Test set: %d examples (LOCKED)", len(test_human))
    logger.info("Test distribution: %s", get_label_distribution([e.label for e in test_human]))

    # Save splits
    save_labeled_data(train_seed, DATA_DIR / "train_seed.jsonl")
    save_labeled_data(test_human, DATA_DIR / "test_human.jsonl")

    # =========================================================================
    # Step 2: Auto-label unlabeled responses (with embedding reuse)
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Auto-label unlabeled responses")
    logger.info("=" * 70)

    auto_labeled_path = DATA_DIR / "auto_labeled_90pct.jsonl"
    train_cache_path = DATA_DIR / "embeddings_cache.npz"
    auto_embeddings_cache = DATA_DIR / "auto_embeddings_cache.npz"

    if args.skip_auto_label and auto_labeled_path.exists():
        logger.info("Skipping auto-labeling, loading existing file...")
        auto_labeled = load_labeled_data(auto_labeled_path)
        logger.info("Loaded %d auto-labeled examples", len(auto_labeled))
        auto_texts = None
        auto_embeddings = None
    else:
        # This computes embeddings once and reuses them for classification
        auto_labeled, auto_texts, auto_embeddings = auto_label_with_embeddings(
            confidence_threshold=args.confidence_threshold,
            minority_threshold=args.minority_threshold,
            batch_size=args.batch_size,
        )
        save_labeled_data(auto_labeled, auto_labeled_path)

        # Cache auto embeddings for potential reuse
        if auto_embeddings is not None and len(auto_embeddings) > 0:
            np.savez_compressed(auto_embeddings_cache, embeddings=auto_embeddings, texts=auto_texts)
            logger.info("Cached auto embeddings to %s", auto_embeddings_cache)

    # =========================================================================
    # Step 3: Compute/assemble final training embeddings
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Compute embeddings")
    logger.info("=" * 70)

    if args.skip_embeddings and train_cache_path.exists():
        logger.info("Skipping train embeddings, using cache...")
    else:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()

        # Compute embeddings for train_seed (human-labeled)
        train_seed_texts = [e.text for e in train_seed]
        logger.info("Computing embeddings for %d train_seed texts...", len(train_seed_texts))
        train_seed_embeddings = embedder.encode(train_seed_texts, normalize=True)

        # Get embeddings for auto_labeled (reuse if available)
        auto_labeled_texts = [e.text for e in auto_labeled]

        if auto_texts is not None and auto_embeddings is not None:
            # Reuse embeddings from auto-labeling step
            logger.info("Reusing embeddings for %d auto-labeled texts...", len(auto_labeled_texts))

            # Build text->index mapping for fast lookup
            text_to_idx = {t: i for i, t in enumerate(auto_texts)}

            # Extract embeddings for high-confidence examples
            auto_labeled_emb_list = []
            missing_count = 0
            for ex in auto_labeled:
                idx = text_to_idx.get(ex.text)
                if idx is not None:
                    auto_labeled_emb_list.append(auto_embeddings[idx])
                else:
                    # Shouldn't happen, but compute if missing
                    missing_count += 1
                    emb = embedder.encode([ex.text], normalize=True)[0]
                    auto_labeled_emb_list.append(emb)

            if missing_count > 0:
                logger.warning("Had to recompute %d missing embeddings", missing_count)

            auto_labeled_embeddings = np.array(auto_labeled_emb_list)
        else:
            # Load from cache or compute fresh
            if auto_embeddings_cache.exists():
                logger.info("Loading auto embeddings from cache...")
                cache_data = np.load(auto_embeddings_cache, allow_pickle=True)
                cached_embeddings = cache_data["embeddings"]
                cached_texts = cache_data["texts"].tolist()

                text_to_idx = {t: i for i, t in enumerate(cached_texts)}
                auto_labeled_emb_list = []
                for ex in auto_labeled:
                    idx = text_to_idx.get(ex.text)
                    if idx is not None:
                        auto_labeled_emb_list.append(cached_embeddings[idx])
                    else:
                        emb = embedder.encode([ex.text], normalize=True)[0]
                        auto_labeled_emb_list.append(emb)
                auto_labeled_embeddings = np.array(auto_labeled_emb_list)
            else:
                logger.info(
                    "Computing embeddings for %d auto-labeled texts...", len(auto_labeled_texts)
                )
                auto_labeled_embeddings = embedder.encode(auto_labeled_texts, normalize=True)

        # Combine train_seed + auto_labeled embeddings
        train_embeddings = np.vstack([train_seed_embeddings, auto_labeled_embeddings])
        logger.info("Combined train embeddings: shape=%s", train_embeddings.shape)

        # Save with metadata
        np.savez_compressed(
            train_cache_path,
            embeddings=train_embeddings,
            n_seed=len(train_seed),
            n_auto=len(auto_labeled),
        )
        logger.info("Saved train embeddings to %s", train_cache_path)

    # Test embeddings (SEPARATE from train)
    test_cache_path = DATA_DIR / "test_embeddings.npz"

    if args.skip_embeddings and test_cache_path.exists():
        logger.info("Skipping test embeddings, using cache...")
    else:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()

        test_texts = [e.text for e in test_human]
        logger.info("Computing embeddings for %d test texts...", len(test_texts))
        test_embeddings = embedder.encode(test_texts, normalize=True)
        np.savez_compressed(test_cache_path, embeddings=test_embeddings)
        logger.info("Saved test embeddings to %s", test_cache_path)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Human-labeled total: %d", len(human_examples))
    logger.info("  - Train seed: %d (80%%)", len(train_seed))
    logger.info("  - Test set: %d (20%%, LOCKED)", len(test_human))
    logger.info(
        "Auto-labeled (majority >= %.0f%%, minority >= %.0f%%): %d",
        args.confidence_threshold * 100,
        args.minority_threshold * 100,
        len(auto_labeled),
    )
    logger.info("")
    logger.info("Files created:")
    logger.info("  - %s", DATA_DIR / "train_seed.jsonl")
    logger.info("  - %s", DATA_DIR / "test_human.jsonl")
    logger.info("  - %s", auto_labeled_path)
    logger.info("  - %s", train_cache_path)
    logger.info("  - %s", test_cache_path)
    logger.info("")
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
