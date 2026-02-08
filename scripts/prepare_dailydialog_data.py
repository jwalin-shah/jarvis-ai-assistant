#!/usr/bin/env python3
"""Prepare DailyDialog data with native dialog act labels.

Loads DailyDialog from HuggingFace (OpenRL/daily_dialog parquet mirror),
extracts per-utterance features (384-dim embeddings + 19 hand-crafted + 14 SpaCy),
and saves train/test splits with native dialog act labels.

Native labels (ISO 24617-2 standard):
- inform: declarative statements, sharing information
- question: interrogative utterances, seeking information
- directive: commands, requests, suggestions requiring action
- commissive: commitments, promises, offers

Features (417 total):
- 384: Normalized BERT embeddings (text normalized with expand_slang)
- 19: Hand-crafted (length, punctuation, mobilization, context)
- 14: SpaCy syntactic (imperatives, modals, person, POS patterns)

Requirements:
- NER service must be running: uv run python scripts/ner_server.py
  (provides SpaCy POS tagging + syntactic features)

Output: data/dailydialog_native/{train,test}.npz, metadata.json

Usage:
    uv run python scripts/prepare_dailydialog_data.py --dry-run   # preview distribution
    uv run python scripts/prepare_dailydialog_data.py              # full run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def log_memory(label: str) -> None:
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    vms_mb = mem_info.vms / 1024 / 1024

    # Also log system-wide memory
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(
        f"[MEMORY] {label}: "
        f"RSS={rss_mb:.1f} MB, VMS={vms_mb:.1f} MB  "
        f"| System: {vm.percent:.1f}% used, swap={swap.used/1024**3:.2f} GB"
    )

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# DailyDialog dialog act labels
DIALOG_ACT_MAP = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "dailydialog_native"

# ---------------------------------------------------------------------------
# Hand-crafted feature extraction (matches category_classifier.py)
# ---------------------------------------------------------------------------

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)


def extract_hand_crafted_features(
    text: str,
    context_messages: list[str],
    mobilization_pressure: str,
    mobilization_type: str,
) -> np.ndarray:
    """Extract 19 hand-crafted features from a message + context."""
    features: list[float] = []

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Response mobilization one-hots (7)
    for level in ("high", "medium", "low", "none"):
        features.append(1.0 if mobilization_pressure == level else 0.0)
    for rtype in ("commitment", "answer", "emotional"):
        features.append(1.0 if mobilization_type == rtype else 0.0)

    # Tone flags (2)
    features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    # Context features (3)
    features.append(float(len(context_messages)))
    avg_ctx_len = (
        float(np.mean([len(m) for m in context_messages])) if context_messages else 0.0
    )
    features.append(avg_ctx_len)
    features.append(1.0 if len(context_messages) == 0 else 0.0)

    # Style features (2)
    words = text.split()
    total_words = len(words)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# DailyDialog loading
# ---------------------------------------------------------------------------


def load_dailydialog() -> list[dict]:
    """Load DailyDialog and extract per-utterance examples with native labels.

    Returns list of dicts with keys: text, last_message, label, context.
    """
    from datasets import load_dataset

    print("Loading DailyDialog...")
    # Use OpenRL parquet mirror (original li2017dailydialog repo requires
    # deprecated dataset scripts, incompatible with datasets>=4.0)
    ds = load_dataset("OpenRL/daily_dialog", split="train")
    print(f"  {len(ds)} dialogues loaded")

    examples: list[dict] = []

    for dialogue in ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]

        if len(utterances) < 2:
            continue

        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            act = acts[i]
            # Map to string label
            label = DIALOG_ACT_MAP.get(act, "inform")  # default to inform if unknown

            # Context: up to 5 previous utterances
            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            examples.append({
                "text": text,
                "last_message": last_msg,
                "label": label,
                "context": context,
                "source": "dailydialog",
            })

    print(f"  {len(examples)} per-utterance examples extracted")
    return examples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_data(
    seed: int = 42,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    """Load, extract features, and save DailyDialog training data.

    Returns dict with stats.
    """
    from jarvis.classifiers.response_mobilization import classify_response_pressure
    from jarvis.embedding_adapter import get_embedder

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load DailyDialog
    t0 = time.perf_counter()
    log_memory("START")
    examples = load_dailydialog()
    log_memory("After loading DailyDialog")

    print(f"\nTotal raw examples: {len(examples)}")

    # Save counts for metadata
    total_raw = len(examples)

    # Distribution before filtering
    raw_counts = Counter(ex["label"] for ex in examples)
    print("\nRaw label distribution:")
    for label, count in sorted(raw_counts.items(), key=lambda x: -x[1]):
        pct = count / total_raw * 100
        print(f"  {label:12s} {count:6d} ({pct:.1f}%)")

    # Compute balance ratio
    min_count = min(raw_counts.values())
    max_count = max(raw_counts.values())
    balance_ratio = max_count / max(min_count, 1)
    print(f"\nBalance ratio: {balance_ratio:.1f}x (max/min)")

    if dry_run:
        print("\n--- DRY RUN: stopping before feature extraction ---")
        return {
            "total_raw": len(examples),
            "raw_distribution": dict(raw_counts),
            "balance_ratio": balance_ratio,
        }

    # Step 2: Filter too-short texts
    filtered = [ex for ex in examples if len(ex["text"].strip()) >= 3]

    # Apply limit if specified (for testing)
    if limit is not None and len(filtered) > limit:
        filtered = filtered[:limit]
        print(f"\nLimited to {limit} examples for testing")

    total_filtered = len(filtered)
    print(f"\nAfter filtering (len >= 3): {total_filtered}")

    # Free memory - don't need original examples anymore
    del examples
    log_memory("After deleting examples")

    # Step 3: Compute mobilization + hand-crafted features IN BATCHES
    print("\nExtracting hand-crafted features in batches...")
    n_examples = len(filtered)

    # Pre-allocate array (stream to disk if memory is tight)
    hc_matrix = np.zeros((n_examples, 19), dtype=np.float32)

    # Process in batches to avoid memory buildup
    batch_size = 10000
    for batch_start in range(0, n_examples, batch_size):
        batch_end = min(batch_start + batch_size, n_examples)

        for i in range(batch_start, batch_end):
            ex = filtered[i]
            mob = classify_response_pressure(ex["last_message"])
            hc = extract_hand_crafted_features(
                text=ex["text"],
                context_messages=ex["context"],
                mobilization_pressure=mob.pressure.value,
                mobilization_type=mob.response_type.value,
            )
            hc_matrix[i] = hc

        if batch_end % 10000 == 0 or batch_end == n_examples:
            print(f"  {batch_end}/{n_examples} features extracted")

    hand_crafted_dims = hc_matrix.shape[1]
    print(f"  Hand-crafted features shape: {hc_matrix.shape}")
    log_memory("After hand-crafted features")

    # Step 3b: Extract SpaCy syntactic features via NER service
    print("\nExtracting SpaCy syntactic features via NER service...")
    from jarvis.nlp.ner_client import get_syntactic_features_batch, is_service_running

    if not is_service_running():
        print("WARNING: NER service not running. SpaCy features will be zeros.")
        print("  Start the service with: uv run python scripts/ner_server.py")
        spacy_matrix = np.zeros((n_examples, 14), dtype=np.float32)
    else:
        print("  NER service detected, extracting features in batches...")
        # Extract texts for SpaCy processing
        all_texts_raw = [ex["text"] for ex in filtered]

        # Process in batches of 5000 (Unix socket can handle larger batches than MLX)
        batch_size = 5000
        spacy_matrix = np.zeros((n_examples, 14), dtype=np.float32)

        for i in range(0, n_examples, batch_size):
            batch_end = min(i + batch_size, n_examples)
            batch_texts = all_texts_raw[i:batch_end]

            batch_features = get_syntactic_features_batch(batch_texts)
            spacy_matrix[i:batch_end] = np.array(batch_features, dtype=np.float32)

            if batch_end % 10000 == 0 or batch_end == n_examples:
                print(f"  {batch_end}/{n_examples} SpaCy features extracted", flush=True)

        del all_texts_raw

    spacy_dims = spacy_matrix.shape[1]
    print(f"  SpaCy features shape: {spacy_matrix.shape}")
    log_memory("After SpaCy features")

    # Step 4: Extract texts and labels FIRST, then free filtered before loading embedder
    print("\nExtracting texts and labels from filtered data...")
    all_texts = [ex["last_message"] for ex in filtered]
    all_labels = [ex["label"] for ex in filtered]
    n_examples = len(filtered)
    log_memory("After extracting texts/labels")

    # FREE MEMORY: Delete filtered (76k dicts with context strings = ~750 MB!)
    del filtered
    print(f"  Freed filtered list ({n_examples} examples with context)")
    log_memory("After deleting filtered")

    # Step 4b: Normalize texts before embedding
    print("\nNormalizing texts for embedding...")
    from jarvis.text_normalizer import normalize_text

    normalized_texts = []
    for i, text in enumerate(all_texts):
        # Normalize: expand slang (no spell_check to save time)
        normalized = normalize_text(text, expand_slang=True, spell_check=False)
        # Fall back to original if normalization returns empty
        normalized_texts.append(normalized if normalized else text)

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n_examples} texts normalized", flush=True)

    print(f"  Normalized {len(normalized_texts)} texts")
    log_memory("After text normalization")

    # Now load embedder (much more memory available)
    print("\nComputing embeddings in batches...")
    print("  About to load embedder...")
    log_memory("BEFORE get_embedder()")

    from jarvis.embedding_adapter import get_embedder
    embedder = get_embedder()

    log_memory("AFTER get_embedder() - THIS IS THE PROBLEM!")

    # Get embedding dimension (save for metadata)
    test_emb = embedder.encode([normalized_texts[0]], normalize=True)
    embedding_dims = test_emb.shape[1]
    del test_emb
    log_memory("After test embedding")

    # Use memmap to write directly to disk (avoid holding all in RAM)
    embeddings_path = output_dir / "embeddings_temp.dat"
    embeddings_mmap = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode='w+',
        shape=(n_examples, embedding_dims)
    )
    log_memory("After creating memmap")

    # Process in batches of 5000 (MLX memory limit prevents VMS spike)
    batch_size = 5000
    for i in range(0, n_examples, batch_size):
        batch_end = min(i + batch_size, n_examples)
        batch_texts = normalized_texts[i:batch_end]

        batch_emb = embedder.encode(batch_texts, normalize=True)
        embeddings_mmap[i:batch_end] = batch_emb

        # Free memory
        del batch_texts, batch_emb

        # Clear MLX cache AND force GC every batch to prevent VMS accumulation
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

        if (batch_end) % 10000 == 0 or batch_end == n_examples:
            print(f"  {batch_end}/{n_examples} embeddings computed", flush=True)
            log_memory(f"After batch {batch_end}")

    # Free text lists now that embeddings are done
    del all_texts, normalized_texts
    log_memory("After deleting text lists")

    # Flush to disk and convert to regular array
    embeddings_mmap.flush()
    log_memory("After memmap flush (before conversion)")
    embeddings = np.array(embeddings_mmap)
    log_memory("After memmap → array conversion (THIS LOADS ALL INTO RAM)")
    del embeddings_mmap
    embeddings_path.unlink()  # Clean up temp file
    log_memory("After deleting memmap")

    # Clear MLX cache and free embedder
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    del embedder
    log_memory("After clearing MLX cache and deleting embedder")

    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 5: Build feature matrix and save
    # Use labels extracted earlier (filtered already deleted)
    y = np.array(all_labels)
    del all_labels
    log_memory("After creating label array")

    # Save metadata before we delete arrays
    labels = sorted(set(y))
    feature_dims = embeddings.shape[1] + hc_matrix.shape[1] + spacy_matrix.shape[1]

    # Combine features (384 + 19 + 14 = 417)
    print(f"\nCombining features:")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Hand-crafted: {hc_matrix.shape}")
    print(f"  SpaCy: {spacy_matrix.shape}")
    X = np.hstack([embeddings, hc_matrix, spacy_matrix])
    print(f"  Total: {X.shape}")
    log_memory("After np.hstack (TRIPLE MEMORY USAGE)")

    # Free memory - don't need these anymore
    del embeddings, hc_matrix, spacy_matrix
    log_memory("After deleting feature matrices")

    print(f"\nFeature matrix: {X.shape}, Labels: {y.shape}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    log_memory("After train/test split")

    # Free memory - don't need full dataset anymore
    del X, y
    log_memory("After deleting X, y")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Compute final distributions
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    np.savez(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez(output_dir / "test.npz", X=X_test, y=y_test)

    label_map = {label: i for i, label in enumerate(labels)}

    metadata = {
        "source": "OpenRL/daily_dialog",
        "labels": labels,
        "label_map": label_map,
        "total_raw": total_raw,
        "total_filtered": total_filtered,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_dims": feature_dims,
        "embedding_dims": embedding_dims,
        "hand_crafted_dims": hand_crafted_dims,
        "spacy_dims": spacy_dims,
        "text_normalization": "expand_slang",
        "spacy_features": [
            "has_imperative", "you_modal", "request_verb", "starts_modal", "directive_question",
            "i_will", "promise_verb", "first_person_count", "agreement",
            "modal_count", "verb_count", "second_person_count", "has_negation", "is_interrogative"
        ],
        "label_distribution_raw": dict(raw_counts),
        "label_distribution_train": {k: int(v) for k, v in train_counts.items()},
        "label_distribution_test": {k: int(v) for k, v in test_counts.items()},
        "balance_ratio": balance_ratio,
        "seed": seed,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"\nSaved to {output_dir}/")
    print(f"Elapsed: {elapsed:.1f}s")
    print(json.dumps(metadata, indent=2))

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare DailyDialog data with native dialog act labels"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview label distribution without feature extraction",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to N examples (for testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prepare_data(seed=args.seed, dry_run=args.dry_run, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
