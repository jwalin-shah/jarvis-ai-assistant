#!/usr/bin/env python3
"""Prepare multi-source training data for 5-category classifier using weak supervision.

Downloads DailyDialog + SAMSum from HuggingFace, applies 27+ labeling functions
(heuristics + mobilization features) via weak supervision, aggregates votes into
labels, extracts features (384-dim embeddings + 19 hand-crafted), balances classes,
and saves train/test splits.

Categories:
- ack: Acknowledgments, reactions, simple agreements (skip SLM, use template)
- info: Information requests, commitments, direct questions (context=5)
- emotional: Emotional support, celebrations, empathy needs (context=3)
- social: Casual conversation, banter, stories (context=3)
- clarify: Requests for clarification, ambiguous messages (context=5)

Labeling: DIY weak supervision (25+ heuristic labeling functions, no external deps).
Aggregation: Weighted majority vote or Dawid-Skene EM.

Output: data/category_training/{train,test}.npz, metadata.json

Usage:
    uv run python scripts/prepare_category_data.py --dry-run                           # preview distribution
    uv run python scripts/prepare_category_data.py                                      # full run (majority vote)
    uv run python scripts/prepare_category_data.py --method dawid_skene                # use Dawid-Skene EM
    uv run python scripts/prepare_category_data.py --min-confidence 0.5                # filter low-confidence
    uv run python scripts/prepare_category_data.py --llm-labels labels.jsonl          # override with LLM labels
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

VALID_CATEGORIES = {"ack", "info", "emotional", "social", "clarify"}

OUTPUT_DIR = PROJECT_ROOT / "data" / "category_training"

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
# DailyDialog: load raw data with metadata (weak supervision replaces mapping)
# ---------------------------------------------------------------------------


def load_dailydialog_raw() -> list[dict]:
    """Load DailyDialog and extract per-utterance examples WITHOUT pre-assigned labels.

    Returns list of dicts with keys: text, last_message, context, metadata (act, emotion).
    Labels will be assigned via weak supervision labeling functions.
    """
    from datasets import load_dataset

    print("Loading DailyDialog...", flush=True)
    # Use OpenRL parquet mirror (original li2017dailydialog repo requires
    # deprecated dataset scripts, incompatible with datasets>=4.0)
    ds = load_dataset("OpenRL/daily_dialog", split="train")
    print(f"  {len(ds)} dialogues loaded", flush=True)

    examples: list[dict] = []

    for dialogue in ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            act = acts[i]
            emotion = emotions[i]

            # Context: up to 5 previous utterances
            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": {"act": int(act), "emotion": int(emotion)},
                "source": "dailydialog",
            })

    print(f"  {len(examples)} per-utterance examples extracted", flush=True)
    return examples


# ---------------------------------------------------------------------------
# SAMSum: load raw data (weak supervision replaces LLM labeling)
# ---------------------------------------------------------------------------


def load_samsum_raw(dry_run: bool = False, max_dry_run: int = 100) -> list[dict]:
    """Load SAMSum and extract per-utterance examples WITHOUT pre-assigned labels.

    Returns list of dicts with keys: text, last_message, context, metadata (None for SAMSum).
    Labels will be assigned via weak supervision labeling functions.
    """
    from datasets import load_dataset

    print("Loading SAMSum...", flush=True)
    # Use knkarthick mirror (Samsung/samsum gated/unavailable on datasets>=4.0)
    ds = load_dataset("knkarthick/samsum", split="train")
    print(f"  {len(ds)} conversations loaded", flush=True)

    # Extract all per-turn examples first
    all_turns: list[dict] = []
    for conv in ds:
        dialogue_text = conv["dialogue"]
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]

        messages: list[tuple[str, str]] = []
        for line in lines:
            # SAMSum format: "Speaker: message"
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 30:
                sender = line[:colon_idx].strip()
                text = line[colon_idx + 1:].strip()
                if text:
                    messages.append((sender, text))

        if len(messages) < 2:
            continue

        for i in range(1, len(messages)):
            text = messages[i][1]
            if len(text.strip()) < 2:
                continue

            context = [m[1] for m in messages[max(0, i - 5):i]]
            last_msg = messages[i - 1][1]

            all_turns.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": None,  # SAMSum has no native metadata
                "source": "samsum",
            })

    print(f"  {len(all_turns)} per-turn examples extracted", flush=True)

    if dry_run:
        all_turns = all_turns[:max_dry_run]

    return all_turns


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_data(
    seed: int = 42,
    dry_run: bool = False,
    max_overrep_ratio: float = 1.0,
    method: str = "majority",
    min_confidence: float = 0.0,
    add_synthetic: bool = True,
    llm_labels_path: str | None = None,
) -> dict:
    """Load, label (via weak supervision), extract features, balance, and save training data.

    Args:
        seed: Random seed for reproducibility.
        dry_run: If True, show label distribution without feature extraction.
        max_overrep_ratio: Max ratio of (class count / minority count).
            1.0 = perfectly balanced (all classes match minority)
            2.0 = allows 2x imbalance (majority can be 2x minority)
        method: Label aggregation method ("majority" or "dawid_skene").
        min_confidence: Minimum confidence threshold to keep an example (0.0 = keep all).
        add_synthetic: If True, add synthetic examples.
        llm_labels_path: Optional path to LLM labels JSONL (overrides weak supervision).

    Returns dict with stats.
    """
    from scripts.label_aggregation import aggregate_labels
    from scripts.labeling_functions import get_registry

    from jarvis.classifiers.response_mobilization import classify_response_pressure
    from jarvis.embedding_adapter import get_embedder

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load both datasets (without labels)
    t0 = time.perf_counter()
    dd_examples = load_dailydialog_raw()
    samsum_examples = load_samsum_raw(dry_run=dry_run, max_dry_run=200)

    all_examples = dd_examples + samsum_examples

    # Step 1b: Add synthetic examples for ack only (clarify/social synthetic removed - caused confusion)
    if add_synthetic:
        from scripts.generate_synthetic_examples import generate_ack_examples

        print("\nGenerating synthetic examples...", flush=True)
        ack_synthetic = generate_ack_examples(n=1500)
        all_examples.extend(ack_synthetic)
        print(f"  Added {len(ack_synthetic)} synthetic ack examples", flush=True)

    print(f"\nTotal raw examples: {len(all_examples)}", flush=True)

    # Step 2: Apply weak supervision labeling functions (skip for synthetic - already labeled)
    print(f"\nApplying labeling functions (method={method})...", flush=True)
    registry = get_registry()
    print(f"  {len(registry.lfs)} labeling functions registered", flush=True)

    # Separate synthetic from real examples
    synthetic_examples = [ex for ex in all_examples if ex.get("source", "").startswith("synthetic_")]
    real_examples = [ex for ex in all_examples if not ex.get("source", "").startswith("synthetic_")]

    # Label only real examples (synthetic are pre-labeled)
    if real_examples:
        labels, confidences = aggregate_labels(real_examples, registry, method=method)
        for i, ex in enumerate(real_examples):
            ex["label"] = labels[i]
            ex["confidence"] = confidences[i]

    # Load LLM labels if provided (overrides weak supervision)
    if llm_labels_path:
        llm_label_path = Path(llm_labels_path)
        if llm_label_path.exists():
            print(f"\nLoading LLM labels from {llm_label_path}...", flush=True)
            llm_labels_map: dict[str, dict] = {}
            with llm_label_path.open() as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # Use text as key (unique identifier)
                        llm_labels_map[record["text"]] = {
                            "label": record["label"],
                            "confidence": record.get("confidence", 0.95),
                            "source_labeling": "llm",
                        }

            # Override weak supervision labels with LLM labels
            llm_override_count = 0
            for ex in real_examples:
                if ex["text"] in llm_labels_map:
                    llm_data = llm_labels_map[ex["text"]]
                    ex["label"] = llm_data["label"]
                    ex["confidence"] = llm_data["confidence"]
                    ex["source_labeling"] = "llm"
                    llm_override_count += 1
                else:
                    ex["source_labeling"] = "heuristic"

            print(f"  Overrode {llm_override_count} labels with LLM labels", flush=True)
        else:
            print(f"WARNING: LLM labels file not found: {llm_label_path}", flush=True)

    # Synthetic examples already have labels, give them confidence=1.0
    for ex in synthetic_examples:
        ex["confidence"] = 1.0

    # Merge back
    all_examples = real_examples + synthetic_examples

    # Extract labels and confidences for stats
    labels = [ex["label"] for ex in all_examples]
    confidences = [ex["confidence"] for ex in all_examples]

    print(f"  Labels assigned to {len(all_examples)} examples", flush=True)

    # Distribution after labeling
    raw_counts = Counter(ex["label"] for ex in all_examples)
    avg_confidence = np.mean(confidences)
    print(f"\nLabel distribution (avg confidence={avg_confidence:.3f}):")
    for label, count in sorted(raw_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_examples) * 100
        avg_conf_per_class = np.mean([
            ex["confidence"] for ex in all_examples if ex["label"] == label
        ])
        print(f"  {label:10s} {count:6d} ({pct:.1f}%) [avg_conf={avg_conf_per_class:.3f}]")

    # Per-source breakdown
    for source in ("dailydialog", "samsum", "synthetic_ack", "synthetic_clarify", "synthetic_social"):
        src_examples = [ex for ex in all_examples if ex["source"] == source]
        if not src_examples:
            continue
        src_counts = Counter(ex["label"] for ex in src_examples)
        print(f"\n  {source} ({len(src_examples)} total):")
        for label, count in sorted(src_counts.items(), key=lambda x: -x[1]):
            pct = count / max(len(src_examples), 1) * 100
            print(f"    {label:10s} {count:6d} ({pct:.1f}%)")

    # LF coverage stats (only on real examples, skip synthetic)
    from scripts.labeling_functions import ABSTAIN

    total_votes = 0
    abstain_votes = 0
    for ex in real_examples[:1000]:  # Sample for speed
        lf_labels = registry.apply_all(
            ex["text"], ex["context"], ex["last_message"], ex.get("metadata")
        )
        total_votes += len(lf_labels)
        abstain_votes += sum(1 for lbl in lf_labels if lbl == ABSTAIN)

    coverage = 1.0 - (abstain_votes / max(total_votes, 1))
    print(f"\nLF coverage (sampled): {coverage:.1%} (non-abstain votes)")

    if dry_run:
        print("\n--- DRY RUN: stopping before feature extraction ---")
        return {
            "total_raw": len(all_examples),
            "raw_distribution": dict(raw_counts),
            "avg_confidence": float(avg_confidence),
            "lf_coverage": float(coverage),
        }

    # Step 3: Filter by length and confidence
    filtered = [
        ex for ex in all_examples
        if len(ex["text"].strip()) >= 3 and ex["confidence"] >= min_confidence
    ]
    print(f"\nAfter filtering (len >= 3, confidence >= {min_confidence}): {len(filtered)}")

    # Step 3: Balance classes
    # Use minority count as baseline, allow max_overrep_ratio × minority
    label_counts = Counter(ex["label"] for ex in filtered)
    minority_count = min(label_counts.values())
    max_per_class = int(minority_count * max_overrep_ratio)

    rng = np.random.default_rng(seed)
    balanced: list[dict] = []
    for label in VALID_CATEGORIES:
        class_examples = [ex for ex in filtered if ex["label"] == label]
        if len(class_examples) > max_per_class:
            indices = rng.choice(len(class_examples), max_per_class, replace=False)
            balanced.extend([class_examples[i] for i in indices])
        else:
            balanced.extend(class_examples)

    rng.shuffle(balanced)

    balanced_counts = Counter(ex["label"] for ex in balanced)
    print(f"\nAfter balancing (max {max_per_class}/class): {len(balanced)}")
    for label, count in sorted(balanced_counts.items(), key=lambda x: -x[1]):
        pct = count / len(balanced) * 100
        print(f"  {label:10s} {count:6d} ({pct:.1f}%)")

    # Step 4: Compute mobilization + hand-crafted features
    print("\nExtracting hand-crafted features...")
    hc_list: list[np.ndarray] = []
    for i, ex in enumerate(balanced):
        mob = classify_response_pressure(ex["last_message"])
        hc = extract_hand_crafted_features(
            text=ex["text"],
            context_messages=ex["context"],
            mobilization_pressure=mob.pressure.value,
            mobilization_type=mob.response_type.value,
        )
        hc_list.append(hc)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(balanced)} features extracted")

    hc_matrix = np.stack(hc_list)
    print(f"  Hand-crafted features shape: {hc_matrix.shape}")

    # Step 5: Compute embeddings (batched)
    print("\nComputing embeddings...")
    embedder = get_embedder()
    all_texts = [ex["last_message"] for ex in balanced]
    embeddings = embedder.encode(all_texts, normalize=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 6: Build feature matrix and save
    X = np.hstack([embeddings, hc_matrix])
    y = np.array([ex["label"] for ex in balanced])

    print(f"\nFeature matrix: {X.shape}, Labels: {y.shape}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    np.savez(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez(output_dir / "test.npz", X=X_test, y=y_test)

    labels = sorted(set(y))
    label_map = {label: i for i, label in enumerate(labels)}

    metadata = {
        "sources": ["OpenRL/daily_dialog", "knkarthick/samsum"],
        "categories": sorted(VALID_CATEGORIES),
        "labeling_method": f"weak_supervision ({method})",
        "num_labeling_functions": len(registry.lfs),
        "aggregation_method": method,
        "min_confidence": min_confidence,
        "avg_confidence": float(avg_confidence),
        "lf_coverage": float(coverage),
        "total_raw": len(all_examples),
        "total_filtered": len(filtered),
        "total_balanced": len(balanced),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_dims": int(X.shape[1]),
        "embedding_dims": int(embeddings.shape[1]),
        "hand_crafted_dims": int(hc_matrix.shape[1]),
        "label_map": label_map,
        "label_distribution_raw": dict(raw_counts),
        "label_distribution_balanced": {k: int(v) for k, v in balanced_counts.items()},
        "max_overrep_ratio": max_overrep_ratio,
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
        description="Prepare multi-source category training data using weak supervision"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview label distribution without feature extraction",
    )
    parser.add_argument(
        "--method", type=str, default="majority", choices=["majority", "dawid_skene"],
        help="Label aggregation method (default: majority)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Minimum confidence threshold to keep an example (default: 0.0)",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Skip synthetic example generation",
    )
    parser.add_argument(
        "--llm-labels", type=str, default=None,
        help="Path to LLM labels JSONL (overrides weak supervision)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    prepare_data(
        seed=args.seed,
        dry_run=args.dry_run,
        method=args.method,
        min_confidence=args.min_confidence,
        add_synthetic=not args.no_synthetic,
        llm_labels_path=args.llm_labels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
