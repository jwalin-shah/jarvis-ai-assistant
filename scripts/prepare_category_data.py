#!/usr/bin/env python3
"""Prepare multi-source training data for 4-category classifier.

Downloads DailyDialog + SAMSum from HuggingFace, maps/labels to 4 categories
(clarify, warm, brief, social), extracts features (384-dim embeddings + 19
hand-crafted), balances classes, and saves train/test splits.

Categories:
- clarify: ambiguous/missing context, defer or ask for clarification
- warm: emotional weight (comfort or celebrate)
- brief: short transactional (confirm, decline, ETA, yes/no)
- social: casual conversational, DEFAULT

DailyDialog labeling: mechanical mapping from act+emotion labels (FREE).
SAMSum labeling: batched Cerebras API (Qwen3-235B, ~$2).

Output: data/category_training/{train,test}.npz, metadata.json

Usage:
    uv run python scripts/prepare_category_data.py --dry-run   # preview distribution
    uv run python scripts/prepare_category_data.py              # full run
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

VALID_CATEGORIES = {"clarify", "warm", "brief", "social"}

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
# DailyDialog: mechanical mapping (FREE)
# ---------------------------------------------------------------------------

# Dialog act labels: 1=inform, 2=question, 3=directive, 4=commissive
# Emotion labels: 0=no_emotion, 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise
NEGATIVE_EMOTIONS = {1, 2, 3, 5}  # anger, disgust, fear, sadness
POSITIVE_EMOTIONS = {4, 6}  # happiness, surprise


def map_dailydialog_category(act: int, emotion: int, text_len: int) -> str:
    """Map DailyDialog act+emotion labels to our 4 categories.

    Mapping table from research doc:
    - directive/commissive → brief (commands/commitments need action response)
    - negative emotion → warm (bad news needs empathy)
    - positive emotion → warm (good news needs celebration)
    - question + no_emotion + short → brief
    - question + no_emotion + long → social
    - inform + no_emotion + short → brief
    - inform + no_emotion + long → social
    """
    # Directive and commissive are always brief
    if act in (3, 4):
        return "brief"

    # Any negative or positive emotion → warm
    if emotion in NEGATIVE_EMOTIONS or emotion in POSITIVE_EMOTIONS:
        return "warm"

    # No emotion: short messages → brief, long → social
    if text_len < 40:
        return "brief"
    return "social"


def load_dailydialog() -> list[dict]:
    """Load DailyDialog and extract per-utterance examples with labels.

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
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            act = acts[i]
            emotion = emotions[i]
            label = map_dailydialog_category(act, emotion, len(text))

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
# SAMSum: Cerebras API labeling (cached)
# ---------------------------------------------------------------------------

SAMSUM_LABEL_PROMPT = """\
Classify each text message into the reply strategy it demands.

Categories (pick the FIRST that applies):
1. clarify - Message is ambiguous, missing context, or can't be interpreted \
(bare "?", voice memos, single emoji with no context, forwarded media)
2. warm - Message carries emotional weight requiring validation \
(bad news, good news, venting, celebration, grief, excitement)
3. brief - Message needs a short transactional reply \
(yes/no questions, confirmations, ETAs, logistics, scheduling)
4. social - Message invites casual conversation (DEFAULT) \
(catching up, banter, opinions, stories, general chat)

Rules:
- Pick the FIRST applicable category in order: clarify > warm > brief > social
- "social" is the DEFAULT - use when others don't clearly fit
- Professional tone does NOT change category; ignore formality

Messages:
{messages}

Reply with ONLY the category numbers (1-4), one per line. \
No explanations, no extra text."""

CATEGORY_NUM_MAP = {"1": "clarify", "2": "warm", "3": "brief", "4": "social"}


def _parse_batch_labels(response_text: str, batch_size: int) -> list[str]:
    """Parse batch labeling response into category names.

    Extracts numbers 1-4 from the response, maps to category names.
    Falls back to 'social' for unparseable lines.
    """
    # Strip chain-of-thought blocks
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    idx = text.find("<think>")
    if idx != -1:
        text = text[:idx]

    labels: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip().rstrip(".")
        # Extract first digit 1-4
        match = re.search(r"[1-4]", line)
        if match:
            labels.append(CATEGORY_NUM_MAP[match.group()])
        elif labels:
            # Skip blank lines between numbers
            continue

    # Pad or truncate to batch_size
    while len(labels) < batch_size:
        labels.append("social")
    return labels[:batch_size]


def load_samsum_with_labels(
    cache_dir: Path,
    batch_size: int = 100,
    dry_run: bool = False,
    max_dry_run: int = 100,
) -> list[dict]:
    """Load SAMSum and label messages via Cerebras API.

    Labels are cached per-batch to disk for resume-on-failure.

    Returns list of dicts with keys: text, last_message, label, context.
    """
    from datasets import load_dataset
    from openai import OpenAI

    from evals.judge_config import JUDGE_API_KEY_ENV, JUDGE_BASE_URL

    print("Loading SAMSum...")
    # Use knkarthick mirror (Samsung/samsum gated/unavailable on datasets>=4.0)
    ds = load_dataset("knkarthick/samsum", split="train")
    print(f"  {len(ds)} conversations loaded")

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
                "source": "samsum",
            })

    print(f"  {len(all_turns)} per-turn examples extracted")

    if dry_run:
        all_turns = all_turns[:max_dry_run]

    # Load cached labels
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_cache_path = cache_dir / "samsum_labels.jsonl"
    cached_labels: dict[int, str] = {}
    if label_cache_path.exists():
        for line in label_cache_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cached_labels[entry["idx"]] = entry["label"]
        print(f"  Loaded {len(cached_labels)} cached labels")

    # Determine which turns need labeling
    to_label_indices = [i for i in range(len(all_turns)) if i not in cached_labels]

    if to_label_indices:
        api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
        if not api_key or api_key == "your-key-here":
            print(f"WARNING: {JUDGE_API_KEY_ENV} not set. Using 'social' for all unlabeled.")
            for i in to_label_indices:
                cached_labels[i] = "social"
        else:
            client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

            # Batch the indices
            batches = []
            for start in range(0, len(to_label_indices), batch_size):
                batches.append(to_label_indices[start:start + batch_size])

            print(f"  Labeling {len(to_label_indices)} turns in {len(batches)} batches "
                  f"(batch_size={batch_size})...")

            labeled = 0
            errors = 0

            # Use qwen-3-235b on Cerebras for labeling
            label_model = "qwen-3-235b-a22b-instruct-2507"

            with open(label_cache_path, "a") as cache_file:
                for batch_num, batch_indices in enumerate(batches):
                    # Format messages for this batch
                    batch_texts = []
                    for idx in batch_indices:
                        batch_texts.append(all_turns[idx]["text"])

                    messages_str = "\n".join(
                        f"{i + 1}. {t}" for i, t in enumerate(batch_texts)
                    )
                    prompt = SAMSUM_LABEL_PROMPT.format(messages=messages_str)

                    try:
                        resp = client.chat.completions.create(
                            model=label_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=batch_size * 3,
                        )
                        response_text = resp.choices[0].message.content.strip()
                        batch_labels = _parse_batch_labels(
                            response_text, len(batch_indices)
                        )

                        for idx, label in zip(batch_indices, batch_labels):
                            cached_labels[idx] = label
                            cache_file.write(
                                json.dumps({"idx": idx, "label": label}) + "\n"
                            )
                            labeled += 1

                        cache_file.flush()

                    except Exception as e:
                        logger.warning("Batch %d error: %s", batch_num, e)
                        errors += 1
                        # Default to social for failed batches
                        for idx in batch_indices:
                            if idx not in cached_labels:
                                cached_labels[idx] = "social"
                                cache_file.write(
                                    json.dumps({"idx": idx, "label": "social"}) + "\n"
                                )
                        cache_file.flush()

                    if (batch_num + 1) % 100 == 0:
                        print(f"    Batch {batch_num + 1}/{len(batches)} "
                              f"(labeled={labeled}, errors={errors})")

            print(f"  Labeled {labeled} turns ({errors} batch errors)")

    # Attach labels to turns
    examples: list[dict] = []
    for i, turn in enumerate(all_turns):
        label = cached_labels.get(i, "social")
        turn["label"] = label
        examples.append(turn)

    return examples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_data(
    seed: int = 42,
    dry_run: bool = False,
    max_overrep_ratio: float = 2.0,
) -> dict:
    """Load, label, extract features, balance, and save training data.

    Returns dict with stats.
    """
    from jarvis.classifiers.response_mobilization import classify_response_pressure
    from jarvis.embedding_adapter import get_embedder

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and label both datasets
    t0 = time.perf_counter()
    dd_examples = load_dailydialog()
    samsum_cache = output_dir / "samsum_cache"
    samsum_examples = load_samsum_with_labels(
        samsum_cache, dry_run=dry_run, max_dry_run=200,
    )

    all_examples = dd_examples + samsum_examples
    print(f"\nTotal raw examples: {len(all_examples)}")

    # Distribution before filtering
    raw_counts = Counter(ex["label"] for ex in all_examples)
    print("\nRaw label distribution:")
    for label, count in sorted(raw_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_examples) * 100
        print(f"  {label:10s} {count:6d} ({pct:.1f}%)")

    # Per-source breakdown
    for source in ("dailydialog", "samsum"):
        src_examples = [ex for ex in all_examples if ex["source"] == source]
        src_counts = Counter(ex["label"] for ex in src_examples)
        print(f"\n  {source} ({len(src_examples)} total):")
        for label, count in sorted(src_counts.items(), key=lambda x: -x[1]):
            pct = count / max(len(src_examples), 1) * 100
            print(f"    {label:10s} {count:6d} ({pct:.1f}%)")

    if dry_run:
        print("\n--- DRY RUN: stopping before feature extraction ---")
        return {
            "total_raw": len(all_examples),
            "raw_distribution": dict(raw_counts),
        }

    # Step 2: Filter too-short texts
    filtered = [ex for ex in all_examples if len(ex["text"].strip()) >= 3]
    print(f"\nAfter filtering (len >= 3): {len(filtered)}")

    # Step 3: Balance classes
    label_counts = Counter(ex["label"] for ex in filtered)
    median_count = int(np.median(list(label_counts.values())))
    max_per_class = int(median_count * max_overrep_ratio)

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
        "labeling_methods": {
            "dailydialog": "mechanical (act+emotion mapping)",
            "samsum": "llm (qwen-3-235b on Cerebras)",
        },
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
        description="Prepare multi-source category training data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview label distribution without feature extraction",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prepare_data(seed=args.seed, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
