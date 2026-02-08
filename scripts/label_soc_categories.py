#!/usr/bin/env python3
"""Label SOC-2508 conversations with message category using LLM judge.

Uses Gemini 2.5 Flash via DeepInfra to classify each conversation into one of
5 categories. Labels are cached to disk so re-runs skip already-labeled items.
Then extracts features (embeddings + hand-crafted) and saves train/test splits.

Categories:
- professional: formal/workplace conversations
- emotional_support: emotional content (breakup, grief, stress, celebration)
- quick_exchange: short logistical messages, Q&A, coordination
- edge_case: ambiguous, very short, or no clear conversational goal
- catching_up: friendly/casual conversation, life updates, chatting

Output: data/soc_categories/{train,test}.npz, labels.jsonl, metadata.json

Usage:
    uv run python scripts/label_soc_categories.py --dry-run  # label 20, print for review
    uv run python scripts/label_soc_categories.py             # label all, extract features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_soc_data import extract_messages  # noqa: E402

logger = logging.getLogger(__name__)

from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key  # noqa: E402
MAX_WORKERS = 10  # Concurrent API requests

VALID_CATEGORIES = {
    "professional", "emotional_support", "quick_exchange", "edge_case", "catching_up",
}

LABELING_PROMPT = """\
You are labeling text message conversations for a classifier training dataset.

Given a conversation between two people (with metadata about their relationship \
and situation), classify the conversation into exactly ONE category.

Categories:
- **professional**: Workplace communication. Formal tone, work topics (meetings, \
deadlines, projects, clients, interviews, job-related). Even if casual in style, \
if the topic is clearly work/career, label it professional.
- **emotional_support**: Emotionally charged exchanges. Someone is sharing feelings, \
seeking comfort, venting, celebrating big news, dealing with loss/breakup/stress, \
or offering emotional support. The emotional content is the main purpose.
- **quick_exchange**: Short, transactional exchanges. Coordinating plans, logistics, \
quick questions with quick answers, confirmations, scheduling. Messages tend to be \
short and goal-oriented. "What time?", "On my way", "Can you grab milk?"
- **edge_case**: Ambiguous or minimal exchanges. Very short messages with no clear \
purpose, unclear context, single-word or emoji-only messages, conversations that \
don't fit other categories. When genuinely unsure, use this.
- **catching_up**: Casual friendly conversation. Life updates, sharing stories, \
general chatting, reconnecting, discussing shared interests, hanging out talk. \
The default for normal friendly texting with no strong signal for other categories.

CONVERSATION METADATA:
{metadata}

MESSAGES:
{messages}

Respond with ONLY a JSON object (no markdown fences):
{{"category": "<category>", "reason": "<1 sentence>"}}"""


# ---------------------------------------------------------------------------
# LLM labeling
# ---------------------------------------------------------------------------


def _load_env_key() -> str:
    """Load judge API key from environment."""
    return get_judge_api_key()


def _format_metadata(conv: dict) -> str:
    """Format SOC conversation metadata for the prompt."""
    experience = conv.get("experience", {})
    parts = []

    rel = experience.get("relationship", "")
    if rel:
        parts.append(f"Relationship: {rel}")

    situation = experience.get("situation", "")
    if situation:
        parts.append(f"Situation: {situation}")

    topic = experience.get("topic", "")
    if topic:
        parts.append(f"Topic: {topic}")

    for key in ("persona1", "persona2"):
        persona = experience.get(key, {})
        style = persona.get("chatting_style", "")
        if style:
            parts.append(f"{key} style: {style}")

    return "\n".join(parts) if parts else "(no metadata available)"


def _format_messages(messages: list[dict], max_messages: int = 15) -> str:
    """Format messages for the prompt, truncating if needed."""
    # Take first few + last few if too many
    if len(messages) > max_messages:
        first = messages[:5]
        last = messages[-(max_messages - 5) :]
        lines = []
        for m in first:
            lines.append(f"{m['sender']}: {m['text']}")
        lines.append(f"... ({len(messages) - max_messages} messages omitted) ...")
        for m in last:
            lines.append(f"{m['sender']}: {m['text']}")
        return "\n".join(lines)

    return "\n".join(f"{m['sender']}: {m['text']}" for m in messages)


JSON_EXTRACT_RE = re.compile(r"\{[^{}]*\}")


def _strip_thinking(text: str) -> str:
    """Strip chain-of-thought blocks, including unclosed ones."""
    # Closed <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed <think>... (truncated response)
    idx = text.find("<think>")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _parse_label(response_text: str) -> tuple[str, str]:
    """Parse LLM response into (category, reason). Returns ('catching_up', '') on failure."""
    text = _strip_thinking(response_text.strip())

    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try direct parse first, then extract first JSON object from full response
    full_text = _strip_thinking(response_text)
    for candidate in [text, None]:
        if candidate is None:
            # Fallback: find first {...} anywhere in the full response
            match = JSON_EXTRACT_RE.search(full_text)
            if not match:
                break
            candidate = match.group(0)
        try:
            data = json.loads(candidate)
            category = data.get("category", "").strip().lower()
            reason = data.get("reason", "")
            if category in VALID_CATEGORIES:
                return category, reason
            logger.warning("Invalid category from LLM: %s", category)
            return "catching_up", f"invalid_category: {category}"
        except (json.JSONDecodeError, AttributeError):
            continue

    logger.warning("Failed to parse LLM response: %s", text[:100])
    return "catching_up", f"parse_error: {text[:50]}"


def label_conversations(
    conversations: list[dict],
    cache_path: Path,
    dry_run: bool = False,
    max_dry_run: int = 20,
) -> dict[int, tuple[str, str]]:
    """Label conversations using LLM, with disk cache.

    Args:
        conversations: List of SOC conversation dicts.
        cache_path: Path to labels.jsonl cache file.
        dry_run: If True, only label max_dry_run conversations.
        max_dry_run: Number of conversations to label in dry_run mode.

    Returns:
        Dict mapping conversation index to (category, reason).
    """
    from openai import OpenAI

    # Load cache
    cached: dict[int, tuple[str, str]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cached[entry["idx"]] = (entry["category"], entry["reason"])
        print(f"Loaded {len(cached)} cached labels from {cache_path}")

    # Determine which conversations need labeling
    to_label = [i for i in range(len(conversations)) if i not in cached]
    if dry_run:
        to_label = to_label[:max_dry_run]

    if not to_label:
        print("All conversations already labeled.")
        return cached

    print(f"Labeling {len(to_label)} conversations with {JUDGE_MODEL} "
          f"({MAX_WORKERS} workers)...")

    api_key = _load_env_key()
    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    labeled = 0
    errors = 0

    def _label_one(idx: int) -> tuple[int, str, str] | None:
        """Label a single conversation. Returns (idx, category, reason)."""
        conv = conversations[idx]
        messages = extract_messages(conv.get("chat_parts", []))
        if len(messages) < 2:
            return idx, "edge_case", "too_few_messages"

        metadata_str = _format_metadata(conv)
        messages_str = _format_messages(messages)
        prompt = LABELING_PROMPT.format(
            metadata=metadata_str, messages=messages_str,
        )

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            response_text = resp.choices[0].message.content.strip()
            category, reason = _parse_label(response_text)

            if reason.startswith("parse_error:"):
                return None  # Will be retried
            return idx, category, reason
        except Exception as e:
            logger.warning("LLM error for conv %d: %s", idx, e)
            return None

    with open(cache_path, "a") as cache_file:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_label_one, idx): idx for idx in to_label}

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    with lock:
                        errors += 1
                    continue

                idx, category, reason = result
                with lock:
                    cached[idx] = (category, reason)
                    cache_file.write(
                        json.dumps({
                            "idx": idx, "category": category, "reason": reason,
                        }) + "\n"
                    )
                    cache_file.flush()
                    labeled += 1
                    if labeled % 50 == 0:
                        print(f"  Labeled {labeled}/{len(to_label)}"
                              f" (errors: {errors})")

    print(f"\nLabeled {labeled} conversations ({errors} errors)")
    return cached


# ---------------------------------------------------------------------------
# Feature extraction (unchanged from before)
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
    """Extract ~19 hand-crafted features from a message + context."""
    features = []

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
        np.mean([len(m) for m in context_messages]) if context_messages else 0.0
    )
    features.append(float(avg_ctx_len))
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
# Main pipeline
# ---------------------------------------------------------------------------


def label_and_extract(
    seed: int = 42,
    dry_run: bool = False,
    max_catching_up_ratio: float = 2.0,
) -> dict:
    """Label SOC-2508 with LLM, extract features, save train/test splits.

    Returns:
        Dict with stats and label distribution.
    """
    from datasets import load_dataset

    from jarvis.classifiers.response_mobilization import classify_response_pressure

    output_dir = PROJECT_ROOT / "data" / "soc_categories"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "labels.jsonl"

    print("Loading SOC-2508 dataset...")
    ds = load_dataset("marcodsn/SOC-2508", split="train")
    conversations = list(ds)
    print(f"Loaded {len(conversations)} conversations")

    # Step 1: LLM labeling (cached)
    conv_labels = label_conversations(conversations, cache_path, dry_run=dry_run)

    # Label distribution at conversation level
    label_counts_conv = Counter(cat for cat, _ in conv_labels.values())
    print("\nConversation-level label distribution:")
    for label, count in sorted(label_counts_conv.items(), key=lambda x: -x[1]):
        pct = count / len(conv_labels) * 100
        print(f"  {label:20s} {count:5d} ({pct:.1f}%)")

    if dry_run:
        # Show labeled samples for review
        print("\n--- Labeled samples for review ---")
        for idx in sorted(conv_labels.keys())[:20]:
            category, reason = conv_labels[idx]
            conv = conversations[idx]
            messages = extract_messages(conv.get("chat_parts", []))
            print(f"\n[{idx}] {category}")
            print(f"  Reason: {reason}")
            for m in messages[:5]:
                print(f"  {m['sender']}: {m['text'][:80]}")
            if len(messages) > 5:
                print(f"  ... ({len(messages) - 5} more messages)")
        return {"total_conversations": len(conv_labels), "labels": dict(label_counts_conv)}

    # Step 2: Extract per-turn examples with features
    print("\nExtracting per-turn examples...")
    examples: list[dict] = []

    for conv_idx, conv in enumerate(conversations):
        if conv_idx not in conv_labels:
            continue

        label, _ = conv_labels[conv_idx]
        chat_parts = conv.get("chat_parts", [])
        messages = extract_messages(chat_parts)

        if len(messages) < 3:
            continue

        for i in range(2, len(messages)):
            reply = messages[i]
            text = reply["text"]

            if text.strip() in ("[Photo]", "[GIF]", "[Video]", "[Link]", "[Sticker]"):
                continue
            if len(text.strip()) < 2:
                continue

            context_msgs = [m["text"] for m in messages[max(0, i - 5) : i]]
            last_msg = messages[i - 1]["text"]

            mob = classify_response_pressure(last_msg)
            mob_pressure = mob.pressure.value
            mob_type = mob.response_type.value

            hc_features = extract_hand_crafted_features(
                text=text,
                context_messages=context_msgs,
                mobilization_pressure=mob_pressure,
                mobilization_type=mob_type,
            )

            examples.append({
                "text": text,
                "last_message": last_msg,
                "label": label,
                "hc_features": hc_features,
                "context": context_msgs,
            })

        if (conv_idx + 1) % 200 == 0:
            print(f"  Processed {conv_idx + 1}/{len(conversations)}, {len(examples)} examples")

    print(f"\nTotal examples before balancing: {len(examples)}")

    label_counts = Counter(ex["label"] for ex in examples)
    print("\nPer-turn label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:5d} ({count / len(examples) * 100:.1f}%)")

    # Step 3: Balance classes
    minority_count = min(
        count for label, count in label_counts.items() if label != "catching_up"
    )
    max_catching_up = int(minority_count * max_catching_up_ratio)

    rng = np.random.default_rng(seed)
    balanced: list[dict] = []
    catching_up_examples = [ex for ex in examples if ex["label"] == "catching_up"]
    other_examples = [ex for ex in examples if ex["label"] != "catching_up"]

    balanced.extend(other_examples)
    if len(catching_up_examples) > max_catching_up:
        indices = rng.choice(len(catching_up_examples), max_catching_up, replace=False)
        balanced.extend([catching_up_examples[i] for i in indices])
    else:
        balanced.extend(catching_up_examples)

    rng.shuffle(balanced)

    print(f"\nTotal examples after balancing: {len(balanced)}")
    balanced_counts = Counter(ex["label"] for ex in balanced)
    for label, count in sorted(balanced_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:5d} ({count / len(balanced) * 100:.1f}%)")

    # Step 4: Compute embeddings (batched, one call)
    print("\nComputing embeddings...")
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    all_texts = [ex["last_message"] for ex in balanced]
    embeddings = embedder.encode(all_texts, normalize=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 5: Build feature matrix and save
    hc_matrix = np.stack([ex["hc_features"] for ex in balanced])
    X = np.hstack([embeddings, hc_matrix])
    y = np.array([ex["label"] for ex in balanced])

    print(f"Feature matrix: {X.shape}, Labels: {y.shape}")

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
        "source": "marcodsn/SOC-2508",
        "labeling_method": "llm",
        "labeling_model": JUDGE_MODEL,
        "total_conversations": len(conv_labels),
        "total_raw_examples": len(examples),
        "total_balanced": len(balanced),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_dims": int(X.shape[1]),
        "embedding_dims": int(embeddings.shape[1]),
        "hand_crafted_dims": int(hc_matrix.shape[1]),
        "label_map": label_map,
        "label_distribution_raw": dict(label_counts),
        "label_distribution_balanced": {k: int(v) for k, v in balanced_counts.items()},
        "seed": seed,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved to {output_dir}/")
    print(json.dumps(metadata, indent=2))

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Label SOC-2508 for category classification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Label 20 and print for review")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    label_and_extract(seed=args.seed, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
