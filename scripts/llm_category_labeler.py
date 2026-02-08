#!/usr/bin/env python3
"""Full LLM-based category labeling for 15-17k ambiguous examples.

Labels examples where weak supervision is uncertain (confidence < 0.5 or disagreement).
Uses Cerebras gpt-oss-120b (free tier: 14.4k req/day, 30 req/min).

Output: JSONL with labeled examples (resume-safe).

Usage:
    uv run python scripts/llm_category_labeler.py
    uv run python scripts/llm_category_labeler.py --max-examples 5000  # smaller run
    uv run python scripts/llm_category_labeler.py --resume  # resume from existing JSONL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import get_judge_client, JUDGE_BASE_URL
from scripts.labeling_functions import get_registry, ABSTAIN
from scripts.label_aggregation import aggregate_labels

VALID_CATEGORIES = ["ack", "info", "emotional", "social", "clarify"]

CLASSIFICATION_PROMPT_TEMPLATE = """Classify each message into ONE category based on how an AI assistant should respond.

Categories:
- ack: Simple acknowledgment, reaction, emoji-only, "ok/yes/no/thanks/bye". No reply needed.
- info: Question, request, scheduling, logistics. Needs factual/action response.
- emotional: Expresses feelings, celebrates, vents. Needs empathy/support.
- social: Casual chat, banter, opinions, stories. Needs friendly engagement.
- clarify: Ambiguous, incomplete, context-dependent. Needs more context first.

Messages:
{messages_block}

Reply with ONLY the category name for each, one per line (e.g., "ack"). No numbers, no explanations."""


def load_ambiguous_examples(max_examples: int = 17000) -> list[dict]:
    """Load examples where weak supervision is uncertain.

    Strategy:
    1. Label all examples with weak supervision
    2. Keep confident labels (confidence >= 0.8, 2+ LF votes) -- free, high accuracy
    3. LLM-label ambiguous pool (confidence < 0.5 or 1 LF vote)
    4. LLM-label disagreement examples (confidence 0.4-0.6, multiple LFs disagreed)

    Args:
        max_examples: Max number of examples to label with LLM.

    Returns:
        List of examples needing LLM labeling.
    """
    from datasets import load_dataset

    print("Loading datasets...", flush=True)
    dd_ds = load_dataset("OpenRL/daily_dialog", split="train")
    samsum_ds = load_dataset("knkarthick/samsum", split="train")

    all_examples: list[dict] = []

    # DailyDialog
    print("  Processing DailyDialog...", flush=True)
    for dialogue in dd_ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            all_examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": {"act": int(acts[i]), "emotion": int(emotions[i])},
                "source": "dailydialog",
            })

    # SAMSum
    print("  Processing SAMSum...", flush=True)
    for conv in samsum_ds:
        dialogue_text = conv["dialogue"]
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]

        messages: list[tuple[str, str]] = []
        for line in lines:
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 30:
                sender = line[:colon_idx].strip()
                text_part = line[colon_idx + 1:].strip()
                if text_part:
                    messages.append((sender, text_part))

        if len(messages) < 2:
            continue

        for i in range(1, len(messages)):
            text = messages[i][1]
            if len(text.strip()) < 2:
                continue

            context = [m[1] for m in messages[max(0, i - 5):i]]
            last_msg = messages[i - 1][1]

            all_examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": None,
                "source": "samsum",
            })

    print(f"  Total raw examples: {len(all_examples)}", flush=True)

    # Apply weak supervision to identify ambiguous examples
    print("\nApplying weak supervision to identify ambiguous examples...", flush=True)
    registry = get_registry()
    labels, confidences = aggregate_labels(all_examples, registry, method="majority")

    for i, ex in enumerate(all_examples):
        ex["heuristic_label"] = labels[i]
        ex["confidence"] = confidences[i]

    # Count LF votes per example (exclude abstain)
    vote_counts: list[int] = []
    for ex in all_examples:
        lf_labels = registry.apply_all(
            ex["text"], ex["context"], ex["last_message"], ex.get("metadata")
        )
        non_abstain = [lbl for lbl in lf_labels if lbl != ABSTAIN]
        vote_counts.append(len(non_abstain))

    for i, ex in enumerate(all_examples):
        ex["lf_vote_count"] = vote_counts[i]

    # Filter ambiguous examples:
    # 1. Low confidence (< 0.5) OR only 1 LF vote
    # 2. Disagreement zone (0.4 <= confidence < 0.6, 2+ LFs voted)
    ambiguous = [
        ex for ex in all_examples
        if (ex["confidence"] < 0.5 or ex["lf_vote_count"] <= 1)
        or (0.4 <= ex["confidence"] < 0.6 and ex["lf_vote_count"] >= 2)
    ]

    print(f"  Ambiguous examples (need LLM): {len(ambiguous)}", flush=True)

    # Show confidence distribution
    conf_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(conf_bins) - 1):
        count = sum(1 for ex in ambiguous if conf_bins[i] <= ex["confidence"] < conf_bins[i + 1])
        print(f"    Confidence [{conf_bins[i]:.1f}-{conf_bins[i + 1]:.1f}): {count}", flush=True)

    # Limit to max_examples
    import numpy as np
    rng = np.random.default_rng(42)

    if len(ambiguous) > max_examples:
        # Stratify by heuristic pre-screening (balance categories)
        stratified: list[dict] = []
        per_category = max_examples // len(VALID_CATEGORIES)

        for category in VALID_CATEGORIES:
            cat_examples = [ex for ex in ambiguous if ex["heuristic_label"] == category]
            if len(cat_examples) >= per_category:
                indices = rng.choice(len(cat_examples), per_category, replace=False)
                stratified.extend([cat_examples[i] for i in indices])
            else:
                stratified.extend(cat_examples)

        rng.shuffle(stratified)
        ambiguous = stratified[:max_examples]

        print(f"  Sampled to {len(ambiguous)} examples (stratified)", flush=True)

    dist = Counter(ex["heuristic_label"] for ex in ambiguous)
    print(f"  Heuristic label distribution (pre-screening):", flush=True)
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(ambiguous) * 100
        print(f"    {cat:10s} {count:6d} ({pct:.1f}%)", flush=True)

    return ambiguous


def classify_batch_llm(
    examples: list[dict],
    model: str,
    client,
    batch_size: int = 20,
) -> list[str]:
    """Classify examples using LLM in batches.

    Args:
        examples: List of examples to classify.
        model: Model name.
        client: OpenAI client.
        batch_size: Number of examples per API call.

    Returns:
        List of predicted categories (same length as examples).
    """
    predictions: list[str] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]

        # Format messages block
        messages_block = "\n".join([
            f"{j + 1}. {ex['text']}"
            for j, ex in enumerate(batch)
        ])

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(messages_block=messages_block)

        # API call
        print(f"  Batch {i // batch_size + 1}/{(len(examples) + batch_size - 1) // batch_size} "
              f"(examples {i + 1}-{min(i + batch_size, len(examples))})", flush=True)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )

            # Check if response is valid
            if not response or not response.choices:
                print(f"    ERROR: Empty response from API", flush=True)
                predictions.extend([ex["heuristic_label"] for ex in batch])
                continue

            # Parse response (reasoning models use .reasoning instead of .content)
            message = response.choices[0].message
            content = message.content or message.reasoning
            if content is None:
                print(f"    ERROR: API returned None for both content and reasoning", flush=True)
                print(f"    Response: {response}", flush=True)
                predictions.extend([ex["heuristic_label"] for ex in batch])
                continue

            content = content.strip()
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            # Validate and normalize
            for line in lines:
                # Remove leading numbers and periods if present
                clean = line.lstrip("0123456789. ").lower().strip()
                if clean in VALID_CATEGORIES:
                    predictions.append(clean)
                else:
                    # Fallback: try to extract category name
                    for cat in VALID_CATEGORIES:
                        if cat in clean:
                            predictions.append(cat)
                            break
                    else:
                        predictions.append("social")  # Default fallback

            # Ensure we have exactly batch_size predictions
            while len(predictions) < i + len(batch):
                predictions.append("social")  # Pad with fallback

            time.sleep(2)  # Rate limiting: 30 req/min = 2s between calls

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            print(f"    Traceback: {traceback.format_exc()}", flush=True)
            # Fallback to heuristic label for failed batch
            predictions.extend([ex["heuristic_label"] for ex in batch])

    return predictions[:len(examples)]  # Trim any excess


def save_labeled_examples(
    examples: list[dict],
    predictions: list[str],
    output_path: Path,
    append: bool = False,
) -> None:
    """Save labeled examples to JSONL (resume-safe).

    Args:
        examples: Examples to save.
        predictions: LLM predictions.
        output_path: Output JSONL path.
        append: If True, append to existing file.
    """
    mode = "a" if append else "w"
    with output_path.open(mode) as f:
        for ex, pred in zip(examples, predictions):
            record = {
                "text": ex["text"],
                "last_message": ex["last_message"],
                "context": ex["context"],
                "label": pred,
                "confidence": 0.95,  # High confidence for LLM labels
                "source": ex["source"],
                "labeling_method": "llm",
            }
            f.write(json.dumps(record) + "\n")


def load_existing_labels(output_path: Path) -> set[str]:
    """Load already-labeled text from JSONL.

    Args:
        output_path: Path to JSONL file.

    Returns:
        Set of already-labeled text strings.
    """
    if not output_path.exists():
        return set()

    labeled = set()
    with output_path.open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                labeled.add(record["text"])

    return labeled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full LLM-based category labeling"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-oss-120b",
        help="Model to use (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=17000,
        help="Max number of examples to label (default: 17000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Batch size for API calls (default: 20)",
    )
    parser.add_argument(
        "--output", type=str, default="llm_category_labels.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already-labeled)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"LLM Category Labeling")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API: {JUDGE_BASE_URL}")
    print(f"Max examples: {args.max_examples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print(f"Resume: {args.resume}")
    print()

    # Get client
    client = get_judge_client()
    if not client:
        print("ERROR: CEREBRAS_API_KEY not set in .env")
        return 1

    # Load ambiguous examples
    examples = load_ambiguous_examples(max_examples=args.max_examples)

    # Check for existing labels (resume)
    output_path = PROJECT_ROOT / args.output
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"\nFound {len(existing)} already-labeled examples", flush=True)
        examples = [ex for ex in examples if ex["text"] not in existing]
        print(f"  Remaining to label: {len(examples)}", flush=True)

    if not examples:
        print("\nNo examples to label. Done.")
        return 0

    # Classify with LLM
    print(f"\nClassifying {len(examples)} examples with {args.model}...", flush=True)
    print(f"Estimated time: {len(examples) // args.batch_size * 2 / 60:.1f} minutes", flush=True)
    print()

    t0 = time.perf_counter()
    predictions = classify_batch_llm(examples, args.model, client, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    print(f"\nClassification complete in {elapsed / 60:.1f} minutes", flush=True)

    # Save results
    save_labeled_examples(examples, predictions, output_path, append=args.resume)
    print(f"\nLabeled examples saved to {output_path}", flush=True)

    # Show distribution
    dist = Counter(predictions)
    print(f"\nLLM label distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        print(f"  {cat:10s} {count:6d} ({pct:.1f}%)")

    # Compare to heuristic pre-screening
    print(f"\nComparison to heuristic pre-screening:")
    agreement = sum(1 for ex, pred in zip(examples, predictions) if ex["heuristic_label"] == pred)
    agreement_pct = agreement / len(examples) * 100
    print(f"  Agreement: {agreement}/{len(examples)} ({agreement_pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Done. {len(examples)} examples labeled in {elapsed / 60:.1f} minutes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
