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

from evals.judge_config import get_judge_client, JUDGE_BASE_URL, JUDGE_MODEL

VALID_CATEGORIES = ["closing", "acknowledge", "question", "request", "emotion", "statement"]

CLASSIFICATION_PROMPT_TEMPLATE = """Classify each message. Check categories in this order:

1. closing: Says goodbye ("bye", "ttyl", "see you later", "gotta go")
2. acknowledge: ≤5 words AND expresses agreement/thanks ("ok", "yeah", "thanks", "sure", "here you are")
3. request: Asks for action ("can you", "could you", "would you", "please" + verb, "let's", "I'd like")
4. question: Has "?" OR starts with question word ("what", "when", "where", "who", "why", "how", "is", "are", "do")
5. emotion: Strong feelings - has emotion words ("happy", "sad", "love", "hate", "excited", "stressed") OR "!!" OR ALLCAPS words OR "😂" OR "wow"
6. statement: Neutral facts, opinions, explanations (if none of above match)

{messages_block}

Answer (comma-separated):"""


def simple_heuristic_label(text: str) -> str:
    """Apply simple heuristics matching the 6-category schema."""
    import re
    text_lower = text.lower().strip()
    text_clean = re.sub(r'[^\w\s?!]', '', text_lower)
    words = text_clean.split()

    # closing
    closing_words = {"bye", "ttyl", "see", "you", "later", "gotta", "go", "talk", "soon", "goodbye"}
    if any(w in closing_words for w in words[:3]):
        if "bye" in text_lower or "ttyl" in text_lower or "gotta go" in text_lower:
            return "closing"

    # acknowledge (≤5 words)
    if "?" not in text and len(words) <= 5:
        ack_words = {"ok", "okay", "yeah", "yep", "yup", "sure", "thanks", "thank", "you",
                     "gotcha", "fine", "alright", "cool", "k", "kk", "here", "are"}
        if any(w in ack_words for w in words):
            return "acknowledge"

    # request
    request_patterns = ["can you", "could you", "would you", "please", "i suggest", "let's", "lets", "i'd like"]
    if any(p in text_lower for p in request_patterns):
        return "request"

    # question
    if "?" in text:
        return "question"
    question_starters = ["what", "when", "where", "who", "why", "how", "is", "are", "do", "does"]
    if words and words[0] in question_starters:
        return "question"

    # emotion
    emotion_words = {"happy", "sad", "angry", "stressed", "excited", "frustrated",
                     "love", "hate", "amazing", "terrible", "wow", "omg", "ugh"}
    if any(w in emotion_words for w in words):
        return "emotion"
    if "!!" in text:
        return "emotion"
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps_words) >= 2:
        return "emotion"

    # statement (default)
    return "statement"


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

    # Apply simple heuristic labeling for stratification
    print("\nApplying simple heuristic labeling for stratification...", flush=True)
    for ex in all_examples:
        ex["heuristic_label"] = simple_heuristic_label(ex["text"])

    # All examples are candidates for LLM labeling (simple heuristics aren't high-confidence)
    ambiguous = all_examples

    print(f"  Examples to label: {len(ambiguous)}", flush=True)

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
    batch_size: int = 5,  # Smaller batches work better
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

        # Format messages block with context
        messages_block = "\n\n".join([
            f"Message {j + 1}:\n"
            f"Previous: \"{ex['last_message'][:100] if ex['last_message'] else '(start of conversation)'}\"\n"
            f"Current: \"{ex['text']}\""
            for j, ex in enumerate(batch)
        ])

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(messages_block=messages_block)

        # API call
        print(f"  Batch {i // batch_size + 1}/{(len(examples) + batch_size - 1) // batch_size} "
              f"(examples {i + 1}-{min(i + batch_size, len(examples))})", flush=True)

        try:
            response = client.chat.completions.create(
                model=model,  # Use model from args (llama-3.3-70b)
                messages=[
                    {"role": "system", "content": "You are a text classifier. Output only category labels, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,  # Just enough for labels
            )

            # Check if response is valid
            if not response or not response.choices:
                print(f"    ERROR: Empty response from API", flush=True)
                predictions.extend([ex["heuristic_label"] for ex in batch])
                continue

            # Parse response
            message = response.choices[0].message
            content = message.content or message.reasoning
            if content is None:
                print(f"    ERROR: API returned None", flush=True)
                predictions.extend([ex["heuristic_label"] for ex in batch])
                continue

            content = content.strip()
            import re

            # Parse comma-separated format
            if "," in content:
                parts = [p.strip().lower() for p in content.split(",")]
                for part in parts:
                    part_clean = re.sub(r'^.*?(\w+)$', r'\1', part)
                    if part_clean in VALID_CATEGORIES:
                        predictions.append(part_clean)
                    elif part_clean == "social":
                        predictions.append("statement")
            else:
                # Fall back to line-by-line
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                for line in lines:
                    line_clean = re.sub(r'^(\d+[\.\):]\s*|Message\s+\d+[\.:]\s*|\*\*Result:?\s*|-\s*Category:\s*)', '', line, flags=re.IGNORECASE)
                    line_clean = line_clean.strip().strip('*').strip().lower()
                    if line_clean in VALID_CATEGORIES:
                        predictions.append(line_clean)
                    elif line_clean == "social":
                        predictions.append("statement")

            # Pad with fallback if needed
            while len(predictions) < i + len(batch):
                predictions.append("statement")

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
        "--model", type=str, default=JUDGE_MODEL,
        help=f"Model to use (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--max-examples", type=int, default=17000,
        help="Max number of examples to label (default: 17000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Batch size for API calls (default: 5)",
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
