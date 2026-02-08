#!/usr/bin/env python3
"""Pilot validation: test LLM category labeling on 200 examples.

Validates the classification prompt and model accuracy before full labeling.
Uses stratified sampling (40 examples per category via heuristic pre-screening).

Usage:
    uv run python scripts/validate_llm_categories.py
    uv run python scripts/validate_llm_categories.py --model zai-glm-4.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge_config import get_judge_client, JUDGE_BASE_URL

VALID_CATEGORIES = ["closing", "acknowledge", "question", "request", "emotion", "statement"]

CLASSIFICATION_PROMPT_TEMPLATE = """Classify each message. Check rules in order, take FIRST match:

1. closing: "bye", "ttyl", "see you later", "gotta go"
2. acknowledge: ≤3 words: "ok", "thanks", "yeah", "gotcha"
3. request: "can you", "could you", "please", "I suggest", "let's"
4. question: has "?" OR starts with question word
5. emotion: "happy", "sad", "love", "hate", "!!", CAPS
6. statement: everything else

EXAMPLES:

Message 1:
Previous: "How was your day?"
Current: "Good thanks"
acknowledge

Message 2:
Previous: "Are you coming?"
Current: "When does it start?"
question

Message 3:
Previous: "I'm at the store"
Current: "Can you get milk?"
request

Message 4:
Previous: "See you soon"
Current: "Bye!"
closing

---

NOW CLASSIFY THESE:

{messages_block}

OUTPUT (one category per line, no other text):"""


def simple_heuristic_label(text: str) -> str:
    """Apply simple heuristics matching the 6-category schema.

    Args:
        text: Message text to classify.

    Returns:
        Category label (closing, acknowledge, question, request, emotion, statement).
    """
    import re

    text_lower = text.lower().strip()
    text_clean = re.sub(r'[^\w\s?!]', '', text_lower)  # Remove punctuation except ?!
    words = text_clean.split()

    # Rule 1: closing
    closing_words = {"bye", "ttyl", "see", "you", "later", "gotta", "go", "talk", "soon", "goodbye"}
    if any(w in closing_words for w in words[:3]):  # Check first 3 words
        if "bye" in text_lower or "ttyl" in text_lower or "gotta go" in text_lower:
            return "closing"

    # Rule 2: acknowledge (≤3 words, no question mark)
    if "?" not in text and len(words) <= 3:
        ack_words = {"ok", "okay", "yeah", "yep", "yup", "sure", "thanks", "thank", "you",
                     "gotcha", "fine", "alright", "cool", "k", "kk"}
        if any(w in ack_words for w in words):
            return "acknowledge"

    # Rule 3: request (imperatives, can you, I suggest)
    request_patterns = ["can you", "could you", "would you", "please", "i suggest", "let's", "lets"]
    if any(p in text_lower for p in request_patterns):
        return "request"

    # Rule 4: question (? or question words)
    if "?" in text:
        return "question"
    question_starters = ["what", "when", "where", "who", "why", "how", "is", "are", "do", "does"]
    if words and words[0] in question_starters:
        return "question"

    # Rule 5: emotion (emotion words, multiple !, CAPS)
    emotion_words = {"happy", "sad", "angry", "stressed", "excited", "frustrated",
                     "love", "hate", "amazing", "terrible", "wow", "omg", "ugh"}
    if any(w in emotion_words for w in words):
        return "emotion"
    if "!!" in text:  # Multiple exclamation marks
        return "emotion"
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps_words) >= 2:
        return "emotion"

    # Rule 6: statement (default)
    return "statement"


def load_stratified_sample(n_per_category: int = 40) -> list[dict]:
    """Load and stratify sample based on heuristic pre-screening.

    Args:
        n_per_category: Number of examples per category target.

    Returns:
        List of examples with text, context, last_message, label (heuristic).
    """
    from datasets import load_dataset

    print(f"Loading datasets for stratified sampling ({n_per_category} per category)...", flush=True)

    # Load both datasets
    dd_ds = load_dataset("OpenRL/daily_dialog", split="train")
    samsum_ds = load_dataset("knkarthick/samsum", split="train")

    # Extract examples
    all_examples: list[dict] = []

    # DailyDialog
    for dialogue in dd_ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, min(len(utterances), 6)):  # First 5 turns per dialogue
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

    # SAMSum (first 500 conversations for speed)
    conv_count = 0
    for conv in samsum_ds:
        if conv_count >= 500:
            break
        conv_count += 1

        dialogue_text = conv["dialogue"]
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]

        messages: list[tuple[str, str]] = []
        for line in lines:
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

            all_examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": None,
                "source": "samsum",
            })

    print(f"  Loaded {len(all_examples)} raw examples", flush=True)

    # Apply simple heuristic labeling for stratification
    print("  Applying simple heuristic labeling for stratification...", flush=True)
    for ex in all_examples:
        ex["heuristic_label"] = simple_heuristic_label(ex["text"])

    # Stratified sampling
    import numpy as np
    rng = np.random.default_rng(42)

    stratified: list[dict] = []
    for category in VALID_CATEGORIES:
        cat_examples = [ex for ex in all_examples if ex["heuristic_label"] == category]

        if len(cat_examples) >= n_per_category:
            indices = rng.choice(len(cat_examples), n_per_category, replace=False)
            stratified.extend([cat_examples[i] for i in indices])
        else:
            # If not enough, take all we have
            print(f"    Warning: Only {len(cat_examples)} examples for {category}, needed {n_per_category}", flush=True)
            stratified.extend(cat_examples)

    rng.shuffle(stratified)

    print(f"  Stratified sample: {len(stratified)} examples", flush=True)
    dist = Counter(ex["heuristic_label"] for ex in stratified)
    for cat, count in sorted(dist.items()):
        print(f"    {cat:10s} {count:3d}", flush=True)

    return stratified


def classify_batch_llm(
    examples: list[dict],
    model: str,
    client,
    batch_size: int = 5,  # Smaller batches work better for these models
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
                model=model,  # Use model from args
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,  # Enough for batch with reasoning
            )

            # Check if response is valid
            if not response or not response.choices:
                print(f"    ERROR: Empty response from API", flush=True)
                predictions.extend(["statement"] * len(batch))
                continue

            # Parse response (reasoning models use .reasoning instead of .content)
            message = response.choices[0].message
            content = message.content or message.reasoning
            if content is None:
                print(f"    ERROR: API returned None for both content and reasoning", flush=True)
                print(f"    Response: {response}", flush=True)
                predictions.extend(["statement"] * len(batch))
                continue

            content = content.strip()

            # Debug: print first batch response to see format
            if i == 0:
                print(f"\n  DEBUG - First batch response:\n{content[:500]}\n", flush=True)

            lines = [line.strip() for line in content.split("\n") if line.strip()]

            # Parse response - expect one category per line
            import re
            # Try multiple patterns:
            # 1. Plain category name (one per line)
            # 2. "**Result: category**" or "- Category: category"
            # 3. Numbered: "1. category" or "Message 1: category"

            for line in lines:
                # Remove common prefixes
                line_clean = re.sub(r'^(\d+[\.\):]\s*|Message\s+\d+[\.:]\s*|\*\*Result:?\s*|-\s*Category:\s*)', '', line, flags=re.IGNORECASE)
                line_clean = line_clean.strip().strip('*').strip()

                # Check if it's a valid category
                if line_clean.lower() in VALID_CATEGORIES:
                    predictions.append(line_clean.lower())
                # Map old "social" to new "statement"
                elif line_clean.lower() == "social":
                    predictions.append("statement")

            # Ensure we have exactly batch_size predictions
            while len(predictions) < i + len(batch):
                predictions.append("statement")  # Pad with fallback

            time.sleep(2)  # Rate limiting: 30 req/min = 2s between calls

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            print(f"    Traceback: {traceback.format_exc()}", flush=True)
            # Fallback to statement for failed batch
            predictions.extend(["statement"] * len(batch))

    return predictions[:len(examples)]  # Trim any excess


def compute_accuracy(predictions: list[str], examples: list[dict]) -> dict:
    """Compute accuracy metrics.

    Args:
        predictions: LLM predictions.
        examples: Examples with heuristic_label.

    Returns:
        Dict with accuracy, per-class accuracy, confusion matrix.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    y_true = [ex["heuristic_label"] for ex in examples]
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)

    print("\n=== Results ===")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, labels=VALID_CATEGORIES, zero_division=0))

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=VALID_CATEGORIES)
    print("        " + "  ".join(f"{cat:>4s}" for cat in VALID_CATEGORIES))
    for i, cat in enumerate(VALID_CATEGORIES):
        print(f"{cat:>8s} " + "  ".join(f"{cm[i, j]:4d}" for j in range(len(VALID_CATEGORIES))))

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot validation: test LLM category labeling"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-oss-120b",
        help="Model to use (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=40,
        help="Number of examples per category (default: 40)",
    )
    parser.add_argument(
        "--output", type=str, default="llm_pilot_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    print(f"=== LLM Category Labeling Pilot ===")
    print(f"Model: {args.model}")
    print(f"API: {JUDGE_BASE_URL}")
    print()

    # Get client
    client = get_judge_client()
    if not client:
        print("ERROR: CEREBRAS_API_KEY not set in .env")
        return 1

    # Load stratified sample
    examples = load_stratified_sample(n_per_category=args.n_per_category)

    # Classify with LLM
    print(f"\nClassifying {len(examples)} examples with {args.model}...", flush=True)
    t0 = time.perf_counter()
    predictions = classify_batch_llm(examples, args.model, client, batch_size=20)
    elapsed = time.perf_counter() - t0

    print(f"\nClassification complete in {elapsed:.1f}s", flush=True)

    # Compute accuracy
    results = compute_accuracy(predictions, examples)
    results["model"] = args.model
    results["n_examples"] = len(examples)
    results["elapsed_seconds"] = elapsed

    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Show sample comparisons (10 per category)
    print("\n=== Sample Comparisons ===")
    for category in VALID_CATEGORIES:
        cat_examples = [
            (i, ex, predictions[i])
            for i, ex in enumerate(examples)
            if ex["heuristic_label"] == category
        ][:10]

        if not cat_examples:
            continue

        print(f"\n{category.upper()} (showing {len(cat_examples)} examples):")
        for i, ex, pred in cat_examples:
            match = "✓" if pred == category else "✗"
            print(f"  {match} [{pred}] {ex['text'][:60]}")

    print(f"\n{'=' * 60}")
    if results["accuracy"] >= 0.80:
        print("✓ PASS: Accuracy >= 80%. Ready for full labeling.")
        return 0
    else:
        print("✗ FAIL: Accuracy < 80%. Review prompt or model before full labeling.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
