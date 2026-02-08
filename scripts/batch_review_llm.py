#!/usr/bin/env python3
"""Manual batch review: Show 10 examples at a time for validation.

Loads the pilot validation results and shows examples with both
heuristic and LLM labels for manual assessment.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from scripts.labeling_functions import get_registry
from scripts.label_aggregation import aggregate_labels
from evals.judge_config import get_judge_client

CLASSIFICATION_PROMPT_TEMPLATE = """Classify each message into ONE category based on how an AI assistant should respond.

Categories:
- ack: Simple acknowledgment, reaction, emoji-only, "ok/yes/no/thanks/bye". No reply needed.
- info: Question, request, scheduling, logistics. Needs factual/action response.
- emotional: Expresses feelings, celebrates, vents. Needs empathy/support.
- social: Casual chat, banter, opinions, stories. Needs friendly engagement.
- clarify: Ambiguous, incomplete, context-dependent. Needs more context first.

For each message, consider the conversation context (previous message) when classifying.

{messages_block}

Reply with ONLY the category name for each, one per line (e.g., "ack"). No numbers, no explanations."""

VALID_CATEGORIES = ["ack", "info", "emotional", "social", "clarify"]


def classify_batch(client, examples, model="gpt-oss-120b"):
    """Classify a batch of examples."""
    messages_block = "\n\n".join([
        f"Message {i + 1}:\n"
        f"Previous: \"{ex['last_message'][:100] if ex['last_message'] else '(start of conversation)'}\"\n"
        f"Current: \"{ex['text']}\""
        for i, ex in enumerate(examples)
    ])
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(messages_block=messages_block)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )

    message = response.choices[0].message
    content = message.content or message.reasoning
    if not content:
        return ["social"] * len(examples)

    content = content.strip()
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    predictions = []
    for line in lines:
        # Skip explanatory lines
        if "classify" in line.lower() or "let's" in line.lower() or "need to" in line.lower():
            continue

        # Extract category
        if "->" in line:
            parts = line.split("->")
            if len(parts) >= 2:
                clean = parts[1].split("(")[0].strip().lower()
            else:
                clean = line.lstrip("0123456789. \"").lower().strip()
        else:
            clean = line.lstrip("0123456789. \"").lower().strip()

        clean = clean.split("(")[0].strip()

        if clean in VALID_CATEGORIES:
            predictions.append(clean)
        else:
            for cat in VALID_CATEGORIES:
                if cat in clean:
                    predictions.append(cat)
                    break
            else:
                predictions.append("social")

    # Pad if needed
    while len(predictions) < len(examples):
        predictions.append("social")

    return predictions[:len(examples)]


def load_sample_batch(batch_num=0, batch_size=10):
    """Load a batch of examples with heuristic labels."""
    print(f"Loading batch {batch_num + 1}...", flush=True)

    # Load DailyDialog
    dd_ds = load_dataset("OpenRL/daily_dialog", split="train")

    examples = []
    for dialogue in dd_ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]
        emotions = dialogue["emotion"]

        if len(utterances) < 2:
            continue

        for i in range(1, min(len(utterances), 6)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            examples.append({
                "text": text,
                "last_message": last_msg,
                "context": context,
                "metadata": {"act": int(acts[i]), "emotion": int(emotions[i])},
                "source": "dailydialog",
            })

    # Get batch
    start = batch_num * batch_size
    batch = examples[start:start + batch_size]

    # Get heuristic labels
    registry = get_registry()
    labels, confidences = aggregate_labels(batch, registry, method="majority")

    for i, ex in enumerate(batch):
        ex["heuristic_label"] = labels[i]
        ex["confidence"] = confidences[i]

    return batch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="Batch number (0-indexed)")
    parser.add_argument("--model", type=str, default="gpt-oss-120b", help="Model to use")
    args = parser.parse_args()

    client = get_judge_client()
    if not client:
        print("ERROR: No API client")
        return 1

    # Load batch
    batch = load_sample_batch(args.batch, batch_size=10)
    print(f"Loaded {len(batch)} examples\n")

    # Get LLM predictions
    print("Getting LLM predictions...", flush=True)
    llm_predictions = classify_batch(client, batch, model=args.model)
    print("Done\n")

    # Show results
    print("=" * 80)
    print(f"BATCH {args.batch + 1} REVIEW")
    print("=" * 80)
    print()

    for i, ex in enumerate(batch):
        heuristic = ex["heuristic_label"]
        llm = llm_predictions[i]
        conf = ex["confidence"]

        print(f"Example {i + 1}:")
        print(f"  Message: {ex['text'][:70]}")
        if len(ex['text']) > 70:
            print(f"           {ex['text'][70:][:70]}")
        print(f"  Context: {ex['last_message'][:60] if ex['last_message'] else '(none)'}")
        print(f"  Heuristic: {heuristic:10s} (confidence: {conf:.2f})")
        print(f"  LLM:       {llm:10s}")

        if heuristic == llm:
            print(f"  ✓ AGREE")
        else:
            print(f"  ✗ DISAGREE")
        print()

    # Summary
    agreement = sum(1 for i, ex in enumerate(batch) if ex["heuristic_label"] == llm_predictions[i])
    print("=" * 80)
    print(f"Agreement: {agreement}/{len(batch)} ({agreement/len(batch)*100:.1f}%)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
