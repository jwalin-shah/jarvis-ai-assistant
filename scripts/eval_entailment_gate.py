#!/usr/bin/env python3
"""Evaluate entailment gate impact on the goldset.

Runs CandidateExtractor with and without entailment on the frozen goldset
and compares precision/recall/F1. Includes:
- Normalized text eval (extraction profile: slang expansion, no spell check)
- Full segmenter pipeline eval (TopicSegmenter → GLiNER + spaCy NER merge)

Usage:
    uv run python scripts/eval_entailment_gate.py
    uv run python scripts/eval_entailment_gate.py --normalized-only
    uv run python scripts/eval_entailment_gate.py --segmenter-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.contacts.candidate_extractor import CandidateExtractor
from scripts.run_extractor_bakeoff import (
    compute_metrics,
    parse_context_messages,
    print_comparison,
)

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")


def run_eval(
    extractor: CandidateExtractor,
    gold: list[dict],
    label: str,
    normalize: bool = False,
) -> dict:
    """Run extractor on goldset and compute metrics."""
    normalizer = None
    if normalize:
        from jarvis.text_normalizer import normalize_for_task

        def normalizer_fn(text: str) -> str:
            result = normalize_for_task(text, "extraction")
            return result if result else text

        normalizer = normalizer_fn

    predictions: dict[int, list[dict]] = {}
    timing: dict[int, float] = {}

    total = len(gold)
    start = time.time()

    for i, rec in enumerate(gold):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{label}] {i + 1}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        msg_id = rec["message_id"]
        text = rec["message_text"]

        if normalizer:
            text = normalizer(text)

        t0 = time.perf_counter()

        candidates = extractor.extract_candidates(
            text=text,
            message_id=msg_id,
            is_from_me=rec.get("is_from_me"),
            prev_messages=parse_context_messages(rec.get("context_prev")),
            next_messages=parse_context_messages(rec.get("context_next")),
            use_gate=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        predictions[msg_id] = [
            {
                "span_text": c.span_text,
                "span_label": c.span_label,
                "fact_type": c.fact_type,
                "score": c.gliner_score,
            }
            for c in candidates
        ]
        timing[msg_id] = elapsed_ms

    total_time = time.time() - start
    print(f"  [{label}] Done in {total_time:.1f}s", flush=True)

    metrics = compute_metrics(gold, predictions, timing)
    metrics["extractor_name"] = label
    metrics["num_messages"] = total
    metrics["total_time_s"] = round(total_time, 2)
    metrics["ms_per_message"] = round(total_time / total * 1000, 1)
    return metrics


# ---------------------------------------------------------------------------
# Full segmenter pipeline eval
# ---------------------------------------------------------------------------


@dataclass
class FakeMessage:
    """Minimal Message-like object for the segmenter."""

    id: int
    text: str
    date: datetime
    is_from_me: bool
    chat_id: str = ""


def _build_conversation(rec: dict, base_time: datetime) -> list[FakeMessage]:
    """Build a mini-conversation from goldset record context.

    Creates Message-like objects from context_prev + current + context_next,
    with synthetic timestamps 10s apart (within one topic segment).
    """
    messages: list[FakeMessage] = []
    t = base_time

    # Parse context_prev
    prev_texts = parse_context_messages(rec.get("context_prev"))
    for i, text in enumerate(prev_texts):
        messages.append(
            FakeMessage(
                id=rec["message_id"] - len(prev_texts) + i,
                text=text,
                date=t,
                is_from_me=False,  # Approximate
            )
        )
        t += timedelta(seconds=10)

    # Current message
    messages.append(
        FakeMessage(
            id=rec["message_id"],
            text=rec["message_text"],
            date=t,
            is_from_me=rec.get("is_from_me", False),
        )
    )
    t += timedelta(seconds=10)

    # Parse context_next
    next_texts = parse_context_messages(rec.get("context_next"))
    for i, text in enumerate(next_texts):
        messages.append(
            FakeMessage(
                id=rec["message_id"] + i + 1,
                text=text,
                date=t,
                is_from_me=False,
            )
        )
        t += timedelta(seconds=10)

    return messages


def run_segmenter_eval(
    extractor: CandidateExtractor,
    gold: list[dict],
    label: str,
) -> dict:
    """Run full segmenter pipeline on goldset.

    For each goldset record:
    1. Build mini-conversation from context_prev + message + context_next
    2. Run TopicSegmenter (normalizes text, extracts spaCy entities)
    3. Run extract_facts_from_segments (GLiNER + spaCy NER merge)
    4. Match predictions for the target message against gold
    """
    from jarvis.contacts.segment_extractor import extract_facts_from_segments
    from jarvis.topics.topic_segmenter import TopicSegmenter

    segmenter = TopicSegmenter(normalization_task="extraction")

    predictions: dict[int, list[dict]] = {}
    timing: dict[int, float] = {}
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    total = len(gold)
    start = time.time()

    for i, rec in enumerate(gold):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{label}] {i + 1}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        msg_id = rec["message_id"]
        t0 = time.perf_counter()

        # Build mini-conversation and segment it
        conversation = _build_conversation(rec, base_time)
        segments = segmenter.segment(conversation)

        # Extract facts from segments
        candidates = extract_facts_from_segments(segments, extractor)

        # Filter to only candidates from our target message
        target_candidates = [c for c in candidates if c.message_id == msg_id]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        predictions[msg_id] = [
            {
                "span_text": c.span_text,
                "span_label": c.span_label,
                "fact_type": c.fact_type,
                "score": c.gliner_score,
            }
            for c in target_candidates
        ]
        timing[msg_id] = elapsed_ms

    total_time = time.time() - start
    print(f"  [{label}] Done in {total_time:.1f}s", flush=True)

    metrics = compute_metrics(gold, predictions, timing)
    metrics["extractor_name"] = label
    metrics["num_messages"] = total
    metrics["total_time_s"] = round(total_time, 2)
    metrics["ms_per_message"] = round(total_time / total * 1000, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate entailment gate on goldset")
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help="Only run normalized text eval (skip raw text baselines)",
    )
    parser.add_argument(
        "--segmenter-only",
        action="store_true",
        help="Only run full segmenter pipeline eval",
    )
    args = parser.parse_args()

    print(f"Loading goldset from {GOLD_PATH}...", flush=True)
    gold = json.load(open(GOLD_PATH))
    print(f"Loaded {len(gold)} records\n", flush=True)

    all_metrics = []

    if not args.normalized_only and not args.segmenter_only:
        # Baseline: no entailment, raw text
        print("=== Running WITHOUT entailment (raw text) ===", flush=True)
        ext_no = CandidateExtractor(use_entailment=False)
        metrics_no = run_eval(ext_no, gold, "no_entailment")
        all_metrics.append(metrics_no)

        # With entailment, raw text
        print("\n=== Running WITH entailment (raw text, threshold=0.12) ===", flush=True)
        ext_yes = CandidateExtractor(use_entailment=True, entailment_threshold=0.12)
        metrics_yes = run_eval(ext_yes, gold, "entailment_0.12")
        all_metrics.append(metrics_yes)

    if not args.segmenter_only:
        # With entailment + normalized text
        print(
            "\n=== Running WITH entailment + normalized text (extraction profile) ===",
            flush=True,
        )
        ext_norm = CandidateExtractor(use_entailment=True, entailment_threshold=0.12)
        metrics_norm = run_eval(ext_norm, gold, "entailment+normalized", normalize=True)
        all_metrics.append(metrics_norm)

    # Full segmenter pipeline
    if args.segmenter_only or not args.normalized_only:
        print(
            "\n=== Running FULL SEGMENTER pipeline (GLiNER + spaCy NER merge) ===",
            flush=True,
        )
        ext_seg = CandidateExtractor(use_entailment=True, entailment_threshold=0.12)
        metrics_seg = run_segmenter_eval(ext_seg, gold, "segmenter+gliner+spacy")
        all_metrics.append(metrics_seg)

    # Compare
    print_comparison(all_metrics)

    # Save results
    out = Path("results/entailment_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)


if __name__ == "__main__":
    main()
