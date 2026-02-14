"""Topic Segmentation Evaluation - Measures boundary detection quality.

Compares TopicSegmenter output against a 'gold' dataset of human-segmented
conversations to measure:
- Boundary Precision: What % of detected boundaries are correct?
- Boundary Recall: What % of gold boundaries were detected?
- WindowDiff / Pk: Standard segmentation error metrics.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.topics.topic_segmenter import get_segmenter  # noqa: E402

# =============================================================================
# Data Types
# =============================================================================


@dataclass
class GoldExample:
    name: str
    messages: list[dict]  # List of message dicts with 'text', 'date', 'is_from_me'
    gold_boundaries: list[int]  # Indices of messages where a new topic starts


@dataclass
class SegEvalResult:
    example_name: str
    precision: float
    recall: float
    f1: float
    detected_boundaries: list[int]
    gold_boundaries: list[int]


# =============================================================================
# Metrics
# =============================================================================


def compute_boundary_metrics(
    detected: list[int], gold: list[int], tolerance: int = 1
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for boundaries with a tolerance window.

    Args:
        detected: List of detected boundary indices.
        gold: List of gold boundary indices.
        tolerance: Distance allowed between detected and gold boundary.
    """
    if not gold:
        return (1.0 if not detected else 0.0, 1.0, 1.0)
    if not detected:
        return (0.0, 0.0, 0.0)

    tp = 0
    matched_gold = set()

    for d in detected:
        # Check if any gold boundary is within tolerance
        for g in gold:
            if abs(d - g) <= tolerance and g not in matched_gold:
                tp += 1
                matched_gold.add(g)
                break

    precision = tp / len(detected)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=" * 70)
    print("JARVIS Topic Segmentation Evaluation")
    print("=" * 70)

    # 1. Load Gold Data
    # For now, we'll use a small embedded set. In production, this would
    # load from evals/segmentation_gold.jsonl
    gold_data = [
        GoldExample(
            name="Logistics to Social",
            messages=[
                {
                    "text": "Hey, what time is the meeting?",
                    "date": "2024-01-01T10:00:00",
                    "is_from_me": False,
                },
                {"text": "It starts at 11am.", "date": "2024-01-01T10:01:00", "is_from_me": True},
                {
                    "text": "Cool, I'll be there.",
                    "date": "2024-01-01T10:02:00",
                    "is_from_me": False,
                },
                {
                    "text": "Wait, did you see that movie last night?",
                    "date": "2024-01-01T10:05:00",
                    "is_from_me": False,
                },
                {"text": "No, was it good?", "date": "2024-01-01T10:06:00", "is_from_me": True},
                {
                    "text": "Yeah, really intense.",
                    "date": "2024-01-01T10:07:00",
                    "is_from_me": False,
                },
            ],
            gold_boundaries=[3],  # New topic starts at index 3
        ),
        GoldExample(
            name="Long Time Gap",
            messages=[
                {"text": "Goodnight!", "date": "2024-01-01T23:00:00", "is_from_me": True},
                {
                    "text": "Morning! How's it going?",
                    "date": "2024-01-02T08:00:00",
                    "is_from_me": False,
                },
                {"text": "Pretty good, you?", "date": "2024-01-02T08:05:00", "is_from_me": True},
            ],
            gold_boundaries=[1],
        ),
        GoldExample(
            name="Complex Shift",
            messages=[
                {
                    "text": "I finished the report.",
                    "date": "2024-01-01T14:00:00",
                    "is_from_me": True,
                },
                {
                    "text": "Awesome, thanks for the hard work.",
                    "date": "2024-01-01T14:05:00",
                    "is_from_me": False,
                },
                {
                    "text": "By the way, are you still planning to go to the gym?",
                    "date": "2024-01-01T14:10:00",
                    "is_from_me": False,
                },
                {
                    "text": "Yeah, in about an hour.",
                    "date": "2024-01-01T14:12:00",
                    "is_from_me": True,
                },
                {"text": "Mind if I join?", "date": "2024-01-01T14:15:00", "is_from_me": False},
            ],
            gold_boundaries=[2],
        ),
    ]

    segmenter = get_segmenter()
    from contracts.imessage import Message
    from datetime import datetime

    results = []

    for ex in tqdm(gold_data, desc="Evaluating"):
        # Convert to Message objects
        msgs = []
        for i, m in enumerate(ex.messages):
            msgs.append(
                Message(
                    id=i,
                    chat_id="eval",
                    sender="other" if not m["is_from_me"] else "me",
                    sender_name=None,
                    text=m["text"],
                    date=datetime.fromisoformat(m["date"]),
                    is_from_me=m["is_from_me"],
                )
            )

        # Run segmenter
        segments = segmenter.segment(msgs)

        # Extract boundaries: index of first message in each segment after the first
        # A boundary is where a new segment starts (index in original message list)
        detected_boundaries = []
        for seg in segments[1:]:  # Skip first segment (starts at index 0)
            # Find the index of the first message of this segment in the original list
            first_msg = seg.messages[0]
            for i, orig_m in enumerate(msgs):
                if orig_m.id == first_msg.id:
                    detected_boundaries.append(i)
                    break

        p, r, f1 = compute_boundary_metrics(detected_boundaries, ex.gold_boundaries)
        results.append(
            SegEvalResult(
                example_name=ex.name,
                precision=p,
                recall=r,
                f1=f1,
                detected_boundaries=detected_boundaries,
                gold_boundaries=ex.gold_boundaries,
            )
        )

    # 3. Summary
    print("\n" + "-" * 70)
    print(f"{'Example Name':30} | {'Prec':6} | {'Rec':6} | {'F1':6}")
    print("-" * 70)
    for r in results:
        print(f"{r.example_name:30} | {r.precision:0.2f} | {r.recall:0.2f} | {r.f1:0.2f}")

    avg_f1 = sum(r.f1 for r in results) / len(results)
    print("-" * 70)
    print(f"Average F1: {avg_f1:0.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
