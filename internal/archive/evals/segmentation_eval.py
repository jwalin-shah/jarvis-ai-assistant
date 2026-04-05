"""Topic Segmentation Evaluation - Measures boundary detection quality.

Compares TopicSegmenter output against a 'gold' dataset of human-segmented
conversations to measure:
- Boundary Precision: What % of detected boundaries are correct?
- Boundary Recall: What % of gold boundaries were detected?
- WindowDiff / Pk: Standard segmentation error metrics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
from jarvis.topics.topic_segmenter import get_segmenter  # noqa: E402


  # noqa: E402
# =============================================================================  # noqa: E402
# Data Types  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class GoldExample:  # noqa: E402
    name: str  # noqa: E402
    messages: list[dict]  # List of message dicts with 'text', 'date', 'is_from_me'  # noqa: E402
    gold_boundaries: list[int]  # Indices of messages where a new topic starts  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class SegEvalResult:  # noqa: E402
    example_name: str  # noqa: E402
    precision: float  # noqa: E402
    recall: float  # noqa: E402
    f1: float  # noqa: E402
    detected_boundaries: list[int]  # noqa: E402
    gold_boundaries: list[int]  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Metrics  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def compute_boundary_metrics(  # noqa: E402
    detected: list[int], gold: list[int], tolerance: int = 1  # noqa: E402
) -> tuple[float, float, float]:  # noqa: E402
    """Compute precision, recall, and F1 for boundaries with a tolerance window.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        detected: List of detected boundary indices.  # noqa: E402
        gold: List of gold boundary indices.  # noqa: E402
        tolerance: Distance allowed between detected and gold boundary.  # noqa: E402
    """  # noqa: E402
    if not gold:  # noqa: E402
        return (1.0 if not detected else 0.0, 1.0, 1.0)  # noqa: E402
    if not detected:  # noqa: E402
        return (0.0, 0.0, 0.0)  # noqa: E402
  # noqa: E402
    tp = 0  # noqa: E402
    matched_gold = set()  # noqa: E402
  # noqa: E402
    for d in detected:  # noqa: E402
        # Check if any gold boundary is within tolerance  # noqa: E402
        for g in gold:  # noqa: E402
            if abs(d - g) <= tolerance and g not in matched_gold:  # noqa: E402
                tp += 1  # noqa: E402
                matched_gold.add(g)  # noqa: E402
                break  # noqa: E402
  # noqa: E402
    precision = tp / len(detected)  # noqa: E402
    recall = tp / len(gold)  # noqa: E402
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # noqa: E402
  # noqa: E402
    return precision, recall, f1  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Main  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("JARVIS Topic Segmentation Evaluation")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    # 1. Load Gold Data  # noqa: E402
    # For now, we'll use a small embedded set. In production, this would  # noqa: E402
    # load from evals/segmentation_gold.jsonl  # noqa: E402
    gold_data = [  # noqa: E402
        GoldExample(  # noqa: E402
            name="Logistics to Social",  # noqa: E402
            messages=[  # noqa: E402
                {  # noqa: E402
                    "text": "Hey, what time is the meeting?",  # noqa: E402
                    "date": "2024-01-01T10:00:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {"text": "It starts at 11am.", "date": "2024-01-01T10:01:00", "is_from_me": True},  # noqa: E402
                {  # noqa: E402
                    "text": "Cool, I'll be there.",  # noqa: E402
                    "date": "2024-01-01T10:02:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {  # noqa: E402
                    "text": "Wait, did you see that movie last night?",  # noqa: E402
                    "date": "2024-01-01T10:05:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {"text": "No, was it good?", "date": "2024-01-01T10:06:00", "is_from_me": True},  # noqa: E402
                {  # noqa: E402
                    "text": "Yeah, really intense.",  # noqa: E402
                    "date": "2024-01-01T10:07:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
            ],  # noqa: E402
            gold_boundaries=[3],  # New topic starts at index 3  # noqa: E402
        ),  # noqa: E402
        GoldExample(  # noqa: E402
            name="Long Time Gap",  # noqa: E402
            messages=[  # noqa: E402
                {"text": "Goodnight!", "date": "2024-01-01T23:00:00", "is_from_me": True},  # noqa: E402
                {  # noqa: E402
                    "text": "Morning! How's it going?",  # noqa: E402
                    "date": "2024-01-02T08:00:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {"text": "Pretty good, you?", "date": "2024-01-02T08:05:00", "is_from_me": True},  # noqa: E402
            ],  # noqa: E402
            gold_boundaries=[1],  # noqa: E402
        ),  # noqa: E402
        GoldExample(  # noqa: E402
            name="Complex Shift",  # noqa: E402
            messages=[  # noqa: E402
                {  # noqa: E402
                    "text": "I finished the report.",  # noqa: E402
                    "date": "2024-01-01T14:00:00",  # noqa: E402
                    "is_from_me": True,  # noqa: E402
                },  # noqa: E402
                {  # noqa: E402
                    "text": "Awesome, thanks for the hard work.",  # noqa: E402
                    "date": "2024-01-01T14:05:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {  # noqa: E402
                    "text": "By the way, are you still planning to go to the gym?",  # noqa: E402
                    "date": "2024-01-01T14:10:00",  # noqa: E402
                    "is_from_me": False,  # noqa: E402
                },  # noqa: E402
                {  # noqa: E402
                    "text": "Yeah, in about an hour.",  # noqa: E402
                    "date": "2024-01-01T14:12:00",  # noqa: E402
                    "is_from_me": True,  # noqa: E402
                },  # noqa: E402
                {"text": "Mind if I join?", "date": "2024-01-01T14:15:00", "is_from_me": False},  # noqa: E402
            ],  # noqa: E402
            gold_boundaries=[2],  # noqa: E402
        ),  # noqa: E402
    ]  # noqa: E402
  # noqa: E402
    segmenter = get_segmenter()  # noqa: E402
    from datetime import datetime  # noqa: E402

    # noqa: E402
    from jarvis.contracts.imessage import Message  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for ex in tqdm(gold_data, desc="Evaluating"):  # noqa: E402
        # Convert to Message objects  # noqa: E402
        msgs = []  # noqa: E402
        for i, m in enumerate(ex.messages):  # noqa: E402
            msgs.append(  # noqa: E402
                Message(  # noqa: E402
                    id=i,  # noqa: E402
                    chat_id="eval",  # noqa: E402
                    sender="other" if not m["is_from_me"] else "me",  # noqa: E402
                    sender_name=None,  # noqa: E402
                    text=m["text"],  # noqa: E402
                    date=datetime.fromisoformat(m["date"]),  # noqa: E402
                    is_from_me=m["is_from_me"],  # noqa: E402
                )  # noqa: E402
            )  # noqa: E402
  # noqa: E402
        # Run segmenter  # noqa: E402
        segments = segmenter.segment(msgs)  # noqa: E402
  # noqa: E402
        # Extract boundaries: index of first message in each segment after the first  # noqa: E402
        # A boundary is where a new segment starts (index in original message list)  # noqa: E402
        detected_boundaries = []  # noqa: E402
        for seg in segments[1:]:  # Skip first segment (starts at index 0)  # noqa: E402
            # Find the index of the first message of this segment in the original list  # noqa: E402
            first_msg = seg.messages[0]  # noqa: E402
            for i, orig_m in enumerate(msgs):  # noqa: E402
                if orig_m.id == first_msg.id:  # noqa: E402
                    detected_boundaries.append(i)  # noqa: E402
                    break  # noqa: E402
  # noqa: E402
        p, r, f1 = compute_boundary_metrics(detected_boundaries, ex.gold_boundaries)  # noqa: E402
        results.append(  # noqa: E402
            SegEvalResult(  # noqa: E402
                example_name=ex.name,  # noqa: E402
                precision=p,  # noqa: E402
                recall=r,  # noqa: E402
                f1=f1,  # noqa: E402
                detected_boundaries=detected_boundaries,  # noqa: E402
                gold_boundaries=ex.gold_boundaries,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # 3. Summary  # noqa: E402
    print("\n" + "-" * 70)  # noqa: E402
    print(f"{'Example Name':30} | {'Prec':6} | {'Rec':6} | {'F1':6}")  # noqa: E402
    print("-" * 70)  # noqa: E402
    for r in results:  # noqa: E402
        print(f"{r.example_name:30} | {r.precision:0.2f} | {r.recall:0.2f} | {r.f1:0.2f}")  # noqa: E402
  # noqa: E402
    avg_f1 = sum(r.f1 for r in results) / len(results)  # noqa: E402
    print("-" * 70)  # noqa: E402
    print(f"Average F1: {avg_f1:0.2f}")  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
