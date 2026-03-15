"""Topic Segmentation Evaluation - Measures boundary detection quality.  # noqa: E501
  # noqa: E501
Compares TopicSegmenter output against a 'gold' dataset of human-segmented  # noqa: E501
conversations to measure:  # noqa: E501
- Boundary Precision: What % of detected boundaries are correct?  # noqa: E501
- Boundary Recall: What % of gold boundaries were detected?  # noqa: E501
- WindowDiff / Pk: Standard segmentation error metrics.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import sys  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from jarvis.topics.topic_segmenter import get_segmenter  # noqa: E402  # noqa: E501


  # noqa: E501
# =============================================================================  # noqa: E501
# Data Types  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class GoldExample:  # noqa: E501
    name: str  # noqa: E501
    messages: list[dict]  # List of message dicts with 'text', 'date', 'is_from_me'  # noqa: E501
    gold_boundaries: list[int]  # Indices of messages where a new topic starts  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class SegEvalResult:  # noqa: E501
    example_name: str  # noqa: E501
    precision: float  # noqa: E501
    recall: float  # noqa: E501
    f1: float  # noqa: E501
    detected_boundaries: list[int]  # noqa: E501
    gold_boundaries: list[int]  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Metrics  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def compute_boundary_metrics(  # noqa: E501
    detected: list[int], gold: list[int], tolerance: int = 1  # noqa: E501
) -> tuple[float, float, float]:  # noqa: E501
    """Compute precision, recall, and F1 for boundaries with a tolerance window.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        detected: List of detected boundary indices.  # noqa: E501
        gold: List of gold boundary indices.  # noqa: E501
        tolerance: Distance allowed between detected and gold boundary.  # noqa: E501
    """  # noqa: E501
    if not gold:  # noqa: E501
        return (1.0 if not detected else 0.0, 1.0, 1.0)  # noqa: E501
    if not detected:  # noqa: E501
        return (0.0, 0.0, 0.0)  # noqa: E501
  # noqa: E501
    tp = 0  # noqa: E501
    matched_gold = set()  # noqa: E501
  # noqa: E501
    for d in detected:  # noqa: E501
        # Check if any gold boundary is within tolerance  # noqa: E501
        for g in gold:  # noqa: E501
            if abs(d - g) <= tolerance and g not in matched_gold:  # noqa: E501
                tp += 1  # noqa: E501
                matched_gold.add(g)  # noqa: E501
                break  # noqa: E501
  # noqa: E501
    precision = tp / len(detected)  # noqa: E501
    recall = tp / len(gold)  # noqa: E501
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # noqa: E501
  # noqa: E501
    return precision, recall, f1  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Main  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("JARVIS Topic Segmentation Evaluation")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    # 1. Load Gold Data  # noqa: E501
    # For now, we'll use a small embedded set. In production, this would  # noqa: E501
    # load from evals/segmentation_gold.jsonl  # noqa: E501
    gold_data = [  # noqa: E501
        GoldExample(  # noqa: E501
            name="Logistics to Social",  # noqa: E501
            messages=[  # noqa: E501
                {  # noqa: E501
                    "text": "Hey, what time is the meeting?",  # noqa: E501
                    "date": "2024-01-01T10:00:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {"text": "It starts at 11am.", "date": "2024-01-01T10:01:00", "is_from_me": True},  # noqa: E501
                {  # noqa: E501
                    "text": "Cool, I'll be there.",  # noqa: E501
                    "date": "2024-01-01T10:02:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {  # noqa: E501
                    "text": "Wait, did you see that movie last night?",  # noqa: E501
                    "date": "2024-01-01T10:05:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {"text": "No, was it good?", "date": "2024-01-01T10:06:00", "is_from_me": True},  # noqa: E501
                {  # noqa: E501
                    "text": "Yeah, really intense.",  # noqa: E501
                    "date": "2024-01-01T10:07:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
            ],  # noqa: E501
            gold_boundaries=[3],  # New topic starts at index 3  # noqa: E501
        ),  # noqa: E501
        GoldExample(  # noqa: E501
            name="Long Time Gap",  # noqa: E501
            messages=[  # noqa: E501
                {"text": "Goodnight!", "date": "2024-01-01T23:00:00", "is_from_me": True},  # noqa: E501
                {  # noqa: E501
                    "text": "Morning! How's it going?",  # noqa: E501
                    "date": "2024-01-02T08:00:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {"text": "Pretty good, you?", "date": "2024-01-02T08:05:00", "is_from_me": True},  # noqa: E501
            ],  # noqa: E501
            gold_boundaries=[1],  # noqa: E501
        ),  # noqa: E501
        GoldExample(  # noqa: E501
            name="Complex Shift",  # noqa: E501
            messages=[  # noqa: E501
                {  # noqa: E501
                    "text": "I finished the report.",  # noqa: E501
                    "date": "2024-01-01T14:00:00",  # noqa: E501
                    "is_from_me": True,  # noqa: E501
                },  # noqa: E501
                {  # noqa: E501
                    "text": "Awesome, thanks for the hard work.",  # noqa: E501
                    "date": "2024-01-01T14:05:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {  # noqa: E501
                    "text": "By the way, are you still planning to go to the gym?",  # noqa: E501
                    "date": "2024-01-01T14:10:00",  # noqa: E501
                    "is_from_me": False,  # noqa: E501
                },  # noqa: E501
                {  # noqa: E501
                    "text": "Yeah, in about an hour.",  # noqa: E501
                    "date": "2024-01-01T14:12:00",  # noqa: E501
                    "is_from_me": True,  # noqa: E501
                },  # noqa: E501
                {"text": "Mind if I join?", "date": "2024-01-01T14:15:00", "is_from_me": False},  # noqa: E501
            ],  # noqa: E501
            gold_boundaries=[2],  # noqa: E501
        ),  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    segmenter = get_segmenter()  # noqa: E501
    from datetime import datetime  # noqa: E501

    # noqa: E501
    from jarvis.contracts.imessage import Message  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for ex in tqdm(gold_data, desc="Evaluating"):  # noqa: E501
        # Convert to Message objects  # noqa: E501
        msgs = []  # noqa: E501
        for i, m in enumerate(ex.messages):  # noqa: E501
            msgs.append(  # noqa: E501
                Message(  # noqa: E501
                    id=i,  # noqa: E501
                    chat_id="eval",  # noqa: E501
                    sender="other" if not m["is_from_me"] else "me",  # noqa: E501
                    sender_name=None,  # noqa: E501
                    text=m["text"],  # noqa: E501
                    date=datetime.fromisoformat(m["date"]),  # noqa: E501
                    is_from_me=m["is_from_me"],  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Run segmenter  # noqa: E501
        segments = segmenter.segment(msgs)  # noqa: E501
  # noqa: E501
        # Extract boundaries: index of first message in each segment after the first  # noqa: E501
        # A boundary is where a new segment starts (index in original message list)  # noqa: E501
        detected_boundaries = []  # noqa: E501
        for seg in segments[1:]:  # Skip first segment (starts at index 0)  # noqa: E501
            # Find the index of the first message of this segment in the original list  # noqa: E501
            first_msg = seg.messages[0]  # noqa: E501
            for i, orig_m in enumerate(msgs):  # noqa: E501
                if orig_m.id == first_msg.id:  # noqa: E501
                    detected_boundaries.append(i)  # noqa: E501
                    break  # noqa: E501
  # noqa: E501
        p, r, f1 = compute_boundary_metrics(detected_boundaries, ex.gold_boundaries)  # noqa: E501
        results.append(  # noqa: E501
            SegEvalResult(  # noqa: E501
                example_name=ex.name,  # noqa: E501
                precision=p,  # noqa: E501
                recall=r,  # noqa: E501
                f1=f1,  # noqa: E501
                detected_boundaries=detected_boundaries,  # noqa: E501
                gold_boundaries=ex.gold_boundaries,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # 3. Summary  # noqa: E501
    print("\n" + "-" * 70)  # noqa: E501
    print(f"{'Example Name':30} | {'Prec':6} | {'Rec':6} | {'F1':6}")  # noqa: E501
    print("-" * 70)  # noqa: E501
    for r in results:  # noqa: E501
        print(f"{r.example_name:30} | {r.precision:0.2f} | {r.recall:0.2f} | {r.f1:0.2f}")  # noqa: E501
  # noqa: E501
    avg_f1 = sum(r.f1 for r in results) / len(results)  # noqa: E501
    print("-" * 70)  # noqa: E501
    print(f"Average F1: {avg_f1:0.2f}")  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
