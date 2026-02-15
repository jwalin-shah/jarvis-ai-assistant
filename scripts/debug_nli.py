import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nli_cross_encoder import get_nli_cross_encoder


def test_nli():
    nli = get_nli_cross_encoder()

    pairs = [
        ("I live in San Francisco", "The person lives in San Francisco"),
        ("I work at Google", "The person works at Google"),
        ("I am allergic to peanuts", "The person is allergic to peanuts"),
        # Casual chat style
        ("Jwalin: I live in SF now", "Jwalin lives in San Francisco"),
        ("Unknown: works at Apple", "Unknown works at Apple"),
        # Negative cases
        ("I hate coffee", "The person likes coffee"),
        ("I am going to the store", "The person works at Google"),
    ]

    print(f"{'Premise':<40} | {'Hypothesis':<40} | {'E':<6} | {'C':<6} | {'N':<6}")
    print("-" * 110)

    results = nli.predict_batch(pairs)
    for (p, h), scores in zip(pairs, results):
        print(f"{p:<40} | {h:<40} | {scores['entailment']:.3f} | {scores['contradiction']:.3f} | {scores['neutral']:.3f}")

if __name__ == "__main__":
    test_nli()
