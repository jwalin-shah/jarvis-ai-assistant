#!/usr/bin/env python3
"""Benchmark the response mobilization classifier on actual test cases."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.classifiers.response_mobilization import (
    classify_response_pressure,
    to_legacy_category,
    ResponsePressure,
)

# Test cases with expected legacy categories
TESTS = [
    # Musings - should be ACKNOWLEDGEABLE (LOW pressure)
    ("I wonder if bulls make another trade", "ACKNOWLEDGEABLE"),
    ("Kind of curious what they will offer", "ACKNOWLEDGEABLE"),
    ("I wonder what happened", "ACKNOWLEDGEABLE"),
    ("Curious how they'll handle it", "ACKNOWLEDGEABLE"),
    ("Wonder if anyone else noticed", "ACKNOWLEDGEABLE"),

    # Opinions - should be ACKNOWLEDGEABLE
    ("No way Dallas could've gotten lottery picks", "ACKNOWLEDGEABLE"),
    ("No way that's real", "ACKNOWLEDGEABLE"),
    ("No chance they win tonight", "ACKNOWLEDGEABLE"),
    ("I don't think it's gonna happen", "ACKNOWLEDGEABLE"),
    ("Doubt they'll actually do it", "ACKNOWLEDGEABLE"),

    # Rhetorical questions - should be ACKNOWLEDGEABLE
    ("Why do dads text like that", "ACKNOWLEDGEABLE"),
    ("How does that even work", "ACKNOWLEDGEABLE"),
    ("Who even says that anymore", "ACKNOWLEDGEABLE"),

    # Actual questions WITH ? - should be ANSWERABLE (HIGH pressure)
    ("What time is the game?", "ANSWERABLE"),
    ("Where are you?", "ANSWERABLE"),
    ("Did you get my text?", "ANSWERABLE"),
    ("What happened at the meeting?", "ANSWERABLE"),

    # Actual questions WITHOUT ? - tricky, could be either
    ("What time is the game", "ANSWERABLE"),
    ("Where are you", "ANSWERABLE"),

    # Requests - should be ACTIONABLE (HIGH pressure)
    ("Can you pick me up?", "ACTIONABLE"),
    ("Can you pick me up", "ACTIONABLE"),
    ("Wanna grab lunch?", "ACTIONABLE"),
    ("Wanna grab lunch", "ACTIONABLE"),
    ("Let me know when you're free", "ACTIONABLE"),
    ("Text me when you get there", "ACTIONABLE"),
    ("Pick me up at 5", "ACTIONABLE"),

    # Reactive - should be REACTIVE (MEDIUM pressure)
    ("Omg I got the job!!", "REACTIVE"),
    ("That's so sad", "REACTIVE"),
    ("This is amazing!!", "REACTIVE"),
    ("I can't believe it!!!", "REACTIVE"),

    # Statements - should be ACKNOWLEDGEABLE (LOW pressure)
    ("I went to the store", "ACKNOWLEDGEABLE"),
    ("The game starts at 7", "ACKNOWLEDGEABLE"),
    ("I'm heading out now", "ACKNOWLEDGEABLE"),
    ("Traffic was bad today", "ACKNOWLEDGEABLE"),

    # Backchannels - should be ACKNOWLEDGEABLE (NONE pressure)
    ("ok", "ACKNOWLEDGEABLE"),
    ("lol", "ACKNOWLEDGEABLE"),
    ("sounds good", "ACKNOWLEDGEABLE"),
    ("yeah", "ACKNOWLEDGEABLE"),
]


def main():
    print("Response Mobilization Classifier Benchmark")
    print("=" * 70)
    print(f"Testing on {len(TESTS)} cases")
    print()

    correct = 0
    misclassifications = []

    for text, expected in TESTS:
        result = classify_response_pressure(text)
        got = to_legacy_category(result)

        if got == expected:
            correct += 1
        else:
            misclassifications.append((text, expected, got, result))

    accuracy = correct / len(TESTS)
    print(f"Accuracy: {correct}/{len(TESTS)} ({accuracy:.1%})")
    print()

    if misclassifications:
        print(f"Misclassifications ({len(misclassifications)}):")
        for text, expected, got, result in misclassifications:
            print(f"  '{text[:45]:<45s}'")
            print(f"      expected={expected:14s} got={got:14s}")
            print(f"      pressure={result.pressure.value}, type={result.response_type.value}")
            # Show which features were detected
            active_features = [k for k, v in result.features.items() if v]
            if active_features:
                print(f"      features: {', '.join(active_features)}")
            print()
    else:
        print("No misclassifications!")

    # Show some example classifications
    print("\n" + "=" * 70)
    print("Sample Classifications")
    print("=" * 70)
    samples = [
        "I wonder if they'll win",
        "Do you know if they won?",
        "Can you help me?",
        "I think they'll win",
        "They won!!",
        "ok cool",
    ]
    for text in samples:
        result = classify_response_pressure(text)
        legacy = to_legacy_category(result)
        print(f"  '{text}'")
        print(f"      -> {result.pressure.value:6s} / {result.response_type.value:12s} ({legacy})")
        print()


if __name__ == "__main__":
    main()
