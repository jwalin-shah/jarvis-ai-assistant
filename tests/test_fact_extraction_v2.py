import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from jarvis.contacts.fact_filter import is_fact_likely

logging.basicConfig(level=logging.INFO)

def test_gate():
    print("\n--- Testing Message Gate ---")
    messages = [
        "I love sushi and live in Austin",
        "My sister works at Google",
        "ok sounds good",
        "see you at 5pm",
        "I'm allergic to peanuts",
    ]

    for msg in messages:
        likely = is_fact_likely(msg)
        status = "KEEP" if likely else "SKIP"
        print(f"[{status}] {msg}")

if __name__ == "__main__":
    test_gate()
