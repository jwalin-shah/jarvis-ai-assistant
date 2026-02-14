from jarvis.contacts.candidate_extractor import CandidateExtractor
import logging

logging.basicConfig(level=logging.INFO)

extractor = CandidateExtractor(label_profile="high_recall", use_entailment=False)
texts = [
    "I live in San Francisco now",
    "My sister Sarah is coming over",
    "I work at Apple as a designer",
    "I love eating sushi",
    "I am allergic to peanuts"
]

print("Testing GLiNER Extraction...")
for text in texts:
    print(f"\nText: {text}")
    candidates = extractor.extract_candidates(text, 0, use_gate=False)
    for c in candidates:
        print(f"  - Found: {c.span_text} ({c.span_label}) -> {c.fact_type} [score: {c.gliner_score:.2f}]")
