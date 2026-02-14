from jarvis.contacts.candidate_extractor import CandidateExtractor
import logging

logging.basicConfig(level=logging.INFO)

extractor = CandidateExtractor(label_profile="high_recall")
seg_msgs_data = [{
    "text": "I live in California now",
    "message_id": 1,
    "is_from_me": False,
    "chat_id": "test_chat"
}]

print("Testing extract_batch...")
candidates = extractor.extract_batch(seg_msgs_data, use_gate=False)
for c in candidates:
    print(f"  - Found: {c.span_text} ({c.span_label}) -> {c.fact_type}")
