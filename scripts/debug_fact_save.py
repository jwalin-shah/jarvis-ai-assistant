from jarvis.contacts.candidate_extractor import CandidateExtractor
from jarvis.contacts.fact_storage import save_candidate_facts
import logging
from jarvis.db import get_db

logging.basicConfig(level=logging.INFO)

extractor = CandidateExtractor(label_profile="high_recall")
text = "I live in California now"
chat_id = "test_chat"

print(f"Text: {text}")
candidates = extractor.extract_candidates(text, 0, use_gate=False)
for c in candidates:
    print(f"  - Found Candidate: {c.span_text} ({c.span_label}) -> {c.fact_type}")

if candidates:
    inserted = save_candidate_facts(candidates, chat_id)
    print(f"\nInserted {inserted} facts into DB")
else:
    print("\nNo candidates found to save")

# Check DB
db = get_db()
with db.connection() as conn:
    rows = conn.execute("SELECT * FROM contact_facts").fetchall()
    print(f"Total facts in DB: {len(rows)}")
    for r in rows:
        print(f"  DB Row: {r['category']} | {r['subject']} | {r['predicate']}")
