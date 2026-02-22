from jarvis.contacts.fact_storage import get_all_facts

facts = get_all_facts()
print(f"--- All Facts in DB ({len(facts)}) ---")
for f in facts:
    print(f"- [{f.category}] {f.subject} -> {f.attribution} (Contact: {f.contact_id[-15:]})")
