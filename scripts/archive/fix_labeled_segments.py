#!/usr/bin/env python3
"""Apply cross-validation corrections to segments_labeled.json.

Fixes identified by Kimi and Claude cross-validation:
1. Remove circular reasoning facts (family from contact names)
2. Remove temporary state facts
3. Fix wrong fact types
4. Fix misattributions
5. Remove joke/banter facts
6. Remove security-sensitive facts (passwords, PINs)
7. Fix confidence scores
8. Remove trivial relationships
9. Add missed facts
"""
import json
from pathlib import Path

INPUT = Path("training_data/segment_eval/segments_labeled.json")
OUTPUT = Path("training_data/segment_eval/segments_labeled_fixed.json")

def main() -> None:
    with open(INPUT) as f:
        data = json.load(f)

    label_map = {d["segment_id"]: d for d in data}
    stats = {"removed": 0, "modified": 0, "added": 0}

    # ── REMOVALS: Facts to delete entirely ──

    removals = {
        # Circular reasoning: family facts from contact names
        "f839ec81-b683-4526-b64a-ea239b8d77ee": "all",  # Dad/Mom/Son from contact names
        "20f4beb9-f6e2-4eab-8f44-74cb8939b62e": "all",  # Mom: mother of Jwalin from name
        "25e5f891-de51-4f47-8d6d-7a263f7dcad2": "all",  # Jwalin: son of Dad from name
        "a46a9afa-6dd9-4a1a-b696-5a4ff06f0a90": "all",  # Nita Aunty: aunt from name
        "d185c4b8-f257-44b7-af56-00dc878e3448": "all",  # Mihir: family of Mom
        "61f50b6b-c679-4fa6-8a50-0576b2fd56a5": "all",  # Mihir is brother, span="8:30"
        # Security-sensitive
        "bf2b131a-06ea-49ba-bc44-a09ca1141133": "all",  # email + password in plaintext
    }

    # Facts to remove by matching person+value substring
    targeted_removals = {
        "2d3fe57a-c440-425c-aff1-3b751e533c01": [
            ("Krusha Shah", "related to Mihir"),  # No evidence, just last name
        ],
        "4f0de35f-791c-4990-afef-6f769754e0d3": [
            ("Mihir Shah", "has a job"),  # "has work tomorrow" ≠ has a job
        ],
        "a1b0e302-013e-4c41-8833-678205161146": [
            ("Mihir Shah", "is employed"),  # "working" could mean anything
        ],
        "564e0669-7fa7-4b1c-a574-e9914f3388bd": [
            ("Jwalin", "Software Engineer"),  # Brian was guessing, not confirmed
        ],
        "80d2bab9-8a41-4e76-a99e-4f1f1d1d151a": [
            ("Nilesh Tanna", "desai_madangopal"),  # Not his ID
        ],
        "a37e4641-307b-40c6-a897-b045fdad9422": [
            ("Mihir Shah", "likes banana chips"),  # "booo" = upset, not preference
            ("Mihir Shah", "cooks"),  # One-time cooking ≠ hobby
            ("Sangati Shah", "attends conferences"),  # One conference ≠ job trait
        ],
        "df082869-7904-4ab5-82e5-37875febe151": [
            ("Mami", "iPhone 8"),  # Temporary device ownership
            ("Kaki", "iPhone X"),  # Temporary device ownership
            ("Mihir Shah", "not have a phone"),  # Temporary
        ],
        "e9995d05-4bc7-4515-a700-1cf0e2a0e084": [
            ("Gaurav", "Does not use drugs"),  # Joke context
            ("Ram", "notable skin"),  # Joke context
        ],
        "df642eec-5254-4abd-9ef9-88a3930f2631": [
            ("Jwalin", "was previously fired"),  # Joke between friends
        ],
        "410a14a8-b52b-4a87-a8bb-18bc1a761535": [
            ("Maya Leone", "works"),  # Too vague
        ],
        "449a1138-6a1c-4a43-824f-5bcdc3c826cd": [
            ("Rithvik Sai", "Grain bowl"),  # Specific order, not enduring fact
        ],
        "62c9077b-045e-43c0-a329-29b86119a9ed": [
            ("Mihir Shah", "reservation at Eleven Madison"),  # Temporary event
        ],
        "d4700a8c-3dcc-418a-b233-3feeb31badcc": [
            ("Tejas Polkham", "habitually late"),  # Banter from friends
        ],
        "56f41838-9805-4e95-a065-718ac623a0fe": [
            ("Mihir Shah", "Is able to drive"),  # Not personality
        ],
        "8c5f6506-04de-4209-bb8e-ccd69ae2b5c4": [
            ("Jwalin", "$150 Apple gift card"),  # Temporary ownership
        ],
        "bedda0a0-6a4c-472a-a165-81134cb30ce2": [
            ("Abhinav Selvaraj", "has a dad"),  # Everyone has a dad
            ("Jwalin", "has a dad"),  # Everyone has a dad
        ],
        "c23fb146-a000-4869-b4b8-c662dd8100a5": [
            ("Jwalin", "has a mom and a dad"),  # Trivial
        ],
        "56078b77-0daa-4f21-b67e-8f11a2fe0340": [
            ("Aarohi", "skiing"),  # Saying "I'm so down" ≠ hobby
            ("Yash Tanna", "skiing"),  # "Let's do it" ≠ hobby
            ("Disha", "skiing"),  # "Sameee" ≠ hobby
        ],
        "3323bd9d-33f5-4228-8237-9a63382ec42d": [
            ("Noe", "plays in sports playoffs"),  # Over-inferred
        ],
        "07a9828a-ad13-46bc-9f7b-948ca09a1dfe": [
            ("jeremy allen", "owns an air mattress"),  # Not a preference
        ],
        "1faf448a-dde1-4a59-98ba-47103ee575fc": [
            ("jeremy allen", "drives and can host"),  # Not personality
        ],
        "e121802b-35c5-4f73-9cac-4bbca0b952a7": [
            ("Mihir Shah", "Family member of Jwalin"),  # Circular from context
        ],
        "1d8bcef4-7175-45d9-8c3c-08faa43a8273": [
            ("Jwalin", "Has parents"),  # Trivial
        ],
        "7bae936a-e1f2-4744-ae5f-289e15c8d187": [
            ("Priyal", "Has a father"),  # Trivial
        ],
        "2058a0a0-0000-0000-0000-000000000000": [],  # placeholder
        "149b6f94-cea5-4a64-8c34-8f60f9b0f7ec": [
            ("Soham", "eating lunch"),  # Temporary activity
            ("Krusha", "eating lunch"),  # Temporary activity
            ("Aadi", "eating lunch"),  # Temporary activity
            ("Sangati Shah", "Has a contact named Mihir"),  # Obvious family
        ],
        "3ef40014-80a3-48f3-b541-bff84a547071": [
            ("isha jha", "has a fake ID"),  # Sensitive + not personality
        ],
        # Kimi-identified removals
        "485bc792-40c2-4a4c-bcf0-887ec418c013": [
            ("Jwalin", "school starts on August 19th"),  # Temporal, not personal
        ],
        "f3dc319e-0b31-4e68-a741-20b87ac2a7c8": [
            ("Pranav Pradhan", "involved with Yuva Kendra"),  # Over-inferred
        ],
        "deb85927-10ee-4221-8153-69c8f2315719": [
            ("Jwalin", "Enrolled in an Algorithms class"),  # Mason said it about his class
        ],
    }

    # ── MODIFICATIONS: Fix fact types, confidence, values ──

    modifications = {
        "c57c82a2-0eee-4d2d-b74f-cfa4e990d36b": [
            {
                "match": ("Asian Tim", "Tall physical stature"),
                "action": "remove",  # Banter, not verified
            },
            {
                "match": ("Asian Tim", "Taking a physics class"),
                "action": "update",
                "updates": {"confidence": 0.7, "value": "Taking a physics class (inferred from 'about to take physics test')"},
            },
        ],
        "3491ae44-7358-48c8-aa26-05c99e9b30a5": [
            {
                "match": ("Asian Tim", "Based in Austin"),
                "action": "update",
                "updates": {"confidence": 0.7, "value": "Likely based in Austin (returned from Denver)"},
            },
        ],
        "c11406ef-6106-4e59-b245-2c659c90dd2b": [
            {
                "match": ("Noe", "business"),
                "action": "update",
                "updates": {"confidence": 0.6, "value": "Possibly studies business (unconfirmed)"},
            },
            {
                "match": ("Jwalin", "cuts flat footed"),
                "action": "update",
                "updates": {"fact_type": "hobby", "value": "Has flat-footed cutting technique in ultimate frisbee"},
            },
        ],
        "4b87f092-5f15-49b0-b541-14b96c27b5af": [
            {
                "match": ("Noe", "email"),
                "action": "update",
                "updates": {"fact_type": "preference"},  # contact_info doesn't exist, preference is closer
            },
        ],
        "93357606-907f-4ba8-b296-4ac224e49e2e": [
            {
                "match": ("Jwalin", "2009 Chevy Cobalt"),
                "action": "update",
                "updates": {"fact_type": "preference"},
            },
        ],
        "a6283cd3-fc67-4bad-9947-4c09fc425783": [
            {
                "match": ("Jwalin", "Patelco"),
                "action": "update",
                "updates": {"fact_type": "preference"},
            },
        ],
        "4c1f7879-b073-4e68-bacb-a040020e50e0": [
            {
                "match": ("Vansh Jain", "University of Texas at Dallas"),
                "action": "update",
                "updates": {"confidence": 1.0},  # "i'm only at UTD" is clear
            },
        ],
        "0eed0ec8-b471-43dc-a127-18f07a363a9a": [
            {
                "match": ("Kimiya Ganjooi", "roller"),
                "action": "update",
                "updates": {"confidence": 0.9},  # "I do have a roller" is clear
            },
        ],
        "a34e2d07-1cd9-4617-a291-7c6a6ea080f6": [
            {
                "match": ("Sheethal", "Lives in Dallas"),
                "action": "update",
                "updates": {"confidence": 0.7, "value": "Likely lives near Dallas"},
            },
        ],
        "3b3e08e2-2e4e-4a76-a5db-cb826d17d435": [
            {
                "match": ("Jwalin", "has disabilities"),
                "action": "update",
                "updates": {"confidence": 0.5, "value": "May have disabilities (mentioned in insurance context)"},
            },
        ],
        "12619ff4-8bb3-475f-8bb7-79835bd10cfa": [
            {
                "match": ("Ashutosh Kulkarni", "cars"),
                "action": "update",
                "updates": {"fact_type": "preference", "value": "Knowledgeable about cars"},
            },
        ],
        "bedda0a0-6a4c-472a-a165-81134cb30ce2": [
            {
                "match": ("Abhinav Selvaraj", "Costco membership"),
                "action": "update",
                "updates": {"fact_type": "preference"},  # Not really a preference but closer
            },
        ],
        "417c5dc9-c343-4b8d-a6f1-5ce8aab50250": [
            {
                "match": ("Anirudh", "Dropped out"),
                "action": "update",
                "updates": {"value": "Dropped out of school to work on his own company"},
            },
        ],
        "6ad78a75-115e-4f57-8442-03eaa8230d81": [
            {
                "match": ("Ashutosh Kulkarni", "Korean culture"),
                "action": "update",
                "updates": {"value": "Described as a koreaboo (fan of Korean culture) by friends", "confidence": 0.7},
            },
        ],
    }

    # ── ADDITIONS: Missed facts to add ──

    additions = {
        "12619ff4-8bb3-475f-8bb7-79835bd10cfa": [
            {
                "person": "Jwalin",
                "fact_type": "preference",
                "value": "Looking to buy a used car",
                "confidence": 1.0,
                "span_text": "I'm looking to buy a used car",
            },
            {
                "person": "Jwalin",
                "fact_type": "family",
                "value": "Has a cousin who recently got married (honeymoon)",
                "confidence": 0.9,
                "span_text": "my cousin and his wife are flying in tomorrow morning and leaving for their honeymoon",
            },
        ],
        "c30244ca-8727-4190-929a-3e26faf5d805": [
            {
                "person": "Jwalin",
                "fact_type": "preference",
                "value": "Wants to attend a hockey game",
                "confidence": 1.0,
                "span_text": "I've been wanting to go to a hockey game",
            },
        ],
        "0c3428bd-af76-49db-b4b8-b271c52feea4": [
            {
                "person": "Deevy Bhimani",
                "fact_type": "education",
                "value": "Recently graduated",
                "confidence": 1.0,
                "span_text": "Congrats on graduating brother",
            },
        ],
    }

    # ── APPLY CHANGES ──

    for seg_id, action in removals.items():
        if seg_id not in label_map:
            continue
        if action == "all":
            old_count = len(label_map[seg_id].get("facts", []))
            label_map[seg_id]["facts"] = []
            stats["removed"] += old_count

    for seg_id, targets in targeted_removals.items():
        if seg_id not in label_map:
            continue
        facts = label_map[seg_id].get("facts", [])
        new_facts = []
        for fact in facts:
            should_remove = False
            for person_match, value_match in targets:
                if (person_match.lower() in fact.get("person", "").lower() and
                        value_match.lower() in fact.get("value", "").lower()):
                    should_remove = True
                    break
            if should_remove:
                stats["removed"] += 1
            else:
                new_facts.append(fact)
        label_map[seg_id]["facts"] = new_facts

    for seg_id, mods in modifications.items():
        if seg_id not in label_map:
            continue
        facts = label_map[seg_id].get("facts", [])
        new_facts = []
        for fact in facts:
            matched = False
            for mod in mods:
                person_match, value_match = mod["match"]
                if (person_match.lower() in fact.get("person", "").lower() and
                        value_match.lower() in fact.get("value", "").lower()):
                    matched = True
                    if mod["action"] == "remove":
                        stats["removed"] += 1
                    elif mod["action"] == "update":
                        for k, v in mod["updates"].items():
                            fact[k] = v
                        new_facts.append(fact)
                        stats["modified"] += 1
                    break
            if not matched:
                new_facts.append(fact)
        label_map[seg_id]["facts"] = new_facts

    for seg_id, new_facts in additions.items():
        if seg_id not in label_map:
            continue
        # Check for duplicates before adding
        existing_values = {f.get("value", "").lower() for f in label_map[seg_id].get("facts", [])}
        for fact in new_facts:
            if fact["value"].lower() not in existing_values:
                label_map[seg_id]["facts"].append(fact)
                stats["added"] += 1

    # ── WRITE OUTPUT ──

    result = list(label_map.values())
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    total_facts = sum(len(d.get("facts", [])) for d in result)
    with_facts = sum(1 for d in result if d.get("facts"))

    print(f"Fixed labels written to {OUTPUT}", flush=True)
    print(f"  Removed: {stats['removed']} facts", flush=True)
    print(f"  Modified: {stats['modified']} facts", flush=True)
    print(f"  Added: {stats['added']} facts", flush=True)
    print(f"  Total remaining: {total_facts} facts across {with_facts} segments", flush=True)


if __name__ == "__main__":
    main()
