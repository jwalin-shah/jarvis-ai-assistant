"""Sample diverse iMessages from chat.db for fact extraction testing.

Pulls 50 messages: 25 likely to contain personal facts, 25 unlikely.
Saves to results/sample_messages.json.

Usage:
    uv run python scripts/sample_messages.py
"""

import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Apple epoch: 2001-01-01 00:00:00 UTC
APPLE_EPOCH_OFFSET = 978307200
# chat.db stores dates as nanoseconds since Apple epoch
NANOSECOND_FACTOR = 1_000_000_000

DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "sample_messages.json"

# Keywords that suggest personal facts
FACT_KEYWORDS = [
    # Family
    "mom", "dad", "brother", "sister", "wife", "husband", "son", "daughter",
    "parent", "family", "grandma", "grandpa", "uncle", "aunt", "cousin",
    # Places
    "moved to", "live in", "from", "born in", "grew up", "visiting",
    "trip to", "flight to", "staying at", "neighborhood",
    # Work/education
    "work at", "working at", "job", "hired", "promoted", "quit",
    "started at", "intern", "manager", "engineer", "school", "college",
    "university", "graduated", "studying", "major",
    # Health
    "doctor", "allergic", "allergy", "surgery", "diagnosed", "medication",
    "prescription", "hospital", "dentist", "therapy", "pregnant",
    # Hobbies/interests
    "playing", "practice", "training", "marathon", "gym", "yoga",
    "cooking", "recipe", "guitar", "piano", "painting", "hiking",
    "camping", "surfing", "climbing", "photography",
    # Food preferences
    "vegetarian", "vegan", "gluten", "favorite food", "love eating",
    "allergic to", "can't eat", "don't eat", "lactose",
    # Pets
    "dog", "cat", "puppy", "kitten", "pet",
    # Life events
    "birthday", "wedding", "engaged", "married", "divorced",
    "anniversary", "moving", "bought a house", "new apartment",
]

# Patterns that indicate non-fact messages
NONFACT_PATTERNS = [
    "ok", "lol", "haha", "yeah", "yep", "nope", "sure", "thanks",
    "thank you", "np", "no problem", "good morning", "good night",
    "hey", "hi", "hello", "what's up", "sup", "yo", "bye",
    "see you", "ttyl", "omg", "lmao", "ikr", "smh", "tbh",
    "sounds good", "got it", "k", "kk", "ooh", "nice", "cool",
    "awesome", "great", "perfect", "bet", "word", "facts",
    "for real", "fr", "nah", "idk", "wya", "otw", "bet",
]


def apple_ts_to_iso(ts: int | None) -> str | None:
    """Convert Apple nanosecond timestamp to ISO 8601 string."""
    if ts is None or ts == 0:
        return None
    seconds = ts / NANOSECOND_FACTOR + APPLE_EPOCH_OFFSET
    try:
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def connect_readonly() -> sqlite3.Connection:
    """Open chat.db in read-only mode."""
    if not DB_PATH.exists():
        print(f"ERROR: chat.db not found at {DB_PATH}", flush=True)
        print("Make sure you have Full Disk Access enabled.", flush=True)
        sys.exit(1)

    uri = f"file:{DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except (sqlite3.OperationalError, PermissionError) as e:
        print(f"ERROR: Cannot open chat.db: {e}", flush=True)
        print("Grant Full Disk Access to your terminal in System Settings.", flush=True)
        sys.exit(1)


def fetch_fact_candidates(conn: sqlite3.Connection, target: int = 25) -> list[dict]:
    """Fetch messages likely to contain personal facts.

    Strategy: longer messages (>30 chars) from me, containing fact keywords.
    Pulls a large pool and randomly samples to get diversity.
    """
    # Build LIKE clauses for keyword matching
    like_clauses = " OR ".join(
        f"LOWER(message.text) LIKE '%' || ? || '%'" for _ in FACT_KEYWORDS
    )

    query = f"""
        SELECT
            message.ROWID as id,
            message.text,
            message.is_from_me,
            message.date,
            chat.guid as chat_id
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        WHERE message.text IS NOT NULL
          AND message.text != ''
          AND LENGTH(message.text) > 30
          AND ({like_clauses})
        ORDER BY RANDOM()
        LIMIT 500
    """

    cursor = conn.cursor()
    try:
        cursor.execute(query, [kw.lower() for kw in FACT_KEYWORDS])
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"WARNING: Keyword query failed ({e}), falling back to length-based", flush=True)
        # Fallback: just get longer messages from me
        fallback_query = """
            SELECT
                message.ROWID as id,
                message.text,
                message.is_from_me,
                message.date,
                chat.guid as chat_id
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            WHERE message.text IS NOT NULL
              AND message.text != ''
              AND LENGTH(message.text) > 50
            ORDER BY RANDOM()
            LIMIT 500
        """
        cursor.execute(fallback_query)
        rows = cursor.fetchall()

    # Convert to dicts and deduplicate by text content
    seen_texts: set[str] = set()
    candidates = []
    for row in rows:
        text = row["text"]
        if not text or text in seen_texts:
            continue
        # Skip reactions (start with "Loved", "Liked", "Laughed at", etc.)
        if text.startswith(("\ufffc", "Loved", "Liked", "Laughed at", "Emphasized",
                            "Disliked", "Questioned")):
            continue
        seen_texts.add(text)
        candidates.append({
            "id": row["id"],
            "text": text,
            "is_from_me": bool(row["is_from_me"]),
            "date": apple_ts_to_iso(row["date"]),
            "chat_id": row["chat_id"],
        })

    # Prioritize is_from_me messages (more likely to contain user's facts)
    from_me = [c for c in candidates if c["is_from_me"]]
    from_others = [c for c in candidates if not c["is_from_me"]]

    # Try to get ~15 from me + ~10 from others for diversity
    result = []
    if len(from_me) >= 15:
        result.extend(random.sample(from_me, 15))
    else:
        result.extend(from_me)

    remaining = target - len(result)
    if remaining > 0:
        pool = from_others if from_others else candidates
        sample_size = min(remaining, len(pool))
        if sample_size > 0:
            result.extend(random.sample(pool, sample_size))

    # If still short, fill from any remaining candidates
    remaining = target - len(result)
    if remaining > 0:
        used_ids = {r["id"] for r in result}
        extras = [c for c in candidates if c["id"] not in used_ids]
        result.extend(random.sample(extras, min(remaining, len(extras))))

    return result[:target]


def fetch_nonfact_candidates(conn: sqlite3.Connection, target: int = 25) -> list[dict]:
    """Fetch messages unlikely to contain personal facts.

    Strategy: short messages, greetings, reactions, filler words.
    """
    query = """
        SELECT
            message.ROWID as id,
            message.text,
            message.is_from_me,
            message.date,
            chat.guid as chat_id
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        WHERE message.text IS NOT NULL
          AND message.text != ''
          AND LENGTH(message.text) <= 20
        ORDER BY RANDOM()
        LIMIT 500
    """

    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

    seen_texts: set[str] = set()
    candidates = []
    for row in rows:
        text = row["text"]
        if not text or text.strip() in seen_texts:
            continue
        # Skip reactions (tapback messages)
        if text.startswith(("\ufffc", "Loved", "Liked", "Laughed at", "Emphasized",
                            "Disliked", "Questioned")):
            continue
        seen_texts.add(text.strip())
        candidates.append({
            "id": row["id"],
            "text": text,
            "is_from_me": bool(row["is_from_me"]),
            "date": apple_ts_to_iso(row["date"]),
            "chat_id": row["chat_id"],
        })

    # Prefer messages matching known non-fact patterns for diversity
    pattern_matches = []
    other = []
    for c in candidates:
        text_lower = c["text"].strip().lower()
        if any(text_lower == pat or text_lower.startswith(pat) for pat in NONFACT_PATTERNS):
            pattern_matches.append(c)
        else:
            other.append(c)

    result = []
    # Get ~15 pattern matches + ~10 other short messages
    if len(pattern_matches) >= 15:
        result.extend(random.sample(pattern_matches, 15))
    else:
        result.extend(pattern_matches)

    remaining = target - len(result)
    if remaining > 0 and other:
        result.extend(random.sample(other, min(remaining, len(other))))

    # Fill from any remaining if needed
    remaining = target - len(result)
    if remaining > 0:
        used_ids = {r["id"] for r in result}
        extras = [c for c in candidates if c["id"] not in used_ids]
        result.extend(random.sample(extras, min(remaining, len(extras))))

    return result[:target]


def main() -> None:
    print("Sampling messages from iMessage database...", flush=True)
    print(f"  DB: {DB_PATH}", flush=True)

    conn = connect_readonly()

    # Quick stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM message WHERE text IS NOT NULL AND text != ''")
    total = cursor.fetchone()[0]
    print(f"  Total messages with text: {total:,}", flush=True)

    # Sample both categories
    print("  Fetching fact-likely messages (25)...", flush=True)
    fact_msgs = fetch_fact_candidates(conn, target=25)
    print(f"    Got {len(fact_msgs)} fact candidates", flush=True)

    print("  Fetching non-fact messages (25)...", flush=True)
    nonfact_msgs = fetch_nonfact_candidates(conn, target=25)
    print(f"    Got {len(nonfact_msgs)} non-fact candidates", flush=True)

    conn.close()

    # Combine and tag
    for msg in fact_msgs:
        msg["expected_has_facts"] = True
    for msg in nonfact_msgs:
        msg["expected_has_facts"] = False

    all_messages = fact_msgs + nonfact_msgs
    random.shuffle(all_messages)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_messages)} messages to {OUTPUT_PATH}", flush=True)
    print(f"  Fact-likely: {len(fact_msgs)}", flush=True)
    print(f"  Non-fact:    {len(nonfact_msgs)}", flush=True)

    # Show a few examples
    print("\nSample fact-likely messages:", flush=True)
    for msg in fact_msgs[:3]:
        preview = msg["text"][:80].replace("\n", " ")
        print(f"  [{msg['id']}] {'(me)' if msg['is_from_me'] else '(other)'} {preview}...", flush=True)

    print("\nSample non-fact messages:", flush=True)
    for msg in nonfact_msgs[:3]:
        print(f"  [{msg['id']}] {'(me)' if msg['is_from_me'] else '(other)'} {msg['text']}", flush=True)


if __name__ == "__main__":
    main()
