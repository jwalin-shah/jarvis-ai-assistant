#!/usr/bin/env python3
"""
Comprehensive Contact Profiler

Profiles all contacts with:
- LLM-based relationship classification
- Temporal patterns (frequency, recency)
- Reciprocity analysis (who initiates, response times)
- Topic extraction and themes
- Style features

Usage:
    python scripts/profile_contacts.py                    # Profile all contacts
    python scripts/profile_contacts.py --limit 50        # Limit to 50 contacts
    python scripts/profile_contacts.py --refresh         # Re-profile all (ignore cache)
    python scripts/profile_contacts.py --output custom.json  # Custom output path

Output:
    results/profiles/contact_profiles.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Output paths
RESULTS_DIR = Path("results/profiles")
DEFAULT_OUTPUT = RESULTS_DIR / "contact_profiles.json"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TemporalProfile:
    """Temporal patterns for a contact."""
    last_contact_days: int
    frequency: str  # daily, frequent, occasional, inactive
    avg_msgs_per_week: float
    total_messages: int
    conversation_span_days: int
    peak_hours: list[int]
    peak_days: list[str]


@dataclass
class ReciprocityProfile:
    """Reciprocity analysis for a contact."""
    you_initiate_pct: int
    your_response_time_mins: Optional[float]
    their_response_time_mins: Optional[float]
    your_msg_ratio: float
    balance: str  # you_dominate, they_dominate, balanced
    total_conversations: int


@dataclass
class ContentProfile:
    """Content analysis for a contact."""
    topics: list[str]  # Top topic phrases
    themes: list[str]  # High-level themes
    avg_msg_length: float
    emoji_rate: float
    abbreviation_rate: float


@dataclass
class ContactProfile:
    """Complete profile for a contact."""
    name: str
    category: str  # family, partner, best_friend, friend, acquaintance, professional, group
    confidence: float
    is_group: bool
    temporal: TemporalProfile
    reciprocity: ReciprocityProfile
    content: ContentProfile
    last_updated: str


# ============================================================================
# DATA LOADING
# ============================================================================

def load_contacts(limit: Optional[int] = None, min_messages: int = 15):
    """Load contacts from iMessage database."""
    from core.imessage.reader import MessageReader

    reader = MessageReader()
    conversations = reader.get_conversations(limit=1000)

    contacts = []
    seen = set()

    for conv in conversations:
        name = conv.display_name or ""

        # Skip invalid names
        if name in seen or not name or len(name) < 2:
            continue
        if name.isdigit() and len(name) <= 6:
            continue
        if name.startswith("+1") and len(name) == 12:
            continue

        try:
            messages = reader.get_messages(conv.chat_id, limit=500)
            if not messages or len(messages) < min_messages:
                continue

            # Separate messages by sender
            my_msgs = []
            their_msgs = []

            for m in messages:
                # Skip reactions
                if not m.text:
                    continue
                if any(r in m.text.lower() for r in ["loved", "liked", "emphasized", "laughed at"]):
                    continue

                msg_data = {
                    "text": m.text,
                    "timestamp": m.timestamp,
                    "is_from_me": m.is_from_me,
                }

                if m.is_from_me:
                    my_msgs.append(msg_data)
                else:
                    their_msgs.append(msg_data)

            if len(my_msgs) < 5 or len(their_msgs) < 5:
                continue

            contacts.append({
                "name": name,
                "chat_id": conv.chat_id,
                "my_messages": my_msgs,
                "their_messages": their_msgs,
                "all_messages": sorted(my_msgs + their_msgs,
                                      key=lambda x: x["timestamp"] if x["timestamp"] else datetime.min),
                "is_group": "," in name or "+" in name,
            })
            seen.add(name)

            if limit and len(contacts) >= limit:
                break

        except Exception:
            continue

    return contacts


# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def analyze_temporal(all_messages: list, my_messages: list, their_messages: list) -> TemporalProfile:
    """Extract temporal patterns from conversation."""

    if not all_messages:
        return TemporalProfile(
            last_contact_days=999,
            frequency="inactive",
            avg_msgs_per_week=0,
            total_messages=0,
            conversation_span_days=0,
            peak_hours=[],
            peak_days=[],
        )

    # Use timezone-aware now
    from datetime import timezone
    now = datetime.now(timezone.utc)

    # Get valid timestamps
    timestamps = [m["timestamp"] for m in all_messages if m["timestamp"]]
    if not timestamps:
        return TemporalProfile(
            last_contact_days=999,
            frequency="inactive",
            avg_msgs_per_week=0,
            total_messages=len(all_messages),
            conversation_span_days=0,
            peak_hours=[],
            peak_days=[],
        )

    # Last contact
    last_msg = max(timestamps)
    last_contact_days = (now - last_msg).days if isinstance(last_msg, datetime) else 999

    # Conversation span
    first_msg = min(timestamps)
    span_days = max(1, (last_msg - first_msg).days) if isinstance(first_msg, datetime) else 1

    # Messages per week
    avg_msgs_per_week = len(all_messages) / (span_days / 7) if span_days > 0 else len(all_messages)

    # Frequency classification
    if last_contact_days > 90:
        frequency = "inactive"
    elif last_contact_days > 30:
        frequency = "occasional"
    elif avg_msgs_per_week >= 20:
        frequency = "daily"
    elif avg_msgs_per_week >= 5:
        frequency = "frequent"
    else:
        frequency = "occasional"

    # Peak hours and days
    hours = [ts.hour for ts in timestamps if isinstance(ts, datetime)]
    days = [ts.strftime("%A") for ts in timestamps if isinstance(ts, datetime)]

    hour_counts = Counter(hours)
    day_counts = Counter(days)

    peak_hours = [h for h, _ in hour_counts.most_common(3)]
    peak_days = [d for d, _ in day_counts.most_common(2)]

    return TemporalProfile(
        last_contact_days=last_contact_days,
        frequency=frequency,
        avg_msgs_per_week=round(avg_msgs_per_week, 1),
        total_messages=len(all_messages),
        conversation_span_days=span_days,
        peak_hours=peak_hours,
        peak_days=peak_days,
    )


# ============================================================================
# RECIPROCITY ANALYSIS
# ============================================================================

def analyze_reciprocity(all_messages: list, my_messages: list, their_messages: list) -> ReciprocityProfile:
    """Analyze conversation reciprocity."""

    if not all_messages or len(all_messages) < 5:
        return ReciprocityProfile(
            you_initiate_pct=50,
            your_response_time_mins=None,
            their_response_time_mins=None,
            your_msg_ratio=0.5,
            balance="balanced",
            total_conversations=0,
        )

    # Message ratio
    total = len(my_messages) + len(their_messages)
    your_ratio = len(my_messages) / total if total > 0 else 0.5

    # Who initiates conversations? (after 4+ hours of silence)
    conversation_gap = timedelta(hours=4)
    initiations = {"you": 0, "them": 0}

    sorted_msgs = [m for m in all_messages if m["timestamp"]]
    sorted_msgs.sort(key=lambda x: x["timestamp"])

    for i, msg in enumerate(sorted_msgs):
        if i == 0:
            initiations["you" if msg["is_from_me"] else "them"] += 1
        else:
            prev_ts = sorted_msgs[i-1]["timestamp"]
            curr_ts = msg["timestamp"]
            if prev_ts and curr_ts and (curr_ts - prev_ts) > conversation_gap:
                initiations["you" if msg["is_from_me"] else "them"] += 1

    total_initiations = initiations["you"] + initiations["them"]
    you_initiate_pct = int(initiations["you"] / total_initiations * 100) if total_initiations > 0 else 50

    # Response times
    your_response_times = []
    their_response_times = []

    for i in range(1, len(sorted_msgs)):
        prev = sorted_msgs[i-1]
        curr = sorted_msgs[i]

        if not prev["timestamp"] or not curr["timestamp"]:
            continue

        time_diff = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 60

        # Only count responses within 24 hours
        if time_diff > 24 * 60 or time_diff < 0:
            continue

        if prev["is_from_me"] and not curr["is_from_me"]:
            their_response_times.append(time_diff)
        elif not prev["is_from_me"] and curr["is_from_me"]:
            your_response_times.append(time_diff)

    your_response = round(np.median(your_response_times), 1) if your_response_times else None
    their_response = round(np.median(their_response_times), 1) if their_response_times else None

    # Balance classification
    if your_ratio > 0.65:
        balance = "you_dominate"
    elif your_ratio < 0.35:
        balance = "they_dominate"
    else:
        balance = "balanced"

    return ReciprocityProfile(
        you_initiate_pct=you_initiate_pct,
        your_response_time_mins=your_response,
        their_response_time_mins=their_response,
        your_msg_ratio=round(your_ratio, 2),
        balance=balance,
        total_conversations=total_initiations,
    )


# ============================================================================
# CONTENT ANALYSIS
# ============================================================================

STOP_WORDS = {
    "i", "you", "the", "a", "an", "to", "and", "is", "it", "that", "of", "in",
    "for", "on", "my", "me", "we", "be", "was", "are", "have", "has", "do",
    "did", "will", "would", "could", "should", "can", "just", "so", "but",
    "if", "or", "not", "this", "what", "when", "how", "who", "where", "why",
    "im", "its", "dont", "yeah", "yea", "yes", "no", "ok", "okay", "lol",
    "haha", "like", "get", "got", "go", "going", "went", "come", "know",
    "think", "want", "need", "see", "let", "make", "take", "good", "one",
    "out", "up", "about", "with", "from", "they", "them", "their", "there",
    "here", "now", "then", "also", "too", "very", "really", "some", "all",
    "ur", "u", "r", "gonna", "wanna", "gotta", "thats", "ill", "youre",
}

THEME_PATTERNS = {
    "sports/gaming": r'\b(game|play|watch|score|team|win|lose|fantasy|draft)\b',
    "food/dining": r'\b(eat|dinner|lunch|food|restaurant|cook|hungry|brunch)\b',
    "work/career": r'\b(work|job|meeting|interview|boss|office|project|deadline)\b',
    "education": r'\b(study|class|exam|school|college|homework|grad)\b',
    "travel": r'\b(trip|travel|flight|hotel|vacation|airport)\b',
    "social": r'\b(party|drink|bar|club|birthday|hangout|hang)\b',
    "entertainment": r'\b(movie|show|netflix|watch|episode|music|concert)\b',
    "family": r'\b(family|mom|dad|brother|sister|parent|home)\b',
    "relationships": r'\b(date|relationship|love|boyfriend|girlfriend)\b',
    "logistics": r'\b(car|drive|uber|pickup|address|coming|leaving)\b',
}


def analyze_content(all_messages: list, my_messages: list) -> ContentProfile:
    """Analyze conversation content."""

    texts = [m["text"] for m in all_messages if m["text"]]
    my_texts = [m["text"] for m in my_messages if m["text"]]
    all_text = " ".join(texts).lower()

    # Extract bigrams as topics
    bigrams = Counter()
    for text in texts:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1

    topics = [p for p, c in bigrams.most_common(8) if c >= 2]

    # Detect themes
    themes = []
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, all_text):
            themes.append(theme)

    # Style metrics
    if my_texts:
        avg_length = np.mean([len(t) for t in my_texts])
        emoji_rate = sum(1 for t in my_texts if re.search(r'[\U0001F600-\U0001F64F]', t)) / len(my_texts)
        abbrev_rate = sum(1 for t in my_texts if re.search(r'\b(u|ur|rn|tmrw|idk|ngl|tbh)\b', t.lower())) / len(my_texts)
    else:
        avg_length = 0
        emoji_rate = 0
        abbrev_rate = 0

    return ContentProfile(
        topics=topics[:6],
        themes=themes[:4],
        avg_msg_length=round(avg_length, 1),
        emoji_rate=round(emoji_rate, 2),
        abbreviation_rate=round(abbrev_rate, 2),
    )


# ============================================================================
# LLM CLASSIFICATION
# ============================================================================

def build_classification_prompt(name: str, conversation_sample: str,
                                temporal: TemporalProfile,
                                reciprocity: ReciprocityProfile,
                                content: ContentProfile) -> str:
    """Build a structured prompt for LLM classification."""

    # Build context summary
    context_parts = [
        f"Last contact: {temporal.last_contact_days} days ago",
        f"Frequency: {temporal.frequency} ({temporal.avg_msgs_per_week} msgs/week)",
        f"You initiate: {reciprocity.you_initiate_pct}%",
        f"Balance: {reciprocity.balance}",
    ]
    if content.themes:
        context_parts.append(f"Themes: {', '.join(content.themes[:3])}")

    context = " | ".join(context_parts)

    prompt = f"""Classify the relationship between "me" and "{name}".

CONTEXT: {context}

RECENT MESSAGES:
{conversation_sample}

CATEGORIES (choose ONE):
- FAMILY: Parent, sibling, relative, family group chat
- PARTNER: Romantic partner, significant other
- BEST_FRIEND: Very close friend, confidant, talk daily
- FRIEND: Regular friend, hang out sometimes
- ACQUAINTANCE: Casual contact, specific activity only
- PROFESSIONAL: Coworker, business contact, networking
- GROUP: Group chat with multiple people

Reply with ONLY the category name.

CATEGORY:"""

    return prompt


def classify_with_llm(name: str, conversation_sample: str,
                      temporal: TemporalProfile,
                      reciprocity: ReciprocityProfile,
                      content: ContentProfile,
                      loader) -> tuple[str, float]:
    """Use LLM to classify relationship."""

    prompt = build_classification_prompt(name, conversation_sample, temporal, reciprocity, content)

    try:
        result = loader.generate(prompt, max_tokens=20, temperature=0.1)
        response_text = result.text if hasattr(result, 'text') else str(result)
        response_text = response_text.strip().upper()

        valid_categories = {
            "FAMILY": "family",
            "PARTNER": "partner",
            "BEST_FRIEND": "best_friend",
            "FRIEND": "friend",
            "ACQUAINTANCE": "acquaintance",
            "PROFESSIONAL": "professional",
            "GROUP": "group",
        }

        for key, value in valid_categories.items():
            if key in response_text:
                # Estimate confidence based on response clarity
                confidence = 0.85 if response_text.startswith(key) else 0.7
                return value, confidence

        return "friend", 0.5  # Default

    except Exception as e:
        print(f"    LLM error for {name}: {e}")
        return "friend", 0.3


def classify_with_heuristics(name: str, temporal: TemporalProfile,
                             reciprocity: ReciprocityProfile,
                             content: ContentProfile,
                             is_group: bool) -> tuple[str, float]:
    """Fallback heuristic classification (no LLM)."""

    if is_group:
        return "group", 1.0

    name_lower = name.lower()
    all_themes = " ".join(content.themes)

    # Strong name signals
    if any(w in name_lower for w in ["mom", "dad", "family", "kaki", "uncle", "aunt", "grandma"]):
        return "family", 0.9

    # Check themes and patterns
    if "family" in all_themes:
        return "family", 0.7

    if temporal.frequency == "daily" and reciprocity.balance == "balanced":
        if "work/career" in all_themes:
            return "professional", 0.6
        return "best_friend", 0.6

    if "work/career" in all_themes and "social" not in all_themes:
        return "professional", 0.6

    if temporal.frequency in ["daily", "frequent"]:
        return "friend", 0.6

    return "acquaintance", 0.5


# ============================================================================
# MAIN PROFILER
# ============================================================================

def profile_contacts(limit: Optional[int] = None,
                     use_llm: bool = True,
                     refresh: bool = False,
                     output_path: Optional[Path] = None,
                     model_id: str = "lfm2-2.6b-exp") -> dict:
    """Profile all contacts and save results."""

    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for cached results
    if not refresh and output_path.exists():
        print(f"Loading cached results from {output_path}")
        print("Use --refresh to re-profile all contacts")
        with open(output_path) as f:
            return json.load(f)

    print("=" * 70)
    print("COMPREHENSIVE CONTACT PROFILER")
    print("=" * 70)

    # Load contacts
    print("\nLoading contacts from iMessage...")
    contacts = load_contacts(limit=limit)
    print(f"Found {len(contacts)} contacts with sufficient messages")

    if not contacts:
        print("No contacts found!")
        return {"profiles": [], "by_category": {}}

    # Load LLM if needed
    loader = None
    if use_llm:
        print("\nLoading LLM for classification...")
        try:
            from core.models.loader import ModelLoader
            # Use LFM2-2.6B-Exp for best instruction following (88%+ on IFEval)
            loader = ModelLoader(model_id=model_id)
            print(f"Using model: {loader.model_id}")
        except Exception as e:
            print(f"Could not load LLM: {e}")
            print("Falling back to heuristic classification")
            use_llm = False

    # Profile each contact
    profiles = []
    by_category = defaultdict(list)

    print(f"\nProfiling {len(contacts)} contacts...")
    print("-" * 70)

    for i, contact in enumerate(contacts):
        name = contact["name"]
        is_group = contact["is_group"]

        # Extract features
        temporal = analyze_temporal(
            contact["all_messages"],
            contact["my_messages"],
            contact["their_messages"]
        )

        reciprocity = analyze_reciprocity(
            contact["all_messages"],
            contact["my_messages"],
            contact["their_messages"]
        )

        content = analyze_content(
            contact["all_messages"],
            contact["my_messages"]
        )

        # Classify
        if is_group:
            category, confidence = "group", 1.0
        elif use_llm and loader:
            # Build conversation sample
            sample_msgs = []
            for m in contact["all_messages"][-20:]:
                prefix = "me:" if m["is_from_me"] else "them:"
                sample_msgs.append(f"{prefix} {m['text'][:80]}")
            conversation_sample = "\n".join(sample_msgs[-10:])

            category, confidence = classify_with_llm(
                name, conversation_sample, temporal, reciprocity, content, loader
            )
        else:
            category, confidence = classify_with_heuristics(
                name, temporal, reciprocity, content, is_group
            )

        # Create profile
        profile = ContactProfile(
            name=name,
            category=category,
            confidence=confidence,
            is_group=is_group,
            temporal=temporal,
            reciprocity=reciprocity,
            content=content,
            last_updated=datetime.now().isoformat(),
        )

        profiles.append(profile)
        by_category[category].append(profile)

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(contacts):
            print(f"  Processed {i + 1}/{len(contacts)} contacts...")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for category in ["family", "partner", "best_friend", "friend", "acquaintance", "professional", "group"]:
        members = by_category.get(category, [])
        if not members:
            continue

        active = sum(1 for m in members if m.temporal.last_contact_days < 30)
        names = [m.name for m in sorted(members, key=lambda x: x.temporal.last_contact_days)[:5]]

        print(f"\n{category.upper()}: {len(members)} contacts ({active} active)")
        print(f"  Top: {', '.join(names)}")

    # Save results
    results = {
        "total_contacts": len(profiles),
        "generated_at": datetime.now().isoformat(),
        "used_llm": use_llm,
        "by_category": {k: [p.name for p in v] for k, v in by_category.items()},
        "profiles": [asdict(p) for p in profiles],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to {output_path}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Profile all contacts with LLM classification")
    parser.add_argument("--limit", type=int, help="Limit number of contacts to profile")
    parser.add_argument("--no-llm", action="store_true", help="Use heuristics only (no LLM)")
    parser.add_argument("--refresh", action="store_true", help="Re-profile all contacts (ignore cache)")
    parser.add_argument("--output", type=str, help="Custom output path")
    parser.add_argument("--model", type=str, default="lfm2-2.6b-exp",
                       help="Model to use for classification (default: lfm2-2.6b-exp)")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    profile_contacts(
        limit=args.limit,
        use_llm=not args.no_llm,
        refresh=args.refresh,
        output_path=output_path,
        model_id=args.model,
    )


if __name__ == "__main__":
    main()
