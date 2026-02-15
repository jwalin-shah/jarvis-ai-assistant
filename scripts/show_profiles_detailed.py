import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage.reader import ChatDBReader
from jarvis.contacts.contact_profile import ContactProfileBuilder


def show_profiles():
    reader = ChatDBReader()
    builder = ContactProfileBuilder(min_messages=5)

    targets = [("iMessage;-;+14084643141", "Lavanya"), ("iMessage;-;+14087867207", "Mateo")]

    for chat_id, name in targets:
        # Resolve real name again just to be sure
        conv = reader.get_conversation(chat_id)
        display_name = conv.display_name if conv else name

        messages = reader.get_messages(chat_id, limit=300)
        profile = builder.build_profile(chat_id, messages, contact_name=display_name)

        print("\n" + "=" * 60)
        print(f"CONTACT PROFILE: {profile.contact_name}")
        print("=" * 60)

        print("\n[IDENTITY & RELATIONSHIP]")
        print(
            f"  Relationship: {profile.relationship} ({profile.relationship_confidence * 100:.1f}% confidence)"
        )
        if profile.relationship_reasoning:
            print(f"  Reasoning:    {profile.relationship_reasoning}")
        print(f"  ID:           {profile.contact_id}")

        print("\n[TEXTING STYLE]")
        print(f"  Formality:    {profile.formality} ({profile.formality_score:.2f})")
        print(
            f"  Message Len:  Avg {profile.avg_message_length:.0f} chars ({profile.typical_length})"
        )
        print(f"  Lowercase:    {'Yes' if profile.uses_lowercase else 'No'}")
        print(f"  Emoji Freq:   {profile.emoji_frequency:.2f}")
        print(f"  Common Slang: {', '.join(profile.common_abbreviations)}")
        print(f"  Greetings:    {', '.join(profile.greeting_style)}")

        print("\n[TOPICS]")
        if profile.top_topics:
            for t in profile.top_topics:
                print(f"  • {t}")
        else:
            print("  (Discovery pending full index)")

        print(f"\n[EXTRACTED FACTS ({len(profile.extracted_facts)})]")
        if not profile.extracted_facts:
            print("  (No verified facts in database)")
        else:
            # Group by category
            from collections import defaultdict

            cats = defaultdict(list)
            for f in profile.extracted_facts:
                cats[f["category"]].append(f)

            for cat, facts in cats.items():
                print(f"  {cat.upper()}:")
                for f in facts[:10]:
                    # Combine subject and value naturally
                    connector = f" {f['predicate']} " if f["predicate"] else ": "
                    print(f"    ✓ {f['subject']}{connector}{f['value']}")


if __name__ == "__main__":
    show_profiles()
