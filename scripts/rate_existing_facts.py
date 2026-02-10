"""Rate existing extracted facts for precision measurement.

Loads facts from contact_facts.txt, fetches source messages,
and creates a manual rating template.

Usage:
    uv run python scripts/rate_existing_facts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_contact_facts_file(filepath: Path) -> list[dict]:
    """Parse contact_facts.txt format into structured facts."""
    facts = []
    current_contact = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Contact header: "## Contact: RCS;-;+14803477659"
            if line.startswith("## Contact:"):
                current_contact = line.split(":")[1].strip()
                continue

            # Fact line: "  [category] subject predicate (conf=X.XX)"
            if line.startswith("  ["):
                match = re.match(
                    r"\s+\[(\w+)\]\s+(.+?)\s+(\w+)\s+\(conf=([\d.]+)\)",
                    line,
                )
                if match:
                    category, subject, predicate, confidence = match.groups()
                    fact = {
                        "contact": current_contact,
                        "category": category,
                        "subject": subject,
                        "predicate": predicate,
                        "confidence": float(confidence),
                        "source_text": "",
                    }
                    facts.append(fact)
                    continue

            # Source line: "    src: \"...\""
            if line.startswith("    src:") and facts:
                src_text = line.split('"')[1] if '"' in line else ""
                if src_text:
                    facts[-1]["source_text"] = src_text

    return facts


def main():
    facts_file = Path("contact_facts.txt")
    if not facts_file.exists():
        print(f"Error: {facts_file} not found")
        sys.exit(1)

    print(f"Loading facts from {facts_file}...", flush=True)
    facts = parse_contact_facts_file(facts_file)
    print(f"Loaded {len(facts)} facts", flush=True)

    # Create rating template
    output_file = Path("precision_rating.md")

    with open(output_file, "w") as f:
        f.write("# Fact Extraction Precision Test\n\n")
        f.write(f"**Total facts to rate:** {len(facts)}\n\n")

        f.write("## Instructions\n\n")
        f.write("Rate each fact on a 1-5 scale:\n")
        f.write("- **5** = Highly specific, valid, useful\n")
        f.write("- **4** = Valid, good context\n")
        f.write("- **3** = Borderline (missing context or slightly vague)\n")
        f.write("- **2** = Mostly wrong (fact doesn't match message)\n")
        f.write("- **1** = Invalid/useless\n\n")
        f.write("Edit this file and replace `[ ]` with your rating: `[5]`, `[4]`, etc.\n\n")

        f.write("---\n\n")

        for i, fact in enumerate(facts, 1):
            f.write(f"## Fact {i}: {fact['contact'][:20]}\n\n")
            f.write(f"**Source Message:**\n")
            f.write(f"> {fact['source_text']}\n\n")

            f.write(f"**Extracted Fact:**\n")
            f.write(f"- **Category:** {fact['category']}\n")
            f.write(f"- **Subject:** {fact['subject']}\n")
            f.write(f"- **Predicate:** {fact['predicate']}\n")
            f.write(f"- **Confidence:** {fact['confidence']:.2f}\n\n")

            f.write(f"**Your Rating:** [ ]\n\n")
            f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write("After rating all facts, fill in this summary:\n\n")
        f.write(f"- Total rated: ___ / {len(facts)}\n")
        f.write("- Rating 5 (excellent): ___\n")
        f.write("- Rating 4 (good): ___\n")
        f.write("- Rating 3 (borderline): ___\n")
        f.write("- Rating 2 (wrong): ___\n")
        f.write("- Rating 1 (useless): ___\n\n")
        f.write("**Precision (3+):** ___ / {} = ___%\n".format(len(facts)))
        f.write("**(Borderline + Good + Excellent combined)\n\n")
        f.write("**Precision (4+):** ___ / {} = ___%\n".format(len(facts)))
        f.write("**(Good + Excellent only - stricter)\n\n")

    print(f"\n✓ Rating template created: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Open {output_file}")
    print(f"2. Read the source message + fact for each one")
    print(f"3. Rate each fact 1-5 (replace [ ] with [1], [2], etc.)")
    print(f"4. Fill in the summary section at the end")
    print(f"5. Calculate precision = (ratings 3+) / {len(facts)}")
    print(f"\nThis will show us baseline precision before quality filters.")


if __name__ == "__main__":
    main()
