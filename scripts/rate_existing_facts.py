"""Rate existing extracted facts for precision measurement.

Loads facts from contact_facts.txt, fetches source messages,
and creates a manual rating template.

Usage:
    uv run python scripts/rate_existing_facts.py
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def _setup_logging() -> None:
    """Configure logging with FileHandler + StreamHandler."""
    log_file = Path("rate_existing_facts.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_contact_facts_file(filepath: Path) -> list[dict]:
    """Parse contact_facts.txt format into structured facts."""
    facts = []
    current_contact = None

    try:
        with filepath.open() as f:
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
    except OSError as exc:
        print(f"Error reading facts file '{filepath}': {exc}", flush=True)
        raise SystemExit(1) from exc

    return facts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts-file",
        type=Path,
        default=Path("contact_facts.txt"),
        help="Input contact facts text file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("precision_rating.md"),
        help="Output markdown rating template file (default: %(default)s).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print fact-writing progress every N facts for loops >10 (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    _setup_logging()
    logging.info("Starting rate_existing_facts.py")
    args = parse_args(argv)
    facts_file = args.facts_file
    if not facts_file.exists():
        print(f"Error: {facts_file} not found", flush=True)
        sys.exit(1)

    print(f"Loading facts from {facts_file}...", flush=True)
    facts = parse_contact_facts_file(facts_file)
    print(f"Loaded {len(facts)} facts", flush=True)

    # Create rating template
    output_file = args.output_file

    try:
        with output_file.open("w") as f:
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

            if len(facts) > 10:
                print(f"Writing {len(facts)} facts to rating template...", flush=True)

            for i, fact in enumerate(facts, 1):
                f.write(f"## Fact {i}: {fact['contact'][:20]}\n\n")
                f.write("**Source Message:**\n")
                f.write(f"> {fact['source_text']}\n\n")

                f.write("**Extracted Fact:**\n")
                f.write(f"- **Category:** {fact['category']}\n")
                f.write(f"- **Subject:** {fact['subject']}\n")
                f.write(f"- **Predicate:** {fact['predicate']}\n")
                f.write(f"- **Confidence:** {fact['confidence']:.2f}\n\n")

                f.write("**Your Rating:** [ ]\n\n")
                f.write("---\n\n")

                if (
                    len(facts) > 10
                    and args.progress_every > 0
                    and (i % args.progress_every == 0 or i == len(facts))
                ):
                    print(f"  wrote {i}/{len(facts)} facts", flush=True)

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
    except OSError as exc:
        print(f"Error writing output file '{output_file}': {exc}", flush=True)
        raise SystemExit(1) from exc

    print(f"\nRating template created: {output_file}", flush=True)
    print(f"\nNext steps:", flush=True)
    print(f"1. Open {output_file}", flush=True)
    print(f"2. Read the source message + fact for each one", flush=True)
    print(f"3. Rate each fact 1-5 (replace [ ] with [1], [2], etc.)", flush=True)
    print(f"4. Fill in the summary section at the end", flush=True)
    print(f"5. Calculate precision = (ratings 3+) / {len(facts)}", flush=True)
    print(f"\nThis will show us baseline precision before quality filters.", flush=True)
    logging.info("Finished rate_existing_facts.py")


if __name__ == "__main__":
    main()
