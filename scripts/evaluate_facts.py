"""Evaluate extracted facts quality and completeness.

Measures precision (validity of extracted facts) and suggests recall evaluation.
Flags suspicious facts (vague, bot-generated, low-quality).

Usage:
    uv run python scripts/evaluate_facts.py [--facts FILE] [--messages DB]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QualityIssue:
    """A quality issue found in a fact."""

    fact_id: str
    issue_type: str  # "vague", "bot_message", "short_phrase", "low_confidence", "overly_specific"
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str | None = None


class FactEvaluator:
    """Evaluate extracted facts for quality and coverage."""

    # Common bot message patterns
    BOT_KEYWORDS = {
        "cvs pharmacy",
        "rx ready",
        "pharmacy",
        "job at",
        "check out this job",
        "linkedin",
        "career",
        "apply now",
        "recruiter",
        "indeed",
        "glassdoor",
        "delivery",
        "order",
        "tracking",
        "confirmation",
        "receipt",
        "invoice",
    }

    # Vague/generic phrases that are low-value
    VAGUE_PHRASES = {
        "me",
        "you",
        "it",
        "that",
        "this",
        "him",
        "her",
        "them",
        "thing",
        "stuff",
        "something",
        "anything",
        "nothing",
        "everything",
        "attachment",
        "video",
        "photo",
        "image",
        "link",
        "emoji",
    }

    # Single-word facts that lose context
    SHORT_PHRASES = set(w for w in VAGUE_PHRASES if len(w) <= 4)

    # Confidence thresholds
    CONFIDENCE_THRESHOLD = 0.7  # Below this = warning
    MINIMUM_PHRASE_LENGTH = 3  # words

    def __init__(self):
        self.issues: list[QualityIssue] = []
        self.stats = defaultdict(int)

    def evaluate_fact(self, fact_dict: dict, source_text: str = "") -> list[QualityIssue]:
        """Evaluate a single fact and return quality issues found."""
        issues = []
        fact_id = f"{fact_dict.get('category', '?')}/{fact_dict.get('subject', '?')}"

        subject = fact_dict.get("subject", "").lower().strip()
        source = source_text.lower()
        confidence = fact_dict.get("confidence", 0)

        # Issue 1: Confidence too low
        if confidence < self.CONFIDENCE_THRESHOLD:
            issues.append(
                QualityIssue(
                    fact_id=fact_id,
                    issue_type="low_confidence",
                    severity="medium",
                    description=f"Confidence {confidence:.2f} below threshold {self.CONFIDENCE_THRESHOLD}",
                    suggestion="Consider increasing extraction confidence thresholds or using NLI verification",
                )
            )
            self.stats["low_confidence"] += 1

        # Issue 2: Vague subject
        if subject in self.VAGUE_PHRASES:
            issues.append(
                QualityIssue(
                    fact_id=fact_id,
                    issue_type="vague",
                    severity="high",
                    description=f"Subject '{subject}' is too vague/generic",
                    suggestion=f"Extract full phrase from context, not just '{subject}'",
                )
            )
            self.stats["vague"] += 1

        # Issue 3: Short phrase (loses context)
        word_count = len(subject.split())
        if word_count < self.MINIMUM_PHRASE_LENGTH and subject not in {"sf", "nba", "mlb"}:
            issues.append(
                QualityIssue(
                    fact_id=fact_id,
                    issue_type="short_phrase",
                    severity="medium",
                    description=f"'{subject}' is only {word_count} word(s), likely loses context",
                    suggestion=f"Extract longer phrase for clarity (e.g., 'driving in sf' not 'sf')",
                )
            )
            self.stats["short_phrase"] += 1

        # Issue 4: From bot message
        if self._is_bot_message(source):
            issues.append(
                QualityIssue(
                    fact_id=fact_id,
                    issue_type="bot_message",
                    severity="high",
                    description=f"Fact extracted from automated message (pharmacy, job posting, etc.)",
                    suggestion="Filter bot messages before extraction (e.g., check for known bot keywords)",
                )
            )
            self.stats["bot_message"] += 1

        return issues

    def _is_bot_message(self, text: str) -> bool:
        """Check if message appears to be from a bot."""
        return any(keyword in text for keyword in self.BOT_KEYWORDS)

    def evaluate_facts_batch(self, facts: list[dict], sources: dict[str, str] | None = None):
        """Evaluate multiple facts and collect stats."""
        sources = sources or {}
        all_issues = []

        for fact in tqdm(facts, desc="Evaluating facts", total=len(facts)):
            fact_id = f"{fact.get('category', '?')}/{fact.get('subject', '?')}"
            source_text = sources.get(fact_id, fact.get("source_text", ""))
            issues = self.evaluate_fact(fact, source_text)
            all_issues.extend(issues)

        self.issues = all_issues
        return all_issues


def parse_contact_facts_file(filepath: Path) -> tuple[list[dict], dict[str, str]]:
    """Parse contact_facts.txt format.

    Returns: (facts_list, sources_dict)
    """
    facts = []
    sources = {}
    current_contact = None
    current_category = None

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
                    }
                    fact_id = f"{category}/{subject}"
                    facts.append(fact)
                    current_category = category
                    continue

            # Source line: "    src: \"...\""
            if line.startswith("    src:") and facts:
                src_text = line.split('"')[1] if '"' in line else ""
                if src_text:
                    fact_id = f"{facts[-1]['category']}/{facts[-1]['subject']}"
                    sources[fact_id] = src_text

    return facts, sources


def generate_report(
    facts: list[dict],
    sources: dict[str, str],
    evaluator: FactEvaluator,
) -> str:
    """Generate a detailed evaluation report."""
    issues = evaluator.evaluate_facts_batch(facts, sources)

    report = []
    report.append("=" * 80)
    report.append("FACT EXTRACTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Total facts extracted:        {len(facts)}")
    report.append(f"Facts with issues:            {len(set(i.fact_id for i in issues))}")
    report.append(f"Total issues found:           {len(issues)}")
    report.append("")

    # Issue breakdown
    if issues:
        report.append("ISSUE BREAKDOWN")
        report.append("-" * 40)
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue.issue_type] += 1

        for issue_type in sorted(issue_counts.keys()):
            count = issue_counts[issue_type]
            pct = (count / len(facts)) * 100
            report.append(f"  {issue_type:20s}: {count:3d} ({pct:5.1f}%)")
        report.append("")

    # Category distribution
    report.append("FACTS BY CATEGORY")
    report.append("-" * 40)
    by_category = defaultdict(list)
    for fact in facts:
        by_category[fact["category"]].append(fact)

    for category in sorted(by_category.keys()):
        cats = by_category[category]
        avg_conf = sum(f["confidence"] for f in cats) / len(cats)
        issues_in_cat = len([i for i in issues if f"/{i.fact_id.split('/')[1]}" in str([f for f in cats])])
        report.append(f"  {category:15s}: {len(cats):3d} facts (avg conf: {avg_conf:.2f})")
    report.append("")

    # High-severity issues
    high_issues = [i for i in issues if i.severity == "high"]
    if high_issues:
        report.append("HIGH-SEVERITY ISSUES (RECOMMEND FIX)")
        report.append("-" * 40)
        for issue in sorted(set(i.issue_type for i in high_issues)):
            matching = [i for i in high_issues if i.issue_type == issue]
            report.append(f"\n{issue.upper()} ({len(matching)} facts):")
            for i, issue_obj in enumerate(matching[:5]):  # Show first 5
                report.append(f"  - {issue_obj.fact_id}: {issue_obj.description}")
                if issue_obj.suggestion:
                    report.append(f"    → {issue_obj.suggestion}")
            if len(matching) > 5:
                report.append(f"  ... and {len(matching) - 5} more")
        report.append("")

    # Confidence distribution
    report.append("CONFIDENCE DISTRIBUTION")
    report.append("-" * 40)
    by_conf = defaultdict(int)
    for fact in facts:
        conf_bucket = f"{int(fact['confidence'] * 10) / 10:.1f}"
        by_conf[conf_bucket] += 1

    for conf in sorted(by_conf.keys()):
        count = by_conf[conf]
        pct = (count / len(facts)) * 100
        bar = "█" * int(pct / 2)
        report.append(f"  {conf}: {bar:25s} {count:2d} ({pct:5.1f}%)")
    report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("""
1. BOT MESSAGE FILTERING
   - Filter CVS pharmacy, job posting bots before extraction
   - Check for patterns: "CVS Pharmacy", "job at", "Check out this job"
   - Skip messages with >5 bot keywords

2. SUBJECT QUALITY IMPROVEMENT
   - Reject facts with vague subjects (me, you, that, etc)
   - Require minimum 3-word phrase length
   - Extract full context, not just noun: "driving in sf" not "sf"

3. CONFIDENCE CALIBRATION
   - Current hard-coded scores (0.6-0.8) don't reflect quality
   - Use fact length + bot-filter + vagueness as confidence inputs
   - Consider NLI verification for <0.7 confidence facts

4. PRECISION VS RECALL TRADEOFF
   - Current: High recall (catch everything) + low precision (lots of noise)
   - Recommendation: Reduce recall by 30%, improve precision by 50%
   - Filter facts that fail: bot check + vagueness check + short phrase check

5. GROUND TRUTH EVALUATION
   To measure recall, you need:
   - Manually annotate 50-100 messages with expected facts
   - Extract with current system
   - Calculate: recall = (facts_found / facts_expected)
   - Currently: precision ≈ 0.6-0.7 (my estimate)
   """)

    return "\n".join(report)


LOG_PATH = Path("evaluate_facts.log")
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with both file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    _setup_logging()
    logger.info("Starting evaluate_facts.py")

    parser = argparse.ArgumentParser(description="Evaluate extracted facts quality")
    parser.add_argument(
        "--facts",
        type=Path,
        default=Path("contact_facts.txt"),
        help="Path to contact_facts.txt",
    )
    args = parser.parse_args()

    if not args.facts.exists():
        print(f"Error: {args.facts} not found", flush=True)
        sys.exit(1)

    print(f"Loading facts from {args.facts}...", flush=True)
    facts, sources = parse_contact_facts_file(args.facts)
    print(f"Loaded {len(facts)} facts from {len(set(f['contact'] for f in facts))} contacts", flush=True)
    print(flush=True)

    evaluator = FactEvaluator()
    report = generate_report(facts, sources, evaluator)
    print(report, flush=True)

    # Save report
    report_path = args.facts.parent / "fact_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}", flush=True)
    logger.info(f"Report saved to {report_path}")
    logger.info("Finished evaluate_facts.py")


if __name__ == "__main__":
    main()
