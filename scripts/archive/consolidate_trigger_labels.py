#!/usr/bin/env python3
"""
Consolidate trigger labels from 8-label taxonomy to 5-label taxonomy.

Creates new consolidated files while keeping originals.

Mapping:
    bad_news  -> statement
    good_news -> statement
    yn_question -> question
    info_question -> question
    invitation -> commitment
    statement, ack, greeting -> unchanged
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import argparse

LABEL_MAP = {
    "statement": "statement",
    "ack": "ack",
    "greeting": "greeting",
    "bad_news": "statement",
    "good_news": "statement",
    "yn_question": "question",
    "info_question": "question",
    "invitation": "commitment",
}


class LabelConsolidator:
    def __init__(self, label_map: Optional[Dict[str, str]] = None):
        self.label_map = label_map if label_map else LABEL_MAP
        self.migration_log: List[Tuple[str, str, str]] = []
        self.counter_before = Counter()
        self.counter_after = Counter()

    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        records = []
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)

        consolidated = []
        for record in records:
            original_label = record.get("label", "unknown")
            text = record.get("text", "")[:50]

            self.counter_before[original_label] += 1

            if original_label in self.label_map:
                new_label = self.label_map[original_label]
                if new_label != original_label:
                    self.migration_log.append((original_label, new_label, text))
            else:
                new_label = original_label
                print(f"Warning: Unknown label '{original_label}' for text: {text[:30]}...")

            self.counter_after[new_label] += 1

            record["original_label"] = original_label
            record["label"] = new_label
            consolidated.append(record)

        with open(output_path, "w") as f:
            for record in consolidated:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "input_file": input_path.name,
            "output_file": output_path.name,
            "total_records": len(records),
        }

    def generate_report(self, output_dir: Path) -> str:
        lines = [
            "# Trigger Label Consolidation Report",
            "",
            "## Mapping",
            "",
            "| Old | New |",
            "|-----|-----|",
            "| bad_news | statement |",
            "| good_news | statement |",
            "| yn_question | question |",
            "| info_question | question |",
            "| invitation | commitment |",
            "",
            "## Distribution",
            "",
            "| Label | Before | After |",
            "|-------|--------|-------|",
        ]

        all_labels = set(self.counter_before.keys()) | set(self.counter_after.keys())
        for label in sorted(all_labels):
            before = self.counter_before[label]
            after = self.counter_after[label]
            lines.append(f"| {label} | {before} | {after} |")

        lines.extend(
            [
                "",
                f"Total migrations: {len(self.migration_log)}",
                "",
                "## Files",
                "- Originals preserved",
                "- New files with _consolidated suffix",
            ]
        )

        content = "\n".join(lines)
        report_path = output_dir / "trigger_label_consolidation_report.md"
        with open(report_path, "w") as f:
            f.write(content)
        return content


def main():
    parser = argparse.ArgumentParser(description="Consolidate trigger labels")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument(
        "--training-file", type=str, default="trigger_training_full_corrected.jsonl"
    )
    parser.add_argument(
        "--candidates-file", type=str, default="trigger_candidates_labeled_corrected.jsonl"
    )
    parser.add_argument("--no-candidates", action="store_true")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    consolidator = LabelConsolidator()

    print("Trigger Label Consolidation")
    print("=" * 50)

    training_input = input_dir / args.training_file
    training_output = input_dir / args.training_file.replace("_corrected", "_consolidated")

    if training_input.exists():
        print(f"\nProcessing training file:")
        print(f"  Input:  {training_input}")
        print(f"  Output: {training_output}")
        stats = consolidator.process_file(training_input, training_output)
        print(f"  Records: {stats['total_records']}")
    else:
        print(f"Warning: Training file not found: {training_input}")

    if not args.no_candidates:
        candidates_input = input_dir / args.candidates_file
        candidates_output = input_dir / args.candidates_file.replace("_corrected", "_consolidated")

        if candidates_input.exists():
            print(f"\nProcessing candidates file:")
            print(f"  Input:  {candidates_input}")
            print(f"  Output: {candidates_output}")
            stats = consolidator.process_file(candidates_input, candidates_output)
            print(f"  Records: {stats['total_records']}")
        else:
            print(f"Warning: Candidates file not found: {candidates_input}")

    print(f"\nGenerating report...")
    report = consolidator.generate_report(input_dir)
    report_path = input_dir / "trigger_label_consolidation_report.md"
    print(f"Report saved: {report_path}")

    print("\nConsolidation complete!")
    print(f"Original files kept in place.")
    print(f"New consolidated files created.")


if __name__ == "__main__":
    main()
