#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import ResponseClassifier
from src.config import load_config


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify conversation history into response types")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="data/processed/conversation_pairs.jsonl")
    parser.add_argument("--output", default="data/processed/conversation_pairs_labeled.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classifier = ResponseClassifier(cfg)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}. Run scripts/02_parse_conversations.py first.")
    rows = load_jsonl(in_path)
    print(f"[03] Loaded {len(rows)} conversation pairs")

    for row in rows:
        result = classifier.classify(row["their_message"])
        row["category"] = result.category

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cat_counts = Counter(r["category"] for r in rows)
    print("\n[03] Category distribution")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat}: {count}")

    time_by_cat: dict[str, list[int]] = defaultdict(list)
    len_by_cat: dict[str, list[int]] = defaultdict(list)
    per_contact: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        c = r["category"]
        time_by_cat[c].append(int(r.get("response_time_seconds", 0)))
        len_by_cat[c].append(len(r.get("my_reply", "").split()))
        per_contact[r.get("contact", "unknown")][c] += 1

    print("\n[03] Avg response time (seconds) by category")
    for cat, values in sorted(time_by_cat.items()):
        print(f"  - {cat}: {statistics.mean(values):.1f}")

    print("\n[03] Avg reply length (words) by category")
    for cat, values in sorted(len_by_cat.items()):
        print(f"  - {cat}: {statistics.mean(values):.1f}")

    print("\n[03] Per-contact style patterns (top category)")
    for contact, ctr in per_contact.items():
        top_cat, top_n = ctr.most_common(1)[0]
        print(f"  - {contact}: {top_cat} ({top_n})")

    print(f"\n[03] Wrote labeled dataset -> {out}")


if __name__ == "__main__":
    main()
