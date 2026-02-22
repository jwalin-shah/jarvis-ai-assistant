#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import TextReplyPipeline


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end text reply pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="data/processed/conversation_pairs_labeled.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = TextReplyPipeline(cfg)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(
            f"Input not found: {in_path}. Run scripts/03_classify_history.py to create labeled pairs."
        )
    rows = load_jsonl(in_path)
    rows = rows[-args.limit :]

    selected_scores = []
    length_diffs = []

    print(f"[06] Evaluating {len(rows)} held-out examples")
    for i, row in enumerate(rows, start=1):
        result = pipeline.run(
            incoming_message=row["their_message"],
            recent_messages=row.get("context", []),
            contact_name="Contact",
            relationship="friend",
        )
        selected_scores.append(result.selected_score)
        length_diffs.append(abs(len(result.final_reply.split()) - len(row.get("my_reply", "").split())))

        print(f"\n--- Example {i} ---")
        print(f"Incoming: {row['their_message']}")
        print(f"Actual:   {row.get('my_reply', '')}")
        print(f"Selected: {result.final_reply}")
        print(f"Category: {result.category} (conf={result.category_confidence:.2f})")
        for c, s in zip(result.candidates, result.scores):
            print(f"  [{s:.3f}] <strategy>{c.strategy}</strategy> {c.reply}")

    avg_score = statistics.mean(selected_scores) if selected_scores else 0.0
    avg_len_diff = statistics.mean(length_diffs) if length_diffs else 0.0

    print("\n[06] Aggregate metrics")
    print(f"  - Avg selected RM score: {avg_score:.3f}")
    print(f"  - Avg length difference vs real replies: {avg_len_diff:.3f} words")
    print("  - Category distribution accuracy: requires human-labeled categories (not auto-computed)")


if __name__ == "__main__":
    main()
