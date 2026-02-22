#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.generator import ReplyGenerator


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rejected alternatives for RM training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="data/processed/conversation_pairs_labeled.jsonl")
    parser.add_argument("--output", default="data/training/preference_pairs.jsonl")
    parser.add_argument("--rejected-per-chosen", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen = ReplyGenerator(cfg)
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}. Run scripts/03_classify_history.py first.")
    rows = load_jsonl(in_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            category = row.get("category", "casual")
            rejected_replies = gen.generate_generic_replies(
                incoming_message=row["their_message"],
                recent_messages=row.get("context", []),
                n_samples=max(4, args.rejected_per_chosen * 2),
            )
            picked = 0
            for reply in rejected_replies:
                if reply.strip() == row["my_reply"].strip():
                    continue
                rec = {
                    "contact": row.get("contact", "unknown"),
                    "context": row.get("context", []),
                    "their_message": row["their_message"],
                    "chosen": row["my_reply"],
                    "rejected": reply,
                    "category": category,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                picked += 1
                total += 1
                if picked >= args.rejected_per_chosen:
                    break

    print(f"[04] Generated {total} preference pairs -> {out_path}")


if __name__ == "__main__":
    main()
