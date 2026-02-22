#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline import TextReplyPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="On-device text reply CLI")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--contact", default="Friend")
    parser.add_argument("--relationship", default="friend")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    pipeline = TextReplyPipeline(cfg)

    print("Text Reply CLI (type 'quit' to exit)")
    while True:
        incoming = input("\nIncoming message: ").strip()
        if incoming.lower() in {"quit", "exit"}:
            break

        result = pipeline.run(
            incoming_message=incoming,
            recent_messages=[],
            contact_name=args.contact,
            relationship=args.relationship,
        )

        print(f"\nReply: {result.final_reply}")
        print(f"Category: {result.category} (conf={result.category_confidence:.2f})")
        if result.warning:
            print(f"Warning: {result.warning}")


if __name__ == "__main__":
    main()
