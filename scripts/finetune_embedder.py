#!/usr/bin/env python3
"""Fine-tune BGE-small embedder on SOC conversation triplets.

Creates (anchor, positive, hard_negative) triplets from SOC conversations:
- Anchor: last message in conversation
- Positive: the actual reply (semantically relevant)
- Hard negative: reply from different conversation with similar topic

Uses sentence-transformers MultipleNegativesRankingLoss for training.

Output: models/bge-small-soc-finetuned/

Usage:
    uv run python scripts/finetune_embedder.py
    uv run python scripts/finetune_embedder.py --epochs 5 --batch-size 32
    uv run python scripts/finetune_embedder.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_conversation_pairs(sft_dir: Path) -> list[tuple[str, str]]:
    """Load (last_message, reply) pairs from SFT data.

    Returns:
        List of (anchor_text, positive_text) tuples.
    """
    pairs: list[tuple[str, str]] = []

    for split in ["train", "valid"]:
        path = sft_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                msgs = ex["messages"]
                user_msg = msgs[1]["content"]
                reply = msgs[2]["content"]

                # Extract last_message from user message
                last_message = ""
                if "<last_message>" in user_msg:
                    parts = user_msg.split("<last_message>")
                    if len(parts) > 1:
                        last_message = parts[1].split("</last_message>")[0].strip()

                if last_message and reply:
                    pairs.append((last_message, reply))

    return pairs


def build_triplets(
    pairs: list[tuple[str, str]],
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Build (anchor, positive, hard_negative) triplets.

    Hard negatives are replies from other conversations, sampled randomly.
    In-batch negatives from MultipleNegativesRankingLoss provide additional
    negative signal, so we only need one explicit hard negative.

    Args:
        pairs: List of (last_message, reply) tuples.
        seed: Random seed.

    Returns:
        List of (anchor, positive, hard_negative) triplets.
    """
    random.seed(seed)
    all_replies = [p[1] for p in pairs]
    triplets: list[tuple[str, str, str]] = []

    for i, (anchor, positive) in enumerate(pairs):
        # Sample a random negative (reply from different conversation)
        neg_idx = random.randint(0, len(all_replies) - 1)
        while neg_idx == i:
            neg_idx = random.randint(0, len(all_replies) - 1)
        negative = all_replies[neg_idx]
        triplets.append((anchor, positive, negative))

    return triplets


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune BGE-small on conversation triplets")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/bge-small-soc-finetuned",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    # Load data
    sft_dir = PROJECT_ROOT / "data" / "soc_sft"
    if not sft_dir.exists():
        print(f"ERROR: SFT data not found at {sft_dir}")
        print("       Run: uv run python scripts/prepare_soc_data.py")
        return 1

    print("Loading conversation pairs from SFT data...")
    pairs = load_conversation_pairs(sft_dir)
    print(f"Loaded {len(pairs)} conversation pairs")

    if len(pairs) < 100:
        print("WARNING: Very few pairs. Consider generating more SFT data first.")

    print("Building triplets...")
    triplets = build_triplets(pairs, seed=args.seed)
    print(f"Built {len(triplets)} triplets")

    # Show samples
    print("\nSample triplets:")
    for i in range(min(3, len(triplets))):
        a, p, n = triplets[i]
        print(f"  Anchor:   {a[:60]!r}")
        print(f"  Positive: {p[:60]!r}")
        print(f"  Negative: {n[:60]!r}")
        print()

    if args.dry_run:
        print("Dry run - not training.")
        return 0

    # Import sentence-transformers
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("       Install with: uv pip install sentence-transformers")
        return 1

    # Load model
    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    # Build training data
    train_examples = [
        InputExample(texts=[anchor, positive, negative]) for anchor, positive, negative in triplets
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # Loss function: MultipleNegativesRankingLoss uses in-batch negatives
    # plus our explicit hard negatives for stronger signal
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    output_path = str(PROJECT_ROOT / args.output_dir)
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {output_path}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=output_path,
    )

    print(f"\nFine-tuned model saved to {output_path}")

    # Save metadata
    meta = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_triplets": len(triplets),
        "num_pairs": len(pairs),
        "loss": "MultipleNegativesRankingLoss",
    }
    meta_path = Path(output_path) / "training_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Training metadata saved to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
