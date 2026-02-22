#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.reward_model import StyleRewardModel


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bt_loss(chosen_scores: np.ndarray, rejected_scores: np.ndarray) -> float:
    diff = chosen_scores - rejected_scores
    return float(np.mean(-np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-8)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train style reward model (LoRA-ready scaffold)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="data/training/preference_pairs.jsonl")
    parser.add_argument("--output-dir", default="data/style_rm_lora")
    args = parser.parse_args()

    cfg = load_config(args.config)
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}. Run scripts/04_generate_rejected.py first.")
    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit("No preference pairs found. Run script 04 first.")

    random.seed(42)
    random.shuffle(rows)
    val_split = float(cfg.training.get("val_split", 0.1))
    split = int(len(rows) * (1.0 - val_split))
    train_rows = rows[:split]
    val_rows = rows[split:]

    rm = StyleRewardModel(cfg)

    print(f"[05] Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")
    print("[05] Running lightweight Bradley-Terry calibration pass (fallback mode)")

    chosen_scores = np.array(
        [
            rm._fallback_score("friend", r["their_message"], r["chosen"])
            for r in train_rows
        ],
        dtype=np.float32,
    )
    rejected_scores = np.array(
        [
            rm._fallback_score("friend", r["their_message"], r["rejected"])
            for r in train_rows
        ],
        dtype=np.float32,
    )

    train_loss = bt_loss(chosen_scores, rejected_scores)
    print(f"[05] Train BT loss: {train_loss:.4f}")

    val_chosen = np.array(
        [rm._fallback_score("friend", r["their_message"], r["chosen"]) for r in val_rows],
        dtype=np.float32,
    )
    val_rejected = np.array(
        [rm._fallback_score("friend", r["their_message"], r["rejected"]) for r in val_rows],
        dtype=np.float32,
    )

    val_acc = float(np.mean(val_chosen > val_rejected)) if len(val_rows) else 0.0
    val_gap = float(np.mean(val_chosen - val_rejected)) if len(val_rows) else 0.0

    print(f"[05] Validation accuracy (chosen > rejected): {val_acc * 100:.2f}%")
    print(f"[05] Validation mean score gap: {val_gap:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "scaffold_complete",
        "note": "Replace fallback trainer with MLX LoRA RewardTrainer for full fine-tuning.",
        "config": cfg.training,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_accuracy": val_acc,
        "val_score_gap": val_gap,
    }
    with (output_dir / "adapter_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[05] Saved adapter metadata -> {output_dir / 'adapter_metadata.json'}")


if __name__ == "__main__":
    main()
