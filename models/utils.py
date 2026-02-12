"""Shared utilities for MLX model loading."""

from __future__ import annotations

from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"


def find_model_snapshot(model_dir: Path) -> Path:
    """Find the snapshot directory in a HuggingFace cache model directory.

    Args:
        model_dir: Path to the model directory (e.g., ~/.cache/huggingface/hub/models--foo--bar)

    Returns:
        Path to the first snapshot directory.

    Raises:
        FileNotFoundError: If no snapshots exist.
    """
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots in {model_dir}")
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"Empty snapshots dir: {snapshots_dir}")
    return snapshots[0]


def hf_model_dir(repo_id: str) -> Path:
    """Get the HuggingFace cache directory for a model repo.

    Args:
        repo_id: HuggingFace repo ID (e.g., "BAAI/bge-small-en-v1.5")

    Returns:
        Path to the model cache directory.
    """
    return HF_CACHE / f"models--{repo_id.replace('/', '--')}"


def map_hf_bert_key(hf_name: str) -> str | None:
    """Map a HuggingFace BERT weight key to our MLX model key.

    Args:
        hf_name: Original HuggingFace weight key (without "bert." prefix).

    Returns:
        Mapped key name, or None if the key should be skipped (e.g., position_ids).
    """
    if "position_ids" in hf_name:
        return None

    name = hf_name

    # Encoder layers
    if "encoder.layer." in name:
        name = name.replace("encoder.layer.", "encoder.layers.")
        name = name.replace(".attention.self.query", ".attention.query")
        name = name.replace(".attention.self.key", ".attention.key")
        name = name.replace(".attention.self.value", ".attention.value")
        name = name.replace(".attention.output.dense", ".attention_output_dense")
        name = name.replace(".attention.output.LayerNorm", ".attention_output_LayerNorm")
        name = name.replace(".intermediate.dense", ".intermediate_dense")
        name = name.replace(".output.dense", ".output_dense")
        name = name.replace(".output.LayerNorm", ".output_LayerNorm")

    # Pooler
    name = name.replace("pooler.dense", "pooler_dense")

    return name
