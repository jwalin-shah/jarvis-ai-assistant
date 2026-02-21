"""Generate all personal fine-tuning config YAML files.

Creates 16 configs: 4 models x 2 adapters (LoRA/DoRA) x 2 data variants.
Output: ft_configs/personal_*.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from jarvis.utils.logging import setup_script_logging

# Model matrix
MODELS = [
    {
        "id": "0.3b",
        "path": "mlx-community/LFM2-350M-4bit",
        "batch_size": 4,
        "lora_rank": 4,
        "lora_layers": 8,
    },
    {
        "id": "0.7b",
        "path": "lmstudio-community/LFM2-700M-MLX-8bit",
        "batch_size": 4,
        "lora_rank": 8,
        "lora_layers": 8,
    },
    {
        "id": "1.2b",
        "path": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "batch_size": 2,
        "lora_rank": 8,
        "lora_layers": 8,
    },
    {
        "id": "2.6b",
        "path": "mlx-community/LFM2-2.6B-4bit",
        "batch_size": 1,
        "lora_rank": 8,
        "lora_layers": 8,
    },
]

ADAPTERS = ["lora", "dora"]
DATA_VARIANTS = [
    {"id": "variable", "path": "data/personal/raw_style_variable"},
    {"id": "rawstyle", "path": "data/personal/raw_style"},
]

# Common training params
COMMON = {
    "train": True,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "grad_checkpoint": True,
    "learning_rate": 2.0e-4,
    "iters": 1000,
    "steps_per_eval": 50,
    "save_every": 250,
    "max_seq_length": 512,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ft_configs"),
        help="Directory to write generated YAML configs (default: %(default)s).",
    )
    return parser.parse_args(argv)


def generate_configs(out_dir: Path = Path("ft_configs")) -> None:
    import yaml

    try:
        out_dir.mkdir(exist_ok=True)
    except OSError as exc:
        print(f"Error creating output directory '{out_dir}': {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    count = 0
    total = len(MODELS) * len(ADAPTERS) * len(DATA_VARIANTS)
    for model in MODELS:
        for adapter in ADAPTERS:
            for data in DATA_VARIANTS:
                name = f"personal_{model['id']}_{adapter}_{data['id']}"
                adapter_path = f"adapters/personal/{model['id']}-{adapter}-{data['id']}"

                config = {
                    "model": model["path"],
                    "data": data["path"],
                    "adapter_path": adapter_path,
                    "lora_layers": model["lora_layers"],
                    "lora_rank": model["lora_rank"],
                    "batch_size": model["batch_size"],
                    **COMMON,
                }

                if adapter == "dora":
                    config["fine_tune_type"] = "dora"

                path = out_dir / f"{name}.yaml"
                try:
                    with path.open("w") as f:
                        # Add header comment
                        f.write(
                            f"# Personal fine-tune: {model['id']} {adapter.upper()} {data['id']}\n"
                        )
                        f.write(f"# Run: uv run mlx_lm.lora --config {path}\n\n")
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                except OSError as exc:
                    print(f"Error writing config '{path}': {exc}", file=sys.stderr, flush=True)
                    raise SystemExit(1) from exc

                count += 1
                print(f"  [{count}/{total}] Generated {path}", flush=True)

    print(f"\nGenerated {count} config files in {out_dir}/", flush=True)


if __name__ == "__main__":
    setup_script_logging("generate_ft_configs")
    logging.info("Starting generate_ft_configs.py")
    args = parse_args()
    generate_configs(args.output_dir)
    logging.info("Finished generate_ft_configs.py")
