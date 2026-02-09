"""Generate all personal fine-tuning config YAML files.

Creates 16 configs: 4 models x 2 adapters (LoRA/DoRA) x 2 data variants.
Output: ft_configs/personal_*.yaml
"""

from __future__ import annotations

from pathlib import Path

import yaml

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
    {"id": "cataware", "path": "data/personal/category_aware"},
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


def generate_configs() -> None:
    out_dir = Path("ft_configs")
    out_dir.mkdir(exist_ok=True)

    count = 0
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
                with open(path, "w") as f:
                    # Add header comment
                    f.write(f"# Personal fine-tune: {model['id']} {adapter.upper()} {data['id']}\n")
                    f.write(f"# Run: uv run mlx_lm.lora --config {path}\n\n")
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                count += 1
                print(f"  Generated {path}", flush=True)

    print(f"\nGenerated {count} config files in {out_dir}/", flush=True)


if __name__ == "__main__":
    generate_configs()
