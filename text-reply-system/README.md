# On-Device Text Reply System (MLX)

Local CLI system for generating text replies in your style on Apple Silicon.

## Goals
- On-device only (no cloud APIs)
- Optimized for 8GB Apple Silicon machines
- End-to-end reply in under ~2 seconds for common categories
- Uses a small classifier/reward model + larger generator with Soft Best-of-N

## Project Layout

```text
text-reply-system/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── training/
├── scripts/
│   ├── 01_export_imessage.py
│   ├── 02_parse_conversations.py
│   ├── 03_classify_history.py
│   ├── 04_generate_rejected.py
│   ├── 05_train_style_rm.py
│   └── 06_evaluate.py
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── generator.py
│   ├── reward_model.py
│   ├── pipeline.py
│   ├── soft_bon.py
│   └── config.py
└── app.py
```

## Quick Start

```bash
cd text-reply-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Export iMessage data:
```bash
python scripts/01_export_imessage.py
```
2. Parse conversation pairs:
```bash
python scripts/02_parse_conversations.py
```
3. Classify and prepare training data:
```bash
python scripts/03_classify_history.py
python scripts/04_generate_rejected.py
```
4. Train style reward model adapter:
```bash
python scripts/05_train_style_rm.py
```
5. Evaluate pipeline:
```bash
python scripts/06_evaluate.py
```
6. Run interactive CLI:
```bash
python app.py
```

## Notes
- MLX model APIs evolve quickly. This code uses `mlx-lm` where available and has safe fallbacks.
- If LFM model IDs are unavailable in your environment, update `config.yaml` to Qwen fallback IDs.
- No iMessage sending integration is included (generation-only).
