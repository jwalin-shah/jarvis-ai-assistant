#!/bin/bash
# Overnight fine-tuning comparison: 350M vs 700M vs 1.2B
# Identical hyperparameters, same data, different base models
#
# Usage: ./scripts/training/run_overnight_comparison.sh

set -e

echo "=========================================="
echo "OVERNIGHT FINE-TUNING COMPARISON"
echo "=========================================="
echo ""
echo "Models: 350M, 700M, 1.2B (all 4-bit)"
echo "LoRA rank: 8, LR: 1e-4"
echo "Data: variable context, per-contact cap 500"
echo ""

# Step 1: Extract data
echo "📦 Step 1: Extracting training data..."
if [ ! -f "data/personal/raw_style_variable/train.jsonl" ]; then
    uv run python scripts/training/extract_finetuning_data.py \
        --min-date 2023-01-01 \
        --max-per-contact 500 \
        --output-dir data/personal/raw_style_variable
else
    echo "   Data already exists, skipping extraction"
fi

echo ""
echo "📊 Data ready:"
wc -l data/personal/raw_style_variable/*.jsonl
echo ""

# Step 2: Train 350M
echo "🚀 Step 2: Training 350M model..."
uv run mlx_lm.lora --config ft_configs/personal_350m_lora.yaml
echo "✅ 350M complete"
echo ""

# Step 3: Train 700M
echo "🚀 Step 3: Training 700M model..."
uv run mlx_lm.lora --config ft_configs/personal_700m_lora.yaml
echo "✅ 700M complete"
echo ""

# Step 4: Train 1.2B
echo "🚀 Step 4: Training 1.2B model..."
uv run mlx_lm.lora --config ft_configs/personal_1.2b_lora.yaml
echo "✅ 1.2B complete"
echo ""

echo "=========================================="
echo "ALL TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Adapters saved to:"
echo "  - adapters/personal/350m-lora"
echo "  - adapters/personal/700m-lora"
echo "  - adapters/personal/1.2b-lora"
echo ""
echo "Next step: judge eval against test set"
echo "Baseline to beat: 3.87/10, 20% pass"
