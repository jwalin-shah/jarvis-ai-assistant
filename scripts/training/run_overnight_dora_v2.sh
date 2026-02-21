#!/bin/bash
# Overnight DoRA v2 fine-tuning comparison: 350M vs 700M vs 1.2B
# Corrected hyperparameters: DoRA, rank 16, LR 2e-5, GLU layers
#
# Usage: ./scripts/training/run_overnight_dora_v2.sh

set -e

echo "=========================================="
echo "OVERNIGHT DORA V2 FINE-TUNING"
echo "=========================================="
echo ""
echo "Models: 350M, 700M, 1.2B (all 4-bit)"
echo "DoRA rank: 16, alpha: 32, LR: 2e-5"
echo "Targets: q_proj, v_proj, w1, w2, w3 (GLU + attention)"
echo "Data: variable context, per-contact cap 500"
echo ""

# Data already exists from previous run
echo "📊 Data ready:"
wc -l data/personal/raw_style_variable/*.jsonl
echo ""

# Step 1: Train 350M (resume from step 500 checkpoint)
echo "🚀 Step 1: Training 350M model with DoRA v2 (resume from step 500)..."
if [ -f "adapters/personal/350m-dora-v2/adapters.safetensors" ]; then
    echo "  Found existing adapter, resuming training..."
fi
uv run mlx_lm.lora --config ft_configs/personal_350m_dora_v2.yaml || echo "  350M training interrupted or completed"
echo "✅ 350M done"
echo ""

# Step 2: Train 700M
echo "🚀 Step 2: Training 700M model with DoRA v2..."
uv run mlx_lm.lora --config ft_configs/personal_700m_dora_v2.yaml
echo "✅ 700M complete"
echo ""

# Step 3: Train 1.2B
echo "🚀 Step 3: Training 1.2B model with DoRA v2..."
uv run mlx_lm.lora --config ft_configs/personal_1.2b_dora_v2.yaml
echo "✅ 1.2B complete"
echo ""

echo "=========================================="
echo "ALL TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "DoRA adapters saved to:"
echo "  - adapters/personal/350m-dora-v2"
echo "  - adapters/personal/700m-dora-v2"
echo "  - adapters/personal/1.2b-dora-v2"
echo ""
echo "Next step: judge eval against test set"
echo "Baseline to beat: 3.87/10, 20% pass"
