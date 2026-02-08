#!/usr/bin/env python3
"""Show category distribution at each stage: full dataset → subset → oversampled."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load full dataset
data_dir = Path("data/dailydialog_native")
train_data = np.load(data_dir / "train.npz", allow_pickle=True)
metadata = json.loads((data_dir / "metadata.json").read_text())

X_full, y_full = train_data["X"], train_data["y"]

print("=" * 70)
print("FULL TRAINING DATASET")
print("=" * 70)
print(f"Total samples: {len(y_full):,}\n")

labels = sorted(set(y_full))
full_counts = {}
for label in labels:
    count = (y_full == label).sum()
    pct = 100 * count / len(y_full)
    full_counts[label] = count
    print(f"{label:15s}: {count:7,} ({pct:5.1f}%)")

# 10k stratified subset
print("\n" + "=" * 70)
print("10K STRATIFIED SUBSET (before oversampling)")
print("=" * 70)
print(f"Total samples: 10,000\n")

X_subset, _, y_subset, _ = train_test_split(
    X_full, y_full, train_size=10000, stratify=y_full, random_state=42
)

subset_counts = {}
for label in labels:
    count = (y_subset == label).sum()
    pct = 100 * count / len(y_subset)
    subset_counts[label] = count
    print(f"{label:15s}: {count:7,} ({pct:5.1f}%)")

# After oversampling (commissive=2.4, directive=3.0)
print("\n" + "=" * 70)
print("AFTER OVERSAMPLING (commissive=2.4x, directive=3.0x)")
print("=" * 70)

oversample_map = {"commissive": 2.4, "directive": 3.0}

X_list = [X_subset]
y_list = [y_subset]

for label, factor in oversample_map.items():
    mask = y_subset == label
    label_X = X_subset[mask]
    label_y = y_subset[mask]

    n_original = len(label_y)
    n_to_add = int(n_original * (factor - 1.0))

    if n_to_add > 0:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(label_y), size=n_to_add, replace=True)
        X_list.append(label_X[indices])
        y_list.append(label_y[indices])

y_resampled = np.concatenate(y_list)
print(f"Total samples: {len(y_resampled):,}\n")

resampled_counts = {}
for label in labels:
    count = (y_resampled == label).sum()
    pct = 100 * count / len(y_resampled)
    resampled_counts[label] = count
    print(f"{label:15s}: {count:7,} ({pct:5.1f}%)")

# Summary comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"{'Category':<15s} {'Full Dataset':<15s} {'10K Subset':<15s} {'After Oversample':<20s}")
print("-" * 70)

for label in labels:
    full_pct = 100 * full_counts[label] / len(y_full)
    subset_pct = 100 * subset_counts[label] / len(y_subset)
    resampled_pct = 100 * resampled_counts[label] / len(y_resampled)

    print(f"{label:<15s} "
          f"{full_counts[label]:>6,} ({full_pct:5.1f}%)  "
          f"{subset_counts[label]:>5,} ({subset_pct:5.1f}%)  "
          f"{resampled_counts[label]:>5,} ({resampled_pct:5.1f}%)")

print(f"\n{'TOTAL':<15s} {len(y_full):>6,}          {len(y_subset):>5,}          {len(y_resampled):>5,}")

# Recommendations
print("\n" + "=" * 70)
print("NOTES")
print("=" * 70)
print("• Stratified subset preserves original distribution (good!)")
print("• Oversampling duplicates minority classes to balance training")
print("• directive boosted from 15.4% → 31.7% to fix low F1 (0.51 → better)")
print("• Full dataset has more samples - might need different ratios there")
