#!/usr/bin/env python3
"""Quick test: Linear SVM with same oversampling as RBF."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load data
data_dir = Path("data/dailydialog_native")
train_data = np.load(data_dir / "train.npz", allow_pickle=True)
metadata = json.loads((data_dir / "metadata.json").read_text())

X_full, y_full = train_data["X"], train_data["y"]
embedding_dims = metadata["embedding_dims"]
hand_crafted_dims = metadata["hand_crafted_dims"]

# Create 10k subset
from sklearn.model_selection import train_test_split
X_subset, _, y_subset, _ = train_test_split(
    X_full, y_full, train_size=10000, stratify=y_full, random_state=42
)

print("Original distribution:")
for label in sorted(set(y_subset)):
    count = (y_subset == label).sum()
    pct = 100 * count / len(y_subset)
    print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")

# Apply same oversampling: commissive=2.4x, directive=1.6x
oversample_map = {"commissive": 2.4, "directive": 1.6}

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
        print(f"Oversampled {label}: {n_original} → {n_original + n_to_add} ({factor}x)")

X_resampled = np.vstack(X_list)
y_resampled = np.concatenate(y_list)

# Shuffle
rng = np.random.RandomState(42)
indices = rng.permutation(len(y_resampled))
X_resampled = X_resampled[indices]
y_resampled = y_resampled[indices]

print("\nAfter oversampling:")
for label in sorted(set(y_resampled)):
    count = (y_resampled == label).sum()
    pct = 100 * count / len(y_resampled)
    print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")

# Build pipeline
embedding_cols = list(range(embedding_dims))
hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

preprocessor = ColumnTransformer(
    transformers=[
        ("embeddings", "passthrough", embedding_cols),
        ("hand_crafted", StandardScaler(), hc_cols),
    ],
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", LinearSVC(C=40, class_weight="balanced", max_iter=5000, random_state=42)),
])

# 5-fold CV
print("\nTesting Linear SVM (C=40) with oversampling...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipeline, X_resampled, y_resampled,
    cv=cv, scoring="f1_macro", n_jobs=3
)

print(f"\n{'=' * 50}")
print(f"Linear SVM + Oversample")
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"{'=' * 50}")

print("\nComparison:")
print(f"  Linear, no oversample:  0.6099")
print(f"  RBF, no oversample:     0.6295 (+3.2%)")
print(f"  RBF + oversample:       0.7150 (+17.2%)")
print(f"  Linear + oversample:    {scores.mean():.4f} (+{100*(scores.mean()-0.6099)/0.6099:.1f}%)")
