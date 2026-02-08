#!/usr/bin/env python3
"""Show per-class F1 scores with directive=3.0x oversampling."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
X_subset, _, y_subset, _ = train_test_split(
    X_full, y_full, train_size=10000, stratify=y_full, random_state=42
)

# Apply NEW oversampling: commissive=2.4x, directive=3.0x
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

X_resampled = np.vstack(X_list)
y_resampled = np.concatenate(y_list)

# Shuffle
rng = np.random.RandomState(42)
indices = rng.permutation(len(y_resampled))
X_resampled = X_resampled[indices]
y_resampled = y_resampled[indices]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

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
    ("svm", SVC(
        C=7.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        max_iter=50000,
        cache_size=1000,
        random_state=42
    )),
])

print("Training RBF SVM with directive=3.0x oversampling...")
print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples\n")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Per-class scores
print("=" * 60)
print("PER-CLASS F1 SCORES (directive=3.0x)")
print("=" * 60)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
print(classification_report(y_test, y_pred, digits=4))

# Comparison table
print("=" * 60)
print("IMPROVEMENT vs directive=1.6x")
print("=" * 60)

old_scores = {
    "commissive": 0.7273,
    "directive": 0.5102,  # ← The bottleneck!
    "inform": 0.7686,
    "question": 0.8297,
}

print(f"{'Category':<15s} {'Old F1':<10s} {'New F1':<10s} {'Change':<10s}")
print("-" * 60)

for label in sorted(old_scores.keys()):
    old_f1 = old_scores[label]
    new_f1 = report[label]["f1-score"]
    change = new_f1 - old_f1
    change_pct = 100 * change / old_f1

    emoji = "🚀" if change > 0.05 else "✅" if change > 0 else "⚠️"
    print(f"{label:<15s} {old_f1:<10.4f} {new_f1:<10.4f} "
          f"{change:+.4f} ({change_pct:+5.1f}%) {emoji}")

# Confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
labels = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Print header
print(f"{'True \\ Pred':>15s}", end="")
for label in labels:
    print(f"{label[:10]:>12s}", end="")
print()
print("-" * 60)

# Print rows
for i, true_label in enumerate(labels):
    print(f"{true_label[:15]:>15s}", end="")
    for j, pred_label in enumerate(labels):
        print(f"{cm[i, j]:>12d}", end="")
    print()
