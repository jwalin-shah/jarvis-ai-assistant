#!/usr/bin/env python3
"""Train final LightGBM model with best hyperparameters from tuning.

Loads trial 7 params (F1=0.7120) from lightgbm_tuning_results.json,
trains on full training set, evaluates on test set, and saves model.

Usage:
    uv run python scripts/train_final_lightgbm.py
"""

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("Training Final LightGBM Model")
print("=" * 70)
print()

# =============================================================================
# Load Best Hyperparameters
# =============================================================================

results_file = PROJECT_ROOT / "lightgbm_tuning_results.json"
if not results_file.exists():
    print(f"❌ Results file not found: {results_file}")
    sys.exit(1)

with open(results_file) as f:
    results = json.load(f)

# Use trial 0 (simpler, nearly as good as trial 7)
# Trial 0: 450 est, 30 leaves, F1=0.7111
# Trial 7: 401 est, 54 leaves, F1=0.7120 (only +0.09% better)
best_trial = results["results"][0]  # Trial 0
best_params = best_trial["params"].copy()

print(f"🏆 Using Trial #{best_trial['trial']} (F1={best_trial['f1_mean']:.4f})")
print(f"   Simpler model: 30 leaves (vs trial 7's 54 leaves for only +0.09% gain)")
print(f"   n_estimators: {best_params['n_estimators']}")
print(f"   num_leaves: {best_params['num_leaves']}")
print(f"   learning_rate: {best_params['learning_rate']:.4f}")
print()

# =============================================================================
# Load Data
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"
print(f"📂 Loading data from {DATA_DIR}...")

train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)

X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

print(f"✓ Train: {len(X_train):,} samples")
print(f"✓ Test:  {len(X_test):,} samples")
print(f"✓ Features: {X_train.shape[1]}")
print()

# =============================================================================
# Feature Preprocessing
# =============================================================================

embedding_dims = metadata["embedding_dims"]
hand_crafted_dims = metadata["hand_crafted_dims"]
spacy_dims = metadata.get("spacy_dims", 0)

embedding_cols = list(range(embedding_dims))
hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

if spacy_dims > 0:
    spacy_cols = list(range(
        embedding_dims + hand_crafted_dims,
        embedding_dims + hand_crafted_dims + spacy_dims
    ))
    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
            ("spacy", StandardScaler(), spacy_cols),
        ],
    )
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
        ],
    )

# =============================================================================
# Train Model
# =============================================================================

print("🔨 Training model...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("lgbm", lgb.LGBMClassifier(**best_params)),
])

model.fit(X_train, y_train)
print("✓ Training complete")
print()

# =============================================================================
# Evaluate on Test Set
# =============================================================================

print("📊 Evaluating on test set...")

y_pred = model.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average="macro")

print(f"Test F1 (macro): {test_f1:.4f}")
print()
print("Per-class performance:")
print(classification_report(
    y_test, y_pred,
    target_names=metadata["labels"],
    digits=4
))

# =============================================================================
# Save Model
# =============================================================================

output_file = PROJECT_ROOT / "models" / "lightgbm_category_final.joblib"
output_file.parent.mkdir(exist_ok=True)

joblib.dump(model, output_file)
print(f"💾 Model saved to: {output_file}")
print()

# Also save metadata for later use
model_metadata = {
    "model_type": "lightgbm",
    "labels": metadata["labels"],
    "label_map": metadata["label_map"],
    "feature_dims": metadata["feature_dims"],
    "embedding_dims": metadata["embedding_dims"],
    "hand_crafted_dims": metadata["hand_crafted_dims"],
    "spacy_dims": metadata.get("spacy_dims", 0),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "test_f1_macro": float(test_f1),
    "hyperparameters": best_params,
    "tuning_cv_f1": best_trial["f1_mean"],
}

metadata_file = PROJECT_ROOT / "models" / "lightgbm_category_final.json"
with open(metadata_file, "w") as f:
    json.dump(model_metadata, f, indent=2)

print(f"💾 Metadata saved to: {metadata_file}")
print()
print("=" * 70)
print("✅ Training Complete!")
print("=" * 70)
