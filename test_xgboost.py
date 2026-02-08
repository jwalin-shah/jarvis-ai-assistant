#!/usr/bin/env python3
"""Quick XGBoost test with label encoding fix."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70, flush=True)
print("🧪 XGBoost Test with Label Encoding", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# Load data
DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"
train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
X_train, y_train = train_data["X"], train_data["y"]

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

print(f"✓ Train: {len(X_train):,} samples", flush=True)
print(f"✓ Features: {X_train.shape[1]}", flush=True)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
print(f"✓ Classes: {list(label_encoder.classes_)} → {list(range(len(label_encoder.classes_)))}", flush=True)
print(flush=True)

# Setup preprocessor
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

# Create XGBoost pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )),
])

# Run 3-fold CV
print("Running 3-fold cross-validation...", flush=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

start = time.time()
scores = cross_val_score(
    model, X_train, y_train_encoded,
    cv=cv,
    scoring="f1_macro",
    n_jobs=1,
    verbose=1,
)
elapsed = time.time() - start

print(flush=True)
print("=" * 70, flush=True)
print("📊 XGBoost Results", flush=True)
print("=" * 70, flush=True)
print(f"F1 Score: {scores.mean():.4f} ± {scores.std():.4f}", flush=True)
print(f"Fold scores: {[f'{s:.4f}' for s in scores]}", flush=True)
print(f"Time: {elapsed:.1f}s", flush=True)
print("=" * 70, flush=True)
