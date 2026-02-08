#!/usr/bin/env python3
"""Fast model comparison: Linear SVC, Nystroem, LightGBM, XGBoost vs RBF SVM.

Runs lightweight 3-fold CV on each model type to quickly identify the best approach.
Safe to run in parallel with Optuna - uses separate memory space.

Usage:
    uv run python scripts/model_comparison.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70, flush=True)
print("🏁 Fast Model Comparison (3-fold CV)", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# =============================================================================
# Load Data
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"
print(f"📂 Loading data from {DATA_DIR}...", flush=True)

try:
    # Load from .npz format (same as Optuna script)
    train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    # Encode string labels to integers for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"✓ Train: {len(X_train):,} samples", flush=True)
    print(f"✓ Test:  {len(X_test):,} samples", flush=True)
    print(f"✓ Features: {X_train.shape[1]} ({metadata['embedding_dims']} emb + "
          f"{metadata['hand_crafted_dims']} hc + {metadata.get('spacy_dims', 0)} spacy)", flush=True)
    print(f"✓ Classes: {list(label_encoder.classes_)} → {list(range(len(label_encoder.classes_)))}", flush=True)
    print(flush=True)

except Exception as e:
    print(f"❌ Failed to load data: {e}", flush=True)
    sys.exit(1)

# =============================================================================
# Feature Preprocessing Setup
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
# Cross-Validation Setup
# =============================================================================

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# =============================================================================
# Model Definitions
# =============================================================================

models = {
    "LinearSVC": Pipeline([
        ("preprocessor", preprocessor),
        ("svm", LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            dual="auto",
            random_state=42,
        )),
    ]),

    "Nystroem(300)+LinearSVC": Pipeline([
        ("preprocessor", preprocessor),
        ("nystroem", Nystroem(kernel="rbf", gamma=0.1, n_components=300, random_state=42)),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)),
    ]),

    "Nystroem(500)+LinearSVC": Pipeline([
        ("preprocessor", preprocessor),
        ("nystroem", Nystroem(kernel="rbf", gamma=0.1, n_components=500, random_state=42)),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)),
    ]),
}

# Add LightGBM if available
try:
    import lightgbm as lgb

    models["LightGBM"] = Pipeline([
        ("preprocessor", preprocessor),
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.1,
            class_weight="balanced",
            n_jobs=1,  # Keep light for parallel run
            random_state=42,
            verbose=-1,
        )),
    ])
    print("✓ LightGBM available", flush=True)
except ImportError:
    print("⚠️  LightGBM not available (pip install lightgbm)", flush=True)

# Add XGBoost if available
try:
    import xgboost as xgb

    models["XGBoost"] = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=1,  # Keep light for parallel run
            random_state=42,
        )),
    ])
    print("✓ XGBoost available", flush=True)
except ImportError:
    print("⚠️  XGBoost not available (pip install xgboost)", flush=True)

# Add baseline RBF SVM (small sample for comparison)
models["RBF-SVM(5k-sample)"] = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        max_iter=10000,
        cache_size=500,
        random_state=42,
    )),
])

print(flush=True)
print(f"🔬 Testing {len(models)} models with 3-fold CV", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# =============================================================================
# Run Comparison
# =============================================================================

results = []

for i, (name, model) in enumerate(models.items(), 1):
    print(f"[{i}/{len(models)}] {name}...", end=" ", flush=True)

    start = time.time()

    try:
        # For RBF baseline, subsample to 5k for speed
        if "RBF-SVM" in name:
            # Stratified subsample
            from sklearn.model_selection import train_test_split
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train_encoded,
                train_size=5000,
                stratify=y_train_encoded,
                random_state=42
            )
            scores = cross_val_score(
                model, X_sub, y_sub,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
            )
        else:
            scores = cross_val_score(
                model, X_train, y_train_encoded,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
            )

        elapsed = time.time() - start
        mean_f1 = scores.mean()
        std_f1 = scores.std()

        results.append({
            "model": name,
            "f1_mean": float(mean_f1),
            "f1_std": float(std_f1),
            "time_seconds": elapsed,
            "fold_scores": [float(s) for s in scores],
        })

        print(f"F1={mean_f1:.4f}±{std_f1:.4f} ({elapsed:.1f}s)", flush=True)

    except Exception as e:
        print(f"❌ FAILED: {e}", flush=True)
        results.append({
            "model": name,
            "error": str(e),
        })

print(flush=True)
print("=" * 70, flush=True)
print("📊 Results Summary", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# Sort by F1 score
valid_results = [r for r in results if "f1_mean" in r]
valid_results.sort(key=lambda x: x["f1_mean"], reverse=True)

print(f"{'Model':<30} {'F1 Score':<15} {'Time':<10} {'Speedup':<10}", flush=True)
print("-" * 70, flush=True)

# Find RBF baseline time for speedup calculation
rbf_result = next((r for r in results if "RBF-SVM" in r["model"]), None)
rbf_time = rbf_result["time_seconds"] if rbf_result else None

for r in valid_results:
    model_name = r["model"]
    f1_str = f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
    time_str = f"{r['time_seconds']:.1f}s"

    # Calculate speedup vs RBF (accounting for 5k subsample)
    if rbf_time and "RBF-SVM" not in model_name:
        # Estimate full RBF time: (60k/5k)^2 = 144x longer for full dataset
        estimated_full_rbf = rbf_time * 144
        speedup = estimated_full_rbf / r['time_seconds']
        speedup_str = f"{speedup:.1f}x"
    else:
        speedup_str = "-"

    print(f"{model_name:<30} {f1_str:<15} {time_str:<10} {speedup_str:<10}", flush=True)

print(flush=True)

# =============================================================================
# Save Results
# =============================================================================

output_file = PROJECT_ROOT / "model_comparison_results.json"
with open(output_file, "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(X_train),
        "n_features": X_train.shape[1],
        "cv_folds": 3,
        "results": results,
    }, f, indent=2)

print(f"💾 Results saved to: {output_file}", flush=True)
print(flush=True)

# =============================================================================
# Recommendations
# =============================================================================

if valid_results:
    best = valid_results[0]
    print("=" * 70, flush=True)
    print("🎯 Recommendation", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print(f"Best model: {best['model']}", flush=True)
    print(f"F1 Score:   {best['f1_mean']:.4f} ± {best['f1_std']:.4f}", flush=True)
    print(f"Time:       {best['time_seconds']:.1f}s for 3-fold CV", flush=True)
    print(flush=True)

    if "LinearSVC" in best['model']:
        print("💡 LinearSVC is winning! This suggests your features are already", flush=True)
        print("   well-separated and don't need kernel tricks. Huge speed gains!", flush=True)
    elif "Nystroem" in best['model']:
        print("💡 Nystroem approximation gives you RBF-like performance with", flush=True)
        print("   10-30x speedup. Great middle ground!", flush=True)
    elif "LightGBM" in best['model'] or "XGBoost" in best['model']:
        print("💡 Gradient boosting is outperforming SVM! Consider using this", flush=True)
        print("   for your final model - faster and often more accurate.", flush=True)

    print(flush=True)
    print("Next steps:", flush=True)
    print("1. Run hyperparameter optimization on the best model type", flush=True)
    print("2. Compare test set performance (not just CV)", flush=True)
    print("3. Consider using this in production instead of RBF SVM", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("✨ Comparison complete!", flush=True)
print("=" * 70, flush=True)
