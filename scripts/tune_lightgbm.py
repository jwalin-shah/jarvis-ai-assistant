#!/usr/bin/env python3
"""LightGBM Hyperparameter Tuning with Optuna (2-hour experiment).

Runs 50 trials with 5-fold CV to find optimal LightGBM hyperparameters.
No sampling - uses class_weight='balanced' only.

Usage:
    uv run python scripts/tune_lightgbm.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging with immediate flush for progress monitoring
file_handler = logging.FileHandler("lightgbm_tuning.log", mode='w')  # Overwrite previous run
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)

# Ensure immediate flush (critical for tail -f monitoring)
for handler in logger.handlers:
    handler.stream.reconfigure(line_buffering=True) if hasattr(handler.stream, 'reconfigure') else None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger.info("=" * 70)
logger.info("🔬 LightGBM Hyperparameter Tuning (50 trials × 5-fold CV)")
logger.info("=" * 70)
logger.info("")

# =============================================================================
# Load Data
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"
logger.info(f"📂 Loading data from {DATA_DIR}...")

try:
    train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    X_train, y_train = train_data["X"], train_data["y"]

    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(f"✓ Train: {len(X_train):,} samples")
    logger.info(f"✓ Features: {X_train.shape[1]} ({metadata['embedding_dims']} emb + "
                f"{metadata['hand_crafted_dims']} hc + {metadata.get('spacy_dims', 0)} spacy)")
    logger.info("")

except Exception as e:
    logger.error(f"❌ Failed to load data: {e}")
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
# Optuna Objective Function
# =============================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trial_results = []

def objective(trial):
    """Optuna objective function for LightGBM hyperparameter tuning."""

    # Suggest hyperparameters (reduced ranges for faster trials)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),  # Was 500
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),  # Was 100
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "n_jobs": 1,
        "random_state": 42,
        "verbose": -1,
    }

    # Create pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("lgbm", lgb.LGBMClassifier(**params)),
    ])

    # Run cross-validation
    start = time.time()
    logger.info(f"  → Trial {trial.number + 1}: n_est={params['n_estimators']}, "
                f"leaves={params['num_leaves']}, lr={params['learning_rate']:.4f}")
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )
    elapsed = time.time() - start

    mean_f1 = scores.mean()
    std_f1 = scores.std()

    # Store result
    result = {
        "trial": trial.number,
        "params": params.copy(),
        "f1_mean": float(mean_f1),
        "f1_std": float(std_f1),
        "fold_scores": [float(s) for s in scores],
        "time_seconds": elapsed,
    }
    trial_results.append(result)

    # Auto-save every 5 trials
    if (trial.number + 1) % 5 == 0:
        output_file = PROJECT_ROOT / "lightgbm_tuning_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_samples": len(X_train),
                "n_features": X_train.shape[1],
                "cv_folds": 5,
                "n_trials": len(trial_results),
                "results": trial_results,
            }, f, indent=2)
        logger.info(f"💾 Progress saved ({len(trial_results)} trials)")

    logger.info(f"Trial {trial.number + 1}/50: F1={mean_f1:.4f}±{std_f1:.4f} ({elapsed:.1f}s)")

    return mean_f1

# =============================================================================
# Run Optuna Optimization
# =============================================================================

logger.info("🚀 Starting Optuna optimization...")
logger.info("Expected duration: ~2 hours")
logger.info("")

# Create study
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    study_name="lightgbm_tuning",
)

# Run optimization
start_time = time.time()
study.optimize(objective, n_trials=50, show_progress_bar=True)
total_time = time.time() - start_time

logger.info("")
logger.info("=" * 70)
logger.info("✅ Optimization Complete!")
logger.info("=" * 70)
logger.info("")

# =============================================================================
# Results Summary
# =============================================================================

best_trial = study.best_trial
logger.info(f"🏆 Best Trial: #{best_trial.number}")
logger.info(f"📊 Best F1 Score: {best_trial.value:.4f}")
logger.info("")
logger.info("🎯 Best Hyperparameters:")
for key, value in best_trial.params.items():
    if isinstance(value, float):
        logger.info(f"  {key}: {value:.6f}")
    else:
        logger.info(f"  {key}: {value}")

logger.info("")
logger.info(f"⏱️  Total time: {total_time / 60:.1f} minutes")
logger.info(f"⚡ Avg time per trial: {total_time / 50:.1f}s")
logger.info("")

# =============================================================================
# Save Final Results
# =============================================================================

output_file = PROJECT_ROOT / "lightgbm_tuning_results.json"
with open(output_file, "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(X_train),
        "n_features": X_train.shape[1],
        "cv_folds": 5,
        "n_trials": len(trial_results),
        "best_trial": best_trial.number,
        "best_f1": float(best_trial.value),
        "best_params": best_trial.params,
        "total_time_seconds": total_time,
        "results": trial_results,
    }, f, indent=2)

logger.info(f"💾 Results saved to: {output_file}")
logger.info("")

# Save best model for future use
logger.info("🔨 Training final model with best parameters...")
best_params = best_trial.params.copy()
best_params.update({"class_weight": "balanced", "n_jobs": 1, "random_state": 42, "verbose": -1})

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("lgbm", lgb.LGBMClassifier(**best_params)),
])

best_model.fit(X_train, y_train)

model_file = PROJECT_ROOT / "models" / "lightgbm_tuned.joblib"
model_file.parent.mkdir(exist_ok=True)
joblib.dump(best_model, model_file)

logger.info(f"💾 Best model saved to: {model_file}")
logger.info("")

# =============================================================================
# Performance Analysis
# =============================================================================

# Sort all trials by F1 score
sorted_results = sorted(trial_results, key=lambda x: x["f1_mean"], reverse=True)

logger.info("=" * 70)
logger.info("📈 Top 10 Trials")
logger.info("=" * 70)
logger.info("")
logger.info(f"{'Trial':<8} {'F1 Score':<20} {'n_est':<8} {'leaves':<8} {'lr':<10}")
logger.info("-" * 70)

for r in sorted_results[:10]:
    trial_num = r["trial"]
    f1_str = f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
    n_est = r["params"]["n_estimators"]
    leaves = r["params"]["num_leaves"]
    lr = r["params"]["learning_rate"]
    logger.info(f"{trial_num:<8} {f1_str:<20} {n_est:<8} {leaves:<8} {lr:<10.4f}")

logger.info("")
logger.info("=" * 70)
logger.info("🎉 Tuning Complete!")
logger.info("=" * 70)
