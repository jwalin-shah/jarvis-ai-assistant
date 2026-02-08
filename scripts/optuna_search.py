#!/usr/bin/env python3
"""Bayesian hyperparameter optimization with Optuna.

Uses Optuna's TPE (Tree-structured Parzen Estimator) to intelligently search:
- SVM hyperparameters: C, gamma
- Class resampling ratios: 4 values (commissive, directive, inform, question)

Much more efficient than grid search for high-dimensional spaces.

Usage:
    # Quick search (20 trials, ~5 hours)
    uv run python scripts/optuna_search.py --n-trials 20

    # Thorough search (50 trials, ~12 hours)
    uv run python scripts/optuna_search.py --n-trials 50

    # Resume a previous study
    uv run python scripts/optuna_search.py --study-name my_study --n-trials 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.utils.memory import get_memory_info, get_swap_info

# Setup logging
LOG_FILE = PROJECT_ROOT / "optuna_search.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def apply_resampling(X, y, resample_map, seed=42):
    """Resample classes: oversample (factor>1) or downsample (factor<1)."""
    if not resample_map:
        return X, y

    X_list = []
    y_list = []

    for label in sorted(set(y)):
        mask = y == label
        label_X = X[mask]
        label_y = y[mask]
        n_original = len(label_y)

        factor = resample_map.get(label, 1.0)

        if factor > 1.0:
            # Oversample
            n_to_add = int(n_original * (factor - 1.0))
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(label_y), size=n_to_add, replace=True)
            X_list.append(label_X)
            X_list.append(label_X[indices])
            y_list.append(label_y)
            y_list.append(label_y[indices])
        elif factor < 1.0:
            # Downsample
            n_keep = int(n_original * factor)
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(label_y), size=n_keep, replace=False)
            X_list.append(label_X[indices])
            y_list.append(label_y[indices])
        else:
            # Keep as-is
            X_list.append(label_X)
            y_list.append(label_y)

    X_resampled = np.vstack(X_list)
    y_resampled = np.concatenate(y_list)

    # Shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y_resampled))

    return X_resampled[indices], y_resampled[indices]


class Objective:
    """Optuna objective function for SVM hyperparameter optimization."""

    def __init__(self, X_train, y_train, metadata, seed=42, n_jobs=1):
        self.X_train = X_train
        self.y_train = y_train
        self.metadata = metadata
        self.seed = seed
        self.n_jobs = n_jobs

        # Feature columns
        embedding_dims = metadata["embedding_dims"]
        hand_crafted_dims = metadata["hand_crafted_dims"]
        spacy_dims = metadata.get("spacy_dims", 0)

        self.embedding_cols = list(range(embedding_dims))
        self.hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

        if spacy_dims > 0:
            self.spacy_cols = list(range(
                embedding_dims + hand_crafted_dims,
                embedding_dims + hand_crafted_dims + spacy_dims
            ))
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("embeddings", "passthrough", self.embedding_cols),
                    ("hand_crafted", StandardScaler(), self.hc_cols),
                    ("spacy", StandardScaler(), self.spacy_cols),
                ],
            )
        else:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("embeddings", "passthrough", self.embedding_cols),
                    ("hand_crafted", StandardScaler(), self.hc_cols),
                ],
            )

        # Get unique labels
        self.labels = sorted(set(y_train))

        # Cross-validation
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # Trial counter
        self.trial_count = 0

    def __call__(self, trial):
        """Optuna objective function."""
        self.trial_count += 1

        # Sample hyperparameters
        # SVM hyperparameters
        C = trial.suggest_float("C", 0.1, 100.0, log=True)

        # Gamma: either 'scale' or numeric value (constrained to reasonable range)
        gamma_type = trial.suggest_categorical("gamma_type", ["scale", "numeric"])
        if gamma_type == "scale":
            gamma = "scale"
        else:
            gamma = trial.suggest_float("gamma_value", 1e-4, 1.0, log=True)  # Was 1e-5 to 10.0

        # Class resampling ratios (per label)
        resample_map = {}
        for label in self.labels:
            if label in ["commissive", "directive"]:
                # Minority classes: allow oversampling (1.0 to 3.0, not 4.0)
                ratio = trial.suggest_float(f"resample_{label}", 1.0, 3.0)
            elif label in ["inform"]:
                # Majority class: allow downsampling (0.2 to 1.0)
                ratio = trial.suggest_float(f"resample_{label}", 0.2, 1.0)
            else:
                # Question: moderate adjustment (0.5 to 1.5)
                ratio = trial.suggest_float(f"resample_{label}", 0.5, 1.5)
            resample_map[label] = ratio

        # Log trial info
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {self.trial_count} (completed: {len(trial.study.trials)})")
        logger.info(f"  C: {C:.4f}")
        logger.info(f"  gamma: {gamma if gamma_type == 'scale' else f'{gamma:.6f}'}")
        logger.info(f"  Resampling:")
        for label, ratio in sorted(resample_map.items()):
            logger.info(f"    {label:12s}: {ratio:.2f}x")

        # Apply resampling to training data
        X_resampled, y_resampled = apply_resampling(
            self.X_train, self.y_train, resample_map, self.seed
        )
        logger.info(f"  Training size after resample: {len(X_resampled):,}")

        # Build pipeline
        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("svm", SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight="balanced",
                max_iter=50000,
                cache_size=1000,
                tol=1e-3,
                random_state=self.seed,
            )),
        ])

        # Cross-validation
        try:
            scores = cross_val_score(
                pipeline,
                X_resampled,
                y_resampled,
                cv=self.cv,
                scoring="f1_macro",
                n_jobs=self.n_jobs,
            )
            mean_f1 = scores.mean()
            std_f1 = scores.std()

            logger.info(f"  F1 (CV): {mean_f1:.4f} ± {std_f1:.4f}")
            logger.info(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")

            # Check memory
            swap_info = get_swap_info()
            if swap_info["used_mb"] > 1000:
                logger.warning(f"  ⚠️  High swap: {swap_info['used_mb']:.1f}MB")

            return mean_f1

        except Exception as e:
            logger.error(f"  ❌ Trial failed: {e}")
            raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dailydialog_native"),
        help="Data directory",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name for resuming (default: auto-generate)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1 for 8GB RAM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Bayesian Hyperparameter Optimization (Optuna)")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"n_jobs: {args.n_jobs}")

    # Check initial memory
    swap_info = get_swap_info()
    print(f"\nInitial swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")

    # Load pre-split data
    print(f"\nLoading pre-split data from {args.data_dir}/...")
    train_data = np.load(args.data_dir / "train.npz", allow_pickle=True)
    test_data = np.load(args.data_dir / "test.npz", allow_pickle=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"Train: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    print(f"Features: {X_train.shape[1]} ({metadata['embedding_dims']} emb + "
          f"{metadata['hand_crafted_dims']} hc + {metadata.get('spacy_dims', 0)} spacy)")

    # Create study
    study_name = args.study_name or f"svm_optuna_{int(time.time())}"
    db_path = PROJECT_ROOT / "optuna_studies.db"

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}")
    print(f"Database: {db_path}")
    if len(study.trials) > 0:
        print(f"Resuming from {len(study.trials)} previous trials")

    # Create objective
    objective = Objective(X_train, y_train, metadata, seed=args.seed, n_jobs=args.n_jobs)

    # Optimize
    print(f"\nStarting optimization at {time.strftime('%H:%M:%S')}...")
    print(f"Estimated time: ~{args.n_trials * 15} minutes ({args.n_trials * 15 / 60:.1f} hours)\n")

    start_time = time.time()
    try:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
    finally:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Optimization completed at {time.strftime('%H:%M:%S')}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

    # Best trial
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print("BEST TRIAL")
    print(f"{'='*60}")
    print(f"Trial number: {best_trial.number}")
    print(f"F1 score: {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        if key == "gamma_value" and best_trial.params.get("gamma_type") != "numeric":
            continue
        print(f"  {key}: {value}")

    # Train final model with best hyperparameters
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")

    # Extract best params
    best_C = best_trial.params["C"]
    if best_trial.params["gamma_type"] == "scale":
        best_gamma = "scale"
    else:
        best_gamma = best_trial.params["gamma_value"]

    best_resample = {
        label: best_trial.params[f"resample_{label}"]
        for label in sorted(set(y_train))
    }

    print(f"C: {best_C:.4f}")
    print(f"gamma: {best_gamma if isinstance(best_gamma, str) else f'{best_gamma:.6f}'}")
    print(f"Resampling:")
    for label, ratio in sorted(best_resample.items()):
        print(f"  {label:12s}: {ratio:.2f}x")

    # Apply resampling
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train, y_train, best_resample, args.seed
    )
    print(f"\nTraining size after resample: {len(X_train_resampled):,}")

    # Build final model
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

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("svm", SVC(
            kernel="rbf",
            C=best_C,
            gamma=best_gamma,
            class_weight="balanced",
            max_iter=50000,
            cache_size=1000,
            tol=1e-3,
            random_state=args.seed,
            verbose=True,
        )),
    ])

    # Train on full resampled training set
    print("\nTraining final model...")
    final_pipeline.fit(X_train_resampled, y_train_resampled)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = final_pipeline.predict(X_test)

    print("\nPer-class metrics:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
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
        for j in range(len(labels)):
            print(f"{cm[i, j]:>12d}", end="")
        print()

    # Save model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "category_svm_optuna_best.pkl"

    joblib.dump(final_pipeline, model_path)
    print(f"\n✓ Best model saved to: {model_path.name}")

    # Save test set
    test_file = model_dir / "test_data.npz"
    np.savez(test_file, X=X_test, y=y_test)
    print(f"✓ Test set saved to: {test_file.name}")

    # Save results
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    output_file = PROJECT_ROOT / f"optuna_results_{int(time.time())}.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best_trial.number,
            "f1_cv": best_trial.value,
            "params": best_trial.params,
        },
        "test_set_metrics": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_file.name}")

    # Final swap check
    swap_info = get_swap_info()
    print(f"\nFinal swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")

    print(f"\n{'='*60}")
    print("To resume this study later:")
    print(f"  uv run python scripts/optuna_search.py --study-name {study_name} --n-trials 10")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
