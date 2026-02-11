"""Training commands for JARVIS models.

This module implements training tools for:
- Category classification (LinearSVC)
- Response mobilization (LightGBM)
- Message replyability (Logistic Regression)
- Fact filtering (Custom classifier)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.tools.base import Tool, ToolResult
from jarvis.tools.registry import tool


@tool("train_category", "Train LinearSVC category classifier", "1.0.0")
class CategoryTrainer(Tool):
    """Train category classifier on labeled SOC data.

    Loads training data from NPZ files, trains a LinearSVC with
    GridSearchCV, and saves the model for production use.

    Input:
        - data/train.npz: Training features and labels
        - data/test.npz: Test features and labels
        - data/metadata.json: Feature dimensions and metadata

    Output:
        - Model files in ~/.jarvis/embeddings/{model}/category_classifier_model/
        - Training metrics and classification report

    Example:
        tool = CategoryTrainer({
            "data_dir": Path("data/soc_categories"),
            "label_map": "4class",
            "seed": 42,
        })
        result = tool.run()
    """

    # Configuration keys that are required
    required_config = ["data_dir"]

    # Configuration keys that should be valid paths
    path_config_keys = ["data_dir", "output"]

    def validate(self) -> list[str]:
        """Validate training configuration."""
        errors = super().validate()

        data_dir = self.config.get("data_dir")
        if data_dir:
            data_dir = Path(data_dir)

            # Check required files exist
            required_files = ["train.npz", "test.npz", "metadata.json"]
            for filename in required_files:
                if not (data_dir / filename).exists():
                    errors.append(f"Missing required file: {data_dir / filename}")

            # Validate label_map
            label_map = self.config.get("label_map", "4class")
            if label_map not in ("4class", "3class"):
                errors.append(f"Invalid label_map: {label_map} (must be 4class or 3class)")

        return errors

    def run(self, **kwargs) -> ToolResult:
        """Execute category classifier training.

        Args:
            **kwargs: Override config values for this run

        Returns:
            ToolResult with training metrics and model path
        """
        # Merge runtime kwargs with config
        config = {**self.config, **kwargs}

        data_dir = Path(config["data_dir"])
        label_map = config.get("label_map", "4class")
        seed = config.get("seed", 42)

        self.logger.info(f"Starting category training from {data_dir}")
        self.logger.info(f"Label map: {label_map}, Seed: {seed}")

        try:
            # Load data
            train_data = np.load(data_dir / "train.npz", allow_pickle=True)
            test_data = np.load(data_dir / "test.npz", allow_pickle=True)
            metadata = json.loads((data_dir / "metadata.json").read_text())

            X_train, y_train = train_data["X"], train_data["y"]
            X_test, y_test = test_data["X"], test_data["y"]

            # Apply label mapping
            if label_map == "3class":
                y_train, y_test = self._apply_3class_mapping(y_train, y_test)

            self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

            # Import sklearn here for lazy loading
            from sklearn.compose import ColumnTransformer
            from sklearn.metrics import accuracy_score, classification_report, f1_score
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC

            # Build pipeline
            embedding_dims = metadata["embedding_dims"]
            hand_crafted_dims = metadata["hand_crafted_dims"]

            preprocessor = ColumnTransformer(
                [
                    ("embeddings", StandardScaler(), slice(0, embedding_dims)),
                    (
                        "handcrafted",
                        "passthrough",
                        slice(embedding_dims, embedding_dims + hand_crafted_dims),
                    ),
                ]
            )

            pipeline = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("svc", LinearSVC(dual=False, random_state=seed)),
                ]
            )

            # Grid search
            param_grid = {
                "svc__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "svc__class_weight": ["balanced", None],
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

            self.logger.info("Running GridSearchCV...")
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=1,
                verbose=0,
            )
            search.fit(X_train, y_train)

            # Evaluate
            y_pred = search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            self.logger.info(f"Best params: {search.best_params_}")
            self.logger.info(f"Test accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Save model
            output_path = self._save_model(
                search.best_estimator_,
                metadata,
                config.get("output"),
            )

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            return ToolResult(
                success=True,
                message=f"Training complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}",
                data={
                    "best_params": search.best_params_,
                    "classification_report": report,
                },
                artifacts=[output_path] if output_path else [],
                metrics={
                    "accuracy": accuracy,
                    "f1_weighted": f1,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                },
            )

        except Exception as e:
            self.logger.exception("Training failed")
            return ToolResult(
                success=False,
                message=f"Training failed: {e}",
            )

    def _apply_3class_mapping(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply 3-class label mapping."""
        mapping = {
            "inform": "inform",
            "question": "question",
            "directive": "action",
            "commissive": "action",
        }
        y_train = np.array([mapping.get(str(y), str(y)) for y in y_train])
        y_test = np.array([mapping.get(str(y), str(y)) for y in y_test])
        self.logger.info("Applied 3-class mapping (directive+commissive → action)")
        return y_train, y_test

    def _save_model(
        self,
        model: Any,
        metadata: dict,
        output_dir: Path | None,
    ) -> Path | None:
        """Save trained model to disk."""
        import joblib

        from jarvis.config import get_category_classifier_path

        if output_dir is None:
            output_dir = get_category_classifier_path()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "model.joblib"
        metadata_path = output_dir / "metadata.json"

        joblib.dump(model, model_path)
        metadata_path.write_text(json.dumps(metadata, indent=2))

        self.logger.info(f"Model saved to {model_path}")
        return model_path
