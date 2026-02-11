"""Pytest fixtures for tool unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_training_data(tmp_path: Path) -> Path:
    """Create minimal valid training data for tests."""
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()

    # Create synthetic features
    n_features = 100
    n_train = 20
    n_test = 10

    # Training data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.array(["inform"] * 10 + ["question"] * 10)
    np.savez(data_dir / "train.npz", X=X_train, y=y_train)

    # Test data
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.array(["inform"] * 5 + ["question"] * 5)
    np.savez(data_dir / "test.npz", X=X_test, y=y_test)

    # Metadata
    metadata = {
        "embedding_dims": 96,
        "hand_crafted_dims": 4,
        "labels": ["inform", "question"],
        "version": "1.0.0",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata))

    return data_dir


@pytest.fixture
def invalid_training_data(tmp_path: Path) -> Path:
    """Create incomplete training data (missing files) for error testing."""
    data_dir = tmp_path / "invalid_data"
    data_dir.mkdir()

    # Only create train.npz, missing test.npz and metadata
    X = np.random.randn(10, 100)
    y = np.array(["inform"] * 10)
    np.savez(data_dir / "train.npz", X=X, y=y)

    return data_dir


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> Path:
    """Create a mock model directory structure."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create dummy model file
    import joblib
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent")
    model.fit([[1], [2]], ["a", "b"])
    joblib.dump(model, model_dir / "model.joblib")

    (model_dir / "metadata.json").write_text(json.dumps({"version": "1.0.0"}))

    return model_dir
