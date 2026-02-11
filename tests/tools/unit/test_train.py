"""Unit tests for training commands."""

from pathlib import Path

import pytest

from jarvis.tools.commands.train import CategoryTrainer


class TestCategoryTrainer:
    """Tests for CategoryTrainer."""

    def test_initialization(self, temp_training_data: Path):
        """Trainer initializes with valid config."""
        trainer = CategoryTrainer(
            {
                "data_dir": temp_training_data,
                "seed": 123,
            }
        )
        assert trainer.config["seed"] == 123

    def test_validation_missing_data_dir(self):
        """Validation fails when data_dir is missing."""
        trainer = CategoryTrainer({})
        errors = trainer.validate()
        assert any("data_dir" in e.lower() for e in errors)

    def test_validation_missing_required_files(self, tmp_path: Path):
        """Validation fails when required files don't exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        trainer = CategoryTrainer({"data_dir": empty_dir})
        errors = trainer.validate()

        assert any("train.npz" in e for e in errors)
        assert any("test.npz" in e for e in errors)
        assert any("metadata.json" in e for e in errors)

    def test_validation_valid_data(self, temp_training_data: Path):
        """Validation passes with valid data."""
        trainer = CategoryTrainer({"data_dir": temp_training_data})
        errors = trainer.validate()
        assert errors == []

    def test_validation_invalid_label_map(self, temp_training_data: Path):
        """Validation fails with invalid label_map."""
        trainer = CategoryTrainer(
            {
                "data_dir": temp_training_data,
                "label_map": "invalid",
            }
        )
        errors = trainer.validate()
        assert any("label_map" in e.lower() for e in errors)

    @pytest.mark.parametrize("label_map", ["4class", "3class"])
    def test_validation_valid_label_maps(self, temp_training_data: Path, label_map: str):
        """Validation passes with valid label maps."""
        trainer = CategoryTrainer(
            {
                "data_dir": temp_training_data,
                "label_map": label_map,
            }
        )
        errors = trainer.validate()
        assert not any("label_map" in e.lower() for e in errors)

    def test_dry_run_with_valid_data(self, temp_training_data: Path):
        """Dry run passes with valid configuration."""
        trainer = CategoryTrainer({"data_dir": temp_training_data})
        result = trainer.dry_run()
        assert result.success is True

    def test_run_produces_result(self, temp_training_data: Path, tmp_path: Path):
        """Training run produces a ToolResult."""
        output_dir = tmp_path / "output"

        trainer = CategoryTrainer(
            {
                "data_dir": temp_training_data,
                "output": output_dir,
                "seed": 42,
            }
        )

        result = trainer.run()

        # Should succeed
        assert result.success is True

        # Should have metrics
        assert "accuracy" in result.metrics
        assert "f1_weighted" in result.metrics

        # Should have generated artifacts
        assert len(result.artifacts) > 0

    def test_run_with_3class_mapping(self, temp_training_data: Path, tmp_path: Path):
        """Training works with 3-class label mapping."""
        trainer = CategoryTrainer(
            {
                "data_dir": temp_training_data,
                "output": tmp_path / "output",
                "label_map": "3class",
            }
        )

        result = trainer.run()
        assert result.success is True
        assert "3-class" in result.message.lower() or "accuracy" in result.message.lower()
