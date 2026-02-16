"""TEST-03: Generation timeout tests for MLXModelLoader.

Verifies that:
1. Timeout raises ModelGenerationError
2. Subsequent generation still works after a timeout
3. No timeout when timeout_seconds is None
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from jarvis.core.exceptions import ErrorCode, ModelGenerationError


class TestGenerationTimeout:
    """Tests for generation timeout behavior in MLXModelLoader."""

    def _make_loader_with_mock_model(self):
        """Create an MLXModelLoader with mocked model/tokenizer."""
        from models.loader import MLXModelLoader, ModelConfig

        config = ModelConfig()
        config.generation_timeout_seconds = 1.0  # 1 second timeout
        loader = MLXModelLoader(config)

        # Mock model and tokenizer
        loader._model = MagicMock()
        loader._tokenizer = MagicMock()
        loader._tokenizer.apply_chat_template = MagicMock(
            return_value="<user>test</user><assistant>"
        )
        loader._tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        loader._loaded_at = time.perf_counter()

        return loader

    @patch("models.loader.generate")
    @patch("models.loader.make_sampler")
    @patch("models.loader.make_repetition_penalty")
    def test_timeout_raises_model_generation_error(
        self, mock_rep_penalty, mock_sampler, mock_generate
    ):
        """Generation exceeding timeout raises ModelGenerationError with MDL_TIMEOUT."""
        loader = self._make_loader_with_mock_model()

        # Make generate sleep longer than the timeout
        def slow_generate(**kwargs):
            time.sleep(5.0)
            return "slow response"

        mock_generate.side_effect = slow_generate
        mock_sampler.return_value = MagicMock()
        mock_rep_penalty.return_value = MagicMock()

        with pytest.raises(ModelGenerationError) as exc_info:
            loader.generate_sync(
                "test prompt",
                timeout_seconds=0.5,
            )

        assert exc_info.value.code == ErrorCode.MDL_TIMEOUT

    @patch("models.loader.generate")
    @patch("models.loader.make_sampler")
    @patch("models.loader.make_repetition_penalty")
    def test_generation_works_after_timeout(self, mock_rep_penalty, mock_sampler, mock_generate):
        """After a timeout, subsequent generation should still work."""
        loader = self._make_loader_with_mock_model()

        mock_sampler.return_value = MagicMock()
        mock_rep_penalty.return_value = MagicMock()

        call_count = {"n": 0}

        def generate_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: sleep to trigger timeout
                time.sleep(5.0)
                return "should not reach"
            else:
                # Second call: fast response
                return "<user>test</user><assistant>fast response"

        mock_generate.side_effect = generate_side_effect

        # First call should timeout
        with pytest.raises(ModelGenerationError):
            loader.generate_sync("test prompt", timeout_seconds=0.3)

        # Wait for the background thread to settle after timeout cancellation
        time.sleep(0.5)

        # Second call should succeed
        result = loader.generate_sync("test prompt", timeout_seconds=5.0)
        assert result.text is not None

    @patch("models.loader.generate")
    @patch("models.loader.make_sampler")
    @patch("models.loader.make_repetition_penalty")
    def test_no_timeout_when_none(self, mock_rep_penalty, mock_sampler, mock_generate):
        """When timeout_seconds is None, no timeout is applied."""
        loader = self._make_loader_with_mock_model()
        loader.config.generation_timeout_seconds = None

        mock_sampler.return_value = MagicMock()
        mock_rep_penalty.return_value = MagicMock()
        mock_generate.return_value = "<user>test</user><assistant>response text"

        # Should not raise
        result = loader.generate_sync(
            "test prompt",
            timeout_seconds=None,
        )
        assert result.text is not None

    def test_empty_prompt_raises_immediately(self):
        """Empty prompt raises ModelGenerationError before timeout logic."""
        loader = self._make_loader_with_mock_model()

        with pytest.raises(ModelGenerationError) as exc_info:
            loader.generate_sync("", timeout_seconds=1.0)

        assert exc_info.value.code == ErrorCode.MDL_INVALID_REQUEST

    def test_unloaded_model_raises_immediately(self):
        """Unloaded model raises ModelGenerationError before timeout logic."""
        from models.loader import MLXModelLoader, ModelConfig

        loader = MLXModelLoader(ModelConfig())
        # Model is not loaded

        with pytest.raises(ModelGenerationError) as exc_info:
            loader.generate_sync("test prompt")

        assert exc_info.value.code == ErrorCode.MDL_LOAD_FAILED
