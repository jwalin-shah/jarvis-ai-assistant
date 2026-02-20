"""Behavior tests for health-aware generation.

Tests focus on public API behavior, not implementation details.
Given input X, expect output Y - tests should pass even if internals are rewritten.
"""

from unittest.mock import patch

from contracts.memory import MemoryMode, MemoryState
from contracts.models import GenerationRequest, GenerationResponse
from jarvis.generation import (
    can_use_llm,
    generate_summary,
    generate_with_fallback,
    get_generation_status,
)


class TestCanUseLLM:
    """Behavior: System reports whether LLM can be used based on health."""

    def test_allows_llm_when_system_in_full_mode(self):
        """When memory mode is FULL, LLM is allowed."""
        mock_state = MemoryState(
            available_mb=10000,
            used_mb=6000,
            model_loaded=False,
            current_mode=MemoryMode.FULL,
            pressure_level="green",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            allowed, reason = can_use_llm()

        assert allowed is True
        assert reason == ""

    def test_allows_llm_when_system_in_lite_mode(self):
        """When memory mode is LITE, LLM is still allowed."""
        mock_state = MemoryState(
            available_mb=6000,
            used_mb=10000,
            model_loaded=False,
            current_mode=MemoryMode.LITE,
            pressure_level="yellow",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            allowed, reason = can_use_llm()

        assert allowed is True
        assert reason == ""

    def test_blocks_llm_when_system_in_minimal_mode(self):
        """When memory mode is MINIMAL, LLM is blocked with memory reason."""
        mock_state = MemoryState(
            available_mb=2000,
            used_mb=14000,
            model_loaded=False,
            current_mode=MemoryMode.MINIMAL,
            pressure_level="red",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            allowed, reason = can_use_llm()

        assert allowed is False
        assert "memory" in reason.lower()

    def test_blocks_llm_when_pressure_is_critical(self):
        """When pressure is critical, LLM is blocked regardless of mode."""
        mock_state = MemoryState(
            available_mb=5000,
            used_mb=11000,
            model_loaded=True,
            current_mode=MemoryMode.LITE,
            pressure_level="critical",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            allowed, reason = can_use_llm()

        assert allowed is False
        assert "critical" in reason.lower()

    def test_allows_llm_when_pressure_is_red_but_not_critical(self):
        """Red pressure (non-critical) should still allow LLM usage."""
        mock_state = MemoryState(
            available_mb=4000,
            used_mb=12000,
            model_loaded=True,
            current_mode=MemoryMode.LITE,
            pressure_level="red",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            allowed, reason = can_use_llm()

        assert allowed is True

    def test_blocks_llm_when_memory_check_fails(self):
        """When memory check fails, LLM is blocked as fail-safe."""
        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.side_effect = RuntimeError("Memory controller unavailable")
            allowed, reason = can_use_llm()

        assert allowed is False
        assert "failed" in reason.lower()


class TestGetGenerationStatus:
    """Behavior: Returns system status for health monitoring."""

    def test_returns_status_dictionary_with_expected_fields(self):
        """Status contains model state, generation capability, and memory info."""
        mock_state = MemoryState(
            available_mb=8000,
            used_mb=8000,
            model_loaded=True,
            current_mode=MemoryMode.FULL,
            pressure_level="green",
        )

        with (
            patch("jarvis.generation.get_generator") as mock_gen,
            patch("jarvis.generation.get_memory_controller") as mock_mem,
        ):
            mock_gen.return_value.is_loaded.return_value = True
            mock_mem.return_value.get_state.return_value = mock_state
            mock_mem.return_value.get_mode.return_value = MemoryMode.FULL

            status = get_generation_status()

        assert "model_loaded" in status
        assert "can_generate" in status
        assert "reason" in status
        assert "memory_mode" in status
        assert status["model_loaded"] is True
        assert status["can_generate"] is True
        assert status["memory_mode"] == "full"

    def test_reports_model_not_loaded_when_generator_unavailable(self):
        """When generator fails, status reports model as not loaded."""
        with (
            patch("jarvis.generation.get_generator") as mock_gen,
            patch("jarvis.generation.get_memory_controller") as mock_mem,
        ):
            mock_gen.side_effect = RuntimeError("Generator not available")
            mock_mem.return_value.get_mode.return_value = MemoryMode.LITE

            status = get_generation_status()

        assert status["model_loaded"] is False


class TestGenerateWithFallback:
    """Behavior: Generates text with automatic fallback on failure."""

    def test_returns_error_response_when_llm_unavailable(self):
        """When LLM is blocked, returns error response with reason."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Memory too low")
            response = generate_with_fallback(request)

        assert response.finish_reason == "error"
        assert response.error == "Memory too low"
        assert response.model_name == "fallback"
        assert response.text == ""

    def test_returns_generated_text_when_llm_available(self):
        """When LLM is healthy, returns generated content."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        mock_response = GenerationResponse(
            text="Generated text",
            tokens_used=10,
            generation_time_ms=100.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.get_generator") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value.is_loaded.return_value = True
            mock_gen.return_value.generate.return_value = mock_response

            response = generate_with_fallback(request)

        assert response.text == "Generated text"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 10

    def test_returns_error_response_when_generation_fails(self):
        """When generation throws exception, returns error response."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.get_generator") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value.is_loaded.return_value = True
            mock_gen.return_value.generate.side_effect = RuntimeError("Generation failed")

            response = generate_with_fallback(request)

        assert response.finish_reason == "error"
        assert "Generation failed" in response.error
        assert response.model_name == "fallback"


class TestGenerateSummary:
    """Behavior: Returns conversation summary with fallback when needed."""

    def test_returns_fallback_when_llm_blocked(self):
        """When LLM unavailable, returns fallback summary mentioning participant."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Memory low")
            summary, used_fallback = generate_summary(["msg1", "msg2"], "John")

        assert used_fallback is True
        assert "John" in summary
        assert "Unable to generate summary" in summary

    def test_returns_fallback_for_empty_messages(self):
        """Empty message list returns simple fallback without calling LLM."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (True, "")
            summary, used_fallback = generate_summary([], "Jane")

        assert used_fallback is True
        assert "Jane" in summary
        assert "No messages" in summary

    def test_returns_llm_summary_when_healthy(self):
        """When healthy, returns LLM-generated summary."""
        mock_response = GenerationResponse(
            text="This is a summary of the conversation.",
            tokens_used=20,
            generation_time_ms=150.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.generate_with_fallback") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value = mock_response

            summary, used_fallback = generate_summary(["Hello", "Hi there"], "Bob")

        assert used_fallback is False
        assert summary == "This is a summary of the conversation."

    def test_returns_fallback_when_llm_returns_error(self):
        """When LLM returns error response, falls back to default summary."""
        mock_response = GenerationResponse(
            text="",
            tokens_used=0,
            generation_time_ms=0,
            model_name="fallback",
            used_template=False,
            template_name=None,
            finish_reason="error",
            error="Generation failed",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.generate_with_fallback") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value = mock_response

            summary, used_fallback = generate_summary(["msg"], "Alice")

        assert used_fallback is True
        assert "Alice" in summary
