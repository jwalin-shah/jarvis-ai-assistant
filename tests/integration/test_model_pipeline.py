"""Integration tests for the model pipeline.

These tests verify the interaction between the Generator, Loader, and MLX.
Includes:
1. Mocked pipeline tests (always run) to verify plumbing.
2. Real model tests (run with --run-real-model) to verify actual inference.
"""

import pytest
from unittest.mock import MagicMock, patch
from jarvis.generation import can_use_llm
from contracts.models import GenerationRequest
from models.generator import MLXGenerator
from models.loader import MLXModelLoader, GenerationResult, ModelConfig

# =============================================================================
# Mocked Pipeline Tests (Always Run)
# =============================================================================

def test_generation_pipeline_mocked():
    """Verify the generator correctly calls the loader with valid prompts."""
    # Setup mock loader
    mock_loader = MagicMock(spec=MLXModelLoader)
    mock_loader.is_loaded.return_value = True
    mock_loader.generate_sync.return_value = GenerationResult(
        text="Mocked response",
        tokens_generated=5,
        generation_time_ms=10.0
    )
    # Mock has_prompt_cache to be False to simplify logic path
    mock_loader.has_prompt_cache = False

    # Initialize generator with mock loader
    generator = MLXGenerator(loader=mock_loader, skip_templates=True)
    
    # Create request
    request = GenerationRequest(
        prompt="Hello",
        context_documents=[],
        few_shot_examples=[],
        max_tokens=50
    )
    
    # Execute
    response = generator.generate(request)
    
    # Verify
    assert response.text == "Mocked response"
    assert response.finish_reason == "stop"
    
    # Verify loader call
    mock_loader.generate_sync.assert_called_once()
    call_kwargs = mock_loader.generate_sync.call_args.kwargs
    
    # Check that the prompt passed to generate_sync contains our input
    # (The actual prompt will be formatted by PromptBuilder, so we check for containment)
    assert "Hello" in call_kwargs['prompt'] or "Hello" in mock_loader.generate_sync.call_args[0][0]


# =============================================================================
# Real Model Tests (Run only when requested)
# =============================================================================

@pytest.mark.real_model
def test_real_model_loading_and_generation():
    """Smoke test for the actual MLX model.
    
    This test attempts to:
    1. Load the model (verifies path and memory)
    2. Run a simple generation (verifies tokenizer and tensor operations)
    
    Run with: pytest -v -m real_model
    """
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed")

    # Use a small configuration if possible, or default
    # We assume the default model is present or downloaded.
    # If not, this test will fail, which is intended for the smoke test.
    loader = MLXModelLoader()
    
    try:
        # Load
        # print("Loading model " + loader.config.display_name + "...")
        loader.load()
        assert loader.is_loaded()
        
        # Generate
        prompt = "Say hello."
        # print("Generating for prompt: " + prompt)
        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=10,
            temperature=0.1
        )
        
        # print("Generated: " + result.text)
        assert result.text
        assert len(result.text) > 0
        
    except Exception as e:
        if "Insufficient memory" in str(e):
            pytest.skip(f"Skipping real model test due to insufficient memory: {e}")
        pytest.fail("Real model test failed: " + str(e))
    finally:
        loader.unload()