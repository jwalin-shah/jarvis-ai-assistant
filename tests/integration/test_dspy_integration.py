"""Integration test for DSPy with MLX.

Verifies that the DSPYMLXClient adapter correctly connects 
DSPy's high-level logic with the local MLXGenerator.
"""

import pytest
import dspy
from jarvis.dspy_client import DSPYMLXClient

# Define a simple DSPy Signature for the test
class SimpleReply(dspy.Signature):
    """Generate a short reply to a message."""
    incoming_message = dspy.InputField(desc="The message to reply to")
    reply = dspy.OutputField(desc="A helpful and concise response")

@pytest.mark.real_model
def test_dspy_mlx_integration():
    """Verify DSPy can drive the local MLX model."""
    try:
        import mlx.core
    except ImportError:
        pytest.skip("MLX not installed")

    # 1. Configure DSPy to use our local client
    # We use a very low max_tokens to keep the test fast
    local_lm = DSPYMLXClient(max_tokens=20, temperature=0.1)
    dspy.settings.configure(lm=local_lm)
    
    # 2. Define the predictor (ZeroShot)
    # ZeroShot means "just use the instruction", no compilation yet.
    # This tests the plumbing.
    predictor = dspy.Predict(SimpleReply)
    
    # 3. Execute
    input_text = "What time is it?"
    print(f"
[DSPy] Input: {input_text}")
    
    # This internally calls: 
    # DSPYMLXClient.__call__ -> MLXGenerator.generate -> MLXModelLoader.generate_sync
    result = predictor(incoming_message=input_text)
    
    print(f"[DSPy] Output: {result.reply}")
    
    # 4. Verify
    assert result.reply is not None
    assert len(result.reply) > 0
    # Inspect the history to ensure the prompt was formed correctly
    assert len(local_lm.history) > 0
    print(f"[DSPy] History check passed. Last prompt length: {len(local_lm.history[-1]['prompt'])}")
