"""Integration test for DSPy with MLX.

Verifies that the DSPYMLXClient adapter correctly connects
DSPy's high-level logic with the local MLX model via BaseLM.forward().
"""

import os

import dspy
import pytest
from evals.dspy_client import DSPYMLXClient


class SimpleReply(dspy.Signature):
    """Generate a short reply to a message."""

    incoming_message = dspy.InputField(desc="The message to reply to")
    reply = dspy.OutputField(desc="A helpful and concise response")


@pytest.mark.real_model
@pytest.mark.skipif(
    os.getenv("RUN_DSPY_INTEGRATION") != "1",
    reason="DSPy integration is opt-in. Set RUN_DSPY_INTEGRATION=1 to enable.",
)
def test_dspy_mlx_integration():
    """Verify DSPy can drive the local MLX model via BaseLM.forward()."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX not installed")

    import psutil

    if psutil.virtual_memory().available < 2 * 1024**3:
        pytest.skip("Insufficient available memory (<2GB) to load LLM")

    local_lm = DSPYMLXClient(max_tokens=20, temperature=0.1)
    dspy.configure(lm=local_lm)

    predictor = dspy.Predict(SimpleReply)

    input_text = "What time is it?"
    print(f"\n[DSPy] Input: {input_text}")

    result = predictor(incoming_message=input_text)

    print(f"[DSPy] Output: {result.reply}")

    assert result.reply is not None
    assert len(result.reply) > 0
    assert len(local_lm.history) > 0
    print(f"[DSPy] History check passed. Entries: {len(local_lm.history)}")
