"""Unit tests for ValidityGate."""

from jarvis.nlp.validity_gate import GateConfig, ValidityGate


def test_validity_gate_empty_text():
    """Test that empty or whitespace-only text is rejected."""
    gate = ValidityGate()

    passed, reason = gate.validate("")
    assert not passed
    assert reason == "empty_text"

    passed, reason = gate.validate("   ")
    assert not passed
    assert reason == "empty_text"


def test_validity_gate_reaction():
    """Test that reactions (tapbacks) are rejected."""
    gate = ValidityGate()

    # "Liked 'hello'" is a reaction pattern
    passed, reason = gate.validate('Liked "hello"')
    assert not passed
    assert reason == "reaction"

    # Should pass if reaction check is disabled
    config = GateConfig(reject_reactions=False)
    gate = ValidityGate(config=config)
    passed, _ = gate.validate('Liked "hello"')
    # Note: If it's a reaction, normalize_text might return empty string,
    # but here we are calling validate directly.
    # Actually, is_reaction check comes first.
    # If disabled, it falls through. However, 'Liked "hello"' might be caught by other filters?
    # No, it's just text.
    assert passed


def test_validity_gate_garbage():
    """Test that garbage messages are rejected."""
    gate = ValidityGate()

    passed, reason = gate.validate("__kIMFileTransferGUID")
    assert not passed
    assert reason == "garbage"


def test_validity_gate_spam():
    """Test that spam messages are rejected."""
    gate = ValidityGate()

    passed, reason = gate.validate("Limited time offer")
    assert not passed
    assert reason == "spam"


def test_validity_gate_emoji_only():
    """Test that emoji-only messages are rejected."""
    gate = ValidityGate()

    passed, reason = gate.validate("👍")
    assert not passed
    assert reason == "emoji_only"

    passed, reason = gate.validate("😂 🤣")
    assert not passed
    assert reason == "emoji_only"


def test_validity_gate_acknowledgment():
    """Test that acknowledgment-only messages are rejected."""
    gate = ValidityGate()

    passed, reason = gate.validate("ok")
    assert not passed
    assert reason == "acknowledgment"

    passed, reason = gate.validate("got it")
    assert not passed
    assert reason == "acknowledgment"

    # "Thanks" is also an acknowledgment
    passed, reason = gate.validate("thanks")
    assert not passed
    assert reason == "acknowledgment"


def test_validity_gate_valid_text():
    """Test that valid text passes Gate A."""
    gate = ValidityGate()

    passed, reason = gate.validate("Hello world, how are you?")
    assert passed
    assert reason == "passed"

    # Acknowledgment with more content should pass (depending on is_acknowledgment_only implementation)
    # is_acknowledgment_only returns True if the text IS ONLY an acknowledgment.
    passed, reason = gate.validate("Ok, I will do that.")
    assert passed
    assert reason == "passed"


def test_validity_gate_config_disable():
    """Test that checks can be disabled via config."""
    # Disable all checks
    config = GateConfig(
        reject_reactions=False,
        reject_acknowledgments=False,
        reject_garbage=False,
        reject_spam=False,
        reject_emojis=False
    )
    gate = ValidityGate(config=config)

    # Should pass everything (except empty text which is hardcoded check)
    assert gate.validate('Liked "hello"')[0]
    assert gate.validate("__kIMFileTransferGUID")[0]
    assert gate.validate("Your order has shipped")[0]
    assert gate.validate("👍")[0]
    assert gate.validate("ok")[0]
