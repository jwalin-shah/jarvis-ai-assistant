"""Property-based tests for text normalizer."""

import pytest
from hypothesis import given, strategies as st
from jarvis.text_normalizer import normalize_text, is_reaction

@given(st.text())
def test_normalize_text_never_crashes(text):
    """Test that normalize_text never crashes on any input."""
    normalize_text(text)

@given(st.text())
def test_normalize_text_idempotency(text):
    """Test that normalize_text is idempotent."""
    first = normalize_text(text)
    second = normalize_text(first)
    assert first == second

@given(st.text())
def test_normalize_text_no_invisible_chars(text):
    """Test that normalized text contains no invisible characters."""
    normalized = normalize_text(text)
    # Check for some common invisible chars
    for char in ["\u200b", "\u200c", "\u200d", "\uFEFF"]:
        assert char not in normalized

@pytest.mark.parametrize("prefix", [
    "Liked ", "Loved ", "Disliked ", "Laughed at ", "Emphasized ", "Questioned "
])
@given(st.text())
def test_reactions_return_empty_string(prefix, text):
    """Test that iMessage reactions are always normalized to an empty string."""
    reaction = f'{prefix}"{text}"'
    assert normalize_text(reaction) == ""
    assert is_reaction(reaction) is True
