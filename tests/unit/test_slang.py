"""Unit tests for jarvis/slang.py - Slang Expansion."""


from jarvis.slang import SLANG_MAP, expand_slang, get_slang_map


class TestSlangMap:
    """Tests for the slang expansion dictionary."""

    def test_slang_map_not_empty(self) -> None:
        """SLANG_MAP should contain entries."""
        assert len(SLANG_MAP) > 0

    def test_common_abbreviations_present(self) -> None:
        """Common abbreviations should be in the map."""
        assert "u" in SLANG_MAP
        assert "ur" in SLANG_MAP
        assert "rn" in SLANG_MAP
        assert "btw" in SLANG_MAP
        assert "lmk" in SLANG_MAP
        assert "idk" in SLANG_MAP

    def test_get_slang_map_returns_copy(self) -> None:
        """get_slang_map should return a copy of the map."""
        map1 = get_slang_map()
        map2 = get_slang_map()
        assert map1 == map2
        assert map1 is not map2  # Should be a copy


class TestExpandSlang:
    """Tests for the expand_slang function."""

    def test_expand_empty_text(self) -> None:
        """expand_slang should handle empty text."""
        assert expand_slang("") == ""
        assert expand_slang(None) is None

    def test_expand_basic_abbreviations(self) -> None:
        """expand_slang should expand basic abbreviations."""
        assert expand_slang("u coming rn?") == "you coming right now?"
        assert expand_slang("idk what to do") == "I don't know what to do"
        assert expand_slang("lmk when ur free") == "let me know when your free"

    def test_preserve_case_lowercase(self) -> None:
        """expand_slang should preserve lowercase."""
        result = expand_slang("u coming?")
        assert "you" in result

    def test_preserve_case_uppercase_multichar(self) -> None:
        """expand_slang should preserve uppercase for multi-char words."""
        result = expand_slang("IDK what to do")
        assert "I DON'T KNOW" in result

    def test_preserve_case_single_char_capitalizes(self) -> None:
        """Single-char abbreviations should capitalize (not uppercase)."""
        # Single char "U" -> "You" (capitalize, not uppercase)
        result = expand_slang("U coming?")
        assert "You" in result

    def test_word_boundary_matching(self) -> None:
        """expand_slang should only match whole words."""
        # "burn" should NOT become "beurn" (the 'u' and 'r' shouldn't match)
        assert expand_slang("burn") == "burn"
        assert expand_slang("urn") == "urn"
        assert expand_slang("fur") == "fur"

    def test_no_slang_text_unchanged(self) -> None:
        """Text without slang should be unchanged."""
        original = "Hello, how are you doing today?"
        assert expand_slang(original) == original

    def test_multiple_slang_terms(self) -> None:
        """Multiple slang terms in one message should all be expanded."""
        result = expand_slang("hey u wanna hang rn? lmk")
        assert "you" in result
        assert "want to" in result
        assert "right now" in result
        assert "let me know" in result

    def test_slang_with_punctuation(self) -> None:
        """Slang terms adjacent to punctuation should be expanded."""
        assert expand_slang("u?") == "you?"
        assert expand_slang("idk!") == "I don't know!"
        assert expand_slang("brb,") == "be right back,"

    def test_common_message_patterns(self) -> None:
        """Common iMessage patterns should be expanded correctly."""
        assert "where you at" in expand_slang("wya?")
        assert "on my way" in expand_slang("omw!")
        assert "to be honest" in expand_slang("tbh I'm not sure")
        assert "for your information" in expand_slang("fyi the meeting moved")
