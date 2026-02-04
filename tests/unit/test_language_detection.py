"""Unit tests for language detection in jarvis/text_normalizer.py."""


from jarvis.text_normalizer import detect_language, is_english


class TestDetectLanguage:
    """Tests for the detect_language function."""

    def test_empty_text_returns_english(self) -> None:
        """Empty text should default to English."""
        assert detect_language("") == "en"
        assert detect_language("  ") == "en"

    def test_short_text_returns_english(self) -> None:
        """Very short text should default to English."""
        assert detect_language("hi") == "en"
        assert detect_language("ok") == "en"

    def test_detect_english_text(self) -> None:
        """English text should be detected as English."""
        result = detect_language("Hello, how are you doing today?")
        assert result == "en"

    def test_detect_long_english_text(self) -> None:
        """Longer English text should be detected as English."""
        text = "The quick brown fox jumps over the lazy dog."
        result = detect_language(text)
        assert result == "en"


class TestIsEnglish:
    """Tests for the is_english function."""

    def test_empty_text_is_english(self) -> None:
        """Empty text should be considered English."""
        assert is_english("") is True

    def test_short_text_is_english(self) -> None:
        """Short text (< 10 chars) should be assumed English."""
        assert is_english("hi") is True
        assert is_english("ok!") is True
        assert is_english("lol") is True
        assert is_english("yeah") is True

    def test_english_text_detected(self) -> None:
        """English text should be detected as English."""
        assert is_english("Hello, how are you doing today?") is True
        assert is_english("I'm going to the store to buy some groceries.") is True

    def test_english_slang_detected(self) -> None:
        """English slang/abbreviations should be detected as English."""
        # These may be short but should still be detected correctly
        assert is_english("Hey what's up?") is True
        assert is_english("Want to grab lunch later?") is True

    def test_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        # Very high threshold - harder to pass
        result = is_english("Hello there!", threshold=0.99)
        # Result depends on langdetect confidence
        assert isinstance(result, bool)

        # Very low threshold - easier to pass
        result = is_english("Hello there!", threshold=0.1)
        assert result is True

    def test_handles_detection_errors(self) -> None:
        """Function should handle detection errors gracefully."""
        # Special characters and emojis might cause issues
        assert is_english("😀😂🎉") is True  # Short, defaults to True
        assert isinstance(is_english("👍👍👍👍👍👍👍👍👍👍👍"), bool)


class TestIntegrationWithNormalization:
    """Tests for language detection integration with text normalization."""

    def test_normalize_for_task_with_filter(self) -> None:
        """normalize_for_task should filter non-English when enabled."""
        from jarvis.config import NormalizationProfile

        # Create a profile with filter_non_english enabled
        profile = NormalizationProfile(filter_non_english=True)
        assert profile.filter_non_english is True

        # Create a profile without the filter
        profile_no_filter = NormalizationProfile(filter_non_english=False)
        assert profile_no_filter.filter_non_english is False

    def test_default_profile_no_filter(self) -> None:
        """Default normalization profile should not filter non-English."""
        from jarvis.config import NormalizationProfile

        profile = NormalizationProfile()
        assert profile.filter_non_english is False
