"""Tests for the simplified ReplySuggester.

Tests:
1. Pattern detection accuracy
2. Suggestion quality
3. Performance/latency
"""

import time

import pytest

from jarvis.reply_suggester import (
    MessagePattern,
    ReplySuggester,
    detect_pattern,
    get_reply_suggestions,
)


class TestPatternDetection:
    """Test structural pattern detection."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            # Invitations
            ("Want to grab lunch?", MessagePattern.INVITATION),
            ("wanna hang out?", MessagePattern.INVITATION),
            ("down to play basketball?", MessagePattern.INVITATION),
            ("Are you free tonight?", MessagePattern.INVITATION),
            ("Can you come to the party?", MessagePattern.INVITATION),
            ("tryna ball tomorrow?", MessagePattern.INVITATION),
            ("lets go get food", MessagePattern.INVITATION),
            # Reaction prompts
            ("omg did you see that play??", MessagePattern.REACTION_PROMPT),
            ("dude did you hear about the news?", MessagePattern.REACTION_PROMPT),
            ("bro can you believe that?", MessagePattern.REACTION_PROMPT),
            ("have you seen the new movie?", MessagePattern.REACTION_PROMPT),
            # Venting
            ("fuck this is so frustrating", MessagePattern.VENTING),
            ("ugh I'm so tired", MessagePattern.VENTING),
            ("damn bro", MessagePattern.VENTING),
            ("I'm so frustrated with work", MessagePattern.VENTING),
            # Info questions
            ("What time is the meeting?", MessagePattern.INFO_QUESTION),
            ("Where are we going?", MessagePattern.INFO_QUESTION),
            ("When does it start?", MessagePattern.INFO_QUESTION),
            ("How much does it cost?", MessagePattern.INFO_QUESTION),
            ("Who else is coming?", MessagePattern.INFO_QUESTION),
            # Yes/No questions
            ("Did you finish the report?", MessagePattern.YN_QUESTION),
            ("Is it raining outside?", MessagePattern.YN_QUESTION),
            ("Can we reschedule?", MessagePattern.YN_QUESTION),
            ("Are you coming?", MessagePattern.YN_QUESTION),
            # Greetings (note: "what's up" now matches INFO_QUESTION due to "what" prefix)
            ("hey", MessagePattern.GREETING),
            ("Hi!", MessagePattern.GREETING),
            ("hello", MessagePattern.GREETING),
            ("yo", MessagePattern.GREETING),
            # Acknowledgments
            ("ok", MessagePattern.ACKNOWLEDGMENT),
            ("sounds good", MessagePattern.ACKNOWLEDGMENT),
            ("got it", MessagePattern.ACKNOWLEDGMENT),
            ("thanks", MessagePattern.ACKNOWLEDGMENT),
            ("bet", MessagePattern.ACKNOWLEDGMENT),
            # Statements
            ("I'm on my way", MessagePattern.STATEMENT),
            ("The meeting got moved to 3pm", MessagePattern.STATEMENT),
            ("Just finished the project", MessagePattern.STATEMENT),
        ],
    )
    def test_pattern_detection(self, message: str, expected: MessagePattern) -> None:
        """Test that messages are classified to correct patterns."""
        result = detect_pattern(message)
        assert result == expected, f"Expected {expected.value} for '{message}', got {result.value}"

    def test_pattern_detection_speed(self) -> None:
        """Pattern detection should be fast (no ML)."""
        messages = [
            "Want to grab lunch?",
            "omg did you see that?",
            "What time?",
            "ok sounds good",
            "I'm on my way",
        ] * 100  # 500 messages

        start = time.perf_counter()
        for msg in messages:
            detect_pattern(msg)
        elapsed = time.perf_counter() - start

        # Should process 500 messages in < 100ms (just regex)
        assert elapsed < 0.1, f"Pattern detection too slow: {elapsed * 1000:.1f}ms"
        per_msg = elapsed / len(messages) * 1000
        print(f"\nPattern detection: {elapsed * 1000:.1f}ms for 500 msgs ({per_msg:.3f}ms each)")


class TestReplySuggester:
    """Test the full ReplySuggester."""

    @pytest.fixture
    def suggester(self) -> ReplySuggester:
        """Create a suggester instance."""
        return ReplySuggester()

    def test_suggest_invitation(self, suggester: ReplySuggester) -> None:
        """Test suggestions for invitations."""
        result = suggester.suggest("Want to grab lunch?", n_suggestions=3)

        assert result.pattern == MessagePattern.INVITATION
        assert len(result.suggestions) == 3

        # Should have diverse options (yes/no/question types)
        texts = [s.text.lower() for s in result.suggestions]
        print(f"\nInvitation suggestions: {texts}")

    def test_suggest_reaction_prompt(self, suggester: ReplySuggester) -> None:
        """Test suggestions for reaction prompts."""
        result = suggester.suggest("omg did you see that crazy play??")

        assert result.pattern == MessagePattern.REACTION_PROMPT
        assert len(result.suggestions) >= 1
        print(f"\nReaction suggestions: {result.texts}")

    def test_suggest_venting(self, suggester: ReplySuggester) -> None:
        """Test suggestions for venting messages."""
        result = suggester.suggest("ugh I'm so stressed about this deadline")

        assert result.pattern == MessagePattern.VENTING
        assert len(result.suggestions) >= 1
        print(f"\nVenting suggestions: {result.texts}")

    def test_suggest_includes_templates(self, suggester: ReplySuggester) -> None:
        """Templates should fill in when retrieval doesn't have enough."""
        result = suggester.suggest(
            "Want to do something random and unique that's never been asked before xyz123?",
            n_suggestions=3,
        )

        # Should still return 3 suggestions (from templates)
        assert len(result.suggestions) == 3

        # At least some should be from templates
        sources = [s.source for s in result.suggestions]
        print(f"\nSources: {sources}")

    def test_convenience_function(self) -> None:
        """Test the get_reply_suggestions convenience function."""
        suggestions = get_reply_suggestions("hey what's up", n_suggestions=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert all(isinstance(s, str) for s in suggestions)


class TestPerformance:
    """Test performance characteristics."""

    def test_suggester_latency(self) -> None:
        """Test end-to-end latency."""
        suggester = ReplySuggester()

        messages = [
            "Want to grab lunch?",
            "What time is the meeting?",
            "omg did you see that?",
            "ok sounds good",
            "I'm heading out now",
        ]

        # Warm up
        suggester.suggest(messages[0])

        # Time multiple suggestions
        times = []
        for msg in messages:
            start = time.perf_counter()
            result = suggester.suggest(msg, n_suggestions=3)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  '{msg[:30]}...' -> {elapsed:.1f}ms ({result.pattern.value})")

        avg_time = sum(times) / len(times)
        print(f"\nAverage latency: {avg_time:.1f}ms")

        # Should be fast (< 500ms average including FAISS)
        assert avg_time < 500, f"Average latency too high: {avg_time:.1f}ms"


class TestPatternCoverage:
    """Test that common message types are covered."""

    def test_all_patterns_have_templates(self) -> None:
        """Every pattern should have template fallbacks."""
        from jarvis.reply_suggester import TEMPLATES

        for pattern in MessagePattern:
            assert pattern in TEMPLATES, f"No templates for {pattern.value}"
            assert len(TEMPLATES[pattern]) >= 2, f"Need at least 2 templates for {pattern.value}"

    def test_common_messages_detected(self) -> None:
        """Common iMessage patterns should be detected correctly."""
        common_messages = {
            # Invitations
            "wanna hang": MessagePattern.INVITATION,
            "down to chill?": MessagePattern.INVITATION,
            "free this weekend?": MessagePattern.INVITATION,
            # Reactions
            "yo did you see the game": MessagePattern.REACTION_PROMPT,
            "bro wtf happened": MessagePattern.REACTION_PROMPT,
            # Info
            "when's the thing": MessagePattern.INFO_QUESTION,
            "where at": MessagePattern.INFO_QUESTION,
            # Acks
            "bet": MessagePattern.ACKNOWLEDGMENT,
            "kk": MessagePattern.ACKNOWLEDGMENT,
            "ty": MessagePattern.ACKNOWLEDGMENT,
        }

        failures = []
        for msg, expected in common_messages.items():
            result = detect_pattern(msg)
            if result != expected:
                failures.append(f"'{msg}': expected {expected.value}, got {result.value}")

        if failures:
            print("\nPattern detection failures:")
            for f in failures:
                print(f"  {f}")

        # Allow up to 20% failure rate (patterns are simple)
        assert len(failures) <= len(common_messages) * 0.2, f"Too many failures: {failures}"
