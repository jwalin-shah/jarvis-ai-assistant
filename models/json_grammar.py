"""Grammar-constrained JSON generation for MLX models.

Implements PicoLM-style token masking to guarantee valid JSON output.
Pre-analyzes vocabulary tokens for JSON structural characters, then
masks invalid tokens during generation based on current parser state.

This ensures syntactically valid JSON even from small models (0.7B-1B)
that might otherwise produce malformed output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class TokenJSONInfo:
    """Pre-computed JSON properties for a single token."""

    opens_brace: bool = False
    closes_brace: bool = False
    opens_bracket: bool = False
    closes_bracket: bool = False
    has_quote: bool = False
    has_escape: bool = False
    has_colon: bool = False
    has_comma: bool = False
    is_whitespace: bool = False
    is_digit: bool = False
    is_alpha: bool = False
    raw_text: str = ""


@dataclass
class JSONParserState:
    """Tracks current state of JSON parsing during generation."""

    brace_depth: int = 0
    bracket_depth: int = 0
    in_string: bool = False
    escape_next: bool = False
    expect_key: bool = True
    expect_value: bool = False
    expect_colon: bool = False
    expect_comma: bool = False
    after_value: bool = False


class JSONGrammarProcessor:
    """Logits processor that enforces valid JSON structure.

    Pre-analyzes all tokens in vocabulary at initialization time,
    tracking which tokens contain JSON structural characters.

    During generation, maintains parser state and masks any token
    that would produce invalid JSON.

    Example usage:
        processor = JSONGrammarProcessor(tokenizer)
        for token in generate(..., logits_processors=[processor]):
            # Guaranteed valid JSON tokens
    """

    NEG_INF = -1e9

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self._token_info: list[TokenJSONInfo] = []
        self._state = JSONParserState()
        self._token_count = 0

        self._analyze_vocabulary()
        logger.info(
            "JSONGrammarProcessor initialized: %d tokens analyzed",
            len(self._token_info),
        )

    def _analyze_vocabulary(self) -> None:
        """Pre-analyze all tokens for JSON structural characters."""
        for i in range(self.vocab_size):
            try:
                text = self.tokenizer.decode([i])
            except Exception:
                text = ""

            info = TokenJSONInfo(raw_text=text)

            if not text:
                self._token_info.append(info)
                continue

            info.opens_brace = "{" in text
            info.closes_brace = "}" in text
            info.opens_bracket = "[" in text
            info.closes_bracket = "]" in text
            info.has_quote = '"' in text
            info.has_escape = "\\" in text
            info.has_colon = ":" in text
            info.has_comma = "," in text
            info.is_whitespace = all(c in " \t\n\r" for c in text)
            info.is_digit = all(c in "0123456789.-+eE" for c in text) and any(
                c.isdigit() for c in text
            )
            info.is_alpha = text.lower() in ("true", "false", "null")

            self._token_info.append(info)

    def reset(self) -> None:
        """Reset parser state for a new generation."""
        self._state = JSONParserState()
        self._token_count = 0

    def _is_valid_token(self, token_id: int) -> bool:
        """Check if token is valid given current parser state."""
        if token_id >= len(self._token_info):
            return True

        info = self._token_info[token_id]
        state = self._state
        text = info.raw_text

        if not text:
            return True

        if state.escape_next:
            return True

        if state.in_string:
            if info.has_escape:
                return True
            if info.has_quote:
                return True
            return True

        if info.has_quote:
            return True

        if state.brace_depth == 0 and state.bracket_depth == 0:
            if not state.after_value:
                if info.opens_brace or info.opens_bracket:
                    return True
                if info.is_whitespace:
                    return True
                if info.has_quote:
                    return True
            return False

        if info.is_whitespace:
            return True

        if state.expect_key:
            return info.has_quote

        if state.expect_colon:
            return info.has_colon

        if state.expect_value:
            if info.has_quote:
                return True
            if info.opens_brace:
                return True
            if info.opens_bracket:
                return True
            if info.is_digit:
                return True
            if info.is_alpha:
                return True
            return False

        if state.expect_comma:
            if info.has_comma:
                return True
            if info.closes_brace and state.brace_depth > 0:
                return True
            if info.closes_bracket and state.bracket_depth > 0:
                return True
            return False

        if state.after_value:
            if info.has_comma:
                return True
            if info.closes_brace and state.brace_depth > 0:
                return True
            if info.closes_bracket and state.bracket_depth > 0:
                return True
            return False

        return True

    def _update_state(self, token_id: int) -> None:
        """Update parser state after accepting a token."""
        if token_id >= len(self._token_info):
            return

        info = self._token_info[token_id]
        state = self._state

        if state.escape_next:
            state.escape_next = False
            return

        if info.has_escape and state.in_string:
            state.escape_next = True
            return

        if state.in_string:
            if info.has_quote:
                state.in_string = False
                state.after_value = True
                state.expect_comma = True
            return

        if info.has_quote:
            state.in_string = True
            if state.expect_value or state.brace_depth == 0:
                state.expect_value = False
            elif state.expect_key:
                state.expect_key = False
                state.expect_colon = True
            return

        if info.opens_brace:
            state.brace_depth += 1
            state.expect_key = True
            state.expect_value = False
            state.expect_colon = False
            state.expect_comma = False
            state.after_value = False
            return

        if info.closes_brace:
            state.brace_depth = max(0, state.brace_depth - 1)
            state.after_value = True
            state.expect_key = False
            state.expect_value = False
            state.expect_colon = False
            state.expect_comma = True
            return

        if info.opens_bracket:
            state.bracket_depth += 1
            state.expect_value = True
            state.expect_key = False
            state.expect_colon = False
            state.expect_comma = False
            state.after_value = False
            return

        if info.closes_bracket:
            state.bracket_depth = max(0, state.bracket_depth - 1)
            state.after_value = True
            state.expect_key = False
            state.expect_value = False
            state.expect_colon = False
            state.expect_comma = True
            return

        if info.has_colon:
            state.expect_colon = False
            state.expect_value = True
            return

        if info.has_comma:
            state.expect_comma = False
            state.after_value = False
            if state.brace_depth > 0:
                state.expect_key = True
            else:
                state.expect_value = True
            return

        if info.is_digit or info.is_alpha:
            state.expect_value = False
            state.after_value = True
            state.expect_comma = True

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Mask invalid tokens based on current JSON parser state.

        Args:
            input_ids: Token IDs generated so far (unused, state tracked internally).
            logits: Raw logits from model, shape [batch_size, vocab_size].

        Returns:
            Modified logits with invalid tokens masked to -inf.
        """
        if self._token_count == 0:
            self.reset()

        valid_mask = mx.ones(self.vocab_size, dtype=mx.bool_)

        for token_id in range(min(self.vocab_size, len(self._token_info))):
            if not self._is_valid_token(token_id):
                valid_mask = mx.where(
                    mx.arange(self.vocab_size) == token_id,
                    mx.array(False),
                    valid_mask,
                )

        logits = mx.where(valid_mask, logits, mx.array(self.NEG_INF))

        if input_ids is not None and len(input_ids) > 0:
            last_token = int(input_ids[-1]) if len(input_ids.shape) == 1 else int(input_ids[0, -1])
            self._update_state(last_token)
            self._token_count += 1

        return logits

    def get_allowed_tokens(self) -> list[int]:
        """Get list of currently allowed token IDs (for debugging)."""
        return [
            i for i in range(min(self.vocab_size, len(self._token_info))) if self._is_valid_token(i)
        ]


class SimpleJSONGrammarProcessor:
    """Simplified JSON grammar that just tracks brace/bracket/quote balance.

    Less strict than full JSONGrammarProcessor but simpler and faster.
    Guarantees balanced braces, brackets, and quotes.
    """

    NEG_INF = -1e9

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self._opens_brace: set[int] = set()
        self._closes_brace: set[int] = set()
        self._opens_bracket: set[int] = set()
        self._closes_bracket: set[int] = set()
        self._has_quote: set[int] = set()
        self._has_escape: set[int] = set()

        self._analyze_vocabulary()

        self.brace_depth = 0
        self.bracket_depth = 0
        self.in_string = False
        self.escape_next = False

        logger.info(
            "SimpleJSONGrammarProcessor: braces=%d/%d, brackets=%d/%d, quotes=%d",
            len(self._opens_brace),
            len(self._closes_brace),
            len(self._opens_bracket),
            len(self._closes_bracket),
            len(self._has_quote),
        )

    def _analyze_vocabulary(self) -> None:
        """Pre-compute which tokens contain JSON structural chars."""
        for i in range(self.vocab_size):
            try:
                text = self.tokenizer.decode([i])
            except Exception:
                continue

            if "{" in text:
                self._opens_brace.add(i)
            if "}" in text:
                self._closes_brace.add(i)
            if "[" in text:
                self._opens_bracket.add(i)
            if "]" in text:
                self._closes_bracket.add(i)
            if '"' in text:
                self._has_quote.add(i)
            if "\\" in text:
                self._has_escape.add(i)

    def reset(self) -> None:
        """Reset state for new generation."""
        self.brace_depth = 0
        self.bracket_depth = 0
        self.in_string = False
        self.escape_next = False

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Mask tokens that would break JSON balance."""
        if self.escape_next:
            self.escape_next = False
            return logits

        invalid_tokens: set[int] = set()

        if not self.in_string:
            if self.brace_depth == 0 and self.bracket_depth == 0:
                for i in range(self.vocab_size):
                    if (
                        i not in self._opens_brace
                        and i not in self._opens_bracket
                        and i not in self._has_quote
                    ):
                        invalid_tokens.add(i)

            if self.brace_depth == 0:
                invalid_tokens.update(self._closes_brace)
            if self.bracket_depth == 0:
                invalid_tokens.update(self._closes_bracket)

        for token_id in invalid_tokens:
            if token_id < logits.shape[-1]:
                logits[:, token_id] = self.NEG_INF

        if input_ids is not None and len(input_ids) > 0:
            last_token = int(input_ids[-1]) if len(input_ids.shape) == 1 else int(input_ids[0, -1])
            self._update_state(last_token)

        return logits

    def _update_state(self, token_id: int) -> None:
        """Update balance state after accepting token."""
        if token_id in self._has_escape and self.in_string:
            self.escape_next = True
            return

        if self.in_string:
            if token_id in self._has_quote:
                self.in_string = False
            return

        if token_id in self._has_quote:
            self.in_string = True
        elif token_id in self._opens_brace:
            self.brace_depth += 1
        elif token_id in self._closes_brace:
            self.brace_depth = max(0, self.brace_depth - 1)
        elif token_id in self._opens_bracket:
            self.bracket_depth += 1
        elif token_id in self._closes_bracket:
            self.bracket_depth = max(0, self.bracket_depth - 1)
