"""Source attribution tracking for response grounding.

Tracks which parts of a response are grounded in source material,
enabling transparent attribution and hallucination identification.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AttributionType(str, Enum):
    """Types of source attribution."""

    DIRECT_QUOTE = "direct_quote"  # Verbatim from source
    PARAPHRASE = "paraphrase"  # Rephrased from source
    INFERENCE = "inference"  # Inferred from source
    UNGROUNDED = "ungrounded"  # Not traceable to source


class SourceType(str, Enum):
    """Types of sources."""

    MESSAGE = "message"  # Message from conversation
    CONTEXT = "context"  # General context provided
    TEMPLATE = "template"  # From response template
    KNOWLEDGE = "knowledge"  # General knowledge (unverifiable)


@dataclass
class Attribution:
    """Attribution of a response segment to a source."""

    # Response segment attributed
    response_segment: str
    # Source text it's attributed to
    source_text: str | None
    # Type of attribution
    attribution_type: AttributionType
    # Type of source
    source_type: SourceType = SourceType.CONTEXT
    # Confidence in the attribution (0-1)
    confidence: float = 1.0
    # Similarity score between segment and source
    similarity_score: float = 0.0
    # Position in response (start, end)
    response_span: tuple[int, int] | None = None
    # Position in source (start, end)
    source_span: tuple[int, int] | None = None


@dataclass
class GroundingResult:
    """Result of grounding analysis."""

    # Overall grounding score (0=ungrounded, 1=fully grounded)
    grounding_score: float
    # Percentage of response that is grounded
    grounded_percentage: float
    # Individual attributions
    attributions: list[Attribution] = field(default_factory=list)
    # Count by attribution type
    direct_quote_count: int = 0
    paraphrase_count: int = 0
    inference_count: int = 0
    ungrounded_count: int = 0
    # Whether response passes grounding gate
    passes_gate: bool = True
    # Latency in milliseconds
    latency_ms: float = 0.0
    # Issues found
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "grounding_score": round(self.grounding_score, 4),
            "grounded_percentage": round(self.grounded_percentage, 2),
            "direct_quote_count": self.direct_quote_count,
            "paraphrase_count": self.paraphrase_count,
            "inference_count": self.inference_count,
            "ungrounded_count": self.ungrounded_count,
            "passes_gate": self.passes_gate,
            "latency_ms": round(self.latency_ms, 2),
            "issues": self.issues,
            "attributions": [
                {
                    "segment": attr.response_segment[:100],  # Truncate
                    "source": (attr.source_text[:100] if attr.source_text else None),
                    "type": attr.attribution_type.value,
                    "source_type": attr.source_type.value,
                    "confidence": round(attr.confidence, 4),
                    "similarity": round(attr.similarity_score, 4),
                }
                for attr in self.attributions
            ],
        }


class SegmentExtractor:
    """Extracts meaningful segments from text for attribution."""

    def __init__(self, min_segment_words: int = 3) -> None:
        """Initialize the extractor.

        Args:
            min_segment_words: Minimum words for a valid segment
        """
        self._min_segment_words = min_segment_words

    def extract_segments(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Extract meaningful segments from text.

        Args:
            text: Text to extract segments from

        Returns:
            List of (segment_text, (start, end)) tuples
        """
        segments: list[tuple[str, tuple[int, int]]] = []

        # Split into sentences first
        sentences = self._split_sentences(text)

        current_pos = 0
        for sentence in sentences:
            # Find the sentence position in original text
            start = text.find(sentence, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sentence)

            # Check if segment is meaningful
            if len(sentence.split()) >= self._min_segment_words:
                segments.append((sentence, (start, end)))

            current_pos = end

        return segments

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


class DirectQuoteMatcher:
    """Matches direct quotes between response and source."""

    def __init__(self, min_match_words: int = 4) -> None:
        """Initialize the matcher.

        Args:
            min_match_words: Minimum words for a direct quote match
        """
        self._min_match_words = min_match_words

    def find_quotes(
        self,
        response: str,
        sources: list[str],
    ) -> list[Attribution]:
        """Find direct quotes from sources in response.

        Args:
            response: Response text
            sources: List of source texts

        Returns:
            List of Attribution objects for direct quotes
        """
        attributions: list[Attribution] = []

        # Normalize texts
        response_lower = response.lower()

        for source in sources:
            source_lower = source.lower()

            # Find longest common substrings
            matches = self._find_common_substrings(response_lower, source_lower)

            for match_text, r_start, s_start in matches:
                # Find original case text
                original_response_text = response[r_start : r_start + len(match_text)]
                original_source_text = source[s_start : s_start + len(match_text)]

                attributions.append(
                    Attribution(
                        response_segment=original_response_text,
                        source_text=original_source_text,
                        attribution_type=AttributionType.DIRECT_QUOTE,
                        source_type=SourceType.MESSAGE,
                        confidence=1.0,
                        similarity_score=1.0,
                        response_span=(r_start, r_start + len(match_text)),
                        source_span=(s_start, s_start + len(match_text)),
                    )
                )

        return attributions

    def _find_common_substrings(
        self,
        text1: str,
        text2: str,
    ) -> list[tuple[str, int, int]]:
        """Find common substrings between two texts.

        Returns list of (text, pos_in_text1, pos_in_text2).
        """
        matches: list[tuple[str, int, int]] = []

        words1 = text1.split()
        words2 = text2.split()

        # Build word position map for text2
        word_positions: dict[str, list[int]] = {}
        for i, word in enumerate(words2):
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)

        # Find matching sequences
        i = 0
        while i < len(words1):
            word = words1[i]
            if word in word_positions:
                for j in word_positions[word]:
                    # Try to extend the match
                    match_len = 1
                    while (
                        i + match_len < len(words1)
                        and j + match_len < len(words2)
                        and words1[i + match_len] == words2[j + match_len]
                    ):
                        match_len += 1

                    if match_len >= self._min_match_words:
                        match_text = " ".join(words1[i : i + match_len])
                        # Calculate positions in original texts
                        pos1 = len(" ".join(words1[:i])) + (1 if i > 0 else 0)
                        pos2 = len(" ".join(words2[:j])) + (1 if j > 0 else 0)
                        matches.append((match_text, pos1, pos2))
                        i += match_len - 1
                        break
            i += 1

        return matches


class SemanticMatcher:
    """Matches semantically similar segments between response and source."""

    # Thresholds for classification
    PARAPHRASE_THRESHOLD = 0.75
    INFERENCE_THRESHOLD = 0.5

    def __init__(self) -> None:
        """Initialize the semantic matcher."""
        self._embedder: object | None = None
        self._lock = threading.Lock()

    def _ensure_embedder(self) -> object | None:
        """Lazy load embedder."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    try:
                        from jarvis.embedding_adapter import get_embedder

                        self._embedder = get_embedder()
                    except Exception as e:
                        logger.warning("Failed to load embedder: %s", e)
        return self._embedder

    def match_segments(
        self,
        response_segments: list[tuple[str, tuple[int, int]]],
        sources: list[str],
        exclude_spans: list[tuple[int, int]] | None = None,
    ) -> list[Attribution]:
        """Match response segments to sources semantically.

        Args:
            response_segments: List of (segment, span) tuples
            sources: List of source texts
            exclude_spans: Spans to exclude (already attributed)

        Returns:
            List of Attribution objects
        """
        attributions: list[Attribution] = []
        exclude_spans = exclude_spans or []

        embedder = self._ensure_embedder()
        if embedder is None:
            # Fall back to keyword matching
            return self._keyword_match_segments(response_segments, sources, exclude_spans)

        # Extract source segments
        source_segments: list[str] = []
        for source in sources:
            sentences = re.split(r"(?<=[.!?])\s+", source)
            source_segments.extend([s.strip() for s in sentences if s.strip()])

        if not source_segments:
            return attributions

        try:
            # Encode all segments
            source_embeddings = embedder.encode(source_segments)

            for segment, span in response_segments:
                # Skip if already attributed
                if self._overlaps_with(span, exclude_spans):
                    continue

                # Get segment embedding
                segment_embedding = embedder.encode([segment])[0]

                # Find best matching source
                best_similarity = 0.0
                best_source: str | None = None

                for i, source_emb in enumerate(source_embeddings):
                    similarity = self._cosine_similarity(segment_embedding, source_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_source = source_segments[i]

                # Classify attribution type
                if best_similarity >= self.PARAPHRASE_THRESHOLD:
                    attr_type = AttributionType.PARAPHRASE
                elif best_similarity >= self.INFERENCE_THRESHOLD:
                    attr_type = AttributionType.INFERENCE
                else:
                    attr_type = AttributionType.UNGROUNDED

                attributions.append(
                    Attribution(
                        response_segment=segment,
                        source_text=best_source,
                        attribution_type=attr_type,
                        source_type=SourceType.MESSAGE if best_source else SourceType.KNOWLEDGE,
                        confidence=best_similarity,
                        similarity_score=best_similarity,
                        response_span=span,
                    )
                )

        except Exception as e:
            logger.warning("Semantic matching failed: %s", e)
            return self._keyword_match_segments(response_segments, sources, exclude_spans)

        return attributions

    def _keyword_match_segments(
        self,
        response_segments: list[tuple[str, tuple[int, int]]],
        sources: list[str],
        exclude_spans: list[tuple[int, int]],
    ) -> list[Attribution]:
        """Fallback keyword matching for segments."""
        attributions: list[Attribution] = []

        source_text = " ".join(sources).lower()
        source_words = set(source_text.split())

        for segment, span in response_segments:
            if self._overlaps_with(span, exclude_spans):
                continue

            segment_words = set(segment.lower().split())
            segment_words = {w for w in segment_words if len(w) > 3}

            if not segment_words:
                similarity = 0.0
            else:
                overlap = segment_words & source_words
                similarity = len(overlap) / len(segment_words)

            if similarity >= self.PARAPHRASE_THRESHOLD:
                attr_type = AttributionType.PARAPHRASE
            elif similarity >= self.INFERENCE_THRESHOLD:
                attr_type = AttributionType.INFERENCE
            else:
                attr_type = AttributionType.UNGROUNDED

            attributions.append(
                Attribution(
                    response_segment=segment,
                    source_text=None,
                    attribution_type=attr_type,
                    source_type=SourceType.KNOWLEDGE,
                    confidence=similarity,
                    similarity_score=similarity,
                    response_span=span,
                )
            )

        return attributions

    def _overlaps_with(
        self,
        span: tuple[int, int],
        exclude_spans: list[tuple[int, int]],
    ) -> bool:
        """Check if span overlaps with any excluded span."""
        for ex_start, ex_end in exclude_spans:
            if span[0] < ex_end and span[1] > ex_start:
                return True
        return False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)


class GroundingChecker:
    """Main grounding checker for source attribution.

    Combines direct quote matching and semantic matching to
    attribute response segments to sources.
    """

    GROUNDING_GATE_THRESHOLD = 0.6

    def __init__(self, gate_threshold: float | None = None) -> None:
        """Initialize the grounding checker.

        Args:
            gate_threshold: Minimum grounding score to pass gate
        """
        self._gate_threshold = gate_threshold or self.GROUNDING_GATE_THRESHOLD
        self._segment_extractor = SegmentExtractor()
        self._quote_matcher = DirectQuoteMatcher()
        self._semantic_matcher = SemanticMatcher()

    def check_grounding(
        self,
        response: str,
        sources: str | list[str],
    ) -> GroundingResult:
        """Check grounding of response in sources.

        Args:
            response: Response to check
            sources: Source text(s) to check against

        Returns:
            GroundingResult with attributions and scores
        """
        start_time = time.perf_counter()
        issues: list[str] = []

        # Normalize sources to list
        if isinstance(sources, str):
            source_list = [sources]
        else:
            source_list = list(sources)

        if not source_list or not response.strip():
            return GroundingResult(
                grounding_score=0.0,
                grounded_percentage=0.0,
                passes_gate=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                issues=["No sources provided" if not source_list else "Empty response"],
            )

        # Extract segments from response
        segments = self._segment_extractor.extract_segments(response)

        if not segments:
            return GroundingResult(
                grounding_score=1.0,  # No segments to attribute
                grounded_percentage=100.0,
                passes_gate=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Find direct quotes first
        quote_attributions = self._quote_matcher.find_quotes(response, source_list)

        # Get spans already covered by quotes
        quoted_spans = [attr.response_span for attr in quote_attributions if attr.response_span]

        # Semantic matching for remaining segments
        semantic_attributions = self._semantic_matcher.match_segments(
            segments, source_list, exclude_spans=quoted_spans
        )

        # Combine attributions
        all_attributions = quote_attributions + semantic_attributions

        # Count by type
        direct_quote_count = sum(
            1 for a in all_attributions if a.attribution_type == AttributionType.DIRECT_QUOTE
        )
        paraphrase_count = sum(
            1 for a in all_attributions if a.attribution_type == AttributionType.PARAPHRASE
        )
        inference_count = sum(
            1 for a in all_attributions if a.attribution_type == AttributionType.INFERENCE
        )
        ungrounded_count = sum(
            1 for a in all_attributions if a.attribution_type == AttributionType.UNGROUNDED
        )

        # Calculate grounding score
        total_segments = len(all_attributions)
        if total_segments == 0:
            grounding_score = 1.0
            grounded_percentage = 100.0
        else:
            # Weight: direct quotes = 1.0, paraphrase = 0.9, inference = 0.6, ungrounded = 0.0
            weighted_sum = (
                direct_quote_count * 1.0
                + paraphrase_count * 0.9
                + inference_count * 0.6
                + ungrounded_count * 0.0
            )
            grounding_score = weighted_sum / total_segments
            grounded_count = direct_quote_count + paraphrase_count + inference_count
            grounded_percentage = (grounded_count / total_segments) * 100

        passes_gate = grounding_score >= self._gate_threshold

        if ungrounded_count > 0:
            issues.append(f"{ungrounded_count} ungrounded segment(s) detected")

        latency_ms = (time.perf_counter() - start_time) * 1000

        return GroundingResult(
            grounding_score=grounding_score,
            grounded_percentage=grounded_percentage,
            attributions=all_attributions,
            direct_quote_count=direct_quote_count,
            paraphrase_count=paraphrase_count,
            inference_count=inference_count,
            ungrounded_count=ungrounded_count,
            passes_gate=passes_gate,
            latency_ms=latency_ms,
            issues=issues,
        )


# Global singleton
_grounding_checker: GroundingChecker | None = None
_checker_lock = threading.Lock()


def get_grounding_checker(gate_threshold: float = 0.6) -> GroundingChecker:
    """Get the global grounding checker instance.

    Args:
        gate_threshold: Minimum grounding score to pass gate

    Returns:
        Shared GroundingChecker instance
    """
    global _grounding_checker
    if _grounding_checker is None:
        with _checker_lock:
            if _grounding_checker is None:
                _grounding_checker = GroundingChecker(gate_threshold=gate_threshold)
    return _grounding_checker


def reset_grounding_checker() -> None:
    """Reset the global grounding checker instance."""
    global _grounding_checker
    with _checker_lock:
        _grounding_checker = None
