"""Fact verification against message history.

Provides claim extraction and verification against conversation context,
ensuring responses are factually grounded in the actual message history.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ClaimType(str, Enum):
    """Types of claims that can be extracted from text."""

    FACTUAL = "factual"  # Statement of fact
    TEMPORAL = "temporal"  # Time-related claim
    QUANTITATIVE = "quantitative"  # Numeric claim
    ATTRIBUTION = "attribution"  # Claim about who said/did something
    RELATIONAL = "relational"  # Relationship between entities


class VerificationStatus(str, Enum):
    """Verification status for a claim."""

    VERIFIED = "verified"  # Claim verified against source
    REFUTED = "refuted"  # Claim contradicted by source
    UNVERIFIABLE = "unverifiable"  # Cannot verify from available context
    PARTIALLY_VERIFIED = "partially_verified"  # Some aspects verified


@dataclass
class Claim:
    """A claim extracted from a response."""

    text: str
    claim_type: ClaimType
    confidence: float = 1.0  # Confidence in extraction
    entities: list[str] = field(default_factory=list)  # Entities mentioned
    source_span: tuple[int, int] | None = None  # Position in original text


@dataclass
class VerifiedClaim:
    """A claim with its verification result."""

    claim: Claim
    status: VerificationStatus
    evidence: str | None = None  # Supporting/refuting evidence from source
    similarity_score: float = 0.0  # Similarity to closest source statement
    verification_confidence: float = 1.0


@dataclass
class FactualityResult:
    """Result of factuality checking."""

    # Overall factuality score (0=all false, 1=all true)
    factuality_score: float
    # Number of claims by verification status
    verified_count: int = 0
    refuted_count: int = 0
    unverifiable_count: int = 0
    partially_verified_count: int = 0
    # Individual claim results
    claims: list[VerifiedClaim] = field(default_factory=list)
    # Whether response passes factuality gate
    passes_gate: bool = True
    # Latency in milliseconds
    latency_ms: float = 0.0
    # Issues found
    issues: list[str] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        """Total number of claims extracted."""
        return len(self.claims)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "factuality_score": round(self.factuality_score, 4),
            "total_claims": self.total_claims,
            "verified_count": self.verified_count,
            "refuted_count": self.refuted_count,
            "unverifiable_count": self.unverifiable_count,
            "partially_verified_count": self.partially_verified_count,
            "passes_gate": self.passes_gate,
            "latency_ms": round(self.latency_ms, 2),
            "issues": self.issues,
            "claims": [
                {
                    "text": vc.claim.text,
                    "type": vc.claim.claim_type.value,
                    "status": vc.status.value,
                    "evidence": vc.evidence,
                    "similarity_score": round(vc.similarity_score, 4),
                }
                for vc in self.claims
            ],
        }


class ClaimExtractor:
    """Extracts verifiable claims from text.

    Uses pattern matching and NLP techniques to identify claims
    that can be verified against source context.
    """

    # Patterns for different claim types
    TEMPORAL_PATTERNS = [
        r"(?:on|at|in|during)\s+(\w+\s+\d+|\d+:\d+|\d{4})",
        r"(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+)",
        r"(?:\d+\s+(?:days?|weeks?|months?|years?)\s+ago)",
    ]

    QUANTITATIVE_PATTERNS = [
        r"\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%|percent)?",
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+\w+s?",
    ]

    ATTRIBUTION_PATTERNS = [
        r"(?:\w+)\s+(?:said|told|mentioned|asked|replied|wrote|texted)",
        r"according to\s+(\w+)",
    ]

    def __init__(self, min_claim_words: int = 3) -> None:
        """Initialize the claim extractor.

        Args:
            min_claim_words: Minimum words for a valid claim
        """
        self._min_claim_words = min_claim_words

    def extract_claims(self, text: str) -> list[Claim]:
        """Extract verifiable claims from text.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted Claim objects
        """
        claims: list[Claim] = []

        # Split into sentences
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < self._min_claim_words:
                continue

            # Determine claim type
            claim_type = self._classify_claim_type(sentence)

            # Extract entities
            entities = self._extract_entities(sentence)

            claims.append(
                Claim(
                    text=sentence,
                    claim_type=claim_type,
                    confidence=0.8,  # Default confidence
                    entities=entities,
                )
            )

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _classify_claim_type(self, text: str) -> ClaimType:
        """Classify the type of claim."""
        text_lower = text.lower()

        # Check for temporal patterns
        for pattern in self.TEMPORAL_PATTERNS:
            if re.search(pattern, text_lower):
                return ClaimType.TEMPORAL

        # Check for quantitative patterns
        for pattern in self.QUANTITATIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return ClaimType.QUANTITATIVE

        # Check for attribution patterns
        for pattern in self.ATTRIBUTION_PATTERNS:
            if re.search(pattern, text_lower):
                return ClaimType.ATTRIBUTION

        # Check for relational patterns (contains relationship words)
        relational_words = {"related", "connected", "between", "with", "and"}
        if any(word in text_lower for word in relational_words):
            return ClaimType.RELATIONAL

        # Default to factual
        return ClaimType.FACTUAL

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities: list[str] = []

        # Extract capitalized words (potential proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            # Skip first word (may be capitalized just for sentence start)
            if i == 0:
                continue
            # Clean and check if capitalized
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word[0].isupper():
                entities.append(clean_word)

        # Extract quoted text as entities
        quotes = re.findall(r'"([^"]*)"', text)
        entities.extend(quotes)

        return list(set(entities))


class FactChecker:
    """Verifies claims against source context.

    Checks extracted claims against the conversation history
    and message context to ensure factual accuracy.
    """

    # Threshold for similarity to consider a match
    SIMILARITY_THRESHOLD = 0.6
    # Threshold for factuality gate
    FACTUALITY_GATE_THRESHOLD = 0.7

    def __init__(
        self,
        gate_threshold: float | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize the fact checker.

        Args:
            gate_threshold: Minimum factuality score to pass gate
            similarity_threshold: Minimum similarity for verification
        """
        self._gate_threshold = gate_threshold or self.FACTUALITY_GATE_THRESHOLD
        self._similarity_threshold = similarity_threshold or self.SIMILARITY_THRESHOLD
        self._extractor = ClaimExtractor()
        self._embedder: object | None = None
        self._lock = threading.Lock()

    def _ensure_embedder(self) -> object | None:
        """Lazy load embedder."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    try:
                        from jarvis.embeddings import get_embedder

                        self._embedder = get_embedder()
                    except Exception as e:
                        logger.warning("Failed to load embedder for fact checking: %s", e)
        return self._embedder

    def check_factuality(
        self,
        response: str,
        context: str | list[str],
    ) -> FactualityResult:
        """Check factuality of response against context.

        Args:
            response: Generated response to check
            context: Source context (string or list of messages)

        Returns:
            FactualityResult with verification status for each claim
        """
        start_time = time.perf_counter()
        issues: list[str] = []

        # Normalize context to list of strings
        if isinstance(context, str):
            context_list = [context]
        else:
            context_list = list(context)

        # Extract claims from response
        claims = self._extractor.extract_claims(response)

        if not claims:
            # No claims to verify = perfectly factual
            return FactualityResult(
                factuality_score=1.0,
                passes_gate=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Verify each claim
        verified_claims: list[VerifiedClaim] = []
        verified_count = 0
        refuted_count = 0
        unverifiable_count = 0
        partially_verified_count = 0

        for claim in claims:
            verified_claim = self._verify_claim(claim, context_list)
            verified_claims.append(verified_claim)

            if verified_claim.status == VerificationStatus.VERIFIED:
                verified_count += 1
            elif verified_claim.status == VerificationStatus.REFUTED:
                refuted_count += 1
                issues.append(f"Refuted claim: {claim.text[:50]}...")
            elif verified_claim.status == VerificationStatus.UNVERIFIABLE:
                unverifiable_count += 1
            else:
                partially_verified_count += 1

        # Calculate factuality score
        # Verified = 1.0, Partially = 0.5, Unverifiable = 0.5, Refuted = 0.0
        total_claims = len(claims)
        score_sum = (
            verified_count * 1.0
            + partially_verified_count * 0.5
            + unverifiable_count * 0.5  # Neutral for unverifiable
            + refuted_count * 0.0
        )
        factuality_score = score_sum / total_claims if total_claims > 0 else 1.0

        passes_gate = factuality_score >= self._gate_threshold

        if not passes_gate:
            issues.append(
                f"Factuality score {factuality_score:.2f} below threshold {self._gate_threshold}"
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return FactualityResult(
            factuality_score=factuality_score,
            verified_count=verified_count,
            refuted_count=refuted_count,
            unverifiable_count=unverifiable_count,
            partially_verified_count=partially_verified_count,
            claims=verified_claims,
            passes_gate=passes_gate,
            latency_ms=latency_ms,
            issues=issues,
        )

    def _verify_claim(
        self,
        claim: Claim,
        context_list: list[str],
    ) -> VerifiedClaim:
        """Verify a single claim against context.

        Args:
            claim: Claim to verify
            context_list: List of context strings

        Returns:
            VerifiedClaim with verification result
        """
        # Try semantic similarity first
        embedder = self._ensure_embedder()
        best_similarity = 0.0
        best_evidence: str | None = None

        if embedder is not None:
            try:
                # Get claim embedding
                claim_embedding = embedder.encode([claim.text])[0]

                # Compare against each context item
                for ctx in context_list:
                    ctx_embedding = embedder.encode([ctx])[0]

                    # Cosine similarity
                    import numpy as np

                    dot_product = np.dot(claim_embedding, ctx_embedding)
                    norm_product = np.linalg.norm(claim_embedding) * np.linalg.norm(ctx_embedding)

                    if norm_product > 0:
                        similarity = float(dot_product / norm_product)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_evidence = ctx[:200]  # Truncate evidence
            except Exception as e:
                logger.warning("Embedding similarity check failed: %s", e)

        # Fallback to keyword matching
        if best_similarity < self._similarity_threshold:
            keyword_score = self._keyword_match_score(claim, context_list)
            if keyword_score > best_similarity:
                best_similarity = keyword_score
                # Find best matching context by keywords
                best_evidence = self._find_best_keyword_match(claim, context_list)

        # Determine verification status
        if best_similarity >= 0.8:
            status = VerificationStatus.VERIFIED
        elif best_similarity >= self._similarity_threshold:
            status = VerificationStatus.PARTIALLY_VERIFIED
        elif best_similarity >= 0.3:
            status = VerificationStatus.UNVERIFIABLE
        else:
            # Check for explicit contradictions
            if self._check_contradiction(claim, context_list):
                status = VerificationStatus.REFUTED
            else:
                status = VerificationStatus.UNVERIFIABLE

        return VerifiedClaim(
            claim=claim,
            status=status,
            evidence=best_evidence,
            similarity_score=best_similarity,
            verification_confidence=min(1.0, best_similarity + 0.2),
        )

    def _keyword_match_score(self, claim: Claim, context_list: list[str]) -> float:
        """Compute keyword match score between claim and context.

        Args:
            claim: Claim to match
            context_list: Context strings to match against

        Returns:
            Score from 0 to 1
        """
        claim_words = set(claim.text.lower().split())
        claim_words = {w for w in claim_words if len(w) > 3}  # Filter short words

        if not claim_words:
            return 0.0

        best_score = 0.0
        for ctx in context_list:
            ctx_words = set(ctx.lower().split())
            overlap = claim_words & ctx_words
            score = len(overlap) / len(claim_words)
            best_score = max(best_score, score)

        return best_score

    def _find_best_keyword_match(self, claim: Claim, context_list: list[str]) -> str | None:
        """Find the context string that best matches the claim by keywords."""
        claim_words = set(claim.text.lower().split())
        best_ctx: str | None = None
        best_overlap = 0

        for ctx in context_list:
            ctx_words = set(ctx.lower().split())
            overlap = len(claim_words & ctx_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ctx = ctx[:200]

        return best_ctx

    def _check_contradiction(self, claim: Claim, context_list: list[str]) -> bool:
        """Check if claim is explicitly contradicted by context.

        Args:
            claim: Claim to check
            context_list: Context strings

        Returns:
            True if contradiction detected
        """
        # Simple negation check
        claim_lower = claim.text.lower()

        negation_patterns = [
            (r"(\w+)\s+is\s+(\w+)", r"\1 is not \2"),
            (r"(\w+)\s+was\s+(\w+)", r"\1 was not \2"),
            (r"(\w+)\s+did\s+(\w+)", r"\1 did not \2"),
        ]

        for ctx in context_list:
            ctx_lower = ctx.lower()

            # Check for explicit negations
            for pattern, negated in negation_patterns:
                claim_match = re.search(pattern, claim_lower)
                if claim_match:
                    negated_version = re.sub(pattern, negated, claim_lower)
                    if negated_version in ctx_lower:
                        return True

            # Check for conflicting numbers
            claim_numbers = re.findall(r"\d+", claim_lower)
            ctx_numbers = re.findall(r"\d+", ctx_lower)
            if claim_numbers and ctx_numbers:
                # If same context has different numbers for same pattern
                for num in claim_numbers:
                    for ctx_num in ctx_numbers:
                        if num != ctx_num and abs(int(num) - int(ctx_num)) > 0:
                            # Check if in similar context
                            claim_context = claim_lower.split(num)[0][-20:]
                            if claim_context in ctx_lower:
                                return True

        return False


# Global singleton
_fact_checker: FactChecker | None = None
_fact_checker_lock = threading.Lock()


def get_fact_checker(gate_threshold: float = 0.7) -> FactChecker:
    """Get the global fact checker instance.

    Args:
        gate_threshold: Minimum factuality score to pass gate

    Returns:
        Shared FactChecker instance
    """
    global _fact_checker
    if _fact_checker is None:
        with _fact_checker_lock:
            if _fact_checker is None:
                _fact_checker = FactChecker(gate_threshold=gate_threshold)
    return _fact_checker


def reset_fact_checker() -> None:
    """Reset the global fact checker instance."""
    global _fact_checker
    with _fact_checker_lock:
        _fact_checker = None
