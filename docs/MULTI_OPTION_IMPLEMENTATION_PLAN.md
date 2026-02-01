# Multi-Option Response Implementation Plan

**Date**: 2026-01-31
**Status**: Ready for implementation
**Estimated Effort**: 2-3 days

---

## Executive Summary

This plan implements the "Updated Combined Architecture" from FROM_SCRATCH_PLAN.md:
- Multi-option responses (3 options: AGREE/DECLINE/DEFER) for commitment questions
- DA-filtered FAISS retrieval for type-specific examples
- Diversity enforcement (Smart Reply approach)

---

## Open Questions Resolved

### Question 1: Memory Budget ✅ RESOLVED
> "Does FAISS + embedder + LLM fit in 8GB?"

**Answer**: YES - Measured usage is only **0.10 GB** total.
- Memory before loading: 0.02 GB
- After embedder: 0.09 GB (+0.07 GB)
- After FAISS: 0.09 GB (+0.00 GB)
- After LLM: 0.10 GB (+0.01 GB)
- **Headroom**: 7.90 GB remaining

This is well under budget because:
- MLX models use lazy loading / memory mapping
- FAISS index is relatively small (146 MB)
- bge-small embeddings are 384-dim (not 768 or 1024)

### Question 2: Incremental Updates ✅ DESIGN PROVIDED
> "Best strategy for adding new pairs?"

**Recommended Strategy**: Append-only with periodic rebuild

```python
# Incremental approach (for real-time):
def add_new_pair(trigger, response, contact_id):
    # 1. Add to database immediately
    db.upsert_pair(trigger, response, contact_id)

    # 2. Embed and add to in-memory index
    embedding = embedder.encode(trigger)
    index.add_with_ids([embedding], [pair_id])

    # 3. Mark index as "dirty" for eventual rebuild
    db.mark_index_stale()

# Periodic rebuild (nightly/weekly):
def rebuild_if_stale():
    if db.is_index_stale() and db.stale_count() > 1000:
        build_index_from_db()
        db.mark_index_fresh()
```

For V1: Skip real-time updates, rebuild index on command via `jarvis db build-index`.

### Question 3: Desktop App Needs ✅ AUDITED
> "What endpoints does Tauri app actually require?"

**Audit Results**: Desktop app uses ~40 endpoints from `client.ts`, but core functionality needs only:

| Endpoint | Priority | Used For |
|----------|----------|----------|
| `GET /health` | CRITICAL | Connection status |
| `GET /conversations` | CRITICAL | Conversation list |
| `GET /conversations/{id}/messages` | CRITICAL | Message view |
| `POST /drafts/reply` | CRITICAL | Generate replies |
| `GET /settings` | HIGH | Settings display |
| `PUT /settings` | HIGH | Settings update |
| `GET /settings/models` | HIGH | Model selection |
| `POST /suggestions` | HIGH | Smart reply chips |

**The 35 API routers are feature creep.** For V1 multi-option, only modify `/drafts/reply` and `/suggestions`.

### Question 4: STATEMENT Problem ✅ IDENTIFIED SOLUTION
> "78% of responses are STATEMENT - need better exemplars"

**Measured Distribution**:
```
RESPONSE TYPES:
  STATEMENT: 82,061 (78%)
  ACKNOWLEDGE: 6,501 (6%)
  QUESTION: 6,011 (6%)
  REACT_POSITIVE: 5,297 (5%)
  AGREE: 2,662 (2.5%)
  GREETING: 1,183 (1%)
  DECLINE: 1,146 (1%)
  ANSWER: 417 (<1%)
  DEFER: 362 (<1%)
```

**Solution: Mine Non-STATEMENT Clusters**

Found 20 clusters with 50+ non-STATEMENT responses:

| Cluster | DA Type | Count | Use For |
|---------|---------|-------|---------|
| 7 | AGREE | 180 | Agreement exemplars |
| 74 | AGREE | 127 | Agreement exemplars |
| 9 | AGREE | 124 | Agreement exemplars |
| 213 | ANSWER | 204 | Answer exemplars |
| 160 | REACT_POSITIVE | 178 | Positive reaction exemplars |
| 32 | REACT_POSITIVE | 164 | Positive reaction exemplars |
| 53 | GREETING | 240 | Greeting exemplars |
| 80 | QUESTION | 118 | Follow-up question exemplars |

**Implementation**: Create `get_exemplars_by_type()` that prioritizes these clusters.

### Question 5: Cross-Encoder for V2? ✅ DEFERRED (CORRECT)
> "Research suggests +20-40% accuracy, but latency concerns"

**Decision**: Keep deferred. Current retrieval is fast enough (12k queries/sec).
Revisit only if multi-option accuracy is insufficient.

### Question 6: User Preference Learning ✅ DESIGN PROVIDED
> "How to learn from which option user picks over time?"

**Implementation in `api/routers/feedback.py`** (already exists):

```python
# When user picks an option:
POST /feedback/response {
    "action": "accepted",  # or "rejected", "edited"
    "suggestion_text": "Yeah I'm down!",
    "chat_id": "...",
    "context_messages": [...],
    "metadata": {
        "response_type": "AGREE",  # Track which type was picked
        "other_options": ["DECLINE", "DEFER"],
        "trigger_da": "INVITATION"
    }
}
```

Over time, learn: "For INVITATION from {contact}, user prefers AGREE 70% of time"

### Question 7: Topic Cluster Granularity ✅ ANSWERED
> "240 clusters vs fewer broader topics - what's optimal?"

**Answer**: Keep 240 clusters, but use them for **filtering, not classification**.

Current clusters find topics (food, games, scheduling) not response types.
Use DA classifier for response types, clusters for topic-aware example filtering.

---

## Implementation Tasks

### Task 1: Create `jarvis/response_classifier.py`

**File**: `jarvis/response_classifier.py`
**Lines**: ~200
**Dependencies**: Existing DA classifiers in `~/.jarvis/da_classifiers/`

```python
"""Unified response classifier combining DA classification and structural features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Response functional types."""
    AGREE = "AGREE"
    DECLINE = "DECLINE"
    DEFER = "DEFER"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    STATEMENT = "STATEMENT"
    REACT_POSITIVE = "REACT_POSITIVE"
    REACT_SYMPATHY = "REACT_SYMPATHY"
    GREETING = "GREETING"


class TriggerType(str, Enum):
    """Trigger functional types."""
    INVITATION = "INVITATION"
    YN_QUESTION = "YN_QUESTION"
    WH_QUESTION = "WH_QUESTION"
    INFO_STATEMENT = "INFO_STATEMENT"
    OPINION = "OPINION"
    REQUEST = "REQUEST"
    GOOD_NEWS = "GOOD_NEWS"
    BAD_NEWS = "BAD_NEWS"
    GREETING = "GREETING"
    ACKNOWLEDGE = "ACKNOWLEDGE"


# Mapping from trigger type to valid response types
TRIGGER_TO_VALID_RESPONSES: dict[TriggerType, list[ResponseType]] = {
    TriggerType.INVITATION: [ResponseType.AGREE, ResponseType.DECLINE, ResponseType.DEFER, ResponseType.QUESTION],
    TriggerType.YN_QUESTION: [ResponseType.AGREE, ResponseType.DECLINE, ResponseType.DEFER, ResponseType.ANSWER],
    TriggerType.WH_QUESTION: [ResponseType.ANSWER, ResponseType.QUESTION, ResponseType.DEFER],
    TriggerType.REQUEST: [ResponseType.AGREE, ResponseType.DECLINE, ResponseType.QUESTION],
    TriggerType.OPINION: [ResponseType.AGREE, ResponseType.STATEMENT, ResponseType.QUESTION],
    TriggerType.GOOD_NEWS: [ResponseType.REACT_POSITIVE, ResponseType.QUESTION],
    TriggerType.BAD_NEWS: [ResponseType.REACT_SYMPATHY, ResponseType.QUESTION],
    TriggerType.GREETING: [ResponseType.GREETING, ResponseType.QUESTION],
    TriggerType.INFO_STATEMENT: [ResponseType.ACKNOWLEDGE, ResponseType.QUESTION, ResponseType.STATEMENT],
    TriggerType.ACKNOWLEDGE: [ResponseType.ACKNOWLEDGE, ResponseType.STATEMENT],
}


@dataclass
class ClassificationResult:
    """Result of classifying a message."""
    type: str
    confidence: float
    method: str  # 'da_classifier', 'structural', 'combined'
    da_type: str | None = None
    da_confidence: float | None = None
    structural_type: str | None = None


class UnifiedResponseClassifier:
    """Combines DA classification with structural feature detection."""

    def __init__(self, da_classifier_path: Path | None = None):
        self._da_index: faiss.Index | None = None
        self._da_labels: list[str] = []
        self._da_path = da_classifier_path or Path.home() / ".jarvis" / "da_classifiers" / "response"
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from models.embeddings import get_mlx_embedder
            self._embedder = get_mlx_embedder()
        return self._embedder

    def _load_da_classifier(self) -> bool:
        """Load DA classifier index if available."""
        index_path = self._da_path / "index.faiss"
        labels_path = self._da_path / "labels.npy"

        if not index_path.exists():
            logger.warning("DA classifier index not found at %s", index_path)
            return False

        try:
            self._da_index = faiss.read_index(str(index_path))
            if labels_path.exists():
                self._da_labels = list(np.load(str(labels_path), allow_pickle=True))
            return True
        except Exception as e:
            logger.warning("Failed to load DA classifier: %s", e)
            return False

    def classify(self, text: str, embedder=None) -> ClassificationResult:
        """Classify a message using combined DA + structural features.

        Args:
            text: The text to classify.
            embedder: Optional embedder to use (for caching).

        Returns:
            ClassificationResult with type, confidence, and method used.
        """
        # Step 1: Structural classification (rule-based, high precision)
        structural_type, structural_conf = self._structural_classify(text)

        # If structural classification has high confidence, use it
        if structural_conf >= 0.9:
            return ClassificationResult(
                type=structural_type,
                confidence=structural_conf,
                method="structural",
                structural_type=structural_type,
            )

        # Step 2: DA classification (embedding-based)
        da_type, da_conf = self._da_classify(text, embedder or self.embedder)

        # Step 3: Combine results
        if da_conf >= 0.7:
            return ClassificationResult(
                type=da_type,
                confidence=da_conf,
                method="da_classifier",
                da_type=da_type,
                da_confidence=da_conf,
                structural_type=structural_type,
            )

        # If both have low confidence, prefer structural if available
        if structural_type != "UNKNOWN":
            return ClassificationResult(
                type=structural_type,
                confidence=max(structural_conf, 0.5),
                method="combined",
                da_type=da_type,
                da_confidence=da_conf,
                structural_type=structural_type,
            )

        # Fall back to DA result
        return ClassificationResult(
            type=da_type or "STATEMENT",
            confidence=da_conf or 0.3,
            method="da_classifier",
            da_type=da_type,
            da_confidence=da_conf,
        )

    def _structural_classify(self, text: str) -> tuple[str, float]:
        """Rule-based classification for high-precision patterns."""
        text_lower = text.lower().strip()

        # AGREE patterns (high precision)
        agree_starters = (
            "yeah", "yes", "sure", "definitely", "absolutely",
            "sounds good", "i'm down", "count me in", "let's do it",
            "of course", "for sure", "totally", "i'd love to",
        )
        if any(text_lower.startswith(s) for s in agree_starters):
            return "AGREE", 0.95

        # DECLINE patterns
        decline_starters = (
            "no", "can't", "cannot", "sorry", "unfortunately",
            "i can't", "not ", "i won't", "i'm not", "nah",
            "i don't think", "probably not",
        )
        if any(text_lower.startswith(s) for s in decline_starters):
            return "DECLINE", 0.95

        # DEFER patterns
        defer_starters = (
            "maybe", "possibly", "let me check", "i'll see",
            "not sure", "might", "i'll let you know", "depends",
            "let me think", "i'll get back",
        )
        if any(text_lower.startswith(s) for s in defer_starters):
            return "DEFER", 0.90

        # QUESTION patterns
        if "?" in text:
            return "QUESTION", 0.85

        # REACT_POSITIVE patterns
        react_positive = (
            "congrats", "amazing", "awesome", "that's great",
            "so happy", "wow", "nice", "cool", "yay", "!!",
        )
        if any(p in text_lower for p in react_positive):
            return "REACT_POSITIVE", 0.85

        # ACKNOWLEDGE patterns
        ack_exact = {"ok", "okay", "k", "kk", "got it", "alright", "👍", "sounds good"}
        if text_lower in ack_exact:
            return "ACKNOWLEDGE", 0.95

        return "UNKNOWN", 0.0

    def _da_classify(self, text: str, embedder) -> tuple[str | None, float | None]:
        """DA classification using embedding similarity."""
        if self._da_index is None:
            if not self._load_da_classifier():
                return None, None

        try:
            embedding = embedder.encode([text])[0]
            embedding = embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding)

            distances, indices = self._da_index.search(embedding, k=1)

            if len(indices[0]) > 0 and indices[0][0] < len(self._da_labels):
                label = self._da_labels[indices[0][0]]
                confidence = 1 - distances[0][0]  # Convert distance to similarity
                return label, float(confidence)
        except Exception as e:
            logger.warning("DA classification failed: %s", e)

        return None, None

    def get_valid_response_types(self, trigger_type: str) -> list[str]:
        """Get valid response types for a trigger type."""
        try:
            trigger = TriggerType(trigger_type)
            return [r.value for r in TRIGGER_TO_VALID_RESPONSES.get(trigger, [ResponseType.STATEMENT])]
        except ValueError:
            return ["STATEMENT", "ACKNOWLEDGE", "QUESTION"]


# Singleton
_classifier: UnifiedResponseClassifier | None = None


def get_response_classifier() -> UnifiedResponseClassifier:
    """Get or create the singleton response classifier."""
    global _classifier
    if _classifier is None:
        _classifier = UnifiedResponseClassifier()
    return _classifier
```

---

### Task 2: Create `jarvis/retrieval.py`

**File**: `jarvis/retrieval.py`
**Lines**: ~150
**Dependencies**: Existing FAISS index, database with DA columns

```python
"""DA-filtered FAISS retrieval for type-specific examples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.db import JarvisDB
    from jarvis.index import TriggerIndexSearcher

logger = logging.getLogger(__name__)


# High-quality clusters for each response type (from cluster purity analysis)
BEST_CLUSTERS_BY_TYPE: dict[str, list[int]] = {
    "AGREE": [7, 74, 9],        # 180 + 127 + 124 = 431 exemplars
    "ANSWER": [213],            # 204 exemplars
    "REACT_POSITIVE": [160, 32, 18, 15, 41, 51, 77],  # 178 + 164 + ... exemplars
    "ACKNOWLEDGE": [19, 11, 55, 38, 17],  # 170 + 161 + ...
    "QUESTION": [53, 73, 80],   # 201 + 130 + 118 = 449 exemplars
    "GREETING": [53],           # 240 exemplars
}


def get_typed_examples(
    trigger: str,
    target_response_type: str,
    searcher: "TriggerIndexSearcher",
    db: "JarvisDB",
    k: int = 5,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """Get FAISS results filtered by response DA type.

    Args:
        trigger: The trigger text to search for.
        target_response_type: The desired response type (AGREE, DECLINE, etc.)
        searcher: FAISS index searcher.
        db: Database instance.
        k: Number of examples to return.
        embedder: Optional embedder for caching.

    Returns:
        List of example dicts with trigger_text, response_text, similarity.
    """
    # Get more candidates than needed for filtering headroom
    search_k = k * 5

    try:
        results = searcher.search_with_pairs(
            query=trigger,
            k=search_k,
            threshold=0.3,  # Low threshold to get more candidates
            embedder=embedder,
        )
    except Exception as e:
        logger.warning("FAISS search failed: %s", e)
        return []

    # Filter by response_da_type
    typed_results = [
        r for r in results
        if r.get("response_da_type") == target_response_type
    ]

    # If not enough typed results, try cluster-based filtering
    if len(typed_results) < k:
        best_clusters = BEST_CLUSTERS_BY_TYPE.get(target_response_type, [])
        if best_clusters:
            cluster_results = [
                r for r in results
                if r.get("cluster_id") in best_clusters
            ]
            # Add cluster results that aren't already in typed_results
            existing_ids = {r.get("pair_id") for r in typed_results}
            for r in cluster_results:
                if r.get("pair_id") not in existing_ids:
                    typed_results.append(r)
                    if len(typed_results) >= k:
                        break

    return typed_results[:k]


def get_examples_for_all_types(
    trigger: str,
    valid_types: list[str],
    searcher: "TriggerIndexSearcher",
    db: "JarvisDB",
    k_per_type: int = 3,
    embedder: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Get examples for multiple response types in one search.

    More efficient than calling get_typed_examples multiple times.

    Args:
        trigger: The trigger text.
        valid_types: List of response types to get examples for.
        searcher: FAISS index searcher.
        db: Database instance.
        k_per_type: Examples per type.
        embedder: Optional embedder.

    Returns:
        Dict mapping response_type to list of examples.
    """
    # Get a large pool of candidates
    search_k = len(valid_types) * k_per_type * 5

    try:
        results = searcher.search_with_pairs(
            query=trigger,
            k=search_k,
            threshold=0.3,
            embedder=embedder,
        )
    except Exception as e:
        logger.warning("FAISS search failed: %s", e)
        return {t: [] for t in valid_types}

    # Partition results by type
    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in valid_types}

    for r in results:
        response_type = r.get("response_da_type")
        if response_type in by_type and len(by_type[response_type]) < k_per_type:
            by_type[response_type].append(r)

    # Fill in from best clusters if needed
    for response_type in valid_types:
        if len(by_type[response_type]) < k_per_type:
            best_clusters = BEST_CLUSTERS_BY_TYPE.get(response_type, [])
            for r in results:
                if r.get("cluster_id") in best_clusters:
                    if r not in by_type[response_type] and len(by_type[response_type]) < k_per_type:
                        by_type[response_type].append(r)

    return by_type
```

---

### Task 3: Create `jarvis/multi_option.py`

**File**: `jarvis/multi_option.py`
**Lines**: ~200
**Dependencies**: response_classifier, retrieval, prompts

```python
"""Multi-option response generation with diversity enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.response_classifier import (
    TRIGGER_TO_VALID_RESPONSES,
    TriggerType,
    get_response_classifier,
)
from jarvis.retrieval import get_examples_for_all_types

if TYPE_CHECKING:
    from jarvis.db import Contact, JarvisDB
    from jarvis.index import TriggerIndexSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


@dataclass
class ResponseOption:
    """A single response option."""
    type: str           # AGREE, DECLINE, DEFER, etc.
    response: str       # The generated response text
    confidence: float   # Confidence score (0-1)
    examples_used: int  # Number of examples used for generation


@dataclass
class MultiOptionResult:
    """Result of multi-option generation."""
    options: list[ResponseOption]
    trigger_type: str
    trigger_confidence: float
    is_commitment: bool  # True if this required user commitment (INVITATION, YN_QUESTION)


# Types that represent commitment questions (user must decide yes/no/maybe)
COMMITMENT_TRIGGER_TYPES = {
    TriggerType.INVITATION,
    TriggerType.YN_QUESTION,
    TriggerType.REQUEST,
}


def is_commitment_question(trigger_type: str) -> bool:
    """Check if trigger type requires user commitment."""
    try:
        return TriggerType(trigger_type) in COMMITMENT_TRIGGER_TYPES
    except ValueError:
        return False


def generate_response_options(
    trigger: str,
    trigger_da: str,
    searcher: "TriggerIndexSearcher",
    db: "JarvisDB",
    generator: "MLXGenerator",
    contact: "Contact | None" = None,
    max_options: int = 3,
    embedder: Any = None,
) -> MultiOptionResult:
    """Generate multiple response options for commitment questions.

    Args:
        trigger: The incoming message.
        trigger_da: The trigger's DA type (INVITATION, YN_QUESTION, etc.)
        searcher: FAISS index searcher.
        db: Database instance.
        generator: LLM generator.
        contact: Optional contact for personalization.
        max_options: Maximum number of options to generate.
        embedder: Optional embedder for caching.

    Returns:
        MultiOptionResult with diverse response options.
    """
    from contracts.models import GenerationRequest
    from jarvis.prompts import build_typed_reply_prompt

    # Get valid response types for this trigger
    classifier = get_response_classifier()
    valid_types = classifier.get_valid_response_types(trigger_da)[:max_options]

    # Get examples for each response type
    examples_by_type = get_examples_for_all_types(
        trigger=trigger,
        valid_types=valid_types,
        searcher=searcher,
        db=db,
        k_per_type=3,
        embedder=embedder,
    )

    options: list[ResponseOption] = []

    for response_type in valid_types:
        examples = examples_by_type.get(response_type, [])

        # Build prompt with type guidance
        example_pairs = [(e["trigger_text"], e["response_text"]) for e in examples]
        prompt = build_typed_reply_prompt(
            trigger=trigger,
            response_type=response_type,
            examples=example_pairs,
            contact_name=contact.display_name if contact else None,
        )

        try:
            request = GenerationRequest(
                prompt=prompt,
                max_tokens=60,  # Keep responses short
            )
            response = generator.generate(request)
            response_text = response.text.strip()

            # Calculate confidence based on example quality
            confidence = 0.7  # Base confidence
            if examples:
                avg_similarity = sum(e.get("similarity", 0.5) for e in examples) / len(examples)
                confidence = min(0.95, confidence + avg_similarity * 0.3)

            options.append(ResponseOption(
                type=response_type,
                response=response_text,
                confidence=confidence,
                examples_used=len(examples),
            ))

        except Exception as e:
            logger.warning("Failed to generate %s option: %s", response_type, e)
            continue

    # Apply diversity enforcement
    options = enforce_diversity(options)

    # Classify trigger confidence
    trigger_classification = classifier.classify(trigger, embedder)

    return MultiOptionResult(
        options=options,
        trigger_type=trigger_da,
        trigger_confidence=trigger_classification.confidence,
        is_commitment=is_commitment_question(trigger_da),
    )


def enforce_diversity(options: list[ResponseOption]) -> list[ResponseOption]:
    """Ensure response options are functionally diverse.

    Follows Gmail Smart Reply approach:
    - If first two options are both positive, include a negative/neutral in slot 3
    - If first two options are both negative, include a positive in slot 3

    Args:
        options: List of response options.

    Returns:
        Reordered options with diversity enforced.
    """
    if len(options) < 3:
        return options

    positive_types = {"AGREE", "REACT_POSITIVE"}
    negative_types = {"DECLINE", "REACT_SYMPATHY"}
    neutral_types = {"DEFER", "QUESTION", "ACKNOWLEDGE"}

    # Check sentiment of top options
    top_types = [o.type for o in options[:2]]

    has_positive = any(t in positive_types for t in top_types)
    has_negative = any(t in negative_types for t in top_types)
    has_neutral = any(t in neutral_types for t in top_types)

    # If all positive, try to move negative/neutral to slot 3
    if has_positive and not has_negative and not has_neutral:
        for i, opt in enumerate(options[2:], start=2):
            if opt.type in negative_types or opt.type in neutral_types:
                # Move this option to slot 3 (index 2)
                options.insert(2, options.pop(i))
                break

    # If all negative, try to move positive to slot 3
    elif has_negative and not has_positive:
        for i, opt in enumerate(options[2:], start=2):
            if opt.type in positive_types:
                options.insert(2, options.pop(i))
                break

    return options[:3]
```

---

### Task 4: Modify `jarvis/router.py`

**Changes needed**:
1. Import multi_option module
2. Add trigger DA classification step
3. For commitment questions, return multiple options
4. Keep single response for non-commitment messages

```python
# Add to imports
from jarvis.multi_option import (
    generate_response_options,
    is_commitment_question,
    MultiOptionResult,
)
from jarvis.response_classifier import get_response_classifier

# Add to route() method, after message/intent classification:

def route(self, incoming: str, contact_id: int | None = None, ...):
    # ... existing classification code ...

    # NEW: Classify trigger DA type
    trigger_classifier = get_response_classifier()
    trigger_classification = trigger_classifier.classify(incoming, cached_embedder)
    trigger_da = trigger_classification.type

    # NEW: For commitment questions, generate multiple options
    if is_commitment_question(trigger_da) and not is_context_dependent:
        multi_result = generate_response_options(
            trigger=incoming,
            trigger_da=trigger_da,
            searcher=self.index_searcher,
            db=self.db,
            generator=self.generator,
            contact=contact,
            max_options=3,
            embedder=cached_embedder,
        )

        return {
            "type": "multi_option",
            "options": [
                {
                    "type": opt.type,
                    "response": opt.response,
                    "confidence": opt.confidence,
                }
                for opt in multi_result.options
            ],
            "trigger_type": multi_result.trigger_type,
            "is_commitment": True,
            "similarity_score": search_results[0]["similarity"] if search_results else 0.0,
        }

    # ... rest of existing routing logic for non-commitment messages ...
```

---

### Task 5: Modify `api/routers/drafts.py`

**Changes needed**:
1. Update response schema to support multiple options
2. Add backwards compatibility for single response

```python
# Update response model
class DraftReplyResponse(BaseModel):
    suggestions: list[str]  # Keep for backwards compatibility
    options: list[ResponseOptionModel] | None = None  # NEW: structured options
    route_type: str
    confidence: str
    is_commitment: bool = False
    trigger_type: str | None = None

class ResponseOptionModel(BaseModel):
    type: str
    response: str
    confidence: float

# Update endpoint logic
@router.post("/reply")
async def get_draft_reply(...):
    result = reply_router.route(...)

    if result.get("type") == "multi_option":
        return DraftReplyResponse(
            suggestions=[opt["response"] for opt in result["options"]],
            options=[
                ResponseOptionModel(
                    type=opt["type"],
                    response=opt["response"],
                    confidence=opt["confidence"],
                )
                for opt in result["options"]
            ],
            route_type="multi_option",
            confidence="high",
            is_commitment=True,
            trigger_type=result.get("trigger_type"),
        )

    # Single response (existing behavior)
    return DraftReplyResponse(
        suggestions=[result["response"]],
        options=None,
        route_type=result["type"],
        confidence=result["confidence"],
        is_commitment=False,
    )
```

---

## Testing Plan

### Unit Tests

```python
# tests/unit/test_response_classifier.py
def test_structural_classify_agree():
    classifier = get_response_classifier()
    result = classifier.classify("Yeah I'm down!")
    assert result.type == "AGREE"
    assert result.confidence >= 0.9

def test_structural_classify_decline():
    classifier = get_response_classifier()
    result = classifier.classify("Sorry, can't make it")
    assert result.type == "DECLINE"

def test_is_commitment_question():
    assert is_commitment_question("INVITATION")
    assert is_commitment_question("YN_QUESTION")
    assert not is_commitment_question("INFO_STATEMENT")
```

### Integration Tests

```python
# tests/integration/test_multi_option.py
def test_invitation_generates_three_options():
    router = get_reply_router()
    result = router.route("Want to grab lunch tomorrow?")

    assert result["type"] == "multi_option"
    assert len(result["options"]) == 3

    types = {opt["type"] for opt in result["options"]}
    assert "AGREE" in types or "DECLINE" in types
```

---

## Rollout Plan

### Phase 1: Create New Modules (Day 1)
- [ ] Create `jarvis/response_classifier.py`
- [ ] Create `jarvis/retrieval.py`
- [ ] Create `jarvis/multi_option.py`
- [ ] Run unit tests

### Phase 2: Integration (Day 2)
- [ ] Modify `jarvis/router.py`
- [ ] Modify `api/routers/drafts.py`
- [ ] Run integration tests

### Phase 3: Validation (Day 3)
- [ ] Manual testing with real messages
- [ ] Verify desktop app compatibility
- [ ] Update IMPLEMENTATION_STATUS.md

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Commitment questions return 3 options | 100% |
| Options include AGREE + DECLINE/DEFER | 100% |
| Non-commitment still returns single response | 100% |
| Desktop app works with new schema | 100% |
| Generation latency < 3s per option | 100% |
| Memory usage < 2GB | 100% |
