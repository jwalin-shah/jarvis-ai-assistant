# Feedback & Learning System

> **Last Updated:** 2026-02-10

## Current Infrastructure

```
FEEDBACK COLLECTION
├─ Explicit (API): sent, edited, dismissed, wrote_from_scratch
├─ Stored in: ~/.jarvis/feedback.jsonl (10K max entries)
└─ POST /feedback/response { action, suggestion_text, edited_text, ... }

EVALUATION SCORES (computed per entry)
├─ Tone Score: Does response match formality?
├─ Relevance Score: Semantic similarity to context
├─ Naturalness Score: Robotic phrases, repetition
├─ Length Score: Compared to user's typical length
└─ Overall Score: Weighted average

PATTERN ANALYSIS
├─ GET /feedback/improvements - Edit pattern mining
└─ GET /feedback/stats - Acceptance rate, etc.
```

## Failure Reason Tagging

```python
class FailureReason(str, Enum):
    # Classifier issues (fixable)
    CLASSIFIER_WRONG = "classifier_wrong"
    TONE_WRONG = "tone_wrong"
    TOO_GENERIC = "too_generic"
    CONTEXT_INSUFFICIENT = "context_insufficient"

    # Capability gaps (need new features)
    NEEDS_CALENDAR = "needs_calendar"
    NEEDS_MEMORY = "needs_memory"
    NEEDS_TASK_TRACKING = "needs_tasks"
    NEEDS_CONTACT_INFO = "needs_contacts"
    NEEDS_EXTERNAL_INFO = "needs_external"
```

## What's Missing: The Feedback Loop Gap

**Current state:** Feedback API exists but desktop app doesn't call it.

| Type | How It Works | Status |
|------|--------------|--------|
| **Explicit** | User clicks Send/Edit/Dismiss | API ready, UI not wired |
| **Implicit** | Detect what user actually sent | Not implemented |

## Proposed: Passive Feedback Detection

Watch chat.db for what user actually sends, compare to suggestion:

```python
async def _detect_suggestion_outcome(self, chat_id, user_message, sent_at):
    recent_suggestion = await self._get_recent_suggestion(chat_id, within_minutes=5)
    if not recent_suggestion:
        return

    # Compare using embeddings
    similarity = compute_similarity(user_message, recent_suggestion.text)

    if similarity > 0.92:
        action = "sent"      # Basically unchanged
    elif similarity > 0.55:
        action = "edited"
    else:
        action = "wrote_from_scratch"  # Ignored suggestion

    # Record automatically
    store.record_feedback(action=action, ...)
```

## Proposed: Trigger Complexity Analysis

Long messages with multiple topics get bad suggestions. Analyze before generating:

```python
@dataclass
class TriggerAnalysis:
    word_count: int
    sentence_count: int
    topic_count: int
    question_count: int
    has_time_reference: bool
    has_commitment_request: bool

    @property
    def complexity_level(self) -> str:
        if self.topic_count > 2 or self.question_count > 2:
            return "multi_topic"  # Split or flag
        if self.word_count > 50:
            return "complex"
        return "simple"
```

## Implementation Roadmap

| Priority | Task | Effort |
|----------|------|--------|
| 1 | Wire desktop Send/Edit/Dismiss to feedback API | Easy |
| 2 | Add passive detection in watcher.py | Medium |
| 3 | Add trigger complexity analysis | Medium |
| 4 | CLI: `jarvis feedback stats` | Easy |
| 5 | Real-time style learning per contact | Medium |
| 6 | Classifier retraining pipeline | Hard |
