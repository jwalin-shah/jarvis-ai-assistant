# Context Window Improvements

## 5A. Longer Context

**Goal**: Increase context window for better multi-turn understanding.

### Current State

- `jarvis/router.py` uses last 10 messages
- LFM2.5 supports up to 128k tokens

### Implementation

```python
def get_context_length(conversation):
    if len(conversation) < 20:
        return len(conversation)  # Full context

    # Long conversations: last 30 + key messages
    return 30 + len(get_key_messages(conversation))
```

**Key message selection:** Questions, answers, decisions, topic changes

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Context messages | 10 | 20-50 |
| Prompt length | ~500 tokens | ~1000-2000 |
| Generation latency | 600-3000ms | +30% |

**Effort**: Low (3-5 days)

---

## 5B. Conversation Summarization

**Goal**: Summarize older context to fit in prompt.

### Implementation

```python
def summarize_conversation(messages, max_tokens=200):
    recent = messages[-10:]  # Keep verbatim
    older = messages[:-10]

    summary = generate_summary(older, max_tokens=max_tokens)
    return f"[Earlier: {summary}]\n" + format_messages(recent)
```

**Cached summaries:** Store in jarvis.db, update incrementally

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Context coverage | 10 messages | 100+ (summarized) |
| Prompt tokens | 500 | 700-1000 |
| Additional latency | 0 | 500ms (first time, cached) |

**Effort**: Medium (2 weeks)

---

## 5C. Topic Tracking

**Goal**: Track topic changes for smarter context selection.

### Implementation

```python
def segment_by_topic(messages):
    topics = []
    current_topic = []

    for msg in messages:
        if is_topic_change(msg, current_topic):
            topics.append(current_topic)
            current_topic = [msg]
        else:
            current_topic.append(msg)
    return topics
```

**Topic-aware context:**
- Include full current topic
- Include summaries of relevant past topics
- Skip unrelated topics

**Expected Impact**: Context relevance 0.60 → 0.80+
**Effort**: Medium (2-3 weeks)
**Dependency**: Summarization (5B)

---

## 5D. Entity Tracking

**Goal**: Track named entities across conversation.

### Implementation

```python
def extract_entities(message):
    return {
        'people': [...],
        'places': [...],
        'dates': [...],
        'events': [...]
    }
```

**Entity-enhanced prompts:**
```
[Entities: John (friend), Sunset Grill (restaurant), Saturday 7pm]
```

**Benefits:**
- Handles "he/she/they" references
- Better understanding of plans
- Can surface relevant entity history

**Expected Impact**: Entity resolution 0% → 70%+
**Effort**: High (3-4 weeks)
**Dependency**: spaCy or similar NER
