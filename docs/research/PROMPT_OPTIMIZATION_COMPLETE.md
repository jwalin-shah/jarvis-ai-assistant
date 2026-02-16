# Prompt Optimization - Complete Results

## Executive Summary

Through systematic experimentation, we discovered that **simpler is better** for SLM prompting. The current complex categorization system actually hurts quality.

---

## Key Findings

### 1. Categorization Ablation

| Variant | Judge Score | Anti-AI | Latency |
|---------|-------------|---------|---------|
| **Universal** | **5.58/10** ✅ | **0%** ✅ | **273ms** ✅ |
| Categorized | 5.20/10 | 3.3% | 309ms |
| Category Hint | 5.15/10 | 1.7% | 287ms |

**Finding:** Universal prompt beats 6-category system by 0.38 points with 0% anti-AI rate.

### 2. Universal Prompt Optimization

| Prompt | Score | Anti-AI | Notes |
|--------|-------|---------|-------|
| **Baseline** | **6.27/10** ✅ | **0%** | Current winner |
| Minimal | 5.82/10 | 1.7% | Too vague |
| Negative constraints | 5.75/10 | 0% | Rules don't help |
| Persona | 5.72/10 | 0% | Unnecessary |
| Style-focused | 5.70/10 | 0% | Overly complex |

**Finding:** Simple baseline prompt performs best.

### 3. Context & RAG (Partial Results)

Preliminary findings from partial run:

| Variant | Notes |
|---------|-------|
| No context | Model invents responses |
| 3-5 messages | Good balance |
| 10 messages | Context overload |
| RAG only | Examples can mislead |
| Context + RAG | Best potential |

---

## 🏆 Recommended Production Configuration

### System Prompt
```
You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.
```

### Context Strategy
- **Depth:** 5-7 recent messages (not 10)
- **Format:** Simple chronological list
- **RAG:** Use only 1-2 highly similar examples (not 3+)

### Model Settings
- **Temperature:** 0.1 (deterministic but natural)
- **Max tokens:** 50 (forces brevity)
- **Top-p:** 0.9
- **Repetition penalty:** 1.05

---

## Additional Optimizations to Try

### 1. Best-of-N Selection (Implement This!)

```python
def score_candidate(reply: str, context: dict) -> float:
    """Multi-factor scoring for selecting best reply."""
    scores = {
        'brevity': 1.0 if len(reply) < 50 else 0.7 if len(reply) < 100 else 0.3,
        'lowercase_bonus': 0.15 if reply and reply[0].islower() else 0,
        'no_ai_phrases': -0.5 if has_ai_phrases(reply) else 0,
        'context_match': embedding_sim(reply, context['last_message']),
    }
    return sum(scores.values())

# Generate 3 candidates at different temperatures
candidates = []
for temp in [0.1, 0.2, 0.3]:
    reply = generate(temperature=temp)
    score = score_candidate(reply, context)
    candidates.append((reply, score))

return max(candidates, key=lambda x: x[1])[0]
```

### 2. Dynamic Temperature

```python
def get_temperature(category: str, context: dict) -> float:
    """Adjust temperature based on context."""
    if category == "emotion":
        return 0.15  # More consistent for emotional content
    elif category == "question":
        return 0.1   # Factual, low variance
    else:
        return 0.2   # Casual, more creative
```

### 3. Confidence-Based Fallback

```python
if generation_confidence < 0.6:
    # Low confidence - use template
    return get_template_reply(category)
else:
    return generated_reply
```

### 4. Real-Time Style Adaptation

```python
def analyze_recent_style(messages: list[str]) -> dict:
    """Analyze last 10 messages for style cues."""
    return {
        'uses_emoji': any('😊' in m for m in messages),
        'avg_length': sum(len(m) for m in messages) / len(messages),
        'uses_lowercase': sum(1 for m in messages if m[0].islower()) > len(messages)/2,
    }

# Adapt prompt based on detected style
style = analyze_recent_style(context)
if style['uses_emoji']:
    prompt += " It's okay to use emoji if they used them first."
```

### 5. Response Length Targeting

```python
def get_target_length(examples: list[str]) -> int:
    """Match response length to user's typical length."""
    avg = sum(len(e) for e in examples) / len(examples)
    return int(avg * 0.8)  # Slightly shorter than average

# Use in prompt
prompt += f"\nKeep it under {target_length} characters."
```

---

## Testing on Real Messages

### Option 1: Extract Recent Conversations

```python
# Extract last 50 conversations for testing
def extract_test_conversations(limit: int = 50) -> list[dict]:
    """Extract real conversations from iMessage DB."""
    query = """
    SELECT 
        c.chat_id,
        m.text,
        m.is_from_me,
        datetime(m.date/1000000000 + 978307200, 'unixepoch') as date
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    WHERE m.text IS NOT NULL
    AND length(m.text) > 0
    AND m.date > (strftime('%s','now') - 86400 * 7) * 1000000000
    ORDER BY c.chat_id, m.date DESC
    LIMIT 1000
    """
    # Group by chat, extract conversation threads
    # Return list of {context: [...], last_message: str}
```

### Option 2: A/B Test in Production

```python
# Shadow mode - generate but don't show
if is_in_test_group(user_id):
    # Generate with new prompt
    new_reply = generate_universal(context, message)
    old_reply = generate_categorized(context, message)
    
    # Log both for comparison
    log_ab_test(user_id, message, old_reply, new_reply)
    
    # Still show old reply
    return old_reply
```

### Option 3: User Acceptance Tracking

```python
def track_reply_acceptance(
    generated_reply: str,
    user_final_reply: str,
    was_edited: bool,
    time_to_send: float,
) -> None:
    """Track if users accept, edit, or ignore suggestions."""
    similarity = embedding_sim(generated_reply, user_final_reply)
    
    if similarity > 0.8 and not was_edited:
        metric = "accepted"
    elif similarity > 0.5:
        metric = "modified"
    else:
        metric = "rejected"
    
    log_metric(metric, prompt_version=PROMPT_VERSION)
```

---

## Implementation Roadmap

### Phase 1: Deploy Universal Prompt (This Week)
1. Replace categorized prompts with universal prompt
2. Remove category classifier from reply path
3. Keep classifier for analytics only
4. A/B test with 10% traffic

### Phase 2: Add Best-of-N (Next Week)
1. Generate 3 candidates per request
2. Implement scoring function
3. Select best candidate
4. Measure latency impact

### Phase 3: Context Optimization (Week 3)
1. Test context depths (3, 5, 7, 10)
2. Optimize RAG retrieval (top-1 vs top-3)
3. A/B test on real conversations

### Phase 4: Advanced Features (Week 4+)
1. Dynamic temperature
2. Real-time style adaptation
3. Confidence-based fallbacks

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Judge Score | 5.20 | 6.5+ |
| Anti-AI Rate | 3.3% | <1% |
| User Acceptance | ? | >70% |
| Latency | 309ms | <350ms |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Users miss categorized replies | Keep classifier, use for analytics only |
| Universal prompt too generic | A/B test gradually, monitor feedback |
| Latency from Best-of-N | Generate in parallel, add 50-100ms max |
| Context truncation issues | Test with long conversation histories |

---

## Conclusion

**The data is clear:** Simpler prompts work better for SLMs. The 6-category system added complexity without benefit. 

**Recommended immediate action:**
1. Switch to universal prompt
2. Reduce context to 5-7 messages
3. Implement Best-of-N selection
4. A/B test with real users

Expected improvement: **+1.0 judge score points** (5.2 → 6.2+) with simpler, faster code.
