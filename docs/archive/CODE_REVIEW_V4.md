# Code Review V4: Threading Semantic Chunking

NOTE: Shared text normalization is consolidated in `jarvis/text_normalizer.py` and wired into the clustering pipeline.

## Summary

Current semantic chunking in `jarvis/threading.py` is a good baseline but shallow. It relies on adjacent-message similarity plus time gaps and reply-to links. This works for obvious topic shifts but struggles with short messages, gradual drift, and long reply latency.

## Findings

### ✅ Strengths
- **Fast + simple**: Only compares adjacent messages; low compute cost.
- **No labels required**: Works out-of-the-box for new users.
- **Reply-to linking**: Correctly preserves explicit reply chains when present.

### ⚠️ Weaknesses
- **Local-only similarity**: Compares only current vs previous message, not the thread context.
- **Time-gap dependency**: Splits on time gaps by default, even when user replies late.
- **Single global threshold**: One semantic threshold for all chats + message types.
- **Short-message bias**: Very short responses (“ok”, “lol”, “idk”) often produce low similarity and trigger false splits.
- **No drift handling**: Gradual topic changes aren’t tracked (no rolling centroid or window).

## Recommendations (high impact)
1. **Compare to a rolling thread centroid**, not just the previous message.
2. **Lower similarity threshold for short messages**, or ignore short messages for split decisions.
3. **Make time-gap splitting optional or weighted**, not a hard rule.
4. **Add minimum thread length before splitting**, to avoid fragmenting conversations.

## Code References
- Thread splitting logic: `jarvis/threading.py`
  - `_should_start_new_thread()` uses time gap + adjacent similarity threshold.
  - `_compute_similarity()` embeds and compares two messages only.
