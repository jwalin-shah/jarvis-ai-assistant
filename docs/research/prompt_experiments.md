# Research: Fact Extraction Prompt Experiments (v3 Bakeoff)

## Summary

We conducted a rigorous "bakeoff" (A/B test) of 10+ prompt variants for the 0.7b model to determine the optimal strategy for fact extraction.

## ❌ What Failed (Do Not Repeat)

1. **Plaintext Templates**:
   - prompts asking for bulleted lists (`- Subject | Predicate`) failed consistently.
   - The model often hallucinated extra fields or ignored the separator.
   - **Lesson**: Small models (0.7b) follow JSON schemas better than free-text instructions.

2. **Output Priming**:
   - We tried forcing the model's first token to be `{"` to "prime" it for JSON.
   - **Result**: Disaster. The model got confused, often repeating the brace or hallucinating gibberish.
   - **Lesson**: Let the model generate the full JSON object naturally.

3. **Multi-Segment Batching**:
   - We tried processing 3-5 disparate conversations in one prompt using `[Segment N]` markers.
   - **Result**: The model hallucinated `segment_id`s (outputting ID 3 when only 1 segment existed) and leaked facts between contexts.
   - **Lesson**: Stick to **Single-Segment Processing** for 0.7b models.

4. **Relaxed Parsing**:
   - Allowing the model to output markdown fences (` ```json `) broke our initial strict parser.
   - **Fix**: We now strip fences in post-processing rather than fighting the model in the prompt.

## ✅ What Worked (The Solution)

1. **Strict JSONL Prompt (`combined_v3`)**:
   - Explicit instructions to output _only_ JSON lines.
   - Negative constraints ("Do NOT use pronouns", "Do NOT mention the chat itself").
   - **Result**: 75% grounding (vs 53% baseline).

2. **Heuristic Confidence Scoring**:
   - Instead of asking the model for a confidence score (which it hallucinates as highly confident), we map the _type_ of fact to a score.
   - `works_at` = 1.0, `likes` = 0.8, `schedule` = 0.6.

3. **Post-Processing Guards**:
   - **Empty Input Guard**: If input is empty, return empty list (don't even call LLM).
   - **Grounding Filter**: Verify `fact.value` appears in the source text.
