# Code Reuse, Prompt Consistency, and Load/Unload Audit

Audit of segmentation, extraction, prompt building, and model lifecycle.

---

## 1. Segmentation – Reuse and Consistency

### Entry points

| Caller                                                   | Segmenter used                                 | Then                                                  |
| -------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| **segment_ingest** (backfill)                            | `segment_conversation_basic` (basic_segmenter) | persist + index + **BatchedInstructionFactExtractor** |
| **watcher** (resegment)                                  | `segment_conversation` (topic_segmenter)       | `process_segments(..., extract_facts=False)`          |
| **backfill_complete**                                    | (segments from elsewhere)                      | `process_segments(..., extract_facts=...)`            |
| **Scripts** (evaluate_segmentation, reextract_radhika\*) | `segment_conversation` (topic_segmenter)       | various                                               |

### Shared vs duplicated

- **Shared:** `process_segments()` (persist → index → optional fact extraction) is the single pipeline after segmentation. Watcher and backfill both use it.
- **Not shared:** Two different segmenters:
  - **basic_segmenter**: `segment_conversation_basic` → `BasicSegment` (no topic label). Used by segment_ingest.
  - **topic_segmenter**: `segment_conversation` → `TopicSegment` (with topic_label, etc.). Used by watcher and scripts.
- **Duplicated logic:** Both segmenters repeat the same steps: sort by date, same junk filtering (`is_junk_message`, `normalize_text`), same embedder usage, similar boundary detection. Topic segmenter adds topic labeling and returns a richer type.

### Recommendation

- Keep two segmenters only if “basic vs topic-aware” is a deliberate product choice; otherwise consider one core (e.g. boundaries + filtering) with an optional topic-labeling step.
- If keeping both, at least extract shared filtering + boundary logic into a common helper to avoid drift.

---

## 2. Extraction – Reuse and Consistency

### Entry points

| Caller                                              | Extractor                                                          | API used                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- |
| **segment_pipeline** (watcher, backfill with facts) | `InstructionFactExtractor` (singleton `get_instruction_extractor`) | `extract_facts_from_batch()`                             |
| **tasks/worker** (FACT_EXTRACTION)                  | `InstructionFactExtractor` (same singleton)                        | `extract_facts_from_batch()` called one window at a time |
| **segment_ingest**                                  | `BatchedInstructionFactExtractor` (separate class)                 | `extract_facts_from_segments_batch()`                    |

### Inconsistencies

- **Two extractor implementations:** `InstructionFactExtractor` (JSONL, two-pass, in `instruction_extractor.py`) vs `BatchedInstructionFactExtractor` (bullet format “[Segment N] - [Name] Fact”, different prompts, in `batched_extractor.py`). Same tier names (0.7b, 1.2b, 350m) but different prompt/output formats.
- **Worker semantics:** Task worker now iterates extraction windows and calls `extract_facts_from_batch([window], ...)` per window for stability.
- **Different pipelines:** segment_pipeline uses InstructionFactExtractor + `save_facts` + pass1 logging. segment_ingest uses BatchedInstructionFactExtractor + its own verification/save loop. Behavior and quality can differ.

### Recommendation

- Prefer a single extraction path: one extractor (e.g. InstructionFactExtractor with batching) and one pipeline (e.g. segment_pipeline-style: persist → index → extract with shared save_facts/verification).
- Route segment_ingest and task worker through that path (batch API where possible).
- Deprecate or align BatchedInstructionFactExtractor so prompts and output format match the canonical extractor.

---

## 3. Prompts – Consistency

### Where prompts live

- **Reply / summary / search:** Centralized in `jarvis/prompts/` (constants, builders, `__init__.py`). Single source of truth; used for reply generation, RAG, search, summaries. **Consistent.**
- **Fact extraction:**
  - **InstructionFactExtractor:** Prompts defined inside `jarvis/contacts/instruction_extractor.py` (`_EXTRACTION_SYSTEM_PROMPT`, `_VERIFY_*`, JSONL output).
  - **BatchedInstructionFactExtractor:** Prompts inside `jarvis/contacts/batched_extractor.py` (`_BATCH_USER_PROMPT_TEMPLATE`, `_BATCH_VERIFY_*`, bullet format).
  - No shared extraction prompt constants; formats and rules differ.

### Recommendation

- Move extraction prompt templates and rules into `jarvis/prompts/` (e.g. extraction-specific module or section) and have both extractors use them so wording and format stay consistent unless a given flow truly needs a different prompt.
- Keep “one place per concern”: reply/summary/search in existing prompt modules; extraction in a dedicated extraction prompt module.

---

## 4. Load/Unload – Correctness and Coordination

### ModelManager

- **Tracks:** `llm` (reply), `embedder`, `nli`.
- **prepare_for(model_type):** Unloads other types so only one “family” is active. `prepare_for("llm")` unloads embedder and NLI; it does **not** unload the **extraction** model (InstructionFactExtractor / BatchedInstructionFactExtractor).
- **Reply LLM:** Via `models.loader.get_model()` singleton; `ModelManager._unload_llm()` calls `reset_model()` so reply LLM is unloaded.
- **Extraction models:** Each has its own `MLXModelLoader` instance (in InstructionFactExtractor and BatchedInstructionFactExtractor). They are **not** registered with ModelManager.

### Gaps

1. **Extraction not in ModelManager:** When we `prepare_for("llm")` (e.g. for reply), the extraction model is never unloaded. When we load the extraction model (segment_pipeline or task worker), we never call ModelManager, so reply LLM and extraction model can both stay in memory (two “LLM-style” models).
2. **InstructionFactExtractor doesn’t use ModelManager:** `InstructionFactExtractor.load()` does not call `prepare_for("llm")` or `reset_model()`. So loading the extractor does not unload the reply LLM; we can have both loaded.
3. **BatchedInstructionFactExtractor does use ModelManager:** Its `load()` calls `get_model_manager().prepare_for("llm")`, so it unloads embedder and NLI, but `_unload_llm()` does not unload the InstructionFactExtractor singleton, so if that was loaded earlier it remains.
4. **Unload order:** When switching from extraction back to reply, nothing unloads the extraction model unless we add it.

### Recommendation (implemented)

- Treat extraction as part of the “LLM” family for memory: only one of {reply LLM, extraction model} should be loaded at a time.
- **ModelManager.\_unload_llm():** Also unload the instruction extractor (implemented: calls `reset_instruction_extractor()`).
- **InstructionFactExtractor.load():** Call `get_model_manager().prepare_for("llm")` before loading (implemented).
- **Reply LLM load path:** When the generator loads the reply model, it calls `prepare_for("llm")` first so the extractor (and embedder/NLI) are unloaded (implemented in `models/generator.py`).
- **reset_model():** Now only unloads the reply loader; does not set the singleton to `None`, so the generator’s reference remains valid and a subsequent `load()` reuses the same loader.

---

## 5. Summary Table

| Area                         | Reused?                                                                                  | Consistent?                               | Load/Unload                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Segmentation                 | Partially (process_segments shared; two segmenters, duplicated filtering/boundary logic) | No (basic vs topic, two types)            | N/A (no model in segmenters; embedder used)                                     |
| Extraction                   | No (two extractors, two pipelines; worker/window path differs)                           | No (different prompts and output formats) | Improved (InstructionFactExtractor + generator now coordinate via ModelManager) |
| Reply/summary/search prompts | Yes (jarvis/prompts)                                                                     | Yes                                       | N/A                                                                             |
| Extraction prompts           | No (in each extractor file)                                                              | No                                        | —                                                                               |
| ModelManager                 | —                                                                                        | —                                         | Incomplete (extractor not unloaded with LLM)                                    |

---

## 6. Suggested Next Steps (in order)

1. **Load/unload (critical):** Add extraction to the “LLM” unload path and have InstructionFactExtractor.load() use ModelManager so only one of reply vs extraction is in memory at a time.
2. **Extraction prompts:** Move extraction prompt text and rules into `jarvis/prompts` and use them from both extractors so behavior is consistent unless intentionally different.
3. **Single extraction path:** Route segment_ingest and task worker through the same extractor and pipeline (InstructionFactExtractor + extract_facts_from_batch + shared save_facts/verification); remove or align BatchedInstructionFactExtractor.
4. **Segmentation:** Optionally factor shared filtering and boundary logic from basic and topic segmenters into a shared helper to avoid duplication and drift.
