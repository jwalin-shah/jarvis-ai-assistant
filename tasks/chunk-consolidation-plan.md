# Retrieval Consolidation: Chunks as Source of Truth

## Problem

Two parallel indexing systems that overlap:
- **Trigger Index** (`index.py`) - pairs from dumb time-window heuristics (10-min bundling). Breaks on slow typists, pairs unrelated adjacent turns, loses multi-turn context.
- **Chunk Index** (`chunk_index.py`) - topic-segmented conversation chunks via linguistic detection. Better boundaries, richer context.

Classifiers are trained but barely used in retrieval. Quality scores computed but ignored during search. The current router (`router.py`) is clean: `_search_examples()` → `_generate_response()` with mobilization hints. Single entry point to swap.

## Target Architecture

```
Raw messages → topic_chunker.py → TopicChunks
                                    ├→ enriched with DA type + quality
                                    ├→ contextual embeddings (metadata-prepended)
                                    └→ single searchable index
Router._search_examples() → chunk search → extract pairs on-demand → generate
```

---

## Phase 1: Contextual Chunk Embeddings (Quick Win)

**Goal**: Improve retrieval accuracy with zero new infrastructure.

### 1a. Prepend metadata to `text_for_embedding`

`topic_chunker.py:186-212` already prepends `"Topic: kw1, kw2\n\n"`. Extend to include contact name:

```python
# Current
f"Topic: {', '.join(self.keywords)}\n\n{conversation_text}"

# New
f"Conversation with {contact_name} about {self.label}. Topics: {', '.join(self.keywords)}.\n\n{conversation_text}"
```

This is Anthropic's contextual retrieval lite. ~35% retrieval failure reduction in their benchmarks.

**Requires**: `contact_name` field on TopicChunk (or resolve from `contact_id` at build time).

### 1b. Ensure query-side context matches

In `chunk_index.py:485-488`, the query is already normalized with `normalize_for_task("chunk_embedding")`. No additional query-side context needed since we're searching for incoming messages against conversation chunks.

**Files**: `jarvis/topics/topic_chunker.py`
**Risk**: Low (additive change to existing property)
**Verify**: Rebuild chunk index, spot-check 20 searches for relevance improvement

---

## Phase 2: Enrich Chunks with DA Metadata

**Goal**: Chunks carry the same metadata as pairs so they can serve typed retrieval.

### 2a. Add classification fields to chunk storage

When storing chunks in `_store_chunks()` (`incremental_chunking.py:158-191`), also:
1. Call `extract_last_exchange()` on each chunk with `has_my_response == True`
2. Run response classifier on the (trigger, response) pair
3. Store `response_da_type`, `response_da_conf` alongside existing chunk metadata

New fields in chunk storage:
- `response_da_type: str | None` (e.g., "AGREE", "DECLINE", "DEFER")
- `response_da_conf: float` (classifier confidence)
- `trigger_da_type: str | None`
- `quality_score: float` (reuse extract.py quality scoring logic on the exchange)

### 2b. Add DA fields to ChunkSearchResult

`chunk_index.py:83-97` - extend `ChunkSearchResult` with:
- `response_da_type: str | None`
- `response_da_conf: float`
- `quality_score: float`
- `last_trigger: str | None`
- `last_response: str | None`

These come from chunk metadata stored at build time, not computed at search time.

### 2c. Add metadata filtering to chunk search

`ChunkIndexSearcher.search()` (`chunk_index.py:470-526`) currently filters by `chat_id` only. Add:
- `response_da_type: str | None` - filter by DA type (for typed retrieval)
- `min_quality: float = 0.0` - filter by quality score

Post-filter after FAISS search (same pattern as existing `chat_id` filtering).

**Files**: `jarvis/topics/topic_chunker.py`, `jarvis/chunk_index.py`, `jarvis/incremental_chunking.py`
**Risk**: Medium (additive, but touches indexing pipeline)
**Verify**: Build enriched index, confirm DA types populated on >80% of chunks with responses

---

## Phase 3: Wire Router to Chunk Index

**Goal**: `_search_examples()` searches chunks instead of trigger pairs.

### 3a. Swap `_search_examples()` backend

Current (`router.py:261-280`):
```python
def _search_examples(self, incoming, cached_embedder):
    return self.index_searcher.search_with_pairs(query=incoming, k=5, ...)
```

New:
```python
def _search_examples(self, incoming, cached_embedder):
    chunk_results = self.chunk_searcher.search(query=incoming, k=5, ...)
    # Convert ChunkSearchResult → same dict format _generate_response expects
    examples = []
    for chunk in chunk_results:
        if chunk.last_trigger and chunk.last_response:
            examples.append({
                "trigger_text": chunk.last_trigger,
                "response_text": chunk.last_response,
                "similarity": chunk.similarity,
                "formatted_context": chunk.formatted_text,  # bonus: full conversation context
            })
    return examples
```

The router's `_generate_response()` (`router.py:442-604`) reads `result["trigger_text"]` and `result["response_text"]` from search results (line 472-474), so the interface stays the same.

### 3b. Pass chunk context to generation

Currently few-shot examples are isolated (trigger, response) tuples. With chunks, we can also pass `formatted_text` as richer context to `build_rag_reply_prompt()`. This gives the LLM full conversation flow, not just isolated exchanges.

### 3c. Mobilization-aware retrieval

The router already classifies mobilization (`router.py:398-406`). When mobilization says `ResponseType.COMMITMENT`, pass `response_da_type="AGREE"` (or the set of commitment types) to chunk search so we retrieve relevant examples.

```python
# In route(), after mobilization classification:
da_filter = None
if mobilization.response_type == ResponseType.COMMITMENT:
    da_filter = ["AGREE", "DECLINE", "DEFER"]

search_results = self._search_examples(incoming, cached_embedder, da_filter=da_filter)
```

**Files**: `jarvis/router.py`
**Risk**: Medium (changes retrieval source, needs A/B comparison)
**Verify**:
- Run `scripts/evaluate_retrieval.py` before and after
- Compare: exact_match_rate, semantic_match_rate, mean_top_score
- Spot-check 50 random queries manually

---

## Phase 4: Enable Existing Infrastructure

**Goal**: Turn on features already built but disabled. Independent of Phase 1-3.

### 4a. Enable cross-encoder reranking
- `retrieval.py:839-912` - `CrossEncoderReranker` already implemented
- Set `rerank_enabled: True` in config
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB)
- Reranks top-k FAISS candidates for higher precision

### 4b. Enable BM25 hybrid search
- `retrieval.py` - reciprocal rank fusion already implemented
- Set `bm25_enabled: True` in config
- Catches exact keyword matches that embeddings miss (names, specific phrases)

### 4c. Fix incremental normalization bug
- `index.py:~894` - `IncrementalTriggerIndex.add_pairs()` skips `normalize_for_task()`
- Apply same normalization as bulk build
- Same fix for `compact()`

**Files**: `jarvis/config.py`, `jarvis/retrieval.py`, `jarvis/index.py`
**Risk**: Low (enabling existing code)
**Verify**: Run eval script before/after each toggle

---

## Phase 5: Deprecate Turn-Based Extraction

**Goal**: Stop building pairs from raw time windows. Chunks are the source.

### 5a. New extraction entry point

```python
def extract_pairs_from_chunks(chunks: list[TopicChunk]) -> list[ExtractedPair]:
    """Derive pairs from topic chunks instead of raw time windows."""
    pairs = []
    for chunk in chunks:
        if not chunk.has_my_response:
            continue
        for trigger, response in chunk.extract_all_exchanges():
            pair = ExtractedPair(
                trigger_text=trigger,
                response_text=response,
                chat_id=chunk.chat_id,
                contact_id=chunk.contact_id,
                # Inherit chunk metadata
                topic_label=chunk.label,
                keywords=chunk.keywords,
            )
            # Classify + quality score
            pair.response_da_type = classify_response(response, trigger)
            pair.quality_score = compute_quality(trigger, response)
            pairs.append(pair)
    return pairs
```

### 5b. Deprecate `extract_all_pairs_v2()`
- Mark as deprecated in `jarvis/extract.py`
- Keep for benchmarking comparisons
- New ingestion pipeline: `ingest → chunk → derive pairs`

### 5c. Compare pair quality
- Run both extraction methods on same conversation set
- Compare: pair count, quality distribution, DA type distribution
- Verify chunk-derived pairs are equal or better quality

**Files**: `jarvis/extract.py`, `jarvis/ingest.py`
**Risk**: Medium (changes data pipeline)
**Verify**: Side-by-side comparison of extraction quality

---

## Phase 6: Single Index (Future)

**Goal**: Replace both FAISS indexes with sqlite-vec.

### 6a. Evaluate sqlite-vec
- Test with 145k vectors, 384-dim
- Benchmark: search latency vs FAISS
- Test: `WHERE response_da_type = 'AGREE' AND quality_score > 0.3` + vector search
- Check Float16 quantization accuracy

### 6b. Migrate to sqlite-vec
- Store all embeddings in existing SQLite DB
- Tag each vector: `chat_id`, `contact_id`, `da_type`, `quality_score`, `topic_label`, `timestamp`
- Metadata pre-filtering before vector search (not post-filter)
- Eliminate FAISS + rank_bm25 dependencies

### 6c. Remove legacy
- Remove `jarvis/index.py` FAISS builder/searcher
- Remove `jarvis/chunk_index.py` FAISS builder/searcher
- Remove `extract.py` turn-based extraction
- Single retrieval API

**Risk**: High (full index migration)
**Verify**: Full end-to-end eval, A/B test with real conversations

---

## Execution Order

| Phase | Effort | Impact | Depends On | Parallel? |
|-------|--------|--------|------------|-----------|
| 1 (contextual embeddings) | Small | High | Nothing | Yes |
| 4 (enable reranking/BM25) | Small | High | Nothing | Yes |
| 2 (enrich chunks with DA) | Medium | Medium | Nothing | Yes |
| 3 (wire router to chunks) | Medium | High | Phase 2 | No |
| 5 (deprecate extraction) | Medium | Medium | Phase 3 | No |
| 6 (sqlite-vec) | Large | Medium | Phase 5 | No |

**Start with**: Phase 1 + Phase 4 + Phase 2 in parallel (all independent, high impact).

## Success Metrics

- `scripts/evaluate_retrieval.py` before/after each phase
- Track: exact_match_rate, semantic_match_rate, mean_top_score, p10_top_score
- Target: >10% improvement in semantic_match_rate from Phase 1 + Phase 4
- Manual spot-check: 50 random retrievals assessed for relevance
