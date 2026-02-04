# Lessons Learned

Patterns and mistakes to avoid. Updated after corrections.

---

## 2026-02-03: BERTopic Clustering Experiment

### 1. Topic ID Mismatch in Printing

**Mistake**: When printing BERTopic results, assumed topic IDs were 0..n-1. BERTopic uses actual topic IDs (including -1 for outliers), so `sizes[i]` doesn't correspond to `topic_id=i`.

**Fix**: Return `unique_topics` list alongside `sizes`, then zip them for printing:
```python
return sil, sizes, unique_topics, samples, labels, n_topics, topic_words
```

**Rule**: When iterating over cluster/topic results, always use the actual IDs from the clustering output, not assumed sequential indices.

---

### 2. Algorithm Name Matching in Summaries

**Mistake**: Changed algorithm name from `"bertopic"` to `"bertopic-{config}"` but summary code still checked `algo == "bertopic"`.

**Fix**: Use `algo.startswith("bertopic")` instead of exact match.

**Rule**: When adding variants to an algorithm name (e.g., `bertopic-default`, `bertopic-balanced`), update all downstream code that filters/groups by algorithm name.

---

### 3. Silhouette Score Computed on Wrong Embedding Space

**Mistake**: Computed silhouette on original 768-dim embeddings even though BERTopic clusters in UMAP-reduced space. This gives misleading quality metrics.

**Fix**: Extract UMAP embeddings from `topic_model.umap_model.embedding_` and use those for silhouette:
```python
if hasattr(topic_model, "umap_model") and topic_model.umap_model is not None:
    reduced_embeddings = topic_model.umap_model.embedding_
    sil = silhouette_score(reduced_embeddings[sample_idx], labels[sample_idx])
```

**Rule**: Always compute clustering quality metrics in the same space where clustering was performed.

---

### 4. Cache Invalidation After Preprocessing Changes

**Observation**: When adding new text filters (garbage removal, tapback fixes), cached embeddings become stale because they were computed on unfiltered text.

**Rule**: After changing `normalize_text()` or any preprocessing, either:
- Delete cache files manually: `rm results/clustering/embed_cache/*.npy`
- Use `--force-embed` flag

Consider adding a preprocessing version hash to cache filenames.

---

### 5. Regex Patterns Need End-of-String Flexibility

**Mistake**: Tapback regex used `^Liked\s+".*"$` which failed on truncated quotes like `Laughed at "message was cut off..."`

**Fix**: Remove trailing `$` anchor: `^Liked\s+"`

**Rule**: For user-generated content, regex patterns should be permissive at boundaries. System messages get truncated, have trailing characters, etc.

---

### 6. Iterate Fast, Then Scale

**Workflow learned**:
1. Tune on small subset (`--limit 20000`) with config sweep
2. Compare metrics (outlier rate, topic coherence)
3. Pick best config
4. Run full dataset once with winner

Don't run full 167k messages repeatedly while iterating on parameters.
