# Performance Targets

Target latencies and benchmarks for JARVIS components.

## Component Targets

| Component | Target | Notes |
|-----------|--------|-------|
| **Embeddings** | <100ms | Single text encoding |
| **Generation** | <2s | LLM response (50 tokens) |
| **Retrieval** | <50ms | FAISS similarity search |
| **Total Pipeline** | <3s | End-to-end user request |

## Current Benchmarks

### Latency (from benchmarks)

| Operation | P50 | P95 | P99 | Target |
|-----------|-----|-----|-----|--------|
| Intent classification | 12ms | 25ms | 45ms | <50ms |
| FAISS search (10K vectors) | 3ms | 8ms | 15ms | <50ms |
| LLM generation (50 tokens) | 180ms | 320ms | 500ms | <2s |
| **Full pipeline** | **250ms** | **450ms** | **700ms** | **<3s** |

### Desktop Performance (V1 vs V2)

| Operation | V1 (HTTP) | V2 (Direct + Socket) | Target |
|-----------|-----------|----------------------|--------|
| Load conversations | ~100-150ms | ~1-5ms | <100ms |
| Load messages | ~100-150ms | ~1-5ms | <100ms |
| New message notification | Up to 10s (polling) | Instant (push) | Instant |
| Generate draft | ~50ms + inference | ~1-5ms + inference | <2s total |
| Semantic search | ~50ms + compute | ~1-5ms + compute | <500ms |

### Application Targets

| Metric | Current | Target |
|--------|---------|--------|
| Cold start time | ~4-5s | <2s to interactive |
| Conversation switch | ~100-150ms | <100ms perceived |
| Message send feedback | ~200ms | Instant (optimistic) |
| Search response (10K messages) | ~800ms | <500ms |

## Validation Gates

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency | <3s | 3-5s | >5s |
| G4 | Cold-start latency | <15s | 15-20s | >20s |

## Memory Budget

| Component | Memory | Target |
|-----------|--------|--------|
| LLM (LFM-1.2B 4-bit) | ~1.2GB | <2GB |
| Embeddings model | ~120MB | <200MB |
| FAISS index (50K vectors) | ~75MB | <100MB |
| Classifiers (SVM) | ~20MB | <50MB |
| **Total** | **~1.4GB** | **<5.5GB** |

## Running Benchmarks

```bash
# Latency benchmark
uv run python -m benchmarks.latency.run

# Memory benchmark
uv run python -m benchmarks.memory.run

# Hallucination benchmark
uv run python -m benchmarks.hallucination.run
```

## Monitoring

Track these metrics in production:

| Metric | Target | Formula |
|--------|--------|---------|
| Acceptance Rate | >50% | sent / (sent + edited + dismissed) |
| Edit Rate | <30% | edited / total_actioned |
| Time to Respond | <3s | suggestion shown → user action |
| Classifier Accuracy | >85% | periodic evaluation |
