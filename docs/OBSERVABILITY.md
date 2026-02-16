# JARVIS Observability Stack

## Current Instrumentation (Already Implemented)

Your codebase already has **professional-grade** logging and metrics tracking. Here's what's been recording data all along:

### 1. **Latency Tracking** ✅

```python
# In reply_service.py
latency_ms = {
    "total": ...,
    "generation": ...,
    "embedding": ...,
    "search": ...
}

# Records to metrics system
record_routing_metrics(
    latency_ms=latency_ms,
    ...
)
```

**Tracked Operations:**

- `reply.generate` - Full reply generation latency
- `embed.query` - Query embedding time
- `search.vector` - Vector search time
- `model.load` - Model loading time
- `fact.extract` - Fact extraction pipeline

### 2. **Performance Histograms** ✅

```python
# jarvis/metrics.py
histogram = get_latency_histogram()
histogram.observe("generation", duration_seconds)

# Get stats
stats = histogram.get_stats("generation")
# Returns: p50, p90, p95, p99, min, max, mean
```

**Buckets:** 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s

### 3. **Memory Monitoring** ✅

```python
# Real-time memory tracking
sampler = get_memory_sampler()
sampler.start()
# ... work ...
stats = sampler.get_stats()
# Returns: peak_rss_mb, avg_rss_mb, current_rss_mb
```

### 4. **Request Counting** ✅

```python
# API endpoint tracking
counter = get_request_counter()
counter.increment("/api/reply", method="POST")
stats = counter.get_stats()
# Returns: total_requests, requests_per_second, uptime
```

### 5. **Template Analytics** ✅

```python
# Template matching stats
analytics = get_template_analytics()
hit_rate = analytics.get_hit_rate()  # % queries using templates
top_templates = analytics.get_top_templates(20)
missed_queries = analytics.get_missed_queries(50)
```

### 6. **Generation Logging** ✅

```python
# jarvis/core/generation/logging.py
log_custom_generation(
    prompt=prompt,
    completion=completion,
    model=model_name,
    latency_ms=latency,
    metadata={...}
)
```

Logs stored in: `~/.jarvis/generations/custom_generations.jsonl`

---

## How to Access Your Metrics

### View Latency Stats

```python
from jarvis.metrics import get_latency_histogram

histogram = get_latency_histogram()

# Get generation latency stats
stats = histogram.get_stats("reply.generate")
print(f"p50: {stats['p50_ms']}ms")
print(f"p99: {stats['p99_ms']}ms")
print(f"mean: {stats['mean_ms']}ms")

# All operations
all_stats = histogram.get_stats()
for op, data in all_stats.items():
    print(f"{op}: p99={data['p99_ms']}ms, count={data['count']}")
```

### View Memory Usage

```python
from jarvis.metrics import get_memory_sampler

sampler = get_memory_sampler()
stats = sampler.get_stats()

print(f"Current RAM: {stats['current_rss_mb']}MB")
print(f"Peak RAM: {stats['peak_rss_mb']}MB")
print(f"Avg RAM: {stats['avg_rss_mb']}MB")
```

### View Template Hit Rate

```python
from jarvis.metrics import get_template_analytics

analytics = get_template_analytics()
print(f"Template hit rate: {analytics.get_hit_rate():.1f}%")

# Top templates
for template in analytics.get_top_templates(10):
    print(f"{template['template_name']}: {template['match_count']} hits")
```

---

## What Gets Logged

### INFO Level (Default)

- Model loading/unloading
- Generation completions
- Fact extraction results
- Topic segmentation
- Search queries
- Error conditions

### DEBUG Level (Enable with `JARVIS_LOG_LEVEL=DEBUG`)

- **LLM prompts and responses**
- Embedding batch sizes
- Cache hits/misses
- NLI entailment scores
- SQL queries
- Memory pressure events
- Model warm-up details

---

## Metrics Dashboard (Future)

You could build a simple metrics viewer:

```bash
# View real-time metrics
make metrics-dashboard

# Or export to Prometheus/Grafana
make export-metrics
```

This would show:

- Reply latency histogram (p50/p90/p99)
- Template hit rate over time
- Memory usage graph
- Top expensive operations
- Model load times

---

## Where Metrics Are Used

1. **`reply_service.py`** - Records all reply generation metrics
2. **`embedding/`** - Tracks embedding latency
3. **`search/vec_search.py`** - Records search latency
4. **`contacts/instruction_extractor.py`** - Fact extraction metrics
5. **`model_manager.py`** - Model load/unload times

---

## Summary

You already have **enterprise-grade observability**:

✅ Latency histograms (p50, p90, p95, p99)  
✅ Memory tracking (RSS, VMS, peak)  
✅ Request counting (endpoints, methods)  
✅ Template analytics (hit rate, top templates)  
✅ Generation logging (JSONL format)  
✅ DEBUG logging everywhere (enable with env var)

**No additional logging needed!** The system is already production-ready.
