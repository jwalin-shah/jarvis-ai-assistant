# JARVIS Observability Roadmap

> **Last Updated:** 2026-02-10

> **Version:** 1.0
> **Status:** Draft
> **Target:** Apple Silicon 8GB, Production-Ready Monitoring

---

## Executive Summary

This document defines a comprehensive observability strategy for the JARVIS AI assistant, covering logs, metrics, traces, SLOs, alerting, and dashboards. It provides concrete instrumentation points per subsystem with event schemas and metric names.

---

## 1. Observability Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          JARVIS Observability Stack                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Logs      │  │  Metrics    │  │   Traces    │  │   Events/Events     │ │
│  │  (struct)   │  │(Prometheus) │  │  (OTel)     │  │    (structured)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┴─────────────────┘                    │          │
│                           │                                      │          │
│              ┌────────────┴────────────┐                        │          │
│              │    OpenTelemetry SDK    │◄───────────────────────┘          │
│              └────────────┬────────────┘                                   │
│                           │                                                │
│         ┌─────────────────┼─────────────────┐                              │
│         ▼                 ▼                 ▼                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐                     │
│  │  Prometheus │  │    Tempo   │  │    Loki/Vector  │                     │
│  │   Server    │  │  (Traces)  │  │     (Logs)      │                     │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘                     │
│         └─────────────────┴──────────────────┘                              │
│                           │                                                │
│                    ┌──────┴──────┐                                         │
│                    │   Grafana   │                                         │
│                    │  Dashboards │                                         │
│                    └─────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Subsystem Instrumentation Matrix

| Subsystem | Critical Metrics | Latency SLO | Error Rate SLO | Key Events |
|-----------|-----------------|-------------|----------------|------------|
| **API Layer** | Request rate, p99 latency | < 100ms (p99) | < 0.1% | Request start/complete, errors |
| **Models/MLX** | GPU memory, tokens/sec, load time | < 2000ms (gen) | < 1% | Model load/unload, generation |
| **Prefetch** | Queue depth, hit rate, cost saved | < 500ms | < 0.5% | Prefetch hit/miss, eviction |
| **Scheduler** | Pending sends, retry rate | < 100ms | < 0.01% | Schedule, send, fail, retry |
| **WebSocket** | Connections, msg rate, latency | < 50ms | < 0.1% | Connect, disconnect, msg |
| **Classifiers** | Inference time, cache hit rate | < 50ms | < 0.01% | Classification, fallback |
| **Database** | Query time, connection pool | < 20ms | < 0.01% | Query slow, connection max |
| **iMessage** | Read latency, sync status | < 100ms | < 0.1% | Access grant/revoke, sync |

---

## 3. Metric Definitions

### 3.1 Core Metrics (Prometheus Format)

#### API Metrics

```yaml
# Request counter with labels
jarvis_api_requests_total:
  type: counter
  labels: [method, endpoint, status_code]
  description: Total API requests
  
jarvis_api_request_duration_seconds:
  type: histogram
  labels: [method, endpoint]
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
  description: API request latency
  
jarvis_api_requests_inflight:
  type: gauge
  labels: [method, endpoint]
  description: Current in-flight requests
  
jarvis_api_rate_limited_total:
  type: counter
  labels: [endpoint]
  description: Rate limited request count
```

#### Model/MLX Metrics

```yaml
jarvis_model_loaded:
  type: gauge
  labels: [model_id, device]
  description: Model load status (1=loaded, 0=unloaded)
  
jarvis_model_memory_bytes:
  type: gauge
  labels: [model_id, memory_type]
  # memory_type: weights, kv_cache, activations
  description: Model memory usage
  
jarvis_model_load_duration_seconds:
  type: histogram
  labels: [model_id]
  description: Model load time
  
jarvis_generation_tokens_total:
  type: counter
  labels: [model_id, finish_reason]
  description: Total tokens generated
  
jarvis_generation_tokens_per_second:
  type: gauge
  labels: [model_id]
  description: Current generation throughput
  
jarvis_generation_duration_seconds:
  type: histogram
  labels: [model_id, used_template]
  buckets: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
  description: End-to-end generation latency
  
jarvis_kv_cache_utilization:
  type: gauge
  labels: [model_id]
  description: KV cache utilization (0-1)
  
jarvis_metal_memory_bytes:
  type: gauge
  description: Apple Metal GPU memory usage
  
jarvis_template_match_rate:
  type: gauge
  description: Template match ratio (0-1)
```

#### Prefetch Metrics

```yaml
jarvis_prefetch_predictions_total:
  type: counter
  labels: [prediction_type, result]
  # prediction_type: draft_reply, embedding, contact_profile, model_warm, search_results, vec_index
  # result: hit, miss, expired, rejected
  description: Prefetch prediction outcomes
  
jarvis_prefetch_queue_size:
  type: gauge
  description: Current prefetch queue depth
  
jarvis_prefetch_latency_seconds:
  type: histogram
  labels: [prediction_type]
  description: Prefetch execution latency
  
jarvis_prefetch_cache_hit_rate:
  type: gauge
  labels: [tier]
  # tier: l1, l2, l3
  description: Cache hit rate by tier
  
jarvis_prefetch_cost_saved_ms:
  type: counter
  description: Cumulative latency saved by prefetching
  
jarvis_prefetch_resource_blocked_total:
  type: counter
  labels: [resource]
  # resource: memory, cpu, battery
  description: Prefetch blocked due to resource constraints
```

#### Scheduler Metrics

```yaml
jarvis_scheduler_items_total:
  type: counter
  labels: [status]
  # status: scheduled, sending, sent, failed, cancelled, expired
  description: Scheduled item state transitions
  
jarvis_scheduler_pending_items:
  type: gauge
  description: Currently scheduled items
  
jarvis_scheduler_retry_count:
  type: histogram
  description: Distribution of retry attempts
  
jarvis_scheduler_latency_seconds:
  type: histogram
  labels: [operation]
  # operation: schedule, send, cancel
  description: Scheduler operation latency
  
jarvis_scheduler_missed_schedules:
  type: counter
  description: Schedules missed (overdue > threshold)
```

#### WebSocket Metrics

```yaml
jarvis_websocket_connections:
  type: gauge
  labels: [status]
  # status: active, health_subscribers
  description: Current WebSocket connections
  
jarvis_websocket_connection_duration_seconds:
  type: histogram
  description: Connection lifetime distribution
  
jarvis_websocket_messages_total:
  type: counter
  labels: [direction, message_type]
  # direction: inbound, outbound
  # message_type: generate, token, health_update, error, etc.
  description: WebSocket message count
  
jarvis_websocket_generation_latency_seconds:
  type: histogram
  labels: [streaming]
  description: Generation request latency
  
jarvis_websocket_rate_limited_total:
  type: counter
  labels: [client_id_hash]
  description: Per-client rate limit hits
```

#### Classifier Metrics

```yaml
jarvis_classifier_inference_seconds:
  type: histogram
  labels: [classifier, method]
  # classifier: category, mobilization, relationship, intent
  # method: fast_path, ml_model, heuristic, cascade
  description: Classification latency
  
jarvis_classifier_cache_hit_rate:
  type: gauge
  labels: [classifier]
  description: Classification cache effectiveness
  
jarvis_classifier_confidence:
  type: histogram
  labels: [classifier, category]
  buckets: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  description: Classification confidence distribution
  
jarvis_classifier_fallback_total:
  type: counter
  labels: [classifier, reason]
  description: Fallback to default classification
```

#### Database Metrics

```yaml
jarvis_db_query_duration_seconds:
  type: histogram
  labels: [operation, table]
  buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
  description: Database query latency
  
jarvis_db_connections_active:
  type: gauge
  description: Active database connections
  
jarvis_db_connections_max:
  type: gauge
  description: Maximum connection pool size
  
jarvis_db_slow_queries_total:
  type: counter
  labels: [operation, table]
  description: Queries exceeding threshold (100ms)
```

#### System/Health Metrics

```yaml
jarvis_process_memory_bytes:
  type: gauge
  labels: [type]
  # type: rss, vms, shared, text, data
  description: Process memory usage
  
jarvis_system_memory_available_bytes:
  type: gauge
  description: System available memory
  
jarvis_process_cpu_percent:
  type: gauge
  description: Process CPU utilization
  
jarvis_process_open_fds:
  type: gauge
  description: Open file descriptors
  
jarvis_health_status:
  type: gauge
  labels: [component]
  # component: imessage, model, permissions, memory
  # value: 0=unhealthy, 1=degraded, 2=healthy
  description: Component health status
  
jarvis_uptime_seconds:
  type: counter
  description: Process uptime
```

---

## 4. Structured Logging Schema

### 4.1 Log Levels and Usage

| Level | Usage | Retention |
|-------|-------|-----------|
| `DEBUG` | Detailed flow tracing, variable dumps | 7 days |
| `INFO` | Business events, state changes | 30 days |
| `WARNING` | Recoverable issues, degraded performance | 90 days |
| `ERROR` | Failed operations, exceptions | 1 year |
| `CRITICAL` | Data loss, system failure | Permanent |

### 4.2 Log Entry Schema

```json
{
  "timestamp": "2026-02-10T10:03:57.685Z",
  "level": "INFO",
  "logger": "jarvis.api.generation",
  "message": "Generation completed successfully",
  "trace_id": "abc123-def456",
  "span_id": "span789",
  "service": "jarvis-api",
  "version": "1.2.3",
  "environment": "production",
  "context": {
    "request_id": "req-uuid",
    "user_id_hash": "sha256:abcd...",
    "chat_id_hash": "sha256:efgh..."
  },
  "event": {
    "type": "generation.complete",
    "category": "model_inference"
  },
  "metrics": {
    "latency_ms": 1250,
    "tokens_generated": 42,
    "tokens_per_second": 33.6
  },
  "metadata": {
    "model_id": "mlx-community/Qwen2.5-1.5B",
    "used_template": false,
    "finish_reason": "stop"
  }
}
```

### 4.3 Key Log Events by Subsystem

#### API Layer Events

```yaml
api.request.start:
  level: DEBUG
  fields: [method, path, query_params, client_ip]
  
api.request.complete:
  level: INFO
  fields: [method, path, status_code, latency_ms, response_size]
  
api.request.error:
  level: ERROR
  fields: [method, path, error_type, error_message, stack_trace]
  
api.rate_limit.hit:
  level: WARNING
  fields: [client_ip, endpoint, limit, window]
```

#### Model/Generation Events

```yaml
model.load.start:
  level: INFO
  fields: [model_id, device, memory_requested_mb]
  
model.load.complete:
  level: INFO
  fields: [model_id, load_duration_ms, memory_used_mb]
  
model.load.failed:
  level: ERROR
  fields: [model_id, error, memory_available_mb]
  
model.unload:
  level: INFO
  fields: [model_id, reason, memory_freed_mb]
  
generation.start:
  level: INFO
  fields: [request_id, model_id, max_tokens, temperature, prompt_tokens]
  
generation.token:
  level: DEBUG
  fields: [request_id, token_index, token_length]
  
generation.complete:
  level: INFO
  fields: [request_id, tokens_generated, generation_time_ms, finish_reason]
  
generation.failed:
  level: ERROR
  fields: [request_id, error, loaded_for_request]
  
template.match:
  level: INFO
  fields: [template_name, similarity_score, query_hash]
  
template.miss:
  level: DEBUG
  fields: [best_similarity, query_hash]
```

#### Prefetch Events

```yaml
prefetch.schedule:
  level: DEBUG
  fields: [prediction_type, key, priority, ttl_seconds]
  
prefetch.execute.start:
  level: DEBUG
  fields: [prediction_type, key, queue_wait_ms]
  
prefetch.execute.complete:
  level: INFO
  fields: [prediction_type, key, execution_ms, cached, result_size]
  
prefetch.hit:
  level: INFO
  fields: [prediction_type, key, tier, age_seconds]
  
prefetch.miss:
  level: DEBUG
  fields: [prediction_type, key, reason]
  
prefetch.evict:
  level: DEBUG
  fields: [tier, key, age_seconds, access_count]
  
prefetch.resource_blocked:
  level: WARNING
  fields: [resource, threshold, current_value]
```

#### Scheduler Events

```yaml
scheduler.item.scheduled:
  level: INFO
  fields: [item_id, contact_id, send_at, priority]
  
scheduler.item.send.start:
  level: INFO
  fields: [item_id, attempt_number, retry_count]
  
scheduler.item.send.complete:
  level: INFO
  fields: [item_id, send_duration_ms]
  
scheduler.item.failed:
  level: ERROR
  fields: [item_id, error, final_failure]
  
scheduler.item.cancelled:
  level: INFO
  fields: [item_id, reason]
  
scheduler.item.expired:
  level: WARNING
  fields: [item_id, hours_overdue]
  
scheduler.missed.processed:
  level: INFO
  fields: [count_on_startup]
```

#### WebSocket Events

```yaml
websocket.connect:
  level: INFO
  fields: [client_id, client_ip, auth_method, user_agent]
  
websocket.disconnect:
  level: INFO
  fields: [client_id, duration_seconds, messages_exchanged]
  
websocket.message.receive:
  level: DEBUG
  fields: [client_id, message_type, payload_size]
  
websocket.message.send:
  level: DEBUG
  fields: [client_id, message_type, payload_size]
  
websocket.generation.start:
  level: INFO
  fields: [client_id, generation_id, streaming]
  
websocket.generation.cancel:
  level: INFO
  fields: [client_id, generation_id, tokens_generated]
  
websocket.error:
  level: ERROR
  fields: [client_id, error_type, error_message]
  
websocket.capacity.rejected:
  level: WARNING
  fields: [client_ip, max_connections]
```

#### Classifier Events

```yaml
classifier.inference.start:
  level: DEBUG
  fields: [classifier, input_hash, context_length]
  
classifier.inference.complete:
  level: INFO
  fields: [classifier, result, confidence, method, latency_ms]
  
classifier.cache.hit:
  level: DEBUG
  fields: [classifier, cache_key, age_seconds]
  
classifier.fallback:
  level: WARNING
  fields: [classifier, reason, input_hash]
```

#### Database Events

```yaml
db.query.start:
  level: DEBUG
  fields: [operation, table, query_hash]
  
db.query.complete:
  level: DEBUG
  fields: [operation, table, latency_ms, rows_affected]
  
db.query.slow:
  level: WARNING
  fields: [operation, table, latency_ms, query_hash]
  
db.connection.max:
  level: ERROR
  fields: [pool_size, waiters]
  
db.transaction.start:
  level: DEBUG
  fields: [transaction_id]
  
db.transaction.complete:
  level: DEBUG
  fields: [transaction_id, committed]
```

---

## 5. Distributed Tracing

### 5.1 Trace Context Propagation

```python
# W3C Trace Context headers
traceparent: "00-abc123-def456-01"  # version-trace_id-span_id-flags
tracestate: "jarvis=abc123:def456"
```

### 5.2 Span Naming Convention

```yaml
Spans:
  # API Layer
  - "api.{method}.{path}"                    # e.g., api.POST.generation
  - "api.middleware.auth"
  - "api.middleware.rate_limit"
  
  # Generation
  - "generation.request"
  - "generation.template_match"
  - "generation.model_inference"
  - "generation.token_stream"
  - "generation.rag_retrieval"
  
  # RAG/Search
  - "rag.embed_query"
  - "rag.vector_search"
  - "rag.rerank"
  - "rag.build_context"
  
  # Classification
  - "classifier.category"
  - "classifier.mobilization"
  - "classifier.relationship"
  
  # Database
  - "db.query.{operation}.{table}"
  - "db.transaction"
  
  # External
  - "external.imessage.read"
  - "external.imessage.send"
```

### 5.3 Span Attributes

```yaml
Common Attributes:
  service.name: "jarvis-api"
  service.version: "1.2.3"
  deployment.environment: "production"
  host.name: "macbook-pro-001"
  process.pid: 12345

Request Attributes:
  http.method: "POST"
  http.route: "/api/v1/generation"
  http.status_code: 200
  http.request.body.size: 1024
  http.response.body.size: 512
  user_agent.original: "JARVIS-Desktop/1.2.3"
  client.address: "127.0.0.1"

Generation Attributes:
  jarvis.generation.model_id: "mlx-community/Qwen2.5-1.5B"
  jarvis.generation.max_tokens: 100
  jarvis.generation.temperature: 0.7
  jarvis.generation.tokens_generated: 42
  jarvis.generation.used_template: false
  jarvis.generation.finish_reason: "stop"

Classification Attributes:
  jarvis.classifier.type: "category"
  jarvis.classifier.result: "question"
  jarvis.classifier.confidence: 0.87
  jarvis.classifier.method: "lightgbm"
```

---

## 6. Service Level Objectives (SLOs)

### 6.1 Critical SLOs

| SLO | Target | Window | Alert Threshold |
|-----|--------|--------|-----------------|
| API Availability | 99.9% | 30 days | 99.5% |
| Generation p99 Latency | < 3s | 1 hour | 5s |
| Generation p50 Latency | < 800ms | 1 hour | 1.5s |
| Classification p99 Latency | < 100ms | 1 hour | 200ms |
| Model Load Success Rate | 99.5% | 7 days | 98% |
| Scheduled Send Success | 99.9% | 7 days | 99% |
| WebSocket Connection Stability | 99.5% | 1 day | 99% |
| Memory Usage | < 6GB | 5 min | 7GB |

### 6.2 SLO Error Budgets

```yaml
Error Budget Calculation:
  availability: 0.1% of requests can fail (43m downtime/month)
  latency: 1% of requests can exceed target
  
Alerting:
  burn_rate_fast: 14.4x (2% budget in 1 hour)
  burn_rate_slow: 2x (5% budget in 6 hours)
```

---

## 7. Alerting Rules

### 7.1 Critical Alerts (Page Immediately)

```yaml
alerts:
  - name: HighErrorRate
    expr: |
      (
        sum(rate(jarvis_api_requests_total{status_code=~"5.."}[5m]))
        /
        sum(rate(jarvis_api_requests_total[5m]))
      ) > 0.01
    for: 2m
    severity: critical
    summary: "High error rate detected"
    
  - name: ModelLoadFailure
    expr: |
      increase(jarvis_model_load_failed_total[5m]) > 3
    for: 1m
    severity: critical
    summary: "Model load failures detected"
    
  - name: MemoryCritical
    expr: |
      jarvis_process_memory_bytes{type="rss"} > 7e9
    for: 30s
    severity: critical
    summary: "Process memory critically high"
    
  - name: SchedulerSendFailure
    expr: |
      increase(jarvis_scheduler_items_total{status="failed"}[5m]) > 5
    for: 2m
    severity: critical
    summary: "Multiple scheduled sends failed"
```

### 7.2 Warning Alerts (Notify, No Page)

```yaml
alerts:
  - name: HighLatencyP99
    expr: |
      histogram_quantile(0.99,
        rate(jarvis_api_request_duration_seconds_bucket[5m])
      ) > 0.5
    for: 5m
    severity: warning
    summary: "API p99 latency elevated"
    
  - name: LowTemplateMatchRate
    expr: |
      jarvis_template_match_rate < 0.3
    for: 15m
    severity: warning
    summary: "Template match rate below expected"
    
  - name: PrefetchQueueBacklog
    expr: |
      jarvis_prefetch_queue_size > 50
    for: 10m
    severity: warning
    summary: "Prefetch queue building up"
    
  - name: WebSocketConnectionSpike
    expr: |
      jarvis_websocket_connections{status="active"} > 80
    for: 5m
    severity: warning
    summary: "WebSocket connections near capacity"
    
  - name: DatabaseSlowQueries
    expr: |
      increase(jarvis_db_slow_queries_total[5m]) > 10
    for: 5m
    severity: warning
    summary: "Slow database queries detected"
    
  - name: iMessageAccessLost
    expr: |
      jarvis_health_status{component="imessage"} == 0
    for: 1m
    severity: warning
    summary: "iMessage database access lost"
```

### 7.3 Informational Alerts

```yaml
alerts:
  - name: ModelUnloaded
    expr: |
      jarvis_model_loaded == 0
    for: 0s
    severity: info
    summary: "Model has been unloaded"
    
  - name: GenerationFallbackToTemplate
    expr: |
      increase(jarvis_generation_duration_seconds_count{used_template="true"}[1h]) > 100
    for: 0s
    severity: info
    summary: "Template fallback usage detected"
```

---

## 8. Dashboard Definitions

### 8.1 Executive Summary Dashboard

```yaml
dashboard: "JARVIS - Executive Summary"
refresh: 30s
panels:
  - title: "API Availability (30d)"
    type: stat
    query: |
      1 - (
        sum(rate(jarvis_api_requests_total{status_code=~"5.."}[30d]))
        /
        sum(rate(jarvis_api_requests_total[30d]))
      )
    
  - title: "Error Budget Remaining"
    type: gauge
    query: |
      1 - (
        sum(increase(jarvis_api_requests_total{status_code=~"5.."}[30d]))
        /
        (sum(increase(jarvis_api_requests_total[30d])) * 0.001)
      )
    
  - title: "Avg Generation Latency (1h)"
    type: stat
    query: |
      rate(jarvis_generation_duration_seconds_sum[1h])
      /
      rate(jarvis_generation_duration_seconds_count[1h])
    
  - title: "Daily Active Users"
    type: stat
    query: |
      count(count by (user_id_hash) (jarvis_api_requests_total[1d]))
    
  - title: "Model Load Status"
    type: stat
    query: jarvis_model_loaded
    
  - title: "Memory Usage"
    type: timeseries
    queries:
      - jarvis_process_memory_bytes{type="rss"}
      - jarvis_metal_memory_bytes
```

### 8.2 API Performance Dashboard

```yaml
dashboard: "JARVIS - API Performance"
refresh: 10s
panels:
  - title: "Request Rate by Endpoint"
    type: timeseries
    query: |
      sum by (endpoint) (
        rate(jarvis_api_requests_total[5m])
      )
    
  - title: "Latency Percentiles"
    type: timeseries
    queries:
      - histogram_quantile(0.99, ...)
      - histogram_quantile(0.95, ...)
      - histogram_quantile(0.50, ...)
    
  - title: "Error Rate by Status"
    type: timeseries
    query: |
      sum by (status_code) (
        rate(jarvis_api_requests_total{status_code=~"[45].."}[5m])
      )
    
  - title: "Rate Limited Requests"
    type: counter
    query: jarvis_api_rate_limited_total
    
  - title: "Top 10 Slow Endpoints"
    type: table
    query: |
      topk(10,
        histogram_quantile(0.99,
          rate(jarvis_api_request_duration_seconds_bucket[1h])
        )
      )
```

### 8.3 Model/MLX Dashboard

```yaml
dashboard: "JARVIS - Model Performance"
refresh: 5s
panels:
  - title: "Model Status"
    type: stat
    query: jarvis_model_loaded
    
  - title: "Memory Breakdown"
    type: pie
    query: |
      jarvis_model_memory_bytes
    
  - title: "Generation Throughput"
    type: timeseries
    queries:
      - rate(jarvis_generation_tokens_total[1m])
      - jarvis_generation_tokens_per_second
    
  - title: "Template vs Model Usage"
    type: pie
    query: |
      sum by (used_template) (
        increase(jarvis_generation_duration_seconds_count[1h])
      )
    
  - title: "KV Cache Utilization"
    type: gauge
    query: jarvis_kv_cache_utilization
    
  - title: "Model Load/Unload Events"
    type: table
    query: |
      {__name__=~"jarvis_model_load.*"}
    
  - title: "Generation Latency Heatmap"
    type: heatmap
    query: jarvis_generation_duration_seconds_bucket
```

### 8.4 Prefetch Efficiency Dashboard

```yaml
dashboard: "JARVIS - Prefetch System"
refresh: 10s
panels:
  - title: "Cache Hit Rate by Tier"
    type: timeseries
    query: jarvis_prefetch_cache_hit_rate
    
  - title: "Prefetch Queue Depth"
    type: timeseries
    query: jarvis_prefetch_queue_size
    
  - title: "Cost Saved by Prefetch"
    type: stat
    query: jarvis_prefetch_cost_saved_ms
    
  - title: "Prediction Type Distribution"
    type: pie
    query: |
      sum by (prediction_type) (
        jarvis_prefetch_predictions_total
      )
    
  - title: "Resource Blocks"
    type: timeseries
    query: jarvis_prefetch_resource_blocked_total
    
  - title: "Prefetch Latency Distribution"
    type: heatmap
    query: jarvis_prefetch_latency_seconds_bucket
```

### 8.5 Scheduler Dashboard

```yaml
dashboard: "JARVIS - Message Scheduler"
refresh: 30s
panels:
  - title: "Pending Sends"
    type: stat
    query: jarvis_scheduler_pending_items
    
  - title: "Items by Status"
    type: timeseries
    query: |
      sum by (status) (
        rate(jarvis_scheduler_items_total[5m])
      )
    
  - title: "Retry Distribution"
    type: histogram
    query: jarvis_scheduler_retry_count
    
  - title: "Upcoming Sends (Next Hour)"
    type: table
    query: |
      scheduler_items{status="scheduled", send_at < now() + 3600}
    
  - title: "Missed Schedules"
    type: counter
    query: jarvis_scheduler_missed_schedules
```

### 8.6 WebSocket Dashboard

```yaml
dashboard: "JARVIS - WebSocket Connections"
refresh: 5s
panels:
  - title: "Active Connections"
    type: stat
    query: jarvis_websocket_connections{status="active"}
    
  - title: "Connection History"
    type: timeseries
    queries:
      - jarvis_websocket_connections{status="active"}
      - jarvis_websocket_connections{status="health_subscribers"}
    
  - title: "Message Rate"
    type: timeseries
    query: |
      sum by (direction, message_type) (
        rate(jarvis_websocket_messages_total[1m])
      )
    
  - title: "Generation Latency"
    type: timeseries
    query: |
      histogram_quantile(0.99,
        rate(jarvis_websocket_generation_latency_seconds_bucket[5m])
      )
    
  - title: "Rate Limited Clients"
    type: table
    query: jarvis_websocket_rate_limited_total
    
  - title: "Connection Duration Distribution"
    type: heatmap
    query: jarvis_websocket_connection_duration_seconds_bucket
```

### 8.7 System Health Dashboard

```yaml
dashboard: "JARVIS - System Health"
refresh: 10s
panels:
  - title: "Component Health"
    type: stat
    query: jarvis_health_status
    
  - title: "Process Memory"
    type: timeseries
    queries:
      - jarvis_process_memory_bytes{type="rss"}
      - jarvis_system_memory_available_bytes
    
  - title: "CPU Usage"
    type: timeseries
    query: jarvis_process_cpu_percent
    
  - title: "Open File Descriptors"
    type: timeseries
    query: jarvis_process_open_fds
    
  - title: "Uptime"
    type: stat
    query: jarvis_uptime_seconds
    
  - title: "iMessage Access Status"
    type: stat
    query: jarvis_health_status{component="imessage"}
```

---

## 9. Instrumentation Implementation Guide

### 9.1 Python Instrumentation Pattern

```python
# jarvis/observability/instrumentation.py
"""Centralized observability instrumentation for JARVIS."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import Counter, Histogram, Gauge, Info

# Prometheus metrics (for backward compatibility)
METRICS = {
    "api_requests": Counter(
        "jarvis_api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status_code"]
    ),
    "api_latency": Histogram(
        "jarvis_api_request_duration_seconds",
        "API request latency",
        ["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ),
    "model_memory": Gauge(
        "jarvis_model_memory_bytes",
        "Model memory usage",
        ["model_id", "memory_type"]
    ),
    "generation_latency": Histogram(
        "jarvis_generation_duration_seconds",
        "Generation latency",
        ["model_id", "used_template"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    ),
}

# OpenTelemetry tracer
tracer = trace.get_tracer("jarvis")


def init_observability(
    service_name: str = "jarvis-api",
    service_version: str = "1.0.0",
    prometheus_port: int = 9090,
) -> None:
    """Initialize observability stack."""
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    })
    
    # Traces
    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    
    # Metrics
    reader = PrometheusMetricReader()
    metrics_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(metrics_provider)
    
    # Start Prometheus HTTP server
    from prometheus_client import start_http_server
    start_http_server(prometheus_port)


@contextmanager
def timed_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    counter: Counter | None = None,
    histogram: Histogram | None = None,
    labels: list[str] | None = None,
):
    """Context manager for timed spans with metrics."""
    start = time.perf_counter()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
            status = "success"
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            if histogram and labels:
                histogram.labels(*labels).observe(duration)
            if counter and labels:
                counter.labels(*labels).inc()


def trace_method(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Decorator to trace method execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or func.__qualname__
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                return func(*args, **kwargs)
        return wrapper
    return decorator


def record_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    """Record a metric value."""
    if name in METRICS:
        metric = METRICS[name]
        if isinstance(metric, Gauge):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        elif isinstance(metric, Counter):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
```

### 9.2 Subsystem-Specific Instrumentation

#### API Router Instrumentation

```python
# api/routers/generation.py - Instrumented example

from fastapi import APIRouter, Request
from jarvis.observability.instrumentation import (
    timed_span, tracer, record_metric, METRICS
)

router = APIRouter()

@router.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    """Generate a reply with full observability."""
    start_time = time.perf_counter()
    
    with tracer.start_as_current_span("api.generate") as span:
        span.set_attribute("http.method", "POST")
        span.set_attribute("chat.id_hash", hashlib.sha256(
            body.chat_id.encode()
        ).hexdigest()[:16])
        
        try:
            # Business logic
            with tracer.start_as_current_span("generation.pipeline"):
                result = await process_generation(body)
            
            # Record success metrics
            duration = time.perf_counter() - start_time
            METRICS["api_requests"].labels(
                method="POST",
                endpoint="/generate",
                status_code="200"
            ).inc()
            METRICS["api_latency"].labels(
                method="POST",
                endpoint="/generate"
            ).observe(duration)
            
            span.set_attribute("generation.tokens", result.tokens_used)
            span.set_attribute("generation.duration_ms", duration * 1000)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            METRICS["api_requests"].labels(
                method="POST",
                endpoint="/generate",
                status_code="500"
            ).inc()
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
```

#### Model Instrumentation

```python
# models/generator.py - Instrumented example

from jarvis.observability.instrumentation import tracer, METRICS

class MLXGenerator:
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        with tracer.start_as_current_span("generation.request") as span:
            span.set_attribute("model.id", self.config.model_path)
            span.set_attribute("generation.max_tokens", request.max_tokens)
            
            start = time.perf_counter()
            
            # Template matching
            with tracer.start_as_current_span("generation.template_match"):
                template_response = self._try_template_match(request)
                if template_response:
                    span.set_attribute("generation.used_template", True)
                    METRICS["generation_latency"].labels(
                        model_id=self.config.model_path,
                        used_template="true"
                    ).observe(time.perf_counter() - start)
                    return template_response
            
            # Model inference
            with tracer.start_as_current_span("generation.model_inference") as inf_span:
                result = self._generate_with_model(request, start)
                inf_span.set_attribute("generation.tokens", result.tokens_used)
            
            duration = time.perf_counter() - start
            METRICS["generation_latency"].labels(
                model_id=self.config.model_path,
                used_template="false"
            ).observe(duration)
            
            span.set_attribute("generation.used_template", False)
            span.set_attribute("generation.duration_ms", duration * 1000)
            
            return result
```

---

## 10. Deployment Configuration

### 10.1 Docker Compose (Local Development)

```yaml
# docker-compose.observability.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  tempo:
    image: grafana/tempo:2.3.0
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    volumes:
      - ./config/tempo.yml:/etc/tempo.yml
      - tempo_data:/tmp/tempo
    command: ["-config.file=/etc/tempo.yml"]

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - ./config/loki.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

volumes:
  prometheus_data:
  grafana_data:
  tempo_data:
  loki_data:
```

### 10.2 Prometheus Configuration

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'jarvis-api'
    static_configs:
      - targets: ['host.docker.internal:8742']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

## 11. Appendix

### 11.1 Metric Naming Conventions

```
jarvis_{subsystem}_{metric}_{unit}

Subsystems:
  - api
  - model
  - generation
  - prefetch
  - scheduler
  - websocket
  - classifier
  - db
  - health

Units:
  - _total (counter)
  - _seconds (histogram/timer)
  - _bytes (gauge)
  - _count (gauge)
  - _ratio / _rate (gauge, 0-1)
```

### 11.2 Label Naming Conventions

```
General:
  - service, version, environment
  - instance, host
  
Request:
  - method, endpoint, status_code
  
Model:
  - model_id, device, memory_type
  
Classification:
  - classifier, category, method
```

### 11.3 Privacy Considerations

```yaml
Sensitive Data Handling:
  - Hash user identifiers (SHA-256, first 16 chars)
  - Truncate message content in logs
  - Exclude PII from metrics labels
  - Use query_hash instead of raw query
  - Anonymize client IPs (last octet)

Log Retention:
  - DEBUG: 7 days
  - INFO: 30 days  
  - WARNING: 90 days
  - ERROR: 1 year
  - CRITICAL: Permanent
```

---

## 12. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Prometheus + Grafana infrastructure
- [ ] Implement core metrics (api, model, system)
- [ ] Configure basic logging with structured format
- [ ] Create Executive Summary dashboard

### Phase 2: Core Subsystems (Week 3-4)
- [ ] Instrument classifiers with latency/confidence metrics
- [ ] Add prefetch observability (queue, hit rate, cost)
- [ ] Implement scheduler monitoring
- [ ] Create subsystem-specific dashboards

### Phase 3: Advanced Features (Week 5-6)
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Implement WebSocket metrics
- [ ] Configure alerting rules
- [ ] Set up log aggregation (Loki)

### Phase 4: Refinement (Week 7-8)
- [ ] Define and validate SLOs
- [ ] Create runbooks for alerts
- [ ] Performance optimization based on metrics
- [ ] Documentation and team training

---

*End of Observability Roadmap*
