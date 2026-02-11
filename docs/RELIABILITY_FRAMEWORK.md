# JARVIS Local-First Product Reliability Validation Framework

**Version:** 1.0.0  
**Last Updated:** 2026-02-10  
**Status:** Active Implementation

---

## Executive Summary

This document defines the comprehensive reliability validation framework for JARVIS, a local-first AI assistant. The framework ensures consistent, predictable behavior under all operating conditions including offline modes, degraded states, and system stress scenarios.

### Core Principles

1. **Local-First**: All critical features work without network connectivity
2. **Graceful Degradation**: Features reduce capability rather than fail completely
3. **Transparent UX**: Users always understand system state and available capabilities
4. **Self-Healing**: Automatic recovery from transient failures
5. **Deterministic**: Same inputs produce same outputs regardless of system state

---

## 1. Offline-Mode Behavior Validation

### 1.1 Offline Mode Definition

JARVIS operates in three connectivity tiers:

| Tier | Condition | Capabilities |
|------|-----------|--------------|
| **Full** | All services available | 100% feature set |
| **Degraded** | Model unavailable / Memory pressure | Core features, no AI generation |
| **Offline** | API/Socket disconnected | Read-only local data access |

### 1.2 Component Offline Behavior Matrix

#### 1.2.1 AI Model (MLX) Unavailable

```python
# Behavior: Degraded Mode
When: Model not loaded AND cannot load due to memory/permission
Then:
  - Draft generation → Fallback suggestions + explanation
  - Summary generation → "Unable to summarize" message
  - Intent classification → Rule-based fallback
  - Search → Keyword-only search (no semantic)
```

**Validation Criteria:**
- [ ] Draft endpoint returns HTTP 503 with `Retry-After: 30`
- [ ] Fallback suggestions are contextually appropriate
- [ ] UI displays "AI features unavailable" banner
- [ ] Search automatically falls back to keyword mode

#### 1.2.2 API Server Unavailable

```python
# Behavior: Offline Mode
When: API at localhost:8742 not responding
Then:
  - Desktop app uses direct SQLite access
  - Read operations work (conversations, messages)
  - Write operations queued for retry
  - AI features disabled
```

**Validation Criteria:**
- [ ] Desktop app detects disconnection within 5 seconds
- [ ] Conversation list loads from local cache
- [ ] New messages show "pending sync" indicator
- [ ] Automatic reconnection with exponential backoff

#### 1.2.3 Socket Server Disconnected

```python
# Behavior: Fallback to HTTP/WebSocket
When: Unix socket at ~/.jarvis/jarvis.sock unavailable
Then:
  - Tauri app falls back to WebSocket
  - WebSocket falls back to HTTP polling
  - Streaming disabled, polling only
```

**Validation Criteria:**
- [ ] Connection fallback chain completes within 10 seconds
- [ ] Streaming gracefully degrades to polling
- [ ] No data loss during transition

#### 1.2.4 iMessage Database Inaccessible

```python
# Behavior: Offline Mode
When: Full Disk Access denied or chat.db locked
Then:
  - Show permission instructions
  - Allow manual message entry
  - Cache previously loaded conversations
```

**Validation Criteria:**
- [ ] Clear permission error with fix instructions
- [ ] Previously cached conversations remain visible
- [ ] Manual message composition still works

### 1.3 Offline Data Synchronization

#### Sync Queue Strategy

```python
class OfflineSyncQueue:
    """Queue for operations performed while offline."""
    
    PRIORITY_ORDER = [
        SyncOp.MESSAGE_SEND,      # Highest - user-initiated
        SyncOp.MESSAGE_READ,      # Mark as read
        SyncOp.ANALYTICS_EVENT,   # Lowest - can be dropped
    ]
    
    MAX_QUEUE_SIZE = 1000
    MAX_RETRY_ATTEMPTS = 5
    RETRY_BACKOFF = [1, 5, 15, 60, 300]  # seconds
```

**Validation Criteria:**
- [ ] Queue persists across app restarts
- [ ] Operations execute in priority order on reconnection
- [ ] Failed operations retry with exponential backoff
- [ ] Queue size limit prevents unbounded growth

---

## 2. Degraded-Mode UX Policy

### 2.1 UX State Machine

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   HEALTHY   │───→│  DEGRADED   │───→│   FAILED    │
│  (Green)    │    │   (Amber)   │    │    (Red)    │
└─────────────┘    └─────────────┘    └─────────────┘
       ↑                  │                  │
       └──────────────────┴──────────────────┘
                    (Auto-recovery)
```

### 2.2 Visual Indicators

#### 2.2.1 Health Status Banner

```svelte
<!-- HealthStatus.svelte implementation -->
{#if healthStatus === 'degraded'}
  <div class="health-banner degraded" role="alert">
    <Icon name="warning" />
    <span>Some features are limited. AI responses may be slower.</span>
    <button on:click={showDetails}>Details</button>
  </div>
{/if}
```

**Policy Requirements:**
- [ ] Banner appears within 2 seconds of state change
- [ ] Color coding: Green (healthy), Amber (degraded), Red (failed)
- [ ] Dismissible but reappears on state change
- [ ] "Details" button explains specific limitations

#### 2.2.2 Feature-Level Indicators

| Feature | Healthy | Degraded | Failed |
|---------|---------|----------|--------|
| Draft Gen | Full sparkle icon | Dimmed icon + tooltip | Hidden / disabled |
| Search | Search icon | Icon + "keyword only" label | "Search unavailable" |
| Summary | Summary button | Loading state slower | Button disabled |

### 2.3 User Messaging Guidelines

#### 2.3.1 Message Tone Guidelines

```yaml
Healthy:
  tone: "Encouraging, proactive"
  examples:
    - "AI model ready"
    - "All features available"

Degraded:
  tone: "Informative, reassuring"
  examples:
    - "AI features running slower than usual"
    - "Using backup search mode"
    - "Some suggestions may be generic"

Failed:
  tone: "Clear, actionable"
  examples:
    - "AI features unavailable. Free up memory?"
    - "Cannot access messages. Grant permission?"
    - "Connection lost. Retrying..."
```

#### 2.3.2 Actionable Suggestions

```python
DEGRADED_SUGGESTIONS = {
    FailureReason.MEMORY_PRESSURE: {
        "message": "System memory is low",
        "actions": [
            {"label": "Free Up Memory", "action": "open_activity_monitor"},
            {"label": "Use Smaller Model", "action": "switch_model:0.5B"},
        ]
    },
    FailureReason.MODEL_LOAD_FAILED: {
        "message": "AI model failed to load",
        "actions": [
            {"label": "Retry", "action": "reload_model"},
            {"label": "Check Memory", "action": "show_health"},
        ]
    },
}
```

### 2.4 Input Handling During Degraded States

#### 2.4.1 Draft Generation

```typescript
// Policy: Never block user input
async function generateDraft(params: DraftParams): Promise<DraftResult> {
  const health = await checkHealth();
  
  if (health.status === 'failed') {
    // Return fallback immediately, don't wait
    return {
      suggestions: getFallbackSuggestions(),
      degraded: true,
      message: "AI features temporarily unavailable"
    };
  }
  
  if (health.status === 'degraded') {
    // Try with shorter timeout
    return await Promise.race([
      callModel(params),
      timeout(5000).then(() => ({
        suggestions: getFallbackSuggestions(),
        degraded: true
      }))
    ]);
  }
  
  return await callModel(params);
}
```

#### 2.4.2 Search

```typescript
// Policy: Always return something useful
async function search(query: string): Promise<SearchResult> {
  try {
    if (await isSemanticSearchAvailable()) {
      return await semanticSearch(query);
    }
  } catch (e) {
    logger.warn("Semantic search failed, falling back");
  }
  
  // Always fallback to keyword search
  return await keywordSearch(query);
}
```

---

## 3. Resilience Test Suite Strategy

### 3.1 Test Pyramid

```
                    /\
                   /  \
                  / E2E \        <- Full app scenarios (10%)
                 /________\
                /          \
               / Integration \   <- Component interactions (30%)
              /______________\
             /                \
            /   Unit Tests      \ <- Core resilience logic (60%)
           /____________________\
```

### 3.2 Test Categories

#### 3.2.1 Unit Tests: Circuit Breakers

```python
# tests/reliability/test_circuit_breaker.py
class TestCircuitBreakerResilience:
    """Test circuit breaker under various failure conditions."""
    
    def test_circuit_opens_under_sustained_load(self):
        """Circuit opens when failure rate exceeds threshold."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=1
        ))
        
        # Simulate sustained failures
        for _ in range(5):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()
    
    def test_circuit_half_open_allows_probe(self):
        """HALF_OPEN state allows single test request."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.1
        ))
        
        cb.record_failure()
        time.sleep(0.15)
        
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()  # One call allowed
        assert not cb.can_execute()  # Second call blocked
```

#### 3.2.2 Unit Tests: Retry Logic

```python
# tests/reliability/test_retry_resilience.py
class TestRetryResilience:
    """Test retry mechanisms under various conditions."""
    
    def test_retry_with_jitter_prevents_thundering_herd(self):
        """Random jitter prevents synchronized retries."""
        delays = []
        
        @retry_with_backoff(max_retries=3, base_delay=0.1, jitter=True)
        def flaky():
            raise ConnectionError()
        
        # Collect actual delays from multiple runs
        for _ in range(100):
            try:
                flaky()
            except ConnectionError:
                pass
        
        # Verify jitter creates variance
        unique_delays = set(delays)
        assert len(unique_delays) > 50  # High variance
    
    def test_retry_exhaustion_preserves_context(self):
        """Original exception context preserved after retries."""
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def failing():
            raise ConnectionError("Original error")
        
        with pytest.raises(ConnectionError) as exc_info:
            failing()
        
        assert "Original error" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
```

#### 3.2.3 Integration Tests: Memory Pressure

```python
# tests/reliability/test_memory_resilience.py
class TestMemoryPressureResilience:
    """Test behavior under memory pressure."""
    
    def test_model_unloads_under_pressure(self):
        """Model unloads when memory pressure detected."""
        # Simulate memory pressure
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value = Mock(available=500 * 1024 * 1024)  # 500MB
            
            controller = DefaultMemoryController()
            
            # Should trigger model unload
            assert controller.should_unload_model()
            
            # Verify graceful degradation
            result = generate_draft_fallback()
            assert "memory" in result.message.lower()
    
    def test_batch_size_reduces_under_pressure(self):
        """Batch operations reduce size when memory low."""
        # Test adaptive batch sizing
        pass
```

#### 3.2.4 Integration Tests: Network Resilience

```python
# tests/reliability/test_network_resilience.py
class TestNetworkResilience:
    """Test behavior under network disruptions."""
    
    @pytest.mark.asyncio
    async def test_socket_reconnection(self):
        """Socket client reconnects after disconnection."""
        client = JarvisSocket()
        
        # Connect
        await client.connect()
        assert client.getState() == "connected"
        
        # Simulate disconnect
        await simulate_network_partition()
        
        # Verify auto-reconnect
        await asyncio.wait_for(
            wait_for_state(client, "connected"),
            timeout=30
        )
    
    def test_api_timeout_handling(self):
        """API requests timeout gracefully."""
        with patch('requests.get', side_effect=TimeoutError):
            response = client.get_health()
            
            # Should return cached/stale data, not error
            assert response is not None
            assert response.stale is True
```

#### 3.2.5 E2E Tests: Full Scenarios

```typescript
// tests/e2e/reliability/test_offline_scenarios.spec.ts
test.describe('Offline Scenarios', () => {
  
  test('continues working when API goes offline', async ({ page }) => {
    // 1. Load conversations while online
    await page.goto('/');
    await expect(page.locator('.conversation-list')).toBeVisible();
    
    // 2. Disconnect API
    await route.fulfill({ status: 503 });
    
    // 3. Verify offline indicator appears
    await expect(page.locator('.offline-banner')).toBeVisible();
    
    // 4. Verify cached data still accessible
    await page.click('.conversation-item:first-child');
    await expect(page.locator('.message-list')).toBeVisible();
    
    // 5. Verify AI features disabled
    await page.keyboard.press('Cmd+D');
    await expect(page.locator('.draft-panel')).toContainText('unavailable');
  });
  
  test('recovers when API comes back online', async ({ page }) => {
    // Test reconnection and sync
  });
});
```

### 3.3 Test Fixtures and Utilities

#### 3.3.1 Resilience Test Fixtures

```python
# tests/fixtures/resilience_fixtures.py
@pytest.fixture
def degraded_model():
    """Simulate model in degraded state."""
    with patch('models.get_generator') as mock:
        mock.return_value.is_loaded.return_value = False
        yield mock

@pytest.fixture
def memory_pressure():
    """Simulate memory pressure conditions."""
    with patch('psutil.virtual_memory') as mock:
        mock.return_value.available = 500 * 1024 * 1024  # 500MB
        mock.return_value.percent = 95
        yield mock

@pytest.fixture
def network_partition():
    """Simulate network partition."""
    with patch('socket.socket') as mock:
        mock.side_effect = ConnectionRefusedError()
        yield mock

@pytest.fixture
def slow_responses():
    """Simulate slow API responses."""
    original = requests.request
    
    def slow_request(*args, **kwargs):
        time.sleep(5)
        return original(*args, **kwargs)
    
    with patch('requests.request', slow_request):
        yield
```

#### 3.3.2 Chaos Testing Utilities

```python
# tests/utils/chaos_engineering.py
class ChaosMonkey:
    """Inject failures for resilience testing."""
    
    def __init__(self):
        self.failures = []
        self.enabled = False
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def random_failure(self, probability: float = 0.1):
        """Decorator that randomly fails function calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.enabled and random.random() < probability:
                    raise RandomFailure(f"Chaos injected failure in {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def latency_injection(self, min_ms: int = 100, max_ms: int = 1000):
        """Add random latency to function calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.enabled:
                    time.sleep(random.randint(min_ms, max_ms) / 1000)
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

### 3.4 Performance Under Load

#### 3.4.1 Load Test Scenarios

```python
# tests/performance/reliability_load_tests.py
class TestReliabilityUnderLoad:
    """Verify reliability under high load."""
    
    def test_circuit_breaker_under_load(self):
        """Circuit breaker handles concurrent failures correctly."""
        cb = CircuitBreaker("load_test")
        failures = []
        
        def worker():
            for _ in range(100):
                try:
                    cb.execute(lambda: (_ for _ in ()).throw(ValueError()))
                except Exception as e:
                    failures.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify circuit state is consistent
        assert cb.state in [CircuitState.CLOSED, CircuitState.OPEN]
    
    def test_queue_memory_under_load(self):
        """Task queue doesn't grow unbounded under load."""
        queue = TaskQueue(max_completed_tasks=100)
        
        # Create many tasks
        for i in range(1000):
            task = queue.enqueue(TaskType.BATCH_EXPORT, {"id": i})
            task.status = TaskStatus.COMPLETED
            queue.update(task)
        
        # Verify size is bounded
        stats = queue.get_stats()
        assert stats["total"] <= 200  # Some pending + max completed
```

---

## 4. Implementation Guidelines

### 4.1 Adding Resilience to New Features

#### Checklist for New Features

- [ ] **Circuit Breaker**: Wrap external calls with circuit breaker
- [ ] **Retry Logic**: Add retry for transient failures
- [ ] **Fallback**: Define degraded behavior
- [ ] **Timeout**: Set appropriate timeouts
- [ ] **Metrics**: Log failures and degraded usage
- [ ] **UX**: Add UI indicator for degraded state

#### Example Implementation

```python
# New feature with full resilience
from core.health.degradation import get_degradation_controller
from jarvis.retry import retry_with_backoff
from contracts.health import DegradationPolicy

# 1. Register feature with degradation policy
controller = get_degradation_controller()
controller.register_feature(DegradationPolicy(
    feature_name="smart_reply",
    health_check=lambda: model_is_loaded(),
    degraded_behavior=lambda: get_template_replies(),
    fallback_behavior=lambda: get_generic_replies(),
    recovery_check=lambda: can_load_model(),
    max_failures=3,
))

# 2. Implement with retry and circuit breaker
@retry_with_backoff(max_retries=3, base_delay=0.1)
def generate_smart_reply(context: str) -> str:
    return controller.execute("smart_reply", 
        lambda: call_model(context))
```

### 4.2 Monitoring and Alerting

#### Key Metrics

```python
RELIABILITY_METRICS = {
    # Circuit breaker metrics
    "circuit_open_count": Counter,
    "circuit_recovery_time": Histogram,
    
    # Retry metrics  
    "retry_attempts": Histogram,
    "retry_success_rate": Gauge,
    
    # Degradation metrics
    "degraded_mode_duration": Histogram,
    "fallback_usage_count": Counter,
    
    # Offline metrics
    "offline_duration": Histogram,
    "sync_queue_size": Gauge,
    "sync_success_rate": Gauge,
}
```

#### Health Check Endpoint

```python
@router.get("/health/deep")
async def deep_health_check() -> DeepHealthResponse:
    """Comprehensive health check for reliability monitoring."""
    
    checks = {
        "database": await check_database(),
        "model": await check_model(),
        "socket": await check_socket(),
        "memory": await check_memory(),
        "disk": await check_disk(),
    }
    
    return DeepHealthResponse(
        status="healthy" if all(c.passed for c in checks.values()) else "degraded",
        checks=checks,
        timestamp=datetime.utcnow(),
    )
```

---

## 5. Validation Gates

### 5.1 CI/CD Reliability Gates

```yaml
# .github/workflows/reliability-gates.yml
name: Reliability Gates

on: [push, pull_request]

jobs:
  resilience-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Circuit Breaker Tests
        run: pytest tests/reliability/test_circuit_breaker.py -v
      
      - name: Run Retry Logic Tests
        run: pytest tests/reliability/test_retry_resilience.py -v
      
      - name: Run Memory Pressure Tests
        run: pytest tests/reliability/test_memory_resilience.py -v
      
      - name: Run Network Resilience Tests
        run: pytest tests/reliability/test_network_resilience.py -v
      
      - name: Run Chaos Tests
        run: pytest tests/reliability/test_chaos.py --chaos-mode -v
```

### 5.2 Release Criteria

Before each release, verify:

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Circuit breaker recovery | < 60s | Average recovery time |
| Retry success rate | > 95% | After 3 retries |
| Degraded mode UX | 100% | All features have fallback |
| Offline data loss | 0% | Operations queued successfully |
| Recovery time | < 30s | From failed to healthy |

---

## 6. References

- [AGENTS.md](../AGENTS.md) - Project coding standards
- [TEST_FLAKE_ERADICATION_PROGRAM.md](TEST_FLAKE_ERADICATION_PROGRAM.md) - Test reliability
- [DESIGN.md](DESIGN.md) - System architecture
- Circuit breaker implementation: `core/health/circuit.py`
- Degradation controller: `core/health/degradation.py`
- Fallback responses: `jarvis/fallbacks.py`

---

## 7. Appendix: Test Execution

### Running Reliability Tests

```bash
# Run all reliability tests
make test-reliability

# Run specific test categories
pytest tests/reliability/test_circuit_breaker.py -v
pytest tests/reliability/test_memory_resilience.py -v
pytest tests/reliability/test_network_resilience.py -v

# Run with chaos mode
pytest tests/reliability/ --chaos-mode -v

# Run stress tests
pytest tests/reliability/test_stress.py --stress-mode -v
```

### Test Data Generators

```python
# Generate test scenarios
python -m tests.reliability.generate_scenarios \
    --output tests/reliability/scenarios/ \
    --count 100
```
