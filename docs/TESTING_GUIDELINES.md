# Testing Guidelines and Best Practices

> **Last Updated:** 2026-02-10

## Overview

This document outlines testing patterns, conventions, and best practices for the JARVIS codebase.

## Core Principles

1. **Use Config Values, Not Hardcoded Values**
   - Always use `get_config()` or config defaults instead of hardcoding thresholds, limits, etc.
   - Makes tests resilient to config changes
   - Tests actual behavior users will experience

2. **Comprehensive Coverage**
   - Success cases
   - Error cases  
   - Edge cases (null, empty, boundaries)
   - Invalid inputs
   - Integration scenarios

3. **Descriptive Test Names**
   - Use `test_` prefix
   - Describe what is being tested: `test_returns_empty_when_input_is_none`
   - Group related tests in classes

4. **Use Fixtures for Common Setup**
   - Mock dependencies
   - Create test data
   - Isolate tests

## Testing Patterns

### Pattern 1: Config-Driven Tests

**❌ BAD:**
```python
def test_threshold_check():
    assert router.route("test", threshold=0.85)  # Hardcoded!
```

**✅ GOOD:**
```python
def test_threshold_check():
    config = get_config()
    threshold = config.routing.quick_reply_threshold
    assert router.route("test", threshold=threshold)
```

### Pattern 2: Fixture-Based Config

**✅ BEST:**
```python
@pytest.fixture
def test_config(tmp_path):
    """Create test config."""
    config = JarvisConfig(
        routing=RoutingConfig(quick_reply_threshold=0.8),
        memory_thresholds=MemoryThresholds(full_mode_mb=8000),
    )
    return config

def test_with_custom_config(test_config, monkeypatch):
    """Test with custom config values."""
    monkeypatch.setattr("jarvis.config.get_config", lambda: test_config)
    # Test uses config values, not hardcoded
```

### Pattern 3: Boundary Testing

**✅ GOOD:**
```python
def test_threshold_at_min_boundary(self):
    """Test threshold at minimum valid value."""
    config = JarvisConfig(routing=RoutingConfig(quick_reply_threshold=0.0))
    # Test behavior at boundary
    
def test_threshold_at_max_boundary(self):
    """Test threshold at maximum valid value."""
    config = JarvisConfig(routing=RoutingConfig(quick_reply_threshold=1.0))
    # Test behavior at boundary
    
def test_threshold_below_minimum(self):
    """Test that values below minimum raise ValidationError."""
    with pytest.raises(ValidationError):
        RoutingConfig(quick_reply_threshold=-0.1)
```

### Pattern 4: Edge Cases

**✅ GOOD:**
```python
def test_empty_input(self):
    """Test empty string input."""
    result = normalize_text("")
    assert result == ""

def test_none_input(self):
    """Test None input."""
    result = normalize_text(None)  # type: ignore
    assert result == ""

def test_whitespace_only(self):
    """Test whitespace-only input."""
    result = normalize_text("   \n\t   ")
    assert result == ""

def test_very_long_input(self):
    """Test input exceeding max length."""
    long_text = "x" * 10000
    result = normalize_text(long_text)
    # Should handle gracefully
```

### Pattern 5: Error Cases

**✅ GOOD:**
```python
def test_invalid_input_type(self):
    """Test that invalid input types raise appropriate errors."""
    with pytest.raises(TypeError):
        normalize_text(123)  # type: ignore

def test_missing_dependency(self, monkeypatch):
    """Test behavior when dependency is missing."""
    monkeypatch.setattr("jarvis.embedding_adapter.get_embedder", None)
    with pytest.raises(ImportError):
        # Test error handling
```

### Pattern 6: Integration Scenarios

**✅ GOOD:**
```python
def test_end_to_end_flow(self, test_config, mock_embedder, mock_db):
    """Test complete flow from input to output."""
    # Setup
    router = ReplyRouter(...)
    
    # Execute
    result = router.route("test message")
    
    # Verify
    assert result.response is not None
    assert result.similarity >= 0.0
    assert result.similarity <= 1.0
```

## Common Fixtures

### Config Fixtures

```python
@pytest.fixture
def default_config():
    """Default config for tests."""
    return JarvisConfig()

@pytest.fixture
def custom_config():
    """Custom config with test values."""
    return JarvisConfig(
        routing=RoutingConfig(quick_reply_threshold=0.75),
        memory_thresholds=MemoryThresholds(full_mode_mb=8000),
    )

@pytest.fixture
def config_with_threshold(monkeypatch):
    """Patch get_config() with custom threshold."""
    def _create_config(threshold: float):
        config = JarvisConfig(
            routing=RoutingConfig(quick_reply_threshold=threshold)
        )
        monkeypatch.setattr("jarvis.config.get_config", lambda: config)
        return config
    return _create_config
```

### Mock Fixtures

```python
@pytest.fixture
def mock_embedder():
    """Mock embedder for tests."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.1] * 384])
    embedder.embedding_dim = 384
    return embedder

@pytest.fixture
def mock_db(tmp_path):
    """Mock database for tests."""
    db_path = tmp_path / "test.db"
    db = JarvisDB(db_path)
    # Initialize with test data
    return db
```

## Test Structure

### Class Organization

```python
class TestFeatureName:
    """Tests for FeatureName functionality."""
    
    # Success cases
    def test_basic_functionality(self):
        """Test basic successful case."""
        pass
    
    def test_with_custom_config(self, custom_config):
        """Test with custom configuration."""
        pass
    
    # Error cases
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        pass
    
    def test_missing_dependency(self):
        """Test error handling when dependency missing."""
        pass
    
    # Edge cases
    def test_empty_input(self):
        """Test empty input handling."""
        pass
    
    def test_none_input(self):
        """Test None input handling."""
        pass
    
    def test_boundary_values(self):
        """Test boundary value handling."""
        pass
    
    # Integration
    def test_integration_scenario(self):
        """Test end-to-end integration."""
        pass
```

## Anti-Patterns to Avoid

### ❌ Hardcoded Values

```python
# BAD
def test_threshold():
    assert router.route("test", threshold=0.85)  # Hardcoded!
```

### ❌ Missing Edge Cases

```python
# BAD - only tests happy path
def test_normalize():
    assert normalize_text("hello") == "hello"
```

### ❌ Tests That Depend on Each Other

```python
# BAD - tests depend on order
class TestSequence:
    def test_step1(self):
        self.value = 1
    
    def test_step2(self):
        assert self.value == 1  # Depends on step1!
```

### ❌ Testing Implementation Details

```python
# BAD - tests internal implementation
def test_uses_cache():
    assert router._cache.get("key")  # Testing private method
```

## Best Practices Checklist

- [ ] Use config values instead of hardcoded values
- [ ] Test success cases
- [ ] Test error cases
- [ ] Test edge cases (empty, None, boundaries)
- [ ] Test invalid inputs
- [ ] Test integration scenarios
- [ ] Use descriptive test names
- [ ] Group related tests in classes
- [ ] Use fixtures for common setup
- [ ] Mock external dependencies
- [ ] Tests are independent (no order dependency)
- [ ] Tests are fast (use mocks, not real services)
- [ ] Tests are deterministic (no random values without seeds)

## Examples

See:
- `tests/unit/test_config.py` - Config-driven tests
- `tests/unit/test_text_normalizer.py` - Comprehensive edge case testing
- `tests/unit/test_router.py` - Mock-based testing
