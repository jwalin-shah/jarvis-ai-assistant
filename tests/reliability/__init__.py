"""Reliability validation tests for JARVIS.

This package contains tests for:
- Offline-mode behavior
- Degraded-mode handling
- Circuit breaker functionality
- Retry logic resilience
- Memory pressure handling
- Network resilience

Run with:
    pytest tests/reliability/ -v
    pytest tests/reliability/ --chaos-mode -v  # With chaos engineering
"""
