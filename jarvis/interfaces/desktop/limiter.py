import time


class RateLimiter:
    """Token bucket rate limiter — O(1) per request.

    Each client gets a bucket that refills at ``refill_rate`` tokens/sec up to
    ``max_tokens``.  Every request consumes one token; requests are rejected
    when the bucket is empty.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0) -> None:
        # max_requests tokens refill over window_seconds → steady-state rate
        self._max_tokens = float(max_requests)
        self._refill_rate = max_requests / window_seconds  # tokens per second
        # Per-client state: (tokens_remaining, last_refill_time)
        self._buckets: dict[str, list[float]] = {}
        self._time = time

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed.

        Returns True if under rate limit, False if exceeded.  O(1) per call.
        """
        now = self._time.monotonic()

        bucket = self._buckets.get(client_id)
        if bucket is None:
            # New client: full bucket minus this request
            self._buckets[client_id] = [self._max_tokens - 1.0, now]
            return True

        # Refill tokens based on elapsed time
        elapsed = now - bucket[1]
        tokens = min(self._max_tokens, bucket[0] + elapsed * self._refill_rate)

        if tokens < 1.0:
            # Update timestamp even on rejection so next refill is accurate
            bucket[0] = tokens
            bucket[1] = now
            return False

        bucket[0] = tokens - 1.0
        bucket[1] = now
        return True
