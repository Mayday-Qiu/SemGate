from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Dict


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool
    replenish_rate: float
    burst_capacity: float
    requested_tokens: float


@dataclass
class RateLimitDecision:
    allowed: bool
    limit_key: str
    remaining_tokens: float
    retry_after_s: float
    reason: str

    def to_metadata(self) -> Dict[str, object]:
        return {
            "allowed": self.allowed,
            "limit_key": self.limit_key,
            "remaining_tokens": round(self.remaining_tokens, 6),
            "retry_after_s": round(self.retry_after_s, 6),
            "reason": self.reason,
        }


@dataclass
class _BucketState:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    def __init__(self, config: RateLimitConfig) -> None:
        if config.replenish_rate <= 0:
            raise ValueError("rate limit replenish_rate must be greater than 0")
        if config.burst_capacity <= 0:
            raise ValueError("rate limit burst_capacity must be greater than 0")
        if config.requested_tokens <= 0:
            raise ValueError("rate limit requested_tokens must be greater than 0")
        self._config = config
        self._buckets: Dict[str, _BucketState] = {}
        self._lock = Lock()

    def check(self, key: str) -> RateLimitDecision:
        if not self._config.enabled:
            return RateLimitDecision(
                allowed=True,
                limit_key=key,
                remaining_tokens=self._config.burst_capacity,
                retry_after_s=0.0,
                reason="rate_limit_disabled",
            )

        now = monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _BucketState(tokens=self._config.burst_capacity, updated_at=now)
                self._buckets[key] = bucket

            elapsed_s = max(0.0, now - bucket.updated_at)
            replenished = elapsed_s * self._config.replenish_rate
            bucket.tokens = min(self._config.burst_capacity, bucket.tokens + replenished)
            bucket.updated_at = now

            if bucket.tokens >= self._config.requested_tokens:
                bucket.tokens -= self._config.requested_tokens
                return RateLimitDecision(
                    allowed=True,
                    limit_key=key,
                    remaining_tokens=bucket.tokens,
                    retry_after_s=0.0,
                    reason="token_available",
                )

            missing_tokens = self._config.requested_tokens - bucket.tokens
            retry_after_s = missing_tokens / self._config.replenish_rate
            return RateLimitDecision(
                allowed=False,
                limit_key=key,
                remaining_tokens=bucket.tokens,
                retry_after_s=retry_after_s,
                reason="token_bucket_empty",
            )
