from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Deque, Dict, Optional, Tuple


CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    enabled: bool
    sliding_window_size: int
    minimum_number_of_calls: int
    failure_rate_threshold: float
    slow_call_rate_threshold: float
    wait_duration_in_open_state_s: float
    permitted_calls_in_half_open: int


@dataclass(frozen=True)
class CircuitSnapshot:
    state: str
    sample_count: int
    failure_count: int
    slow_call_count: int
    failure_rate: float
    slow_call_rate: float
    last_opened_reason: Optional[str]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "sample_count": self.sample_count,
            "failure_count": self.failure_count,
            "slow_call_count": self.slow_call_count,
            "failure_rate": round(self.failure_rate, 6),
            "slow_call_rate": round(self.slow_call_rate, 6),
            "last_opened_reason": self.last_opened_reason,
        }


@dataclass(frozen=True)
class CircuitPermission:
    allowed: bool
    reason: str
    snapshot: CircuitSnapshot


@dataclass(frozen=True)
class _CallRecord:
    failed: bool
    slow: bool


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        if config.sliding_window_size <= 0:
            raise ValueError("circuit sliding_window_size must be greater than 0")
        if config.minimum_number_of_calls <= 0:
            raise ValueError("circuit minimum_number_of_calls must be greater than 0")
        if config.permitted_calls_in_half_open <= 0:
            raise ValueError("circuit permitted_calls_in_half_open must be greater than 0")
        self._config = config
        self._state = CLOSED
        self._records: Deque[_CallRecord] = deque(maxlen=config.sliding_window_size)
        self._opened_at: Optional[float] = None
        self._half_open_successes = 0
        self._half_open_calls = 0
        self._last_opened_reason: Optional[str] = None

    def before_call(self) -> CircuitPermission:
        if not self._config.enabled:
            return CircuitPermission(
                allowed=True,
                reason="circuit_breaker_disabled",
                snapshot=self.snapshot(),
            )

        now = monotonic()
        if self._state == OPEN:
            if self._opened_at is not None and now - self._opened_at >= self._config.wait_duration_in_open_state_s:
                self._transition_to_half_open()
            else:
                return CircuitPermission(
                    allowed=False,
                    reason="circuit_open",
                    snapshot=self.snapshot(),
                )

        if self._state == HALF_OPEN:
            if self._half_open_calls >= self._config.permitted_calls_in_half_open:
                return CircuitPermission(
                    allowed=False,
                    reason="half_open_probe_limit_reached",
                    snapshot=self.snapshot(),
                )
            self._half_open_calls += 1
            return CircuitPermission(
                allowed=True,
                reason="half_open_probe_allowed",
                snapshot=self.snapshot(),
            )

        return CircuitPermission(
            allowed=True,
            reason="circuit_closed",
            snapshot=self.snapshot(),
        )

    def record_result(self, outcome: str, duration_ms: float, slow_call_threshold_ms: float) -> CircuitSnapshot:
        if not self._config.enabled:
            return self.snapshot()

        failed = outcome in {"timeout", "network_error", "server_error", "failed"}
        slow = duration_ms >= slow_call_threshold_ms

        if self._state == HALF_OPEN:
            if failed or slow:
                reason = "half_open_probe_failed" if failed else "half_open_probe_slow"
                self._transition_to_open(reason)
                return self.snapshot()
            self._half_open_successes += 1
            if self._half_open_successes >= self._config.permitted_calls_in_half_open:
                self._transition_to_closed()
            return self.snapshot()

        self._records.append(_CallRecord(failed=failed, slow=slow))
        if self._state == CLOSED:
            self._evaluate_closed_window()
        return self.snapshot()

    def snapshot(self) -> CircuitSnapshot:
        sample_count = len(self._records)
        failure_count = sum(1 for record in self._records if record.failed)
        slow_call_count = sum(1 for record in self._records if record.slow)
        failure_rate = (failure_count / sample_count * 100.0) if sample_count else 0.0
        slow_call_rate = (slow_call_count / sample_count * 100.0) if sample_count else 0.0
        return CircuitSnapshot(
            state=self._state,
            sample_count=sample_count,
            failure_count=failure_count,
            slow_call_count=slow_call_count,
            failure_rate=failure_rate,
            slow_call_rate=slow_call_rate,
            last_opened_reason=self._last_opened_reason,
        )

    def _evaluate_closed_window(self) -> None:
        snapshot = self.snapshot()
        if snapshot.sample_count < self._config.minimum_number_of_calls:
            return
        if snapshot.failure_rate >= self._config.failure_rate_threshold:
            self._transition_to_open("failure_rate_threshold_exceeded")
            return
        if snapshot.slow_call_rate >= self._config.slow_call_rate_threshold:
            self._transition_to_open("slow_call_rate_threshold_exceeded")

    def _transition_to_open(self, reason: str) -> None:
        self._state = OPEN
        self._opened_at = monotonic()
        self._half_open_successes = 0
        self._half_open_calls = 0
        self._last_opened_reason = reason

    def _transition_to_half_open(self) -> None:
        self._state = HALF_OPEN
        self._half_open_successes = 0
        self._half_open_calls = 0

    def _transition_to_closed(self) -> None:
        self._state = CLOSED
        self._records.clear()
        self._opened_at = None
        self._half_open_successes = 0
        self._half_open_calls = 0


class CircuitBreakerRegistry:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._breakers: Dict[Tuple[str, str], CircuitBreaker] = {}

    def get(self, backend_id: str, task_type: str) -> CircuitBreaker:
        key = (backend_id, task_type)
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(self._config)
            self._breakers[key] = breaker
        return breaker
