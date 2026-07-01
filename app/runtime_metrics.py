from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from threading import Lock
from typing import Deque, Dict, Optional


DEFAULT_WINDOW_SIZE = 100


@dataclass(frozen=True)
class RuntimeMetricsSnapshot:
    workflow_id: str
    ongoing_count: int = 0
    recent_sample_count: int = 0
    recent_avg_latency_ms: Optional[float] = None
    recent_p95_latency_ms: Optional[float] = None
    recent_error_rate: Optional[float] = None

    @property
    def has_recent_data(self) -> bool:
        return self.recent_sample_count > 0

    def to_metadata(self) -> Dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "ongoing_count": self.ongoing_count,
            "recent_sample_count": self.recent_sample_count,
            "recent_avg_latency_ms": self.recent_avg_latency_ms,
            "recent_p95_latency_ms": self.recent_p95_latency_ms,
            "recent_error_rate": self.recent_error_rate,
        }


class _WorkflowRuntimeMetrics:
    def __init__(self, workflow_id: str, window_size: int) -> None:
        self.workflow_id = workflow_id
        self.ongoing_count = 0
        self.latencies_ms: Deque[float] = deque(maxlen=window_size)
        self.error_flags: Deque[bool] = deque(maxlen=window_size)

    def snapshot(self) -> RuntimeMetricsSnapshot:
        latencies = list(self.latencies_ms)
        error_flags = list(self.error_flags)
        if not latencies:
            return RuntimeMetricsSnapshot(
                workflow_id=self.workflow_id,
                ongoing_count=self.ongoing_count,
            )
        return RuntimeMetricsSnapshot(
            workflow_id=self.workflow_id,
            ongoing_count=self.ongoing_count,
            recent_sample_count=len(latencies),
            recent_avg_latency_ms=round(sum(latencies) / len(latencies), 3),
            recent_p95_latency_ms=round(_percentile(latencies, 0.95), 3),
            recent_error_rate=round(sum(1 for failed in error_flags if failed) / len(error_flags), 6),
        )


class RuntimeMetricsStore:
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._metrics: Dict[str, _WorkflowRuntimeMetrics] = {}
        self._lock = Lock()

    def start(self, workflow_id: str) -> RuntimeMetricsSnapshot:
        with self._lock:
            metrics = self._get_or_create(workflow_id)
            metrics.ongoing_count += 1
            return metrics.snapshot()

    def finish(self, workflow_id: str, latency_ms: float, failed: bool) -> RuntimeMetricsSnapshot:
        with self._lock:
            metrics = self._get_or_create(workflow_id)
            metrics.ongoing_count = max(0, metrics.ongoing_count - 1)
            metrics.latencies_ms.append(max(0.0, latency_ms))
            metrics.error_flags.append(bool(failed))
            return metrics.snapshot()

    def snapshot(self, workflow_id: str) -> RuntimeMetricsSnapshot:
        with self._lock:
            return self._get_or_create(workflow_id).snapshot()

    def _get_or_create(self, workflow_id: str) -> _WorkflowRuntimeMetrics:
        if workflow_id not in self._metrics:
            self._metrics[workflow_id] = _WorkflowRuntimeMetrics(workflow_id, self._window_size)
        return self._metrics[workflow_id]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(percentile * len(ordered)) - 1))
    return ordered[index]
