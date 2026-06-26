from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Optional
from uuid import uuid4

from app.schemas import AgentInvocationRequest, TraceEvent


def started_timer() -> float:
    return perf_counter()


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def workflow_event(
    *,
    request: AgentInvocationRequest,
    node: str,
    event_type: str,
    output_summary: str,
    started_at: float,
    input_summary: str = "",
    status: str = "success",
    estimated_tokens: int = 0,
    estimated_cost: float = 0.0,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TraceEvent:
    return TraceEvent(
        trace_id=request.trace_id,
        request_id=request.request_id,
        event_id=str(uuid4()),
        service="agent_orchestrator",
        node=node,
        event_type=event_type,  # type: ignore[arg-type]
        input_summary=input_summary,
        output_summary=output_summary,
        status=status,  # type: ignore[arg-type]
        latency_ms=elapsed_ms(started_at),
        estimated_tokens=estimated_tokens,
        estimated_cost=estimated_cost,
        error_type=error_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata=metadata or {},
    )
