from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict

from app.schemas import AgentInvocationRequest, Citation, TraceEvent


class AgentState(TypedDict, total=False):
    request: AgentInvocationRequest
    answer: str
    citations: List[Citation]
    evidence: List[Dict[str, Any]]
    draft: str
    status: str
    failure_reason: Optional[str]
    fallback_used: bool
    metrics_tokens: int
    metrics_cost: float
    tools: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    trace_events: Annotated[List[TraceEvent], operator.add]
    errors: Annotated[List[Dict[str, Any]], operator.add]
