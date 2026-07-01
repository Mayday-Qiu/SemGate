from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List

from fastapi import FastAPI

from app.schemas import (
    AgentInvocationRequest,
    AgentWorkflowResponse,
    AgenticMetrics,
)
from services.agent_orchestrator.clients import ModelClient, ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import elapsed_ms, started_timer, workflow_event
from services.agent_orchestrator.workflows.coding import run_coding_workflow
from services.agent_orchestrator.workflows.document_writing import run_document_writing_workflow
from services.agent_orchestrator.workflows.knowledge_qa import run_knowledge_qa_workflow
from services.agent_orchestrator.workflows.media_generation import run_media_generation_workflow


app = FastAPI(title="SemGateway Agent Orchestrator", version="1.0.0")
tool_client = ToolClient()
model_client = ModelClient()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "agent_orchestrator"}


@app.post("/invoke", response_model=AgentWorkflowResponse)
async def invoke(request: AgentInvocationRequest) -> AgentWorkflowResponse:
    started_at = perf_counter()
    if request.selected_workflow in {"knowledge_qa_workflow", "large_knowledge_qa_workflow"}:
        state = await run_knowledge_qa_workflow(request, tool_client)
    elif request.selected_workflow == "coding_workflow":
        state = await run_coding_workflow(request, model_client)
    elif request.selected_workflow == "media_generation_workflow":
        state = await run_media_generation_workflow(request, tool_client)
    elif request.selected_workflow == "document_writing_workflow":
        state = await run_document_writing_workflow(request, tool_client, model_client)
    else:
        state = _run_unknown_workflow_stub(request)

    answer = str(state.get("answer", ""))
    trace_events = list(state.get("trace_events", []))
    tools = _unique_tools(list(state.get("tools", [])))
    citations = list(state.get("citations", []))
    latency_ms = elapsed_ms(started_at)
    estimated_tokens = int(state.get("metrics_tokens") or max(80, int((len(request.input) + len(answer)) / 4)))
    estimated_cost = float(state.get("metrics_cost") or round(estimated_tokens * 0.000001, 6))
    status_value = str(state.get("status", "success"))
    state_metadata = state.get("metadata", {})
    if not isinstance(state_metadata, dict):
        state_metadata = {}

    return AgentWorkflowResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        selected_workflow=request.selected_workflow,
        answer=answer,
        citations=citations,
        status=status_value,  # type: ignore[arg-type]
        metrics=AgenticMetrics(
            latency_ms=latency_ms,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
        ),
        tools=tools,
        trace_events=trace_events,
        metadata={
            "implementation_status": _implementation_status(request.selected_workflow),
            "fallback_used": bool(state.get("fallback_used", False)),
            "failure_reason": state.get("failure_reason"),
            "tool_result_count": len(state.get("tool_results", [])),
            "tool_statuses": _tool_statuses(list(state.get("tool_results", []))),
            **state_metadata,
        },
    )


def _run_unknown_workflow_stub(request: AgentInvocationRequest) -> AgentState:
    started_at = started_timer()
    event = workflow_event(
        request=request,
        node="UnknownWorkflow",
        event_type="error",
        output_summary=f"unknown_workflow={request.selected_workflow}",
        status="failed",
        started_at=started_at,
        error_type="unknown_workflow",
    )
    return {
        "request": request,
        "answer": "Selected workflow is not implemented in agent_orchestrator.",
        "status": "failed",
        "failure_reason": "unknown_workflow",
        "trace_events": [event],
        "errors": [{"type": "unknown_workflow"}],
    }


def _unique_tools(tools: List[str]) -> List[str]:
    seen = set()
    result = []
    for tool in tools:
        if tool in seen:
            continue
        seen.add(tool)
        result.append(tool)
    return result


def _tool_statuses(tool_results: List[Dict[str, Any]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for result in tool_results:
        tool_name = result.get("tool_name")
        status = result.get("status")
        if tool_name and status:
            statuses[str(tool_name)] = str(status)
    return statuses


def _implementation_status(workflow_id: str) -> str:
    if workflow_id in {
        "knowledge_qa_workflow",
        "large_knowledge_qa_workflow",
        "coding_workflow",
        "media_generation_workflow",
        "document_writing_workflow",
    }:
        return "phase_2_bounded_dag"
    return "unknown"
