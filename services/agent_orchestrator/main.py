from __future__ import annotations

from time import perf_counter
from typing import Dict, List

from fastapi import FastAPI

from app.schemas import (
    AgentInvocationRequest,
    AgentWorkflowResponse,
    AgenticMetrics,
    Citation,
    TraceEvent,
)
from services.agent_orchestrator.clients import ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import elapsed_ms, started_timer, workflow_event
from services.agent_orchestrator.workflows.doc_review import run_doc_review_workflow
from services.agent_orchestrator.workflows.incident_analysis import run_incident_analysis_workflow
from services.agent_orchestrator.workflows.tech_qa import run_tech_qa_workflow


app = FastAPI(title="SemRoute Agent Orchestrator", version="0.7.0")
tool_client = ToolClient()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "agent_orchestrator"}


@app.post("/invoke", response_model=AgentWorkflowResponse)
async def invoke(request: AgentInvocationRequest) -> AgentWorkflowResponse:
    started_at = perf_counter()
    if request.selected_workflow == "tech_qa_workflow":
        state = await run_tech_qa_workflow(request, tool_client)
    elif request.selected_workflow == "doc_review_workflow":
        state = await run_doc_review_workflow(request, tool_client)
    elif request.selected_workflow == "incident_analysis_workflow":
        state = await run_incident_analysis_workflow(request, tool_client)
    elif request.selected_workflow == "media_generation_workflow":
        state = _run_media_generation_stub(request)
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
        },
    )


def _run_media_generation_stub(request: AgentInvocationRequest) -> AgentState:
    parse_started_at = started_timer()
    media_type = str(request.metadata.get("media_type", "video" if "video" in request.input.lower() else "image"))
    target_tool = "video_generation_tool" if media_type == "video" else "image_generation_tool"
    parse_event = workflow_event(
        request=request,
        node="ParseMediaRequest",
        event_type="node_end",
        output_summary=f"media_type={media_type}",
        started_at=parse_started_at,
        metadata={"langgraph_enabled": False},
    )
    stub_started_at = started_timer()
    answer = (
        "The media-generation workflow is reserved as a controlled entrypoint for image/video generation tasks. "
        f"Phase 2 keeps it as a non-LangGraph placeholder and would route future execution through {target_tool}."
    )
    stub_event = workflow_event(
        request=request,
        node="MediaGenerationStub",
        event_type="node_end",
        output_summary=f"placeholder_tool={target_tool}",
        started_at=stub_started_at,
        metadata={
            "placeholder_tool": target_tool,
            "implementation_status": "placeholder",
            "reason": "real image/video backend is outside Phase 2",
        },
    )
    tokens = max(80, int((len(request.input) + len(answer)) / 4))
    return {
        "request": request,
        "answer": answer,
        "status": "success",
        "citations": [
            Citation(source_id="media_policy", title="media_generation_policy.md", chunk_id="c_prompt_safety", score=0.71)
        ],
        "tools": [target_tool],
        "trace_events": [parse_event, stub_event],
        "metrics_tokens": tokens,
        "metrics_cost": round(tokens * 0.000001, 6),
    }


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


def _implementation_status(workflow_id: str) -> str:
    if workflow_id in {"tech_qa_workflow", "doc_review_workflow", "incident_analysis_workflow"}:
        return "phase_2_langgraph"
    if workflow_id == "media_generation_workflow":
        return "phase_2_placeholder_no_langgraph"
    return "unknown"
