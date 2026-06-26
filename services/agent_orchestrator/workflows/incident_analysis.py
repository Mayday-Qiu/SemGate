from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest
from services.agent_orchestrator.clients import ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import (
    estimate_tokens,
    evidence_to_citations,
    first_record,
    has_permissions,
    permission_missing,
    tool_error,
    tool_result,
    tool_succeeded,
)


REQUIRED_PERMISSIONS = [
    "kb:project_docs:read",
    "tool:business:read",
    "tool:service_status:read",
    "tool:doc_search:use",
    "tool:evidence_check:use",
]


async def run_incident_analysis_workflow(request: AgentInvocationRequest, tool_client: ToolClient) -> AgentState:
    graph = _build_graph(tool_client)
    return await graph.ainvoke(
        {
            "request": request,
            "status": "success",
            "tools": [],
            "tool_results": [],
            "trace_events": [],
            "errors": [],
            "evidence": [],
            "citations": [],
            "fallback_used": False,
        }
    )


def _build_graph(tool_client: ToolClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_incident(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseIncident",
                    event_type="node_end",
                    output_summary="parsed incident request",
                    started_at=started_at,
                    metadata={"input_length": len(request.input)},
                )
            ]
        }

    async def permission_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        if has_permissions(request, REQUIRED_PERMISSIONS):
            return {
                "trace_events": [
                    workflow_event(
                        request=request,
                        node="PermissionCheck",
                        event_type="node_end",
                        output_summary="permission granted",
                        started_at=started_at,
                        metadata={"required_permissions": REQUIRED_PERMISSIONS},
                    )
                ]
            }

        missing = permission_missing(request, REQUIRED_PERMISSIONS)
        return {
            "status": "refused",
            "failure_reason": "permission_denied",
            "errors": [{"type": "permission_denied", "missing_permissions": missing}],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="PermissionCheck",
                    event_type="error",
                    output_summary="permission denied",
                    status="permission_denied",
                    started_at=started_at,
                    error_type="permission_denied",
                    metadata={"missing_permissions": missing},
                )
            ],
        }

    async def service_status(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        simulate_failure = request.metadata.get("simulate_service_status_failure")
        response = await tool_client.invoke(
            request,
            "service_status_tool",
            {
                "service_name": str(request.metadata.get("service_name", "gateway")),
                "simulate_failure": simulate_failure,
            },
        )
        result = tool_result(response)
        success = tool_succeeded(response)
        return {
            "tools": ["service_status_tool"],
            "tool_results": [response],
            "errors": [] if success else [tool_error(response)],
            "fallback_used": not success,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ServiceStatus",
                    event_type="tool_call",
                    output_summary=f"service_status_tool status={response.get('status')}",
                    status="success" if success else "failed",
                    started_at=started_at,
                    error_type=None if success else str(response.get("error_type")),
                    metadata={"tool_response": response, "service_status": result},
                )
            ],
        }

    async def business_query(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "business_query_tool",
            {"entity_type": "ticket", "query": str(request.metadata.get("ticket_id", "INC-001"))},
        )
        result = tool_result(response)
        success = tool_succeeded(response)
        return {
            "tools": ["business_query_tool"],
            "tool_results": [response],
            "errors": [] if success else [tool_error(response)],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="BusinessQuery",
                    event_type="tool_call",
                    output_summary=f"business_query_tool status={response.get('status')}",
                    status="success" if success else "failed",
                    started_at=started_at,
                    error_type=None if success else str(response.get("error_type")),
                    metadata={"tool_response": response, "record": first_record(result)},
                )
            ],
        }

    async def retrieve_runbook(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "doc_search_tool",
            {"query": f"runbook for {request.input}", "top_k": 3, "strategy": "hybrid"},
        )
        result = tool_result(response)
        evidence = result.get("evidence", []) if tool_succeeded(response) else []
        return {
            "evidence": evidence,
            "tools": ["doc_search_tool"],
            "tool_results": [response],
            "errors": [] if tool_succeeded(response) else [tool_error(response)],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RetrieveRunbook",
                    event_type="tool_call",
                    output_summary=f"doc_search_tool status={response.get('status')}",
                    status="success" if tool_succeeded(response) else "failed",
                    started_at=started_at,
                    error_type=None if tool_succeeded(response) else str(response.get("error_type")),
                    metadata={"tool_response": response},
                )
            ],
        }

    async def evidence_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "evidence_check_tool",
            {"claim": "incident fix plan needs runbook support", "evidence_query": request.input, "top_k": 3},
        )
        return {
            "tools": ["evidence_check_tool"],
            "tool_results": [response],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="EvidenceCheck",
                    event_type="tool_call",
                    output_summary=f"evidence_check_tool status={response.get('status')}",
                    status="success" if tool_succeeded(response) else "failed",
                    started_at=started_at,
                    error_type=None if tool_succeeded(response) else str(response.get("error_type")),
                    metadata={"tool_response": response},
                )
            ],
        }

    async def generate_fix_plan(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        fallback_used = bool(state.get("fallback_used"))
        status_note = "service status tool failed, so the plan uses conservative fallback guidance" if fallback_used else "service status and ticket data were available"
        answer = (
            "Incident analysis workflow completed: "
            f"{status_note}. Recommended next steps: verify gateway downstream latency, inspect recent workflow errors, "
            "and use the retrieved runbook evidence before changing production settings."
        )
        tokens = estimate_tokens(request.input + answer)
        return {
            "answer": answer,
            "citations": evidence_to_citations(state.get("evidence", [])),
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="GenerateFixPlan",
                    event_type="node_end",
                    output_summary="generated incident fix plan",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                    metadata={"fallback_used": fallback_used},
                )
            ],
        }

    async def fallback_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        fallback_used = bool(state.get("fallback_used"))
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="FallbackCheck",
                    event_type="fallback" if fallback_used else "node_end",
                    output_summary="fallback used" if fallback_used else "no fallback needed",
                    status="success",
                    started_at=started_at,
                    metadata={"fallback_used": fallback_used},
                )
            ]
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        answer = state.get("answer")
        if failure_reason and not answer:
            answer = f"Cannot complete incident analysis because {failure_reason}."
        return {
            "answer": answer or "",
            "trace_events": [
                workflow_event(
                    request=request,
                    node="FinalizeResponse",
                    event_type="node_end",
                    output_summary=f"status={state.get('status', 'success')}",
                    started_at=started_at,
                    metadata={"failure_reason": failure_reason},
                )
            ],
        }

    def after_permission(state: AgentState) -> str:
        return "finalize" if state.get("failure_reason") else "service_status"

    builder.add_node("parse", parse_incident)
    builder.add_node("permission", permission_check)
    builder.add_node("service_status", service_status)
    builder.add_node("business", business_query)
    builder.add_node("runbook", retrieve_runbook)
    builder.add_node("evidence_check", evidence_check)
    builder.add_node("generate", generate_fix_plan)
    builder.add_node("fallback", fallback_check)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "permission")
    builder.add_conditional_edges("permission", after_permission, {"service_status": "service_status", "finalize": "finalize"})
    builder.add_edge("service_status", "business")
    builder.add_edge("business", "runbook")
    builder.add_edge("runbook", "evidence_check")
    builder.add_edge("evidence_check", "generate")
    builder.add_edge("generate", "fallback")
    builder.add_edge("fallback", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
