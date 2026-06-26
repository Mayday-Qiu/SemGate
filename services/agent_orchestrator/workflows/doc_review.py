from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest
from services.agent_orchestrator.clients import ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import (
    estimate_tokens,
    evidence_to_citations,
    has_permissions,
    permission_missing,
    tool_error,
    tool_result,
    tool_succeeded,
)


REQUIRED_PERMISSIONS = [
    "kb:project_docs:read",
    "tool:doc_search:use",
    "tool:inspection:use",
    "tool:evidence_check:use",
]


async def run_doc_review_workflow(request: AgentInvocationRequest, tool_client: ToolClient) -> AgentState:
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

    async def parse_document(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseDocument",
                    event_type="node_end",
                    output_summary="parsed inspection request",
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

    async def retrieve_policy(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "doc_search_tool",
            {"query": f"inspection policy for: {request.input}", "top_k": 3, "strategy": "hybrid"},
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
                    node="RetrievePolicy",
                    event_type="tool_call",
                    output_summary=f"doc_search_tool status={response.get('status')}",
                    status="success" if tool_succeeded(response) else "failed",
                    started_at=started_at,
                    error_type=None if tool_succeeded(response) else str(response.get("error_type")),
                    metadata={"tool_response": response},
                )
            ],
        }

    async def run_inspection(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        inspection_type = str(request.metadata.get("inspection_type", "document_structure"))
        response = await tool_client.invoke(
            request,
            "inspection_tool",
            {
                "inspection_type": inspection_type,
                "content": request.input,
                "context": {"evidence": state.get("evidence", [])},
            },
        )
        result = tool_result(response)
        passed = bool(result.get("passed")) if tool_succeeded(response) else False
        return {
            "draft": _inspection_draft(result),
            "tools": ["inspection_tool"],
            "tool_results": [response],
            "errors": [] if tool_succeeded(response) else [tool_error(response)],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RunInspection",
                    event_type="tool_call",
                    output_summary=f"inspection passed={str(passed).lower()}",
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
        claim = state.get("draft") or request.input
        response = await tool_client.invoke(
            request,
            "evidence_check_tool",
            {"claim": claim, "evidence_query": request.input, "top_k": 3, "min_score": 0.55},
        )
        result = tool_result(response)
        supported = bool(result.get("supported")) if tool_succeeded(response) else False
        return {
            "tools": ["evidence_check_tool"],
            "tool_results": [response],
            "errors": [] if supported else [{"type": "weak_evidence", "tool_response": response}],
            "trace_events": [
                workflow_event(
                    request=request,
                    node="EvidenceCheck",
                    event_type="tool_call",
                    output_summary=f"supported={str(supported).lower()}",
                    status="success" if supported else "failed",
                    started_at=started_at,
                    error_type=None if supported else "weak_evidence",
                    metadata={"tool_response": response},
                )
            ],
        }

    async def generate_review(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        draft = state.get("draft", "No inspection details were produced.")
        answer = (
            "Inspection workflow completed. "
            f"{draft} The workflow used project evidence and an inspection tool before producing this review."
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
                    node="GenerateReview",
                    event_type="node_end",
                    output_summary="generated inspection review",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        answer = state.get("answer")
        if failure_reason and not answer:
            answer = f"Cannot complete inspection workflow because {failure_reason}."
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
        return "finalize" if state.get("failure_reason") else "retrieve"

    builder.add_node("parse", parse_document)
    builder.add_node("permission", permission_check)
    builder.add_node("retrieve", retrieve_policy)
    builder.add_node("inspection", run_inspection)
    builder.add_node("evidence_check", evidence_check)
    builder.add_node("generate", generate_review)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "permission")
    builder.add_conditional_edges("permission", after_permission, {"retrieve": "retrieve", "finalize": "finalize"})
    builder.add_edge("retrieve", "inspection")
    builder.add_edge("inspection", "evidence_check")
    builder.add_edge("evidence_check", "generate")
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _inspection_draft(result: Dict[str, Any]) -> str:
    findings = result.get("findings") or []
    missing = result.get("missing_items") or []
    return (
        f"inspection_type={result.get('inspection_type')}, "
        f"passed={result.get('passed')}, "
        f"risk_level={result.get('risk_level')}, "
        f"findings={findings}, missing_items={missing}."
    )
