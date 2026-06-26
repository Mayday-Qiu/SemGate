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
    has_permissions,
    permission_missing,
    tool_error,
    tool_result,
    tool_succeeded,
)


REQUIRED_PERMISSIONS = ["kb:project_docs:read", "tool:doc_search:use", "tool:evidence_check:use"]


async def run_tech_qa_workflow(request: AgentInvocationRequest, tool_client: ToolClient) -> AgentState:
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

    async def parse_request(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseRequest",
                    event_type="node_end",
                    output_summary="parsed knowledge question",
                    started_at=started_at,
                    metadata={"task_type": request.task_profile.task_type},
                )
            ]
        }

    async def permission_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        if has_permissions(request, REQUIRED_PERMISSIONS):
            event = workflow_event(
                request=request,
                node="PermissionCheck",
                event_type="node_end",
                output_summary="permission granted",
                started_at=started_at,
                metadata={"required_permissions": REQUIRED_PERMISSIONS},
            )
            return {"trace_events": [event]}

        missing = permission_missing(request, REQUIRED_PERMISSIONS)
        event = workflow_event(
            request=request,
            node="PermissionCheck",
            event_type="error",
            output_summary="permission denied",
            status="permission_denied",
            started_at=started_at,
            error_type="permission_denied",
            metadata={"missing_permissions": missing},
        )
        return {
            "status": "refused",
            "failure_reason": "permission_denied",
            "errors": [{"type": "permission_denied", "missing_permissions": missing}],
            "trace_events": [event],
        }

    async def retrieve_evidence(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "doc_search_tool",
            {"query": request.input, "top_k": 3, "strategy": "hybrid"},
        )
        result = tool_result(response)
        evidence = result.get("evidence", []) if tool_succeeded(response) else []
        event = workflow_event(
            request=request,
            node="RetrieveEvidence",
            event_type="tool_call",
            output_summary=f"doc_search_tool status={response.get('status')}",
            status="success" if tool_succeeded(response) else "failed",
            started_at=started_at,
            error_type=None if tool_succeeded(response) else str(response.get("error_type")),
            metadata={"tool_response": response},
        )
        if not tool_succeeded(response):
            return {
                "status": "refused",
                "failure_reason": "retrieval_failed",
                "tools": ["doc_search_tool"],
                "tool_results": [response],
                "errors": [tool_error(response)],
                "trace_events": [event],
            }
        return {
            "evidence": evidence,
            "tools": ["doc_search_tool"],
            "tool_results": [response],
            "trace_events": [event],
        }

    async def evidence_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        response = await tool_client.invoke(
            request,
            "evidence_check_tool",
            {"claim": request.input, "evidence_query": request.input, "top_k": 3, "min_score": 0.6},
        )
        result = tool_result(response)
        supported = bool(result.get("supported")) if tool_succeeded(response) else False
        event = workflow_event(
            request=request,
            node="EvidenceCheck",
            event_type="tool_call",
            output_summary=f"supported={str(supported).lower()}",
            status="success" if supported else "failed",
            started_at=started_at,
            error_type=None if supported else "no_evidence",
            metadata={"tool_response": response},
        )
        if not supported:
            return {
                "status": "refused",
                "failure_reason": "no_evidence",
                "tools": ["evidence_check_tool"],
                "tool_results": [response],
                "errors": [{"type": "no_evidence", "tool_response": response}],
                "trace_events": [event],
            }
        return {
            "tools": ["evidence_check_tool"],
            "tool_results": [response],
            "trace_events": [event],
        }

    async def generate_answer(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        evidence = state.get("evidence", [])
        evidence_titles = ", ".join(str(item.get("title", "unknown")) for item in evidence[:2])
        answer = (
            "SemRoute-Gateway uses Gateway-side TaskProfile and WorkflowProfile to select an Agent workflow, "
            "then the selected workflow retrieves evidence and checks whether the answer is supported. "
            f"This answer is grounded by retrieved evidence from: {evidence_titles or 'no evidence'}."
        )
        tokens = estimate_tokens(request.input + answer)
        event = workflow_event(
            request=request,
            node="GenerateAnswer",
            event_type="node_end",
            output_summary="generated grounded answer",
            started_at=started_at,
            estimated_tokens=tokens,
            estimated_cost=round(tokens * 0.000001, 6),
        )
        return {
            "answer": answer,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "trace_events": [event],
        }

    async def citation_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        citations = evidence_to_citations(state.get("evidence", []))
        event = workflow_event(
            request=request,
            node="CitationCheck",
            event_type="node_end",
            output_summary=f"citations={len(citations)}",
            status="success" if citations else "failed",
            started_at=started_at,
            error_type=None if citations else "missing_citation",
        )
        return {
            "citations": citations,
            "trace_events": [event],
            "status": "success" if citations else "refused",
            "failure_reason": None if citations else "missing_citation",
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        answer = state.get("answer")
        if failure_reason and not answer:
            answer = f"Cannot complete knowledge QA because {failure_reason}."
        event = workflow_event(
            request=request,
            node="FinalizeResponse",
            event_type="node_end",
            output_summary=f"status={state.get('status', 'success')}",
            started_at=started_at,
            metadata={"failure_reason": failure_reason},
        )
        return {"answer": answer or "", "trace_events": [event]}

    def after_permission(state: AgentState) -> str:
        return "finalize" if state.get("failure_reason") else "retrieve"

    def after_retrieve(state: AgentState) -> str:
        return "finalize" if state.get("failure_reason") else "evidence_check"

    def after_evidence(state: AgentState) -> str:
        return "finalize" if state.get("failure_reason") else "generate"

    builder.add_node("parse", parse_request)
    builder.add_node("permission", permission_check)
    builder.add_node("retrieve", retrieve_evidence)
    builder.add_node("evidence_check", evidence_check)
    builder.add_node("generate", generate_answer)
    builder.add_node("citation", citation_check)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "permission")
    builder.add_conditional_edges("permission", after_permission, {"retrieve": "retrieve", "finalize": "finalize"})
    builder.add_conditional_edges("retrieve", after_retrieve, {"evidence_check": "evidence_check", "finalize": "finalize"})
    builder.add_conditional_edges("evidence_check", after_evidence, {"generate": "generate", "finalize": "finalize"})
    builder.add_edge("generate", "citation")
    builder.add_edge("citation", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
