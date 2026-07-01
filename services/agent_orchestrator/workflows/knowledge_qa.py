from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest
from services.agent_orchestrator.clients import ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import (
    estimate_tokens,
    evidence_refs,
    evidence_to_citations,
    initial_state,
    merge_metadata,
    tool_error,
    tool_result,
    tool_succeeded,
)


async def run_knowledge_qa_workflow(request: AgentInvocationRequest, tool_client: ToolClient) -> AgentState:
    graph = _build_large_graph(tool_client) if request.selected_workflow == "large_knowledge_qa_workflow" else _build_crag_graph(tool_client)
    return await graph.ainvoke(initial_state(request))


def _build_crag_graph(tool_client: ToolClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_question(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "query": request.input,
            "metadata": merge_metadata(state, workflow_dag="corrective_rag", top_k=3),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseQuestion",
                    event_type="node_end",
                    output_summary="parsed ordinary knowledge question",
                    started_at=started_at,
                )
            ],
        }

    async def plan_retrieval(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        query = state.get("query") or request.input
        return {
            "query": query,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="PlanRetrieval",
                    event_type="node_end",
                    output_summary=f"query={query[:80]}",
                    started_at=started_at,
                    metadata={"retry_count": state.get("retry_count", 0), "top_k": 3},
                )
            ],
        }

    async def retrieve_evidence(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        query = state.get("query") or request.input
        response = await tool_client.invoke(request, "doc_search_tool", {"query": query, "top_k": 3, "strategy": "hybrid"})
        evidence = tool_result(response).get("evidence", []) if tool_succeeded(response) else []
        event = workflow_event(
            request=request,
            node="RetrieveEvidence",
            event_type="tool_call",
            output_summary=f"doc_search_tool status={response.get('status')}",
            status="success" if tool_succeeded(response) else "failed",
            started_at=started_at,
            error_type=None if tool_succeeded(response) else str(response.get("error_type")),
            metadata={"tool_response": response, "query": query},
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
        refs = evidence_refs(evidence)
        event.metadata["evidence_refs"] = refs
        return {
            "evidence": evidence,
            "tools": ["doc_search_tool"],
            "tool_results": [response],
            "metadata": merge_metadata(state, evidence_refs=refs),
            "trace_events": [event],
        }

    async def evidence_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        query = state.get("query") or request.input
        response = await tool_client.invoke(
            request,
            "evidence_check_tool",
            {"claim": request.input, "evidence_query": query, "top_k": 3, "min_score": 0.6},
        )
        supported = bool(tool_result(response).get("supported")) if tool_succeeded(response) else False
        retry_count = int(state.get("retry_count", 0))
        can_rewrite = not supported and retry_count < 1
        failure_reason = None if supported or can_rewrite else "insufficient_evidence"
        return {
            "should_retry": can_rewrite,
            "status": "success" if failure_reason is None else "refused",
            "failure_reason": failure_reason,
            "tools": ["evidence_check_tool"],
            "tool_results": [response],
            "metadata": merge_metadata(state, evidence_supported=supported),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="EvidenceCheck",
                    event_type="tool_call" if supported or can_rewrite else "error",
                    output_summary=f"supported={supported}, rewrite={can_rewrite}",
                    status="success" if supported or can_rewrite else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                    metadata={"tool_response": response, "retry_count": retry_count},
                )
            ],
        }

    async def rewrite_query(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        retry_count = int(state.get("retry_count", 0)) + 1
        query = f"{request.input} 关键事实 依据 来源"
        return {
            "query": query,
            "retry_count": retry_count,
            "should_retry": False,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RewriteQuery",
                    event_type="node_end",
                    output_summary=f"retry_count={retry_count}",
                    started_at=started_at,
                    metadata={"query": query},
                )
            ],
        }

    async def generate_answer(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        evidence = state.get("evidence", [])
        snippets = [str(item.get("text", "")).strip() for item in evidence[:3] if item.get("text")]
        answer = "基于检索证据回答：" + ("；".join(snippets) if snippets else "没有可用证据。")
        tokens = estimate_tokens(request.input + answer)
        return {
            "answer": answer,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="GenerateAnswer",
                    event_type="node_end",
                    output_summary="generated answer from retrieved evidence",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                )
            ],
        }

    async def citation_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        citations = evidence_to_citations(state.get("evidence", []))
        refs = [citation.model_dump(mode="json") for citation in citations]
        can_repair = not citations and state.get("repair_count", 0) < 1 and bool(state.get("evidence"))
        failure_reason = None if citations or can_repair else "missing_citation"
        return {
            "citations": citations,
            "should_repair": can_repair,
            "status": "success" if failure_reason is None else "refused",
            "failure_reason": failure_reason,
            "metadata": merge_metadata(state, citation_count=len(citations), citation_refs=refs),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="CitationCheck",
                    event_type="node_end" if citations or can_repair else "error",
                    output_summary=f"citations={len(citations)}, repair={can_repair}",
                    status="success" if citations or can_repair else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                    metadata={"citation_refs": refs},
                )
            ],
        }

    async def repair_with_evidence(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "repair_count": int(state.get("repair_count", 0)) + 1,
            "should_repair": False,
            "failure_reason": None,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RepairWithEvidence",
                    event_type="node_end",
                    output_summary="rebuilding citations from evidence",
                    started_at=started_at,
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        answer = state.get("answer") or f"Cannot answer because {failure_reason}."
        return {
            "answer": answer,
            "metadata": merge_metadata(
                state,
                retry_count=state.get("retry_count", 0),
                repair_count=state.get("repair_count", 0),
            ),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="Finalize",
                    event_type="node_end",
                    output_summary=f"status={state.get('status', 'success')}",
                    started_at=started_at,
                    metadata={"failure_reason": failure_reason},
                )
            ],
        }

    builder.add_node("parse", parse_question)
    builder.add_node("plan", plan_retrieval)
    builder.add_node("retrieve", retrieve_evidence)
    builder.add_node("check", evidence_check)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("generate", generate_answer)
    builder.add_node("citation", citation_check)
    builder.add_node("repair", repair_with_evidence)
    builder.add_node("finalize", finalize)
    builder.set_entry_point("parse")
    builder.add_edge("parse", "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges("retrieve", lambda state: "finalize" if state.get("failure_reason") else "check", {"check": "check", "finalize": "finalize"})
    builder.add_conditional_edges("check", _after_evidence_check, {"rewrite": "rewrite", "generate": "generate", "finalize": "finalize"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", "citation")
    builder.add_conditional_edges("citation", _after_citation_check, {"repair": "repair", "finalize": "finalize"})
    builder.add_edge("repair", "citation")
    builder.add_edge("finalize", END)
    return builder.compile()


def _build_large_graph(tool_client: ToolClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_deep_question(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "query": request.input,
            "metadata": merge_metadata(state, workflow_dag="bounded_deep_research", top_k=4),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseDeepQuestion",
                    event_type="node_end",
                    output_summary="parsed deep knowledge question",
                    started_at=started_at,
                )
            ],
        }

    async def build_research_brief(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        brief = f"Research brief: answer with multiple angles and cite retrieved project evidence. Question: {request.input}"
        return {
            "metadata": merge_metadata(state, research_brief=brief),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="BuildResearchBrief",
                    event_type="node_end",
                    output_summary=brief[:120],
                    started_at=started_at,
                )
            ],
        }

    async def split_subquestions(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        subquestions = _split_subquestions(request.input)
        return {
            "metadata": merge_metadata(state, subquestions=subquestions),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="SplitSubQuestions",
                    event_type="node_end",
                    output_summary=f"subquestions={len(subquestions)}",
                    started_at=started_at,
                    metadata={"subquestions": subquestions, "max": 4},
                )
            ],
        }

    async def parallel_retrieve(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        subquestions = state.get("metadata", {}).get("subquestions") or [request.input]
        responses = await asyncio.gather(
            *[
                tool_client.invoke(request, "doc_search_tool", {"query": question, "top_k": 4, "strategy": "hybrid"})
                for question in subquestions[:4]
            ]
        )
        evidence = []
        for response in responses:
            if tool_succeeded(response):
                evidence.extend(tool_result(response).get("evidence", []))
        failed = [tool_error(response) for response in responses if not tool_succeeded(response)]
        status_value = "success" if evidence else "refused"
        refs = evidence_refs(evidence)
        return {
            "evidence": evidence,
            "status": status_value,
            "failure_reason": None if evidence else "retrieval_failed",
            "tools": ["doc_search_tool"],
            "tool_results": responses,
            "errors": failed,
            "metadata": merge_metadata(state, evidence_refs=refs),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParallelRetrieveEvidence",
                    event_type="tool_call" if evidence else "error",
                    output_summary=f"subquestions={len(subquestions[:4])}, evidence={len(evidence)}",
                    status="success" if evidence else "failed",
                    started_at=started_at,
                    error_type=None if evidence else "retrieval_failed",
                    metadata={"tool_responses": responses, "evidence_refs": refs},
                )
            ],
        }

    async def evidence_aggregate(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        evidence = _dedupe_evidence(state.get("evidence", []))
        response = await tool_client.invoke(
            request,
            "evidence_check_tool",
            {"claim": request.input, "evidence_query": request.input, "top_k": min(6, max(1, len(evidence))), "min_score": 0.6},
        )
        supported = bool(tool_result(response).get("supported")) if tool_succeeded(response) else False
        failure_reason = None if evidence and supported else "insufficient_evidence"
        refs = evidence_refs(evidence)
        return {
            "evidence": evidence,
            "status": "success" if failure_reason is None else "refused",
            "failure_reason": failure_reason,
            "tools": ["evidence_check_tool"],
            "tool_results": [response],
            "metadata": merge_metadata(
                state,
                evidence_supported=supported,
                aggregated_evidence_count=len(evidence),
                evidence_refs=refs,
            ),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="EvidenceAggregate",
                    event_type="tool_call" if failure_reason is None else "error",
                    output_summary=f"evidence={len(evidence)}, supported={supported}",
                    status="success" if failure_reason is None else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                    metadata={"tool_response": response, "evidence_refs": refs},
                )
            ],
        }

    async def synthesize_answer(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        evidence = state.get("evidence", [])
        answer = _synthesize_from_evidence(evidence)
        tokens = estimate_tokens(request.input + answer)
        return {
            "answer": answer,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="SynthesizeAnswer",
                    event_type="node_end",
                    output_summary=f"evidence_items={len(evidence)}",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                )
            ],
        }

    async def citation_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        citations = evidence_to_citations(state.get("evidence", []))
        refs = [citation.model_dump(mode="json") for citation in citations]
        failure_reason = None if citations else "missing_citation"
        return {
            "citations": citations,
            "status": "success" if failure_reason is None else "refused",
            "failure_reason": failure_reason,
            "metadata": merge_metadata(state, citation_count=len(citations), citation_refs=refs),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="CitationCheck",
                    event_type="node_end" if citations else "error",
                    output_summary=f"citations={len(citations)}",
                    status="success" if citations else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                    metadata={"citation_refs": refs},
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        return {
            "answer": state.get("answer") or f"Cannot complete deep knowledge QA because {failure_reason}.",
            "metadata": merge_metadata(state, aggregated_evidence_count=len(state.get("evidence", []))),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="Finalize",
                    event_type="node_end",
                    output_summary=f"status={state.get('status', 'success')}",
                    started_at=started_at,
                    metadata={"failure_reason": failure_reason},
                )
            ],
        }

    builder.add_node("parse", parse_deep_question)
    builder.add_node("brief", build_research_brief)
    builder.add_node("split", split_subquestions)
    builder.add_node("retrieve", parallel_retrieve)
    builder.add_node("aggregate", evidence_aggregate)
    builder.add_node("synthesize", synthesize_answer)
    builder.add_node("citation", citation_check)
    builder.add_node("finalize", finalize)
    builder.set_entry_point("parse")
    builder.add_edge("parse", "brief")
    builder.add_edge("brief", "split")
    builder.add_edge("split", "retrieve")
    builder.add_conditional_edges("retrieve", lambda state: "finalize" if state.get("failure_reason") else "aggregate", {"aggregate": "aggregate", "finalize": "finalize"})
    builder.add_conditional_edges("aggregate", lambda state: "finalize" if state.get("failure_reason") else "synthesize", {"synthesize": "synthesize", "finalize": "finalize"})
    builder.add_edge("synthesize", "citation")
    builder.add_edge("citation", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _after_evidence_check(state: AgentState) -> str:
    if state.get("failure_reason"):
        return "finalize"
    return "rewrite" if state.get("should_retry") else "generate"


def _after_citation_check(state: AgentState) -> str:
    if state.get("should_repair"):
        return "repair"
    return "finalize"


def _split_subquestions(question: str) -> List[str]:
    seeds = [
        f"{question} 核心事实",
        f"{question} 设计依据",
        f"{question} 风险和限制",
        f"{question} 对比和取舍",
    ]
    return seeds[:4]


def _dedupe_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in evidence:
        key = (item.get("source_id"), item.get("chunk_id"), item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _synthesize_from_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    lines = ["基于多子问题检索后的聚合证据，结论如下："]
    for index, item in enumerate(evidence[:4], start=1):
        text = str(item.get("text", "")).strip()
        title = str(item.get("title", "unknown"))
        lines.append(f"{index}. {title}: {text}")
    return "\n".join(lines)
