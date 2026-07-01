from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest, TaskProfile
from services.agent_orchestrator.clients import ModelClient, ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import estimate_tokens, initial_state, merge_metadata, tool_result, tool_succeeded
from services.agent_orchestrator.workflows.knowledge_qa import run_knowledge_qa_workflow
from services.agent_orchestrator.workflows.media_generation import run_media_generation_workflow


async def run_document_writing_workflow(
    request: AgentInvocationRequest,
    tool_client: ToolClient,
    model_client: ModelClient,
) -> AgentState:
    graph = _build_graph(tool_client, model_client)
    return await graph.ainvoke(initial_state(request))


def _build_graph(tool_client: ToolClient, model_client: ModelClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_writing_task(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseWritingTask",
                    event_type="node_end",
                    output_summary="parsed document writing task",
                    started_at=started_at,
                    metadata={"task_type": request.task_profile.task_type},
                )
            ]
        }

    async def build_document_plan(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        sections = _sections(request.input)
        plan = {
            "sections": sections,
            "need_knowledge": _needs_knowledge(request),
            "need_media": _needs_media(request.input),
            "acceptanceCriteriaItems": [f"包含 {section} 章节" for section in sections],
        }
        return {
            "metadata": merge_metadata(state, document_plan=plan, required_sections=sections),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="BuildDocumentPlan",
                    event_type="node_end",
                    output_summary=f"sections={len(sections)}, knowledge={plan['need_knowledge']}, media={plan['need_media']}",
                    started_at=started_at,
                    metadata={"document_plan": plan},
                )
            ],
        }

    async def need_knowledge(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        needed = bool(state.get("metadata", {}).get("document_plan", {}).get("need_knowledge"))
        return {
            "metadata": merge_metadata(state, need_knowledge=needed),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="NeedKnowledge",
                    event_type="node_end",
                    output_summary=f"need_knowledge={needed}",
                    started_at=started_at,
                )
            ],
        }

    async def a2a_knowledge_request(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        workflow = "large_knowledge_qa_workflow" if _deep_request(request.input) else "knowledge_qa_workflow"
        subrequest = _subrequest(request, workflow, "knowledge_qa", request.input)
        result = await run_knowledge_qa_workflow(subrequest, tool_client)
        citations = list(result.get("citations", []))
        citation_refs = [item.model_dump(mode="json") for item in citations]
        child_tools = list(result.get("tools", []))
        context = {
            "workflow": workflow,
            "answer": result.get("answer", ""),
            "status": result.get("status", "success"),
            "citations": citation_refs,
            "child_nodes": [event.node for event in result.get("trace_events", [])],
            "child_tools": child_tools,
            "child_trace_id": subrequest.trace_id,
        }
        child_workflows = list(state.get("metadata", {}).get("child_workflows", [])) + [workflow]
        child_workflow_tools = sorted(set(state.get("metadata", {}).get("child_workflow_tools", []) + child_tools))
        return {
            "citations": citations,
            "metadata": merge_metadata(
                state,
                knowledge_context=context,
                citation_refs=citation_refs,
                citation_source="a2a_knowledge",
                child_workflows=child_workflows,
                child_workflow_tools=child_workflow_tools,
            ),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="A2AKnowledgeRequest",
                    event_type="node_end",
                    output_summary=f"{workflow} status={context['status']}",
                    started_at=started_at,
                    metadata=context,
                )
            ],
        }

    async def need_media(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        needed = bool(state.get("metadata", {}).get("document_plan", {}).get("need_media"))
        return {
            "metadata": merge_metadata(state, need_media=needed),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="NeedMedia",
                    event_type="node_end",
                    output_summary=f"need_media={needed}",
                    started_at=started_at,
                )
            ],
        }

    async def a2a_media_request(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        subrequest = _subrequest(request, "media_generation_workflow", "media_generation", request.input, {"media_type": "image"})
        result = await run_media_generation_workflow(subrequest, tool_client)
        child_tools = list(result.get("tools", []))
        asset = {
            "workflow": "media_generation_workflow",
            "status": result.get("status", "success"),
            "asset_metadata": result.get("metadata", {}).get("asset_metadata"),
            "child_nodes": [event.node for event in result.get("trace_events", [])],
            "child_tools": child_tools,
        }
        child_workflows = list(state.get("metadata", {}).get("child_workflows", [])) + ["media_generation_workflow"]
        child_workflow_tools = sorted(set(state.get("metadata", {}).get("child_workflow_tools", []) + child_tools))
        return {
            "metadata": merge_metadata(
                state,
                media_asset=asset,
                child_workflows=child_workflows,
                child_workflow_tools=child_workflow_tools,
            ),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="A2AMediaRequest",
                    event_type="node_end",
                    output_summary=f"media status={asset['status']}",
                    started_at=started_at,
                    metadata=asset,
                )
            ],
        }

    async def compose_document(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        metadata = state.get("metadata", {})
        plan = metadata.get("document_plan", {})
        sections = plan.get("sections", _sections(request.input))
        knowledge = metadata.get("knowledge_context", {})
        media = metadata.get("media_asset")
        prompt = f"Write a concise technical document with sections {sections}. Context: {str(knowledge.get('answer', ''))[:500]}"
        response = await model_client.invoke(prompt, model="writing", metadata={"profile": "writing"})
        document = _render_document(sections, str(response.get("answer", "")), knowledge, media)
        tokens = estimate_tokens(prompt + document)
        return {
            "answer": document,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "metadata": merge_metadata(state, model_response=response),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ComposeDocument",
                    event_type="model_call",
                    output_summary="composed document",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                    metadata={"model_response": response},
                )
            ],
        }

    async def inspection_tool_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        sections = state.get("metadata", {}).get("required_sections", _sections(request.input))
        response = await tool_client.invoke(
            request,
            "inspection_tool",
            {"document_text": state.get("answer", ""), "required_sections": sections, "check_type": "sections"},
        )
        result = tool_result(response)
        passed = bool(result.get("passed")) if tool_succeeded(response) else False
        return {
            "tools": ["inspection_tool"],
            "tool_results": [response],
            "metadata": merge_metadata(state, inspection_passed=passed, inspection_result=result, direct_tools=["inspection_tool"]),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="InspectionToolCheck",
                    event_type="tool_call",
                    output_summary=f"inspection passed={passed}",
                    status="success" if passed else "failed",
                    started_at=started_at,
                    error_type=None if passed else "inspection_failed",
                    metadata={"tool_response": response},
                )
            ],
        }

    async def document_verification(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        metadata = state.get("metadata", {})
        issues = _document_issues(metadata)
        inspection_failed = "inspection_failed" in issues
        can_repair = inspection_failed and state.get("repair_count", 0) < 1
        blocking = [issue for issue in issues if issue != "inspection_failed"] or (inspection_failed and not can_repair)
        failure_reason = ",".join(blocking) if blocking else None
        return {
            "should_repair": can_repair,
            "status": "success" if failure_reason is None else "failed",
            "failure_reason": failure_reason,
            "metadata": merge_metadata(state, document_verification={"passed": not issues, "issues": issues}),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="DocumentVerification",
                    event_type="node_end" if failure_reason is None else "error",
                    output_summary=f"issues={issues}, repair={can_repair}",
                    status="success" if failure_reason is None else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                )
            ],
        }

    async def repair_document(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        repair_count = int(state.get("repair_count", 0)) + 1
        missing = state.get("metadata", {}).get("inspection_result", {}).get("missing_sections", [])
        appendix = "\n".join(f"\n## {section}\n待补充。" for section in missing)
        return {
            "answer": f"{state.get('answer', '')}{appendix}",
            "repair_count": repair_count,
            "should_repair": False,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RepairDocument",
                    event_type="node_end",
                    output_summary=f"repair_count={repair_count}",
                    started_at=started_at,
                    metadata={"missing_sections": missing},
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        return {
            "answer": state.get("answer") or f"Cannot complete document writing because {failure_reason}.",
            "metadata": merge_metadata(
                state,
                direct_rag_access=False,
                direct_tools=list(state.get("tools", [])),
                child_workflows=state.get("metadata", {}).get("child_workflows", []),
                child_workflow_tools=state.get("metadata", {}).get("child_workflow_tools", []),
                repair_count=state.get("repair_count", 0),
            ),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="Finalize",
                    event_type="node_end",
                    output_summary=f"status={state.get('status', 'success')}",
                    started_at=started_at,
                    metadata={"failure_reason": failure_reason, "direct_rag_access": False},
                )
            ],
        }

    builder.add_node("parse", parse_writing_task)
    builder.add_node("plan", build_document_plan)
    builder.add_node("need_knowledge", need_knowledge)
    builder.add_node("knowledge", a2a_knowledge_request)
    builder.add_node("need_media", need_media)
    builder.add_node("media", a2a_media_request)
    builder.add_node("compose", compose_document)
    builder.add_node("inspect", inspection_tool_check)
    builder.add_node("verify", document_verification)
    builder.add_node("repair", repair_document)
    builder.add_node("finalize", finalize)
    builder.set_entry_point("parse")
    builder.add_edge("parse", "plan")
    builder.add_edge("plan", "need_knowledge")
    builder.add_conditional_edges("need_knowledge", lambda state: "knowledge" if state.get("metadata", {}).get("need_knowledge") else "need_media", {"knowledge": "knowledge", "need_media": "need_media"})
    builder.add_edge("knowledge", "need_media")
    builder.add_conditional_edges("need_media", lambda state: "media" if state.get("metadata", {}).get("need_media") else "compose", {"media": "media", "compose": "compose"})
    builder.add_edge("media", "compose")
    builder.add_edge("compose", "inspect")
    builder.add_edge("inspect", "verify")
    builder.add_conditional_edges("verify", lambda state: "repair" if state.get("should_repair") else "finalize", {"repair": "repair", "finalize": "finalize"})
    builder.add_edge("repair", "inspect")
    builder.add_edge("finalize", END)
    return builder.compile()


def _subrequest(
    request: AgentInvocationRequest,
    workflow: str,
    task_type: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentInvocationRequest:
    return AgentInvocationRequest(
        request_id=f"{request.request_id}:{workflow}",
        trace_id=request.trace_id,
        consumer_id=request.consumer_id,
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        input=text,
        selected_workflow=workflow,
        task_profile=TaskProfile(task_type=task_type),  # type: ignore[arg-type]
        permissions=request.permissions,
        allowed_tools=request.allowed_tools,
        allowed_workflows=request.allowed_workflows,
        metadata={**request.metadata, **(metadata or {}), "parent_trace_id": request.trace_id},
    )


def _sections(text: str) -> List[str]:
    return ["背景", "设计", "验收"] if any(token in text for token in ("技术", "方案", "document", "文档")) else ["概要", "正文", "结论"]


def _needs_knowledge(request: AgentInvocationRequest) -> bool:
    forced = request.metadata.get("need_knowledge")
    if isinstance(forced, bool):
        return forced
    text = request.input.lower()
    return any(token in text for token in ("semgateway", "gateway", "技术", "方案", "论文", "架构", "依据", "引用"))


def _needs_media(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("diagram", "image", "架构图", "配图", "图片"))


def _deep_request(text: str) -> bool:
    return any(token in text for token in ("深度", "全面", "系统", "论文级", "多角度"))


def _document_issues(metadata: Dict[str, Any]) -> List[str]:
    issues = []
    if not metadata.get("inspection_passed"):
        issues.append("inspection_failed")
    if metadata.get("need_knowledge") and metadata.get("knowledge_context", {}).get("status") != "success":
        issues.append("knowledge_context_failed")
    return issues


def _render_document(sections: List[str], model_answer: str, knowledge: Dict[str, Any], media: Any) -> str:
    lines = []
    knowledge_text = str(knowledge.get("answer", ""))
    for section in sections:
        lines.append(f"## {section}\n{model_answer or '根据受限上下文生成。'}")
    if knowledge_text:
        lines.append(f"## 引用上下文\n{knowledge_text}")
    citations = knowledge.get("citations") or []
    if citations:
        lines.append("## 引用\n" + "\n".join(f"- {item.get('title')} / {item.get('chunk_id')}" for item in citations))
    if media:
        asset = media.get("asset_metadata") if isinstance(media, dict) else None
        lines.append(f"## 媒体资产\nasset_status={asset.get('asset_status') if isinstance(asset, dict) else 'unavailable'}")
    return "\n\n".join(lines)
