from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest
from services.agent_orchestrator.clients import ModelClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import estimate_tokens, initial_state, merge_metadata


async def run_coding_workflow(request: AgentInvocationRequest, model_client: ModelClient) -> AgentState:
    graph = _build_graph(model_client)
    return await graph.ainvoke(initial_state(request))


def _build_graph(model_client: ModelClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_task(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseCodingTask",
                    event_type="node_end",
                    output_summary="parsed coding request",
                    started_at=started_at,
                    metadata={"task_type": request.task_profile.task_type},
                )
            ]
        }

    async def build_context(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        snippets = _metadata_snippets(request.metadata)
        context = {
            "input": request.input,
            "snippets": snippets,
            "mode": "read_only",
            "file_write_enabled": False,
            "shell_enabled": False,
        }
        return {
            "metadata": merge_metadata(state, code_context=context),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="BuildCodeContext",
                    event_type="node_end",
                    output_summary="built read-only code context",
                    started_at=started_at,
                    metadata=context,
                )
            ],
        }

    async def invoke_model(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        context = state.get("metadata", {}).get("code_context", {})
        prompt = (
            "Return a bounded coding response with summary, answer or patch_text, test_plan, and change_boundary. "
            "Do not write files, execute shell commands, or suggest destructive commands.\n\n"
            f"Repair instruction: {state.get('repair_instruction', 'none')}\n\n"
            f"Context snippets:\n{context.get('snippets', [])}\n\n"
            f"User request:\n{request.input}"
        )
        response = await model_client.invoke(prompt, model="coding", metadata={"profile": "coding"})
        model_answer = str(response.get("answer", ""))
        tokens = int(response.get("estimated_prompt_tokens", 0)) + int(response.get("estimated_completion_tokens", 0))
        if tokens <= 0:
            tokens = estimate_tokens(prompt + model_answer)
        return {
            "draft": model_answer,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
            "metadata": merge_metadata(state, model_response=response),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="InvokeCodingModel",
                    event_type="model_call",
                    output_summary=f"model={response.get('model')}, provider={response.get('provider')}",
                    started_at=started_at,
                    estimated_tokens=tokens,
                    estimated_cost=round(tokens * 0.000001, 6),
                    metadata={"model_response": response},
                )
            ],
        }

    async def structure_output(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        draft = state.get("draft", "")
        test_plan = {"status": "suggested" if _wants_tests(request.input) else "required", "items": ["Run the focused test or review the patch against the requested behavior."]}
        structured = {
            "summary": "Prepared a bounded coding response without file writes or shell execution.",
            "answer": draft,
            "patch_text": _extract_label(draft, "patch_text"),
            "test_plan": test_plan,
            "change_boundary": "read_only_patch_suggestion",
        }
        answer = _render_answer(structured)
        tokens = estimate_tokens(request.input + answer)
        return {
            "answer": answer,
            "metrics_tokens": max(int(state.get("metrics_tokens") or 0), tokens),
            "metrics_cost": round(max(int(state.get("metrics_tokens") or 0), tokens) * 0.000001, 6),
            "metadata": merge_metadata(state, coding_output=structured),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="StructurePatchOrAnswer",
                    event_type="node_end",
                    output_summary="structured coding output",
                    started_at=started_at,
                    metadata={"has_patch_text": bool(structured["patch_text"]), "test_plan_status": test_plan["status"]},
                )
            ],
        }

    async def output_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        coding_output = state.get("metadata", {}).get("coding_output", {})
        text = f"{coding_output.get('answer', '')} {coding_output.get('patch_text', '')}".lower()
        unsafe = any(token in text for token in ("rm -rf", "sudo ", "curl | sh", "格式化磁盘", "mkfs", "shutdown "))
        test_plan = coding_output.get("test_plan")
        ok = (
            bool(coding_output.get("summary"))
            and bool(coding_output.get("answer") or coding_output.get("patch_text"))
            and isinstance(test_plan, dict)
            and bool(test_plan.get("items"))
            and bool(coding_output.get("change_boundary"))
            and not unsafe
        )
        can_repair = not ok and state.get("repair_count", 0) < 1
        failure_reason = None if ok or can_repair else "invalid_coding_output"
        return {
            "should_repair": can_repair,
            "status": "success" if failure_reason is None else "failed",
            "failure_reason": failure_reason,
            "metadata": merge_metadata(state, coding_output_valid=ok, unsafe_output=unsafe),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="CodingOutputCheck",
                    event_type="node_end" if ok or can_repair else "error",
                    output_summary=f"valid={ok}, repair={can_repair}",
                    status="success" if ok or can_repair else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                )
            ],
        }

    async def repair_prompt(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        repair_count = int(state.get("repair_count", 0)) + 1
        return {
            "repair_count": repair_count,
            "should_repair": False,
            "repair_instruction": "Return non-empty summary plus answer or patch_text, include test_plan.items and change_boundary, avoid shell/file execution.",
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RepairStructuredOutput",
                    event_type="node_end",
                    output_summary=f"repair_count={repair_count}",
                    started_at=started_at,
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        failure_reason = state.get("failure_reason")
        answer = state.get("answer") or f"Cannot complete coding task because {failure_reason}."
        return {
            "answer": answer,
            "metadata": merge_metadata(
                state,
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

    builder.add_node("parse", parse_task)
    builder.add_node("context", build_context)
    builder.add_node("model", invoke_model)
    builder.add_node("structure", structure_output)
    builder.add_node("check", output_check)
    builder.add_node("repair", repair_prompt)
    builder.add_node("finalize", finalize)
    builder.set_entry_point("parse")
    builder.add_edge("parse", "context")
    builder.add_edge("context", "model")
    builder.add_edge("model", "structure")
    builder.add_edge("structure", "check")
    builder.add_conditional_edges("check", lambda state: "repair" if state.get("should_repair") else "finalize", {"repair": "repair", "finalize": "finalize"})
    builder.add_edge("repair", "model")
    builder.add_edge("finalize", END)
    return builder.compile()


def _wants_tests(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("test", "unit", "pytest", "测试", "单元测试"))


def _render_answer(output: Dict[str, Any]) -> str:
    parts = [str(output["summary"]), str(output["answer"])]
    if output.get("patch_text"):
        parts.append(str(output["patch_text"]))
    test_plan = output.get("test_plan", {})
    parts.append(f"test_plan: {test_plan.get('status', 'unknown')}")
    parts.append(f"change_boundary: {output.get('change_boundary', 'unknown')}")
    return "\n\n".join(part for part in parts if part)


def _metadata_snippets(metadata: Dict[str, Any]) -> List[str]:
    snippets = []
    for key in ("code", "code_snippet", "snippet"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            snippets.append(value.strip()[:2000])
    files = metadata.get("files")
    if isinstance(files, list):
        snippets.extend(str(item)[:2000] for item in files[:3])
    return snippets


def _extract_label(text: str, label: str) -> str:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.strip().lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""
