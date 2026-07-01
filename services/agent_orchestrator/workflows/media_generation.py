from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.schemas import AgentInvocationRequest
from services.agent_orchestrator.clients import ToolClient
from services.agent_orchestrator.state import AgentState
from services.agent_orchestrator.trace import started_timer, workflow_event
from services.agent_orchestrator.workflows.common import (
    estimate_tokens,
    initial_state,
    merge_metadata,
    tool_result,
    tool_succeeded,
)


async def run_media_generation_workflow(request: AgentInvocationRequest, tool_client: ToolClient) -> AgentState:
    graph = _build_graph(tool_client)
    return await graph.ainvoke(initial_state(request))


def _build_graph(tool_client: ToolClient) -> Any:
    builder = StateGraph(AgentState)

    async def parse_request(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        media_type = _media_type(request)
        target_tool = "video_generation_tool" if media_type == "video" else "image_generation_tool"
        return {
            "metadata": merge_metadata(state, media_type=media_type, target_tool=target_tool, media_prompt=request.input),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="ParseMediaRequest",
                    event_type="node_end",
                    output_summary=f"media_type={media_type}",
                    started_at=started_at,
                    metadata={"media_type": media_type, "target_tool": target_tool},
                )
            ],
        }

    async def build_generation_params(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        metadata = state.get("metadata", {})
        media_type = str(metadata.get("media_type", _media_type(request)))
        params = {
            "prompt": str(metadata.get("media_prompt") or request.input),
            "media_type": media_type,
            "size": request.metadata.get("size", "1024x1024" if media_type == "image" else "1280x720"),
            "duration_s": request.metadata.get("duration_s", 5 if media_type == "video" else None),
            "seed": request.metadata.get("seed"),
        }
        return {
            "metadata": merge_metadata(state, generation_params=params),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="BuildGenerationParams",
                    event_type="node_end",
                    output_summary=f"media_type={media_type}",
                    started_at=started_at,
                    metadata={"generation_params": params},
                )
            ],
        }

    async def prompt_safety_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        blocked = _blocked_prompt(request.input)
        return {
            "status": "refused" if blocked else "success",
            "failure_reason": "prompt_safety_blocked" if blocked else None,
            "trace_events": [
                workflow_event(
                    request=request,
                    node="PromptSafetyCheck",
                    event_type="error" if blocked else "node_end",
                    output_summary="prompt blocked" if blocked else "prompt accepted",
                    status="failed" if blocked else "success",
                    started_at=started_at,
                    error_type="prompt_safety_blocked" if blocked else None,
                )
            ],
        }

    async def invoke_media_backend(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        metadata = state.get("metadata", {})
        media_type = str(metadata.get("media_type", _media_type(request)))
        target_tool = str(metadata.get("target_tool", "video_generation_tool" if media_type == "video" else "image_generation_tool"))
        params = metadata.get("generation_params", {})
        prompt = str(params.get("prompt") or metadata.get("media_prompt") or request.input)
        response = await tool_client.invoke(request, target_tool, {"prompt": prompt, "media_type": media_type})
        success = tool_succeeded(response)
        result = tool_result(response)
        asset_metadata = _asset_metadata(result, target_tool, media_type, params, success)
        return {
            "status": "success" if success else "failed",
            "failure_reason": None if success else response.get("error_type") or response.get("status"),
            "tools": [target_tool],
            "tool_results": [response],
            "metadata": merge_metadata(state, media_type=media_type, target_tool=target_tool, asset_metadata=asset_metadata),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="InvokeMediaBackend",
                    event_type="tool_call",
                    output_summary=f"{target_tool} status={response.get('status')}",
                    status="success" if success else "failed",
                    started_at=started_at,
                    error_type=None if success else str(response.get("error_type") or response.get("status")),
                    metadata={"tool_response": response, "asset_metadata": asset_metadata},
                )
            ],
        }

    async def build_asset_metadata(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        asset_metadata = state.get("metadata", {}).get("asset_metadata", {})
        return {
            "trace_events": [
                workflow_event(
                    request=request,
                    node="AssetMetadataBuild",
                    event_type="node_end",
                    output_summary=f"asset_status={asset_metadata.get('asset_status')}",
                    started_at=started_at,
                    metadata={"asset_metadata": asset_metadata},
                )
            ]
        }

    async def media_output_check(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        asset_metadata = state.get("metadata", {}).get("asset_metadata", {})
        ok = bool(asset_metadata.get("asset_status")) and bool(asset_metadata.get("model_backend"))
        can_retry = not ok and state.get("repair_count", 0) < 1 and not state.get("failure_reason")
        failure_reason = state.get("failure_reason") or (None if ok or can_retry else "invalid_asset_metadata")
        return {
            "should_retry": can_retry,
            "status": "success" if failure_reason is None else "failed",
            "failure_reason": failure_reason,
            "metadata": merge_metadata(state, media_output_valid=ok),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="MediaOutputCheck",
                    event_type="node_end" if ok or can_retry else "error",
                    output_summary=f"valid={ok}, retry={can_retry}, asset_status={asset_metadata.get('asset_status')}",
                    status="success" if ok or can_retry else "failed",
                    started_at=started_at,
                    error_type=failure_reason,
                )
            ],
        }

    async def refine_prompt(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        repair_count = int(state.get("repair_count", 0)) + 1
        prompt = f"{request.input}\nRefined request: normalize this into backend generation parameters."
        return {
            "repair_count": repair_count,
            "should_retry": False,
            "metadata": merge_metadata(state, media_prompt=prompt),
            "trace_events": [
                workflow_event(
                    request=request,
                    node="RefineMediaPrompt",
                    event_type="node_end",
                    output_summary=f"repair_count={repair_count}",
                    started_at=started_at,
                )
            ],
        }

    async def finalize(state: AgentState) -> AgentState:
        request = state["request"]
        started_at = started_timer()
        metadata = state.get("metadata", {})
        asset_metadata = metadata.get("asset_metadata", {})
        failure_reason = state.get("failure_reason")
        if failure_reason:
            answer = f"Cannot complete media generation because {failure_reason}."
        else:
            answer = (
                f"Media generation request processed by {asset_metadata.get('model_backend')}. "
                f"asset_status={asset_metadata.get('asset_status')}."
            )
        tokens = estimate_tokens(request.input + answer)
        return {
            "answer": answer,
            "metrics_tokens": tokens,
            "metrics_cost": round(tokens * 0.000001, 6),
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

    def after_safety(state: AgentState) -> str:
        return "finalize" if state.get("failure_reason") else "params"

    def after_invoke(state: AgentState) -> str:
        return "asset"

    builder.add_node("parse", parse_request)
    builder.add_node("safety", prompt_safety_check)
    builder.add_node("params", build_generation_params)
    builder.add_node("invoke", invoke_media_backend)
    builder.add_node("asset", build_asset_metadata)
    builder.add_node("check", media_output_check)
    builder.add_node("refine", refine_prompt)
    builder.add_node("finalize", finalize)
    builder.set_entry_point("parse")
    builder.add_edge("parse", "safety")
    builder.add_conditional_edges("safety", after_safety, {"params": "params", "finalize": "finalize"})
    builder.add_edge("params", "invoke")
    builder.add_conditional_edges("invoke", after_invoke, {"asset": "asset", "finalize": "finalize"})
    builder.add_edge("asset", "check")
    builder.add_conditional_edges("check", lambda state: "refine" if state.get("should_retry") else "finalize", {"refine": "refine", "finalize": "finalize"})
    builder.add_edge("refine", "params")
    builder.add_edge("finalize", END)
    return builder.compile()


def _media_type(request: AgentInvocationRequest) -> str:
    metadata_media_type = request.metadata.get("media_type")
    if metadata_media_type in {"image", "video"}:
        return str(metadata_media_type)
    lowered = request.input.lower()
    return "video" if any(token in lowered for token in ("video", "storyboard", "视频", "分镜")) else "image"


def _blocked_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("illegal", "self-harm", "自残", "违法"))


def _asset_metadata(result: Dict[str, Any], target_tool: str, media_type: str, params: Dict[str, Any], success: bool) -> Dict[str, Any]:
    implementation_status = str(result.get("implementation_status", "placeholder"))
    asset_status = "unavailable"
    if success and implementation_status == "placeholder":
        asset_status = "placeholder"
    elif success:
        asset_status = "generated"
    return {
        **result,
        "asset_status": asset_status,
        "asset_uri": result.get("asset_uri"),
        "media_type": media_type,
        "model_backend": target_tool,
        "prompt_summary": str(params.get("prompt", ""))[:120],
        "size": params.get("size"),
        "duration_s": params.get("duration_s"),
        "seed": params.get("seed"),
        "implementation_status": implementation_status,
    }
