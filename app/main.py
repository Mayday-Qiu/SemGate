from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import ValidationError

from app.agentic_router import AgenticRouter
from app.auth import verify_api_key
from app.config import load_settings
from app.logging_utils import write_jsonl_record
from app.memory_planner import MemoryPlanner, MemoryPlannerResult
from app.plan_validator import PlanValidator
from app.planner_context import PlannerContextBuilder
from app.planner_memory_store import PlannerMemoryStore
from app.planner_policy import PlannerPolicy
from app.rate_limit import RateLimitConfig, RateLimitDecision, TokenBucketRateLimiter
from app.runtime_metrics import RuntimeMetricsStore
from app.schemas import (
    AgentInvocationRequest,
    AgentWorkflowResponse,
    AgenticGatewayRequest,
    AgenticGatewayResponse,
    AgenticMetrics,
    AuthenticatedConsumer,
    GatewayPreviewResponse,
    PlanValidationResult,
    PlannerPolicyDecision,
    TaskContract,
    TaskPlan,
)
from app.task_contract import TaskContractBuilder
from app.task_planner import TaskPlanner
from app.task_profile import TaskProfileBuilder
from app.trace_collector import TraceCollector
from app.verification import VerificationGate
from app.workflow_profiles import load_workflow_profiles, validate_workflow_tool_definitions


settings = load_settings()
task_profile_builder = TaskProfileBuilder()
task_contract_builder = TaskContractBuilder()
verification_gate = VerificationGate()
workflow_profile_store = load_workflow_profiles(settings.workflow_profiles_path)
validate_workflow_tool_definitions(workflow_profile_store, task_profile_builder.known_required_tools())
runtime_metrics_store = RuntimeMetricsStore()
agentic_router = AgenticRouter(runtime_metrics_store)
planner_memory_store = PlannerMemoryStore(settings.planner_memory_dir)
memory_planner = MemoryPlanner()
planner_policy = PlannerPolicy()
planner_context_builder = PlannerContextBuilder()
plan_validator = PlanValidator()
task_planner = TaskPlanner(
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    model_id=settings.planner_model_id,
    temperature=settings.planner_temperature,
    top_p=settings.planner_top_p,
    max_tokens=settings.planner_max_tokens,
    timeout_s=settings.planner_timeout_s,
    enable_thinking=settings.planner_enable_thinking,
)
rate_limiter = TokenBucketRateLimiter(
    RateLimitConfig(
        enabled=settings.rate_limit_enabled,
        replenish_rate=settings.rate_limit_replenish_rate,
        burst_capacity=settings.rate_limit_burst_capacity,
        requested_tokens=settings.rate_limit_requested_tokens,
    )
)
app = FastAPI(title="SemGateway", version="1.0.0")


@dataclass
class AgentCallError(Exception):
    error_type: str
    message: str
    http_status_code: int
    status_code: Optional[int] = None


@dataclass
class PlannerArtifacts:
    task_profile: Any
    policy: PlannerPolicyDecision
    context_summary: Dict[str, Any]
    raw_plan: Optional[Dict[str, Any]]
    validated_plan: Optional[TaskPlan]
    validation: Optional[PlanValidationResult]
    route_hints: List[Dict[str, Any]]


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/v1/preview", response_model=GatewayPreviewResponse)
async def preview_v1(
    request: AgenticGatewayRequest,
    consumer: AuthenticatedConsumer = Depends(verify_api_key),
) -> GatewayPreviewResponse:
    request_id = request.request_id or str(uuid4())
    trace_id = str(uuid4())
    request = request.model_copy(update={"request_id": request_id})
    task_profile = task_profile_builder.build(request)
    user_permissions = _effective_permissions(
        task_profile_builder.resolve_user_permissions(request),
        consumer.permissions,
    )
    planner_artifacts = await _run_planner_pipeline(
        request=request,
        task_profile=task_profile,
        user_permissions=user_permissions,
        trace=None,
    )
    task_profile = planner_artifacts.task_profile
    memory_rules = _load_planner_memory()
    memory_result = _plan_memory_route(task_profile, memory_rules, task_plan=planner_artifacts.validated_plan)
    route_hints = list(planner_artifacts.route_hints)
    if not memory_result.warnings:
        route_hints.extend(memory_result.route_hints)
    route_decision = agentic_router.select(
        trace_id=trace_id,
        task_profile=task_profile,
        workflow_profiles=workflow_profile_store.all(),
        user_permissions=user_permissions,
        consumer=consumer,
        route_hints=route_hints,
    )
    memory_result = _plan_memory_contract(
        memory_result,
        task_profile,
        route_decision.selected_workflow,
        memory_rules,
        task_plan=planner_artifacts.validated_plan,
    )
    task_contract = task_contract_builder.build(
        request,
        task_profile,
        route_decision.selected_workflow,
        contract_patches=memory_result.contract_patches,
        validated_task_plan=planner_artifacts.validated_plan,
    )
    preview_status, reasons, next_actions = task_contract_builder.preview(
        task_contract,
        user_permissions,
        route_decision.selection_reason,
    )
    if planner_artifacts.validation is not None and planner_artifacts.validation.status == "failed":
        preview_status = "blocked"
        reasons = list(reasons) + [f"plan validation failed: {', '.join(planner_artifacts.validation.errors)}"]
        next_actions = ["fix TaskPlanner output or planner context before invoke"]
    if memory_result.warnings:
        reasons = list(reasons) + [f"planner memory warning: {warning}" for warning in memory_result.warnings]
        preview_status = "blocked"
    return GatewayPreviewResponse(
        request_id=request_id,
        trace_id=trace_id,
        preview_status=preview_status,
        selected_workflow=route_decision.selected_workflow,
        task_profile=task_profile,
        task_contract=task_contract,
        route_decision=route_decision,
        required_permissions=task_contract.required_permissions,
        missing_permissions=task_contract_builder.missing_permissions(task_contract, user_permissions),
        required_tools=task_contract.required_tools,
        acceptance_criteria=task_contract.acceptance_criteria,
        reasons=reasons,
        next_actions=next_actions,
        memory_planner=_planner_memory_metadata(memory_result, memory_rules),
        planner_policy=planner_artifacts.policy,
        planner_context_summary=planner_artifacts.context_summary,
        raw_task_plan=planner_artifacts.raw_plan,
        validated_task_plan=planner_artifacts.validated_plan,
        plan_validation=planner_artifacts.validation,
    )


@app.post("/v1/invoke", response_model=AgenticGatewayResponse)
async def invoke_v1(
    request: AgenticGatewayRequest,
    consumer: AuthenticatedConsumer = Depends(verify_api_key),
) -> AgenticGatewayResponse:
    started_at = perf_counter()
    request_id = request.request_id or str(uuid4())
    trace_id = str(uuid4())
    request = request.model_copy(update={"request_id": request_id})
    trace = TraceCollector(
        trace_id=trace_id,
        request_id=request_id,
        log_path=settings.trace_log_path,
        output_dir=settings.trace_output_dir,
    )

    rate_decision = rate_limiter.check(consumer.rate_limit_key)
    rate_limit_metadata = _rate_limit_metadata(consumer.rate_limit_key, rate_decision)
    trace.add(
        service="gateway",
        node="GatewayAuth",
        event_type="node_end",
        input_summary=f"user_id={request.user_id}, tenant_id={request.tenant_id}",
        output_summary=f"consumer_id={consumer.consumer_id}",
        latency_ms=consumer.auth_latency_ms,
        metadata={"consumer": consumer.model_dump(mode="json"), "rate_limit": rate_limit_metadata},
    )
    trace.add(
        service="gateway",
        node="RateLimit",
        event_type="node_end" if rate_decision.allowed else "error",
        output_summary="allowed=true" if rate_decision.allowed else "rate limit exceeded",
        status="success" if rate_decision.allowed else "failed",
        error_type=None if rate_decision.allowed else "rate_limited",
        metadata=rate_limit_metadata,
    )
    if not rate_decision.allowed:
        latency_ms = _elapsed_ms(started_at)
        trace.flush()
        _log_agentic_request(
            request=request,
            consumer=consumer,
            trace_id=trace_id,
            selected_workflow=None,
            latency_ms=latency_ms,
            status_value="rate_limited",
            selection_reason="rate limit exceeded",
            route_decision={},
            task_profile={},
            trace_path=trace.trace_path.as_posix(),
            rate_limit=rate_limit_metadata,
            error="Rate limit exceeded",
            error_type="rate_limited",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "trace_id": trace_id,
                "retry_after_s": rate_limit_metadata["retry_after_s"],
            },
            headers={"Retry-After": str(max(1, int(rate_decision.retry_after_s)))},
        )

    profile_started_at = perf_counter()
    task_profile = task_profile_builder.build(request)
    user_permissions = _effective_permissions(
        task_profile_builder.resolve_user_permissions(request),
        consumer.permissions,
    )
    trace.add(
        service="gateway",
        node="TaskProfileBuild",
        event_type="profile_build",
        input_summary=f"task_type_hint={request.task_type}",
        output_summary=f"inferred_task_type={task_profile.task_type}",
        latency_ms=_elapsed_ms(profile_started_at),
        metadata={"task_profile": task_profile.model_dump(mode="json"), "user_permissions": user_permissions},
    )

    planner_artifacts = await _run_planner_pipeline(
        request=request,
        task_profile=task_profile,
        user_permissions=user_permissions,
        trace=trace,
    )
    task_profile = planner_artifacts.task_profile
    if planner_artifacts.validation is not None and planner_artifacts.validation.status == "failed":
        latency_ms = _elapsed_ms(started_at)
        trace.add(
            service="gateway",
            node="InvokeSelectedWorkflow",
            event_type="error",
            output_summary="planner validation failed",
            status="failed",
            error_type="plan_validation_failed",
            metadata={"invocation_skipped": True, "plan_validation": planner_artifacts.validation.model_dump(mode="json")},
        )
        trace.flush()
        _log_agentic_request(
            request=request,
            consumer=consumer,
            trace_id=trace_id,
            selected_workflow=None,
            latency_ms=latency_ms,
            status_value="refused",
            selection_reason="planner validation failed",
            route_decision={},
            task_profile=task_profile.model_dump(mode="json"),
            trace_path=trace.trace_path.as_posix(),
            rate_limit=rate_limit_metadata,
            error="; ".join(planner_artifacts.validation.errors),
            error_type="plan_validation_failed",
        )
        return AgenticGatewayResponse(
            request_id=request_id,
            trace_id=trace_id,
            selected_workflow=None,
            selection_reason="planner validation failed",
            answer="TaskPlanner output failed Gateway validation.",
            citations=[],
            status="refused",
            metrics=AgenticMetrics(latency_ms=latency_ms),
            metadata={
                "task_profile": task_profile.model_dump(mode="json"),
                "planner": _planner_artifacts_metadata(planner_artifacts),
                "trace_path": trace.trace_path.as_posix(),
            },
        )

    memory_started_at = perf_counter()
    memory_rules = _load_planner_memory()
    memory_result = _plan_memory_route(task_profile, memory_rules, task_plan=planner_artifacts.validated_plan)
    trace.add(
        service="gateway",
        node="MemoryPlannerRead",
        event_type="node_end" if not memory_result.warnings else "error",
        output_summary=(
            f"enabled={memory_result.enabled}, route_hints={len(memory_result.route_hints)}, "
            f"warnings={len(memory_result.warnings)}"
        ),
        status="success" if not memory_result.warnings else "failed",
        latency_ms=_elapsed_ms(memory_started_at),
        error_type=None if not memory_result.warnings else "planner_memory_warning",
        metadata=_planner_memory_metadata(memory_result, memory_rules),
    )
    if memory_result.warnings:
        return _refuse_planner_memory(
            request=request,
            consumer=consumer,
            trace=trace,
            started_at=started_at,
            task_profile=task_profile,
            memory_result=memory_result,
            memory_rules=memory_rules,
            rate_limit_metadata=rate_limit_metadata,
        )

    route_started_at = perf_counter()
    route_hints = list(planner_artifacts.route_hints)
    route_hints.extend(memory_result.route_hints)
    route_decision = agentic_router.select(
        trace_id=trace_id,
        task_profile=task_profile,
        workflow_profiles=workflow_profile_store.all(),
        user_permissions=user_permissions,
        consumer=consumer,
        route_hints=route_hints,
    )
    trace.add(
        service="gateway",
        node="AgenticRouteDecision",
        event_type="route_decision",
        input_summary=f"task_type={task_profile.task_type}",
        output_summary=f"selected_workflow={route_decision.selected_workflow}",
        latency_ms=_elapsed_ms(route_started_at),
        metadata=route_decision.model_dump(mode="json"),
    )

    memory_apply_started_at = perf_counter()
    memory_result = _plan_memory_contract(
        memory_result,
        task_profile,
        route_decision.selected_workflow,
        memory_rules,
        task_plan=planner_artifacts.validated_plan,
    )
    trace.add(
        service="gateway",
        node="MemoryPlannerApply",
        event_type="node_end" if not memory_result.warnings else "error",
        output_summary=(
            f"matched_route_rules={len(memory_result.matched_route_rules)}, "
            f"matched_contract_rules={len(memory_result.matched_contract_rules)}"
        ),
        status="success" if not memory_result.warnings else "failed",
        latency_ms=_elapsed_ms(memory_apply_started_at),
        error_type=None if not memory_result.warnings else "planner_memory_warning",
        metadata=_planner_memory_metadata(memory_result, memory_rules),
    )
    if memory_result.warnings:
        return _refuse_planner_memory(
            request=request,
            consumer=consumer,
            trace=trace,
            started_at=started_at,
            task_profile=task_profile,
            memory_result=memory_result,
            memory_rules=memory_rules,
            rate_limit_metadata=rate_limit_metadata,
            route_decision=route_decision,
        )

    contract_started_at = perf_counter()
    task_contract = task_contract_builder.build(
        request,
        task_profile,
        route_decision.selected_workflow,
        contract_patches=memory_result.contract_patches,
        validated_task_plan=planner_artifacts.validated_plan,
    )
    trace.add(
        service="gateway",
        node="TaskContractBuild",
        event_type="node_end",
        output_summary=f"contract_id={task_contract.contract_id}",
        latency_ms=_elapsed_ms(contract_started_at),
        metadata={"task_contract": task_contract.model_dump(mode="json")},
    )

    if route_decision.selected_workflow is None:
        latency_ms = _elapsed_ms(started_at)
        trace.add(
            service="gateway",
            node="InvokeSelectedWorkflow",
            event_type="error",
            output_summary=route_decision.selection_reason,
            status="failed",
            error_type="no_executable_workflow",
            metadata={"invocation_skipped": True},
        )
        trace.flush()
        _log_agentic_request(
            request=request,
            consumer=consumer,
            trace_id=trace_id,
            selected_workflow=None,
            latency_ms=latency_ms,
            status_value="refused",
            selection_reason=route_decision.selection_reason,
            route_decision=route_decision.model_dump(mode="json"),
            task_profile=task_profile.model_dump(mode="json"),
            trace_path=trace.trace_path.as_posix(),
            rate_limit=rate_limit_metadata,
            error="No executable workflow",
            error_type="no_executable_workflow",
        )
        return AgenticGatewayResponse(
            request_id=request_id,
            trace_id=trace_id,
            contract_id=task_contract.contract_id,
            selected_workflow=None,
            selection_reason=route_decision.selection_reason,
            answer="No executable workflow matched the request and permissions.",
            citations=[],
            status="refused",
            metrics=AgenticMetrics(latency_ms=latency_ms),
            metadata={
                "task_profile": task_profile.model_dump(mode="json"),
                "task_contract": task_contract.model_dump(mode="json"),
                "route_decision": route_decision.model_dump(mode="json"),
                "memory_planner": _planner_memory_metadata(memory_result, memory_rules),
                "planner": _planner_artifacts_metadata(planner_artifacts),
                "trace_path": trace.trace_path.as_posix(),
            },
        )

    return await _invoke_selected_workflow(
        request=request,
        consumer=consumer,
        trace=trace,
        started_at=started_at,
        selected_workflow=route_decision.selected_workflow,
        route_decision=route_decision,
        task_profile=task_profile,
        task_contract=task_contract,
        user_permissions=user_permissions,
        rate_limit_metadata=rate_limit_metadata,
        memory_result=memory_result,
        memory_rules=memory_rules,
        planner_artifacts=planner_artifacts,
    )


async def _invoke_selected_workflow(
    *,
    request: AgenticGatewayRequest,
    consumer: AuthenticatedConsumer,
    trace: TraceCollector,
    started_at: float,
    selected_workflow: str,
    route_decision: Any,
    task_profile: Any,
    task_contract: TaskContract,
    user_permissions: List[str],
    rate_limit_metadata: Dict[str, Any],
    memory_result: MemoryPlannerResult,
    memory_rules: Dict[str, Any],
    planner_artifacts: PlannerArtifacts,
) -> AgenticGatewayResponse:
    runtime_metrics_start = runtime_metrics_store.start(selected_workflow)
    agent_request = AgentInvocationRequest(
        request_id=request.request_id or "",
        trace_id=trace.trace_id,
        consumer_id=consumer.consumer_id,
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        input=request.input,
        selected_workflow=selected_workflow,
        task_profile=task_profile,
        task_contract=task_contract,
        permissions=user_permissions,
        allowed_tools=consumer.allowed_tools,
        allowed_workflows=consumer.allowed_workflows,
        metadata=request.metadata,
    )
    invoke_started_at = perf_counter()
    try:
        agent_response = await _call_agent_orchestrator(agent_request)
    except AgentCallError as exc:
        invoke_latency_ms = _elapsed_ms(invoke_started_at)
        runtime_metrics_finish = runtime_metrics_store.finish(selected_workflow, invoke_latency_ms, failed=True)
        latency_ms = _elapsed_ms(started_at)
        trace.add(
            service="gateway",
            node="InvokeSelectedWorkflow",
            event_type="error",
            input_summary=f"selected_workflow={selected_workflow}",
            output_summary=exc.message,
            status="failed",
            latency_ms=invoke_latency_ms,
            error_type=exc.error_type,
            metadata={
                "runtime_metrics_start": runtime_metrics_start.to_metadata(),
                "runtime_metrics_finish": runtime_metrics_finish.to_metadata(),
            },
        )
        trace.flush()
        _log_agentic_request(
            request=request,
            consumer=consumer,
            trace_id=trace.trace_id,
            selected_workflow=selected_workflow,
            latency_ms=latency_ms,
            status_value="failed",
            selection_reason=route_decision.selection_reason,
            route_decision=route_decision.model_dump(mode="json"),
            task_profile=task_profile.model_dump(mode="json"),
            trace_path=trace.trace_path.as_posix(),
            rate_limit=rate_limit_metadata,
            error=exc.message,
            error_type=exc.error_type,
        )
        raise HTTPException(
            status_code=exc.http_status_code,
            detail={"error": exc.message, "trace_id": trace.trace_id},
        ) from exc

    invoke_latency_ms = _elapsed_ms(invoke_started_at)
    verification_started_at = perf_counter()
    verification = verification_gate.verify(
        task_contract,
        agent_response,
        tool_audit_log_path=settings.tool_audit_log_path,
    )
    response_metadata = {
        "task_profile": task_profile.model_dump(mode="json"),
        "task_contract": task_contract.model_dump(mode="json"),
        "route_decision": route_decision.model_dump(mode="json"),
        "memory_planner": _planner_memory_metadata(memory_result, memory_rules),
        "planner": _planner_artifacts_metadata(planner_artifacts),
        "trace_path": trace.trace_path.as_posix(),
        "consumer_id": consumer.consumer_id,
        "tools": agent_response.tools,
        "rate_limit": rate_limit_metadata,
        "fallback_used": bool(agent_response.metadata.get("fallback_used", False)),
        "agent_metadata": agent_response.metadata,
    }
    final_status, final_answer, response_metadata = _finalize_verified_response(
        agent_response,
        verification.status,
        response_metadata,
    )

    agent_failed = final_status in {"failed", "refused", "rate_limited", "verification_failed"}
    runtime_metrics_finish = runtime_metrics_store.finish(selected_workflow, invoke_latency_ms, failed=agent_failed)
    trace.add(
        service="gateway",
        node="InvokeSelectedWorkflow",
        event_type="node_end",
        input_summary=f"selected_workflow={selected_workflow}",
        output_summary=f"agent_status={agent_response.status}, final_status={final_status}",
        latency_ms=invoke_latency_ms,
        metadata={
            "agent_orchestrator_url": settings.agent_orchestrator_url,
            "selected_workflow": selected_workflow,
            "runtime_metrics_start": runtime_metrics_start.to_metadata(),
            "runtime_metrics_finish": runtime_metrics_finish.to_metadata(),
        },
    )
    trace.extend(agent_response.trace_events)
    latency_ms = _elapsed_ms(started_at)
    metrics = agent_response.metrics.model_copy(update={"latency_ms": latency_ms})
    fallback_used = bool(response_metadata["fallback_used"])
    trace.add(
        service="gateway",
        node="VerificationGate",
        event_type="node_end" if verification.status == "passed" else "error",
        output_summary=f"verification={verification.status}",
        status="success" if verification.status == "passed" else "failed",
        latency_ms=_elapsed_ms(verification_started_at),
        error_type=None if verification.status == "passed" else "verification_failed",
        metadata={"verification": verification.model_dump(mode="json")},
    )
    trace.add(
        service="gateway",
        node="TraceAggregation",
        event_type="node_end",
        output_summary=f"events={trace.event_count + 1}",
        latency_ms=latency_ms,
        estimated_tokens=metrics.estimated_tokens,
        estimated_cost=metrics.estimated_cost,
        metadata={
            "trace_path": trace.trace_path.as_posix(),
            "fallback_used": fallback_used,
            "verification_status": verification.status,
            "runtime_metrics_after": runtime_metrics_finish.to_metadata(),
        },
    )
    trace.flush()

    _log_agentic_request(
        request=request,
        consumer=consumer,
        trace_id=trace.trace_id,
        selected_workflow=selected_workflow,
        latency_ms=latency_ms,
        status_value=final_status,
        selection_reason=route_decision.selection_reason,
        route_decision=route_decision.model_dump(mode="json"),
        task_profile=task_profile.model_dump(mode="json"),
        trace_path=trace.trace_path.as_posix(),
        rate_limit=rate_limit_metadata,
        fallback_used=fallback_used,
    )
    return AgenticGatewayResponse(
        request_id=request.request_id or "",
        trace_id=trace.trace_id,
        contract_id=task_contract.contract_id,
        selected_workflow=selected_workflow,
        selection_reason=route_decision.selection_reason,
        answer=final_answer,
        citations=agent_response.citations,
        status=final_status,  # type: ignore[arg-type]
        verification=verification,
        metrics=metrics,
        metadata=response_metadata,
    )


async def _call_agent_orchestrator(request: AgentInvocationRequest) -> AgentWorkflowResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            response = await client.post(settings.agent_orchestrator_url, json=request.model_dump(mode="json"))
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise AgentCallError(
                    error_type="agent_orchestrator_error",
                    message=f"Agent orchestrator returned {exc.response.status_code}",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=exc.response.status_code,
                ) from exc

            try:
                return AgentWorkflowResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise AgentCallError(
                    error_type="invalid_agent_response",
                    message="Agent orchestrator returned invalid response",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=response.status_code,
                ) from exc
    except httpx.TimeoutException as exc:
        raise AgentCallError(
            error_type="agent_orchestrator_timeout",
            message="Agent orchestrator timed out",
            http_status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        ) from exc
    except httpx.HTTPError as exc:
        raise AgentCallError(
            error_type="agent_orchestrator_network_error",
            message="Agent orchestrator request failed",
            http_status_code=status.HTTP_502_BAD_GATEWAY,
        ) from exc


async def _run_planner_pipeline(
    *,
    request: AgenticGatewayRequest,
    task_profile: Any,
    user_permissions: List[str],
    trace: Optional[TraceCollector],
) -> PlannerArtifacts:
    policy_started_at = perf_counter()
    policy = planner_policy.decide(
        request,
        task_profile,
        planner_enabled=settings.task_planner_enabled,
    )
    task_profile = task_profile.model_copy(update={"planner_policy": policy.model_dump(mode="json")})
    if trace is not None:
        trace.add(
            service="gateway",
            node="PlannerPolicyDecision",
            event_type="node_end",
            output_summary=f"planner_required={policy.planner_required}",
            latency_ms=_elapsed_ms(policy_started_at),
            metadata={"planner_policy": policy.model_dump(mode="json")},
        )
    if not policy.planner_required:
        return PlannerArtifacts(
            task_profile=task_profile,
            policy=policy,
            context_summary={},
            raw_plan=None,
            validated_plan=None,
            validation=None,
            route_hints=[],
        )

    context_started_at = perf_counter()
    context = planner_context_builder.build(
        request=request,
        task_profile=task_profile,
        workflow_profiles=workflow_profile_store.all(),
        user_permissions=user_permissions,
    )
    context_summary = planner_context_builder.summary(context)
    if trace is not None:
        trace.add(
            service="gateway",
            node="PlannerContextBuild",
            event_type="node_end",
            output_summary=f"workflows={context_summary.get('workflow_count')}, tools={context_summary.get('tool_count')}",
            latency_ms=_elapsed_ms(context_started_at),
            metadata={"planner_context_summary": context_summary},
        )

    planner_started_at = perf_counter()
    raw_plan = await _call_task_planner(context)
    if trace is not None:
        planner_failed = bool(raw_plan.get("_planner_error"))
        trace.add(
            service="gateway",
            node="TaskPlannerCall",
            event_type="model_call" if not planner_failed else "error",
            output_summary="planner output received" if not planner_failed else "planner call failed",
            status="success" if not planner_failed else "failed",
            latency_ms=_elapsed_ms(planner_started_at),
            error_type=None if not planner_failed else "task_planner_error",
            metadata={"model": settings.planner_model_id, "enable_thinking": settings.planner_enable_thinking},
        )
    validated_plan, validation = plan_validator.validate(
        raw_plan,
        workflow_profiles=workflow_profile_store.all(),
        user_permissions=user_permissions,
    )
    if validation.status == "failed" and validation.repairable and settings.planner_repair_max_retries > 0:
        repair_started_at = perf_counter()
        repair_plan = await _call_task_planner(context, repair_errors=validation.errors, invalid_plan=raw_plan)
        if trace is not None:
            repair_failed = bool(repair_plan.get("_planner_error"))
            trace.add(
                service="gateway",
                node="PlanRepair",
                event_type="model_call" if not repair_failed else "error",
                output_summary="repair output received" if not repair_failed else "repair call failed",
                status="success" if not repair_failed else "failed",
                latency_ms=_elapsed_ms(repair_started_at),
                error_type=None if not repair_failed else "task_planner_repair_error",
                metadata={"repair_errors": validation.errors, "model": settings.planner_model_id},
            )
        repaired_validated_plan, repaired_validation = plan_validator.validate(
            repair_plan,
            workflow_profiles=workflow_profile_store.all(),
            user_permissions=user_permissions,
        )
        raw_plan = repair_plan
        validated_plan = repaired_validated_plan
        validation = repaired_validation

    if trace is not None:
        trace.add(
            service="gateway",
            node="PlanValidation",
            event_type="node_end" if validation.status == "passed" else "error",
            output_summary=f"plan_validation={validation.status}",
            status="success" if validation.status == "passed" else "failed",
            error_type=None if validation.status == "passed" else "plan_validation_failed",
            metadata={
                "raw_task_plan": raw_plan,
                "validated_task_plan": validated_plan.model_dump(mode="json") if validated_plan else None,
                "plan_validation": validation.model_dump(mode="json"),
            },
        )

    if validated_plan is not None and validation.status == "passed":
        task_profile = _apply_plan_to_task_profile(task_profile, validated_plan)
    return PlannerArtifacts(
        task_profile=task_profile,
        policy=policy,
        context_summary=context_summary,
        raw_plan=raw_plan,
        validated_plan=validated_plan if validation.status == "passed" else None,
        validation=validation,
        route_hints=_route_hints_from_plan(validated_plan) if validation.status == "passed" else [],
    )


async def _call_task_planner(
    context: Dict[str, Any],
    *,
    repair_errors: Optional[List[str]] = None,
    invalid_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    planner_started_at = perf_counter()
    try:
        return await task_planner.plan(context, repair_errors=repair_errors, invalid_plan=invalid_plan)
    except Exception as exc:
        return {
            "_planner_error": f"{type(exc).__name__}: {exc}",
            "_latency_ms": _elapsed_ms(planner_started_at),
        }


def _apply_plan_to_task_profile(task_profile: Any, plan: TaskPlan) -> Any:
    capabilities = list(task_profile.required_capabilities)
    if any(step.workflow == "large_knowledge_qa_workflow" for step in plan.execution_plan):
        for capability in ["deep_rag", "multi_hop_retrieval", "evidence_synthesis"]:
            if capability not in capabilities:
                capabilities.append(capability)
    return task_profile.model_copy(
        update={
            "task_type": plan.primary_task_type,
            "required_capabilities": capabilities,
            "evidence_required": bool(
                task_profile.evidence_required or plan.semantic_features.get("requires_external_evidence")
            ),
        }
    )


def _route_hints_from_plan(plan: Optional[TaskPlan]) -> List[Dict[str, Any]]:
    workflow_id = _planned_selected_workflow(plan)
    if not workflow_id:
        return []
    return [{"workflow_id": workflow_id, "score_boost": 0.2, "source": "task_plan"}]


def _planned_selected_workflow(plan: Optional[TaskPlan]) -> Optional[str]:
    if plan is None:
        return None
    for role in ("final_compose", "primary_answer", "code_draft", "media_generation"):
        for step in plan.execution_plan:
            if step.step_role == role:
                return step.workflow
    return plan.execution_plan[-1].workflow if plan.execution_plan else None


def _planner_artifacts_metadata(artifacts: PlannerArtifacts) -> Dict[str, Any]:
    return {
        "planner_policy": artifacts.policy.model_dump(mode="json"),
        "planner_context_summary": artifacts.context_summary,
        "raw_task_plan": artifacts.raw_plan,
        "validated_task_plan": artifacts.validated_plan.model_dump(mode="json") if artifacts.validated_plan else None,
        "plan_validation": artifacts.validation.model_dump(mode="json") if artifacts.validation else None,
        "route_hints": artifacts.route_hints,
    }


def _plan_memory_route(
    task_profile: Any,
    memory_rules: Dict[str, Any],
    task_plan: Optional[TaskPlan] = None,
) -> MemoryPlannerResult:
    return memory_planner.plan_route(
        enabled=settings.memory_planner_enabled,
        task_profile=task_profile,
        workflow_profiles=workflow_profile_store.all(),
        route_rules=memory_rules["route_rules"],
        warnings=memory_rules["warnings"],
        task_plan=task_plan,
    )


def _plan_memory_contract(
    memory_result: MemoryPlannerResult,
    task_profile: Any,
    selected_workflow: Optional[str],
    memory_rules: Dict[str, Any],
    task_plan: Optional[TaskPlan] = None,
) -> MemoryPlannerResult:
    return memory_planner.plan_contract(
        base=memory_result,
        task_profile=task_profile,
        selected_workflow=selected_workflow,
        workflow_profiles=workflow_profile_store.all(),
        contract_rules=memory_rules["contract_rules"],
        task_plan=task_plan,
    )


def _planner_memory_metadata(memory_result: MemoryPlannerResult, memory_rules: Dict[str, Any]) -> Dict[str, Any]:
    metadata = memory_result.to_metadata()
    metadata["route_rule_count"] = len(memory_rules.get("route_rules", []))
    metadata["contract_rule_count"] = len(memory_rules.get("contract_rules", []))
    return metadata


def _refuse_planner_memory(
    *,
    request: AgenticGatewayRequest,
    consumer: AuthenticatedConsumer,
    trace: TraceCollector,
    started_at: float,
    task_profile: Any,
    memory_result: MemoryPlannerResult,
    memory_rules: Dict[str, Any],
    rate_limit_metadata: Dict[str, Any],
    route_decision: Any = None,
) -> AgenticGatewayResponse:
    latency_ms = _elapsed_ms(started_at)
    selection_reason = "planner memory rules are invalid"
    route_metadata = route_decision.model_dump(mode="json") if route_decision is not None else {}
    trace.add(
        service="gateway",
        node="InvokeSelectedWorkflow",
        event_type="error",
        output_summary=selection_reason,
        status="failed",
        error_type="planner_memory_invalid",
        metadata={"invocation_skipped": True, "memory_planner": _planner_memory_metadata(memory_result, memory_rules)},
    )
    trace.flush()
    _log_agentic_request(
        request=request,
        consumer=consumer,
        trace_id=trace.trace_id,
        selected_workflow=None,
        latency_ms=latency_ms,
        status_value="refused",
        selection_reason=selection_reason,
        route_decision=route_metadata,
        task_profile=task_profile.model_dump(mode="json"),
        trace_path=trace.trace_path.as_posix(),
        rate_limit=rate_limit_metadata,
        error="; ".join(memory_result.warnings),
        error_type="planner_memory_invalid",
    )
    return AgenticGatewayResponse(
        request_id=request.request_id or "",
        trace_id=trace.trace_id,
        contract_id=None,
        selected_workflow=None,
        selection_reason=selection_reason,
        answer="Planner Memory rules are invalid. Fix the rule file before executing this request.",
        citations=[],
        status="refused",
        metrics=AgenticMetrics(latency_ms=latency_ms),
        metadata={
            "task_profile": task_profile.model_dump(mode="json"),
            "route_decision": route_metadata,
            "memory_planner": _planner_memory_metadata(memory_result, memory_rules),
            "trace_path": trace.trace_path.as_posix(),
        },
    )


def _load_planner_memory() -> Dict[str, Any]:
    if not settings.memory_planner_enabled:
        return {"route_rules": [], "contract_rules": [], "warnings": []}
    return planner_memory_store.load()


def _rate_limit_metadata(limit_key: str, decision: RateLimitDecision) -> Dict[str, Any]:
    metadata = decision.to_metadata()
    metadata["limit_key"] = _fingerprint(limit_key)
    metadata["enabled"] = settings.rate_limit_enabled
    metadata["replenish_rate"] = settings.rate_limit_replenish_rate
    metadata["burst_capacity"] = settings.rate_limit_burst_capacity
    metadata["requested_tokens"] = settings.rate_limit_requested_tokens
    return metadata


def _finalize_verified_response(
    agent_response: AgentWorkflowResponse,
    verification_status: str,
    metadata: Dict[str, Any],
) -> tuple[str, str, Dict[str, Any]]:
    final_status = agent_response.status
    final_answer = agent_response.answer
    response_metadata = dict(metadata)
    if agent_response.status == "success" and verification_status == "failed":
        final_status = "verification_failed"
        final_answer = "Workflow output failed Gateway verification. Original answer is stored in metadata.raw_answer."
        response_metadata["raw_answer"] = agent_response.answer
    return final_status, final_answer, response_metadata


def _fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]


def _effective_permissions(request_permissions: List[str], consumer_permissions: List[str]) -> List[str]:
    request_set = set(request_permissions)
    consumer_set = set(consumer_permissions)
    if "*" in consumer_set:
        return sorted(request_set)
    if "*" in request_set:
        return sorted(consumer_set)
    return sorted(request_set & consumer_set)


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _log_agentic_request(
    request: AgenticGatewayRequest,
    consumer: AuthenticatedConsumer,
    trace_id: str,
    selected_workflow: Optional[str],
    latency_ms: float,
    status_value: str,
    selection_reason: str,
    route_decision: Dict[str, Any],
    task_profile: Dict[str, Any],
    trace_path: str,
    rate_limit: Optional[Dict[str, Any]],
    fallback_used: bool = False,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    resource_decision = route_decision.get("resource_decision", {})
    write_jsonl_record(
        settings.log_path,
        {
            "schema": "agentic_gateway_request_log_v2",
            "request_id": request.request_id or "",
            "trace_id": trace_id,
            "consumer_id": consumer.consumer_id,
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
            "task_type_hint": request.task_type,
            "inferred_task_type": task_profile.get("task_type"),
            "selected_workflow": selected_workflow,
            "selected_runtime_target": resource_decision.get("selected_target"),
            "selection_reason": selection_reason,
            "latency_ms": latency_ms,
            "status": status_value,
            "fallback_used": fallback_used,
            "rate_limited": status_value == "rate_limited",
            "rate_limit": {
                "allowed": (rate_limit or {}).get("allowed"),
                "limit_key": (rate_limit or {}).get("limit_key"),
                "retry_after_s": (rate_limit or {}).get("retry_after_s"),
            },
            "route_decision_ref": {"trace_id": trace_id, "trace_path": trace_path},
            "error_type": error_type,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
