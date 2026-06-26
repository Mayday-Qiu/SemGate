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
from app.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
from app.config import load_settings
from app.logging_utils import write_jsonl_record, write_request_log
from app.profiles import load_service_profiles
from app.rate_limit import RateLimitConfig, RateLimitDecision, TokenBucketRateLimiter
from app.registry import load_default_registry
from app.router import CandidateRoute, RoutingDecision, Router, TASK_ROUTING_CONFIGS
from app.runtime_metrics import RuntimeMetricsStore
from app.schemas import (
    AgenticGatewayRequest,
    AgenticGatewayResponse,
    AgentInvocationRequest,
    AgentWorkflowResponse,
    AuthenticatedConsumer,
    BackendInfo,
    BackendResponse,
    GatewayRequest,
    GatewayResponse,
    RequestLog,
)
from app.task_profile import TaskProfileBuilder
from app.trace_collector import TraceCollector
from app.workflow_profiles import load_workflow_profiles


settings = load_settings()
registry = load_default_registry(settings)
profile_store = load_service_profiles(settings.profiles_path)
router = Router(profile_store)
task_profile_builder = TaskProfileBuilder()
workflow_profile_store = load_workflow_profiles(settings.workflow_profiles_path)
runtime_metrics_store = RuntimeMetricsStore()
agentic_router = AgenticRouter(runtime_metrics_store)
rate_limiter = TokenBucketRateLimiter(
    RateLimitConfig(
        enabled=settings.rate_limit_enabled,
        replenish_rate=settings.rate_limit_replenish_rate,
        burst_capacity=settings.rate_limit_burst_capacity,
        requested_tokens=settings.rate_limit_requested_tokens,
    )
)
circuit_breakers = CircuitBreakerRegistry(
    CircuitBreakerConfig(
        enabled=settings.circuit_breaker_enabled,
        sliding_window_size=settings.circuit_sliding_window_size,
        minimum_number_of_calls=settings.circuit_minimum_number_of_calls,
        failure_rate_threshold=settings.circuit_failure_rate_threshold,
        slow_call_rate_threshold=settings.circuit_slow_call_rate_threshold,
        wait_duration_in_open_state_s=settings.circuit_wait_duration_in_open_state_s,
        permitted_calls_in_half_open=settings.circuit_permitted_calls_in_half_open,
    )
)
app = FastAPI(title="SemRoute-Gateway", version="0.1.0")


@dataclass
class BackendCallError(Exception):
    error_type: str
    message: str
    http_status_code: int
    status_code: Optional[int] = None
    terminal: bool = False
    breaker_outcome: Optional[str] = None


@dataclass
class InvocationResult:
    backend_response: Optional[BackendResponse]
    selected_backend: Optional[str]
    status_value: str
    http_status_code: int
    attempts: List[Dict[str, Any]]
    fallback_backend: Optional[str]
    fallback_reason: Optional[str]
    error: Optional[str]


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/backends")
async def backends(_: AuthenticatedConsumer = Depends(verify_api_key)) -> Dict[str, Any]:
    return {"backends": [backend.model_dump(mode="json") for backend in registry.all()]}


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

    rate_limit_started_at = perf_counter()
    rate_decision = rate_limiter.check(consumer.rate_limit_key)
    rate_limit_latency_ms = _elapsed_ms(rate_limit_started_at)
    rate_limit_metadata = _rate_limit_metadata(consumer.rate_limit_key, rate_decision)
    trace.add(
        service="gateway",
        node="GatewayAuth",
        event_type="node_end",
        input_summary=f"user_id={request.user_id}, tenant_id={request.tenant_id}",
        output_summary=f"consumer_id={consumer.consumer_id}",
        latency_ms=consumer.auth_latency_ms,
        metadata={
            "consumer": consumer.model_dump(mode="json"),
            "rate_limit": rate_limit_metadata,
        },
    )
    trace.add(
        service="gateway",
        node="RateLimit",
        event_type="node_end" if rate_decision.allowed else "error",
        output_summary="allowed=true" if rate_decision.allowed else "rate limit exceeded",
        status="success" if rate_decision.allowed else "failed",
        latency_ms=rate_limit_latency_ms,
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
    profile_latency_ms = _elapsed_ms(profile_started_at)
    trace.add(
        service="gateway",
        node="TaskProfileBuild",
        event_type="profile_build",
        input_summary=f"task_type_hint={request.task_type}",
        output_summary=f"inferred_task_type={task_profile.task_type}",
        latency_ms=profile_latency_ms,
        metadata={
            "task_profile": task_profile.model_dump(mode="json"),
            "user_permissions": user_permissions,
        },
    )

    route_started_at = perf_counter()
    route_decision = agentic_router.select(
        trace_id=trace_id,
        task_profile=task_profile,
        workflow_profiles=workflow_profile_store.all(),
        user_permissions=user_permissions,
        consumer=consumer,
    )
    route_latency_ms = _elapsed_ms(route_started_at)
    trace.add(
        service="gateway",
        node="AgenticRouteDecision",
        event_type="route_decision",
        input_summary=f"task_type={task_profile.task_type}",
        output_summary=f"selected_workflow={route_decision.selected_workflow}",
        latency_ms=route_latency_ms,
        metadata=route_decision.model_dump(mode="json"),
    )

    if route_decision.selected_workflow is None:
        latency_ms = _elapsed_ms(started_at)
        trace.add(
            service="gateway",
            node="InvokeSelectedWorkflow",
            event_type="error",
            output_summary=route_decision.selection_reason,
            status="failed",
            latency_ms=0.0,
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
            selected_workflow=None,
            selection_reason=route_decision.selection_reason,
            answer="No executable workflow matched the request and permissions.",
            citations=[],
            status="refused",
            metrics={"latency_ms": latency_ms, "estimated_tokens": 0, "estimated_cost": 0.0},  # type: ignore[arg-type]
            metadata={
                "task_profile": task_profile.model_dump(mode="json"),
                "route_decision": route_decision.model_dump(mode="json"),
                "trace_path": trace.trace_path.as_posix(),
            },
        )

    selected_workflow = route_decision.selected_workflow
    runtime_metrics_start = runtime_metrics_store.start(selected_workflow)
    agent_request = AgentInvocationRequest(
        request_id=request_id,
        trace_id=trace_id,
        consumer_id=consumer.consumer_id,
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        input=request.input,
        selected_workflow=selected_workflow,
        task_profile=task_profile,
        permissions=user_permissions,
        allowed_tools=consumer.allowed_tools,
        allowed_workflows=consumer.allowed_workflows,
        metadata=request.metadata,
    )
    invoke_started_at = perf_counter()
    try:
        agent_response = await _call_agent_orchestrator(agent_request)
    except BackendCallError as exc:
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
            trace_id=trace_id,
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
            detail={"error": exc.message, "trace_id": trace_id},
        ) from exc

    invoke_latency_ms = _elapsed_ms(invoke_started_at)
    agent_failed = agent_response.status in {"failed", "refused", "rate_limited"}
    runtime_metrics_finish = runtime_metrics_store.finish(selected_workflow, invoke_latency_ms, failed=agent_failed)
    trace.add(
        service="gateway",
        node="InvokeSelectedWorkflow",
        event_type="node_end",
        input_summary=f"selected_workflow={selected_workflow}",
        output_summary=f"agent_status={agent_response.status}",
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
    fallback_used = bool(agent_response.metadata.get("fallback_used", False))
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
            "runtime_metrics_after": runtime_metrics_finish.to_metadata(),
        },
    )
    trace.flush()

    _log_agentic_request(
        request=request,
        consumer=consumer,
        trace_id=trace_id,
        selected_workflow=route_decision.selected_workflow,
        latency_ms=latency_ms,
        status_value=agent_response.status,
        selection_reason=route_decision.selection_reason,
        route_decision=route_decision.model_dump(mode="json"),
        task_profile=task_profile.model_dump(mode="json"),
        trace_path=trace.trace_path.as_posix(),
        rate_limit=rate_limit_metadata,
        fallback_used=fallback_used,
        error=None,
        error_type=None,
    )
    return AgenticGatewayResponse(
        request_id=request_id,
        trace_id=trace_id,
        selected_workflow=route_decision.selected_workflow,
        selection_reason=route_decision.selection_reason,
        answer=agent_response.answer,
        citations=agent_response.citations,
        status=agent_response.status,
        metrics=metrics,
        metadata={
            "task_profile": task_profile.model_dump(mode="json"),
            "route_decision": route_decision.model_dump(mode="json"),
            "trace_path": trace.trace_path.as_posix(),
            "consumer_id": consumer.consumer_id,
            "tools": agent_response.tools,
            "rate_limit": rate_limit_metadata,
            "fallback_used": fallback_used,
        },
    )


@app.post("/invoke", response_model=GatewayResponse)
async def invoke(request: GatewayRequest, consumer: AuthenticatedConsumer = Depends(verify_api_key)) -> GatewayResponse:
    started_at = perf_counter()
    request_id = request.request_id or str(uuid4())
    request = request.model_copy(update={"request_id": request_id})

    rate_decision = rate_limiter.check(consumer.rate_limit_key)
    rate_limit_metadata = _rate_limit_metadata(consumer.rate_limit_key, rate_decision)
    if not rate_decision.allowed:
        latency_ms = _elapsed_ms(started_at)
        _log_request(
            request=request,
            selected_backend=None,
            latency_ms=latency_ms,
            status_value="rate_limited",
            error="Rate limit exceeded",
            rate_limit=rate_limit_metadata,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "retry_after_s": rate_limit_metadata["retry_after_s"],
            },
            headers={"Retry-After": str(max(1, int(rate_decision.retry_after_s)))},
        )

    candidates = registry.healthy_for_task(request.task_type)
    routing_decision = router.select(settings.policy, candidates, request.task_type)
    routing_metadata = _routing_metadata(routing_decision)

    if routing_decision.backend is None:
        latency_ms = _elapsed_ms(started_at)
        error_message = f"No healthy backend supports task_type={request.task_type}"
        _log_request(
            request=request,
            selected_backend=None,
            latency_ms=latency_ms,
            status_value="failed",
            error=error_message,
            rate_limit=rate_limit_metadata,
            routing=routing_metadata,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_message,
        )

    invocation = await _invoke_with_resilience(request, routing_decision)
    latency_ms = _elapsed_ms(started_at)
    circuit_metadata = _circuit_metadata(invocation.attempts)

    if invocation.backend_response is None:
        _log_request(
            request=request,
            selected_backend=invocation.selected_backend,
            latency_ms=latency_ms,
            status_value=invocation.status_value,
            error=invocation.error,
            fallback_backend=invocation.fallback_backend,
            fallback_reason=invocation.fallback_reason,
            retry_count=_retry_count(invocation.attempts),
            attempts=invocation.attempts,
            rate_limit=rate_limit_metadata,
            routing=routing_metadata,
            circuit_breakers=circuit_metadata,
            fallback_used=invocation.fallback_backend is not None,
        )
        raise HTTPException(
            status_code=invocation.http_status_code,
            detail=invocation.error or "Backend request failed",
        )

    backend_response = invocation.backend_response
    _log_request(
        request=request,
        selected_backend=backend_response.backend_id,
        latency_ms=latency_ms,
        status_value=invocation.status_value,
        error=None,
        fallback_backend=invocation.fallback_backend,
        fallback_reason=invocation.fallback_reason,
        retry_count=_retry_count(invocation.attempts),
        attempts=invocation.attempts,
        rate_limit=rate_limit_metadata,
        routing=routing_metadata,
        circuit_breakers=circuit_metadata,
        fallback_used=invocation.fallback_backend is not None,
    )
    return GatewayResponse(
        request_id=request_id,
        status=invocation.status_value,  # type: ignore[arg-type]
        selected_backend=backend_response.backend_id,
        output=backend_response.output,
        heuristic_quality_score=backend_response.heuristic_quality_score,
        latency_ms=latency_ms,
        fallback_backend=invocation.fallback_backend,
        fallback_reason=invocation.fallback_reason,
        metadata={
            "backend_latency_ms": backend_response.latency_ms,
            "policy": settings.policy,
            "routing": routing_metadata,
            "attempts": invocation.attempts,
            "fallback": {
                "used": invocation.fallback_backend is not None,
                "backend": invocation.fallback_backend,
                "reason": invocation.fallback_reason,
            },
            "rate_limit": rate_limit_metadata,
            "circuit_breakers": circuit_metadata,
        },
    )


async def _invoke_with_resilience(
    request: GatewayRequest,
    routing_decision: RoutingDecision,
) -> InvocationResult:
    routes = _candidate_routes(routing_decision)
    max_attempts = max(1, settings.fallback_max_attempts)
    attempts: List[Dict[str, Any]] = []
    last_error = "No backend was attempted"
    last_http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    last_status_value = "failed"

    for route in routes[:max_attempts]:
        backend = route.backend
        breaker = circuit_breakers.get(backend.backend_id, str(request.task_type))
        permission = breaker.before_call()
        attempt: Dict[str, Any] = {
            "backend_id": backend.backend_id,
            "candidate_score": route.score,
            "routing": route.metadata,
            "circuit_before": permission.snapshot.to_metadata(),
        }

        if not permission.allowed:
            attempt.update(
                {
                    "result": "short_circuited",
                    "error_type": permission.reason,
                    "latency_ms": 0.0,
                    "circuit_after": permission.snapshot.to_metadata(),
                }
            )
            attempts.append(attempt)
            last_error = f"Backend {backend.backend_id} skipped because {permission.reason}"
            last_http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            last_status_value = "failed"
            continue

        attempt_started_at = perf_counter()
        try:
            backend_response = await _call_backend(backend, request)
        except BackendCallError as exc:
            attempt_latency_ms = _elapsed_ms(attempt_started_at)
            if exc.breaker_outcome is not None:
                circuit_after = breaker.record_result(
                    exc.breaker_outcome,
                    attempt_latency_ms,
                    _task_slo_ms(str(request.task_type)),
                )
            else:
                circuit_after = breaker.snapshot()
            attempt.update(
                {
                    "result": exc.error_type,
                    "status_code": exc.status_code,
                    "latency_ms": attempt_latency_ms,
                    "error": exc.message,
                    "circuit_after": circuit_after.to_metadata(),
                }
            )
            attempts.append(attempt)
            last_error = exc.message
            last_http_status = exc.http_status_code
            last_status_value = "timeout" if exc.error_type == "timeout" else "failed"
            if exc.terminal:
                return InvocationResult(
                    backend_response=None,
                    selected_backend=backend.backend_id,
                    status_value=last_status_value,
                    http_status_code=last_http_status,
                    attempts=attempts,
                    fallback_backend=None,
                    fallback_reason=None,
                    error=last_error,
                )
            continue

        attempt_latency_ms = _elapsed_ms(attempt_started_at)
        circuit_after = breaker.record_result("success", attempt_latency_ms, _task_slo_ms(str(request.task_type)))
        attempt.update(
            {
                "result": "success",
                "status_code": status.HTTP_200_OK,
                "latency_ms": attempt_latency_ms,
                "backend_latency_ms": backend_response.latency_ms,
                "circuit_after": circuit_after.to_metadata(),
            }
        )
        attempts.append(attempt)
        fallback_reason = _fallback_reason(attempts)
        fallback_backend = backend.backend_id if fallback_reason is not None else None
        return InvocationResult(
            backend_response=backend_response,
            selected_backend=backend.backend_id,
            status_value="fallback_success" if fallback_backend is not None else "success",
            http_status_code=status.HTTP_200_OK,
            attempts=attempts,
            fallback_backend=fallback_backend,
            fallback_reason=fallback_reason,
            error=None,
        )

    return InvocationResult(
        backend_response=None,
        selected_backend=None,
        status_value=last_status_value,
        http_status_code=last_http_status,
        attempts=attempts,
        fallback_backend=None,
        fallback_reason=None,
        error=last_error,
    )


async def _call_backend(backend: BackendInfo, request: GatewayRequest) -> BackendResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            response = await client.post(backend.url, json=request.model_dump(mode="json"))
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code >= 500:
                    raise BackendCallError(
                        error_type="server_error",
                        message=f"Backend {backend.backend_id} returned {status_code}",
                        http_status_code=status.HTTP_502_BAD_GATEWAY,
                        status_code=status_code,
                        breaker_outcome="server_error",
                    ) from exc
                raise BackendCallError(
                    error_type="client_error",
                    message=f"Backend {backend.backend_id} rejected request with {status_code}",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=status_code,
                    terminal=True,
                ) from exc

            try:
                return BackendResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise BackendCallError(
                    error_type="invalid_response",
                    message=f"Backend {backend.backend_id} returned invalid response",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=response.status_code,
                    breaker_outcome="server_error",
                ) from exc
    except httpx.TimeoutException as exc:
        raise BackendCallError(
            error_type="timeout",
            message=f"Backend {backend.backend_id} timed out",
            http_status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            breaker_outcome="timeout",
        ) from exc
    except httpx.HTTPError as exc:
        raise BackendCallError(
            error_type="network_error",
            message=f"Backend {backend.backend_id} request failed",
            http_status_code=status.HTTP_502_BAD_GATEWAY,
            breaker_outcome="network_error",
        ) from exc


async def _call_agent_orchestrator(request: AgentInvocationRequest) -> AgentWorkflowResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            response = await client.post(settings.agent_orchestrator_url, json=request.model_dump(mode="json"))
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise BackendCallError(
                    error_type="agent_orchestrator_error",
                    message=f"Agent orchestrator returned {exc.response.status_code}",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=exc.response.status_code,
                ) from exc

            try:
                return AgentWorkflowResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise BackendCallError(
                    error_type="invalid_agent_response",
                    message="Agent orchestrator returned invalid response",
                    http_status_code=status.HTTP_502_BAD_GATEWAY,
                    status_code=response.status_code,
                ) from exc
    except httpx.TimeoutException as exc:
        raise BackendCallError(
            error_type="agent_orchestrator_timeout",
            message="Agent orchestrator timed out",
            http_status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        ) from exc
    except httpx.HTTPError as exc:
        raise BackendCallError(
            error_type="agent_orchestrator_network_error",
            message="Agent orchestrator request failed",
            http_status_code=status.HTTP_502_BAD_GATEWAY,
        ) from exc


def _candidate_routes(routing_decision: RoutingDecision) -> List[CandidateRoute]:
    if routing_decision.ordered_candidates:
        return routing_decision.ordered_candidates
    if routing_decision.backend is None:
        return []
    return [CandidateRoute(backend=routing_decision.backend, score=routing_decision.score, metadata=routing_decision.metadata)]


def _routing_metadata(routing_decision: RoutingDecision) -> Dict[str, Any]:
    return {
        "policy": routing_decision.policy,
        "reason": routing_decision.reason,
        "selected_backend": routing_decision.backend.backend_id if routing_decision.backend else None,
        "selected_backend_score": routing_decision.score,
        "candidates": [route.to_metadata() for route in _candidate_routes(routing_decision)],
        "details": routing_decision.metadata,
    }


def _rate_limit_metadata(limit_key: str, decision: RateLimitDecision) -> Dict[str, Any]:
    metadata = decision.to_metadata()
    metadata["limit_key"] = _fingerprint(limit_key)
    metadata["enabled"] = settings.rate_limit_enabled
    metadata["replenish_rate"] = settings.rate_limit_replenish_rate
    metadata["burst_capacity"] = settings.rate_limit_burst_capacity
    metadata["requested_tokens"] = settings.rate_limit_requested_tokens
    return metadata


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


def _task_slo_ms(task_type: str) -> float:
    config = TASK_ROUTING_CONFIGS.get(task_type)
    if config is None:
        return settings.request_timeout_s * 1000
    return config.task_slo_ms


def _fallback_reason(attempts: List[Dict[str, Any]]) -> Optional[str]:
    if len(attempts) <= 1:
        return None
    for attempt in attempts[:-1]:
        if attempt.get("result") != "success":
            return f"{attempt.get('backend_id')}:{attempt.get('result')}"
    return None


def _retry_count(attempts: List[Dict[str, Any]]) -> int:
    backend_calls = [
        attempt
        for attempt in attempts
        if attempt.get("result") not in {"short_circuited"}
    ]
    return max(0, len(backend_calls) - 1)


def _circuit_metadata(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    snapshots: Dict[str, Any] = {}
    for attempt in attempts:
        backend_id = attempt.get("backend_id")
        if backend_id is None:
            continue
        snapshots[str(backend_id)] = attempt.get("circuit_after") or attempt.get("circuit_before") or {}
    return snapshots


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _log_request(
    request: GatewayRequest,
    selected_backend: Optional[str],
    latency_ms: float,
    status_value: str,
    error: Optional[str],
    fallback_backend: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    retry_count: int = 0,
    attempts: Optional[List[Dict[str, Any]]] = None,
    rate_limit: Optional[Dict[str, Any]] = None,
    routing: Optional[Dict[str, Any]] = None,
    circuit_breakers: Optional[Dict[str, Any]] = None,
    fallback_used: bool = False,
) -> None:
    routing_payload = routing or {}
    write_request_log(
        settings.log_path,
        RequestLog(
            request_id=request.request_id or "",
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            task_type=request.task_type,
            policy=settings.policy,
            selected_backend=selected_backend,
            latency_ms=latency_ms,
            status=status_value,  # type: ignore[arg-type]
            retry_count=retry_count,
            fallback_backend=fallback_backend,
            fallback_reason=fallback_reason,
            error=error,
            candidate_backends=[
                item.get("backend_id", "")
                for item in routing_payload.get("candidates", [])
                if item.get("backend_id")
            ],
            attempts=attempts or [],
            rate_limit=rate_limit or {},
            routing=routing_payload,
            circuit_breakers=circuit_breakers or {},
            fallback_used=fallback_used,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


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
            "route_decision_ref": {
                "trace_id": trace_id,
                "trace_path": trace_path,
            },
            "error_type": error_type,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
