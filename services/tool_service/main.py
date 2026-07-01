from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

from app.safe_logging import safe_metadata, safe_summary
from services.tool_service.audit import write_tool_audit
from services.tool_service.permissions import is_allowed
from services.tool_service.registry import all_tools, get_argument_model, get_tool
from services.tool_service.schemas import ToolInvokeRequest, ToolInvokeResponse
from services.tool_service.tools import execute_tool


app = FastAPI(title="SemGateway Tool Service", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "tool_service"}


@app.get("/tools")
async def tools() -> Dict[str, Any]:
    return {"tools": [definition.model_dump(mode="json") for definition in all_tools()]}


@app.post("/invoke", response_model=ToolInvokeResponse)
async def invoke(request: ToolInvokeRequest) -> ToolInvokeResponse:
    started_at = perf_counter()
    definition = get_tool(request.tool_name)
    if definition is None:
        latency_ms = _elapsed_ms(started_at)
        _audit(
            request=request,
            permission_scope=None,
            status_value="failed",
            latency_ms=latency_ms,
            error_type="unknown_tool",
            error="Unknown tool",
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown tool")

    if not is_allowed(request, definition.tool_name, definition.permission_scope):
        latency_ms = _elapsed_ms(started_at)
        _audit(
            request=request,
            permission_scope=definition.permission_scope,
            status_value="permission_denied",
            latency_ms=latency_ms,
            error_type="permission_denied",
            error="Tool permission denied",
        )
        return ToolInvokeResponse(
            tool_name=request.tool_name,
            status="permission_denied",
            required_permission=definition.permission_scope,
            latency_ms=latency_ms,
            error_type="permission_denied",
            error="Tool permission denied",
        )

    argument_model = get_argument_model(request.tool_name)
    if argument_model is None:
        latency_ms = _elapsed_ms(started_at)
        _audit(
            request=request,
            permission_scope=definition.permission_scope,
            status_value="failed",
            latency_ms=latency_ms,
            error_type="missing_argument_model",
            error="Tool argument model is not registered",
        )
        return ToolInvokeResponse(
            tool_name=request.tool_name,
            status="failed",
            required_permission=definition.permission_scope,
            latency_ms=latency_ms,
            error_type="missing_argument_model",
            error="Tool argument model is not registered",
        )

    try:
        args = argument_model.model_validate(request.arguments)
    except ValidationError as exc:
        latency_ms = _elapsed_ms(started_at)
        error = exc.errors()[0]["msg"] if exc.errors() else "Invalid tool arguments"
        _audit(
            request=request,
            permission_scope=definition.permission_scope,
            status_value="schema_error",
            latency_ms=latency_ms,
            error_type="schema_error",
            error=error,
        )
        return ToolInvokeResponse(
            tool_name=request.tool_name,
            status="schema_error",
            required_permission=definition.permission_scope,
            latency_ms=latency_ms,
            error_type="schema_error",
            error=error,
        )

    try:
        result = await execute_tool(request.tool_name, args)
    except httpx.TimeoutException:
        latency_ms = _elapsed_ms(started_at)
        _audit(
            request=request,
            permission_scope=definition.permission_scope,
            status_value="timeout",
            latency_ms=latency_ms,
            error_type="timeout",
            error="Tool downstream request timed out",
        )
        return ToolInvokeResponse(
            tool_name=request.tool_name,
            status="timeout",
            required_permission=definition.permission_scope,
            latency_ms=latency_ms,
            error_type="timeout",
            error="Tool downstream request timed out",
        )
    except Exception as exc:
        latency_ms = _elapsed_ms(started_at)
        _audit(
            request=request,
            permission_scope=definition.permission_scope,
            status_value="failed",
            latency_ms=latency_ms,
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        return ToolInvokeResponse(
            tool_name=request.tool_name,
            status="failed",
            required_permission=definition.permission_scope,
            latency_ms=latency_ms,
            error_type=exc.__class__.__name__,
            error=str(exc),
        )

    response_status = _status_from_result(result)
    latency_ms = _elapsed_ms(started_at)
    _audit(
        request=request,
        permission_scope=definition.permission_scope,
        status_value=response_status,
        latency_ms=latency_ms,
        error_type=None if response_status == "success" else response_status,
        error=None if response_status == "success" else str(result.get("message", response_status)),
        metadata={"implementation_status": definition.implementation_status},
    )
    return ToolInvokeResponse(
        tool_name=request.tool_name,
        status=response_status,  # type: ignore[arg-type]
        result=result,
        required_permission=definition.permission_scope,
        latency_ms=latency_ms,
        error_type=None if response_status == "success" else response_status,
        error=None if response_status == "success" else str(result.get("message", response_status)),
    )


def _status_from_result(result: Dict[str, Any]) -> str:
    status_value = result.get("status")
    if status_value in {"timeout", "failed"}:
        return str(status_value)
    return "success"


def _audit(
    *,
    request: ToolInvokeRequest,
    permission_scope: Optional[str],
    status_value: str,
    latency_ms: float,
    error_type: Optional[str],
    error: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    write_tool_audit(
        tool_name=request.tool_name,
        user_id=request.user_id,
        consumer_id=request.consumer_id,
        permission_scope=permission_scope,
        input_summary=_input_summary(request.arguments),
        status=status_value,
        latency_ms=latency_ms,
        trace_id=request.trace_id,
        request_id=request.request_id,
        error_type=error_type,
        error=error,
        metadata=metadata,
    )


def _input_summary(arguments: Dict[str, Any]) -> str:
    items = []
    sanitized = safe_metadata(arguments)
    for key in sorted(sanitized):
        if key == "_log_budget":
            continue
        value = sanitized[key]
        items.append(f"{key}={safe_summary(value, 80)}")
    return safe_summary(", ".join(items))


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
