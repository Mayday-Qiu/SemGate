from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from app.schemas import AgentInvocationRequest


class ToolClient:
    def __init__(self, base_url: Optional[str] = None, timeout_s: Optional[float] = None) -> None:
        self._base_url = (base_url or os.getenv("TOOL_SERVICE_URL", "http://localhost:8030")).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else _float_env("AGENT_TOOL_TIMEOUT_S", 3.0)

    async def invoke(
        self,
        request: AgentInvocationRequest,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
            "user_id": request.user_id,
            "consumer_id": request.consumer_id,
            "trace_id": request.trace_id,
            "request_id": request.request_id,
            "permissions": request.permissions,
            "allowed_tools": request.allowed_tools,
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/invoke", json=payload)
            response.raise_for_status()
        return response.json()


class ModelClient:
    def __init__(self, base_url: Optional[str] = None, timeout_s: Optional[float] = None) -> None:
        self._base_url = (base_url or os.getenv("MODEL_BACKEND_URL", "http://localhost:8040")).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else _float_env("AGENT_MODEL_TIMEOUT_S", 5.0)

    async def invoke(self, prompt: str, model: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"prompt": prompt, "model": model, "metadata": metadata or {}}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/invoke", json=payload)
            response.raise_for_status()
        return response.json()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
