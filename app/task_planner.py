from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class TaskPlanner:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_s: float,
        enable_thinking: bool,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._enable_thinking = enable_thinking

    async def plan(
        self,
        context: Dict[str, Any],
        *,
        repair_errors: Optional[List[str]] = None,
        invalid_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._api_key:
            return {"_planner_error": "missing SILICONFLOW_API_KEY"}
        messages = self._messages(context, repair_errors=repair_errors, invalid_plan=invalid_plan)
        payload = self._payload(messages, json_mode=True)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            if response.status_code >= 400 and "response_format" in payload:
                payload = self._payload(messages, json_mode=False)
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
            response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            return _loads_json(str(content))
        except ValueError as exc:
            return {"_planner_error": f"invalid planner json: {exc}", "_raw_content": str(content)[:2000]}

    def _payload(self, messages: List[Dict[str, str]], *, json_mode: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
            "enable_thinking": self._enable_thinking,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _messages(
        self,
        context: Dict[str, Any],
        *,
        repair_errors: Optional[List[str]],
        invalid_plan: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        prompt = {
            "required_json_shape": {
                "plan_version": "2.0",
                "primary_task_type": "knowledge_qa|coding|media_generation|document_writing",
                "secondary_task_types": [],
                "semantic_features": {},
                "execution_plan": [
                    {
                        "step_id": "snake_case",
                        "step_role": "knowledge_context|media_asset|final_compose|primary_answer|code_draft|media_generation",
                        "workflow": "registered workflow",
                        "purpose": "short",
                        "task_type": "optional task type",
                        "required_tools": [],
                        "forbidden_tools": [],
                        "depends_on": [],
                        "required": True,
                    }
                ],
                "contract_hints": {
                    "required_output_fields": [],
                    "required_trace_events": [],
                },
            },
            "planner_context": context,
            "repair_errors": repair_errors or [],
            "invalid_plan": invalid_plan or {},
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are SemGateway TaskPlanner. Return only valid JSON. "
                    "Do not answer the user task. Use only workflows, tools, A2A paths, "
                    "and rules listed in planner_context. Do not output acceptance_requirements; "
                    "Gateway builds acceptance criteria from validated steps."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]


def _loads_json(content: str) -> Dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no json object found")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("planner output is not a json object")
    return payload
