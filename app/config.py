from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    request_timeout_s: float
    log_path: Path
    workflow_profiles_path: Path
    consumers_path: Path
    trace_log_path: Path
    trace_output_dir: Path
    tool_audit_log_path: Path
    agent_orchestrator_url: str
    rate_limit_enabled: bool
    rate_limit_replenish_rate: float
    rate_limit_burst_capacity: float
    rate_limit_requested_tokens: float
    memory_planner_enabled: bool
    planner_memory_dir: Path
    task_planner_enabled: bool
    siliconflow_base_url: str
    siliconflow_api_key: str
    planner_model_id: str
    planner_temperature: float
    planner_top_p: float
    planner_max_tokens: int
    planner_timeout_s: float
    planner_repair_max_retries: int
    planner_enable_thinking: bool


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}")


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("SEMROUTE_API_KEY", "dev-key"),
        request_timeout_s=_get_float("GATEWAY_REQUEST_TIMEOUT_S", 1.5),
        log_path=Path(os.getenv("GATEWAY_LOG_PATH", "logs/gateway.jsonl")),
        workflow_profiles_path=Path(os.getenv("GATEWAY_WORKFLOW_PROFILES_PATH", "configs/workflow_profiles.json")),
        consumers_path=Path(os.getenv("GATEWAY_CONSUMERS_PATH", "configs/consumers.json")),
        trace_log_path=Path(os.getenv("GATEWAY_TRACE_LOG_PATH", "logs/trace_events.jsonl")),
        trace_output_dir=Path(os.getenv("GATEWAY_TRACE_OUTPUT_DIR", "outputs/traces")),
        tool_audit_log_path=Path(os.getenv("GATEWAY_TOOL_AUDIT_LOG_PATH", "logs/tool_audit.jsonl")),
        agent_orchestrator_url=os.getenv("AGENT_ORCHESTRATOR_URL", "http://localhost:8010/invoke"),
        rate_limit_enabled=_get_bool("GATEWAY_RATE_LIMIT_ENABLED", True),
        rate_limit_replenish_rate=_get_float("GATEWAY_RATE_LIMIT_REPLENISH_RATE", 1.0),
        rate_limit_burst_capacity=_get_float("GATEWAY_RATE_LIMIT_BURST_CAPACITY", 5.0),
        rate_limit_requested_tokens=_get_float("GATEWAY_RATE_LIMIT_REQUESTED_TOKENS", 1.0),
        memory_planner_enabled=_get_bool("MEMORY_PLANNER_ENABLED", True),
        planner_memory_dir=Path(os.getenv("PLANNER_MEMORY_DIR", "outputs/planner_memory")),
        task_planner_enabled=_get_bool("TASK_PLANNER_ENABLED", False),
        siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        planner_model_id=os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-8B"),
        planner_temperature=_get_float("PLANNER_TEMPERATURE", 0.0),
        planner_top_p=_get_float("PLANNER_TOP_P", 0.8),
        planner_max_tokens=_get_int("PLANNER_MAX_TOKENS", 1200),
        planner_timeout_s=_get_float("PLANNER_TIMEOUT_SECONDS", 45.0),
        planner_repair_max_retries=_get_int("PLANNER_REPAIR_MAX_RETRIES", 1),
        planner_enable_thinking=_get_bool("PLANNER_ENABLE_THINKING", False),
    )
