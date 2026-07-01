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
    )
