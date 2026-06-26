from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    policy: str
    request_timeout_s: float
    fallback_max_attempts: int
    log_path: Path
    profiles_path: Path
    workflow_profiles_path: Path
    consumers_path: Path
    trace_log_path: Path
    trace_output_dir: Path
    agent_orchestrator_url: str
    rate_limit_enabled: bool
    rate_limit_replenish_rate: float
    rate_limit_burst_capacity: float
    rate_limit_requested_tokens: float
    circuit_breaker_enabled: bool
    circuit_sliding_window_size: int
    circuit_minimum_number_of_calls: int
    circuit_failure_rate_threshold: float
    circuit_slow_call_rate_threshold: float
    circuit_wait_duration_in_open_state_s: float
    circuit_permitted_calls_in_half_open: int
    mock_fast_url: str
    mock_quality_url: str
    mock_unstable_url: str


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
        policy=os.getenv("GATEWAY_POLICY", "fixed"),
        request_timeout_s=_get_float("GATEWAY_REQUEST_TIMEOUT_S", 1.5),
        fallback_max_attempts=_get_int("GATEWAY_FALLBACK_MAX_ATTEMPTS", 2),
        log_path=Path(os.getenv("GATEWAY_LOG_PATH", "logs/gateway.jsonl")),
        profiles_path=Path(os.getenv("GATEWAY_PROFILES_PATH", "outputs/profiles/service_profiles.json")),
        workflow_profiles_path=Path(os.getenv("GATEWAY_WORKFLOW_PROFILES_PATH", "configs/workflow_profiles.json")),
        consumers_path=Path(os.getenv("GATEWAY_CONSUMERS_PATH", "configs/consumers.json")),
        trace_log_path=Path(os.getenv("GATEWAY_TRACE_LOG_PATH", "logs/trace_events.jsonl")),
        trace_output_dir=Path(os.getenv("GATEWAY_TRACE_OUTPUT_DIR", "outputs/traces")),
        agent_orchestrator_url=os.getenv("AGENT_ORCHESTRATOR_URL", "http://localhost:8010/invoke"),
        rate_limit_enabled=_get_bool("GATEWAY_RATE_LIMIT_ENABLED", True),
        rate_limit_replenish_rate=_get_float("GATEWAY_RATE_LIMIT_REPLENISH_RATE", 1.0),
        rate_limit_burst_capacity=_get_float("GATEWAY_RATE_LIMIT_BURST_CAPACITY", 5.0),
        rate_limit_requested_tokens=_get_float("GATEWAY_RATE_LIMIT_REQUESTED_TOKENS", 1.0),
        circuit_breaker_enabled=_get_bool("GATEWAY_CIRCUIT_BREAKER_ENABLED", True),
        circuit_sliding_window_size=_get_int("GATEWAY_CIRCUIT_SLIDING_WINDOW_SIZE", 10),
        circuit_minimum_number_of_calls=_get_int("GATEWAY_CIRCUIT_MINIMUM_NUMBER_OF_CALLS", 4),
        circuit_failure_rate_threshold=_get_float("GATEWAY_CIRCUIT_FAILURE_RATE_THRESHOLD", 50.0),
        circuit_slow_call_rate_threshold=_get_float("GATEWAY_CIRCUIT_SLOW_CALL_RATE_THRESHOLD", 60.0),
        circuit_wait_duration_in_open_state_s=_get_float("GATEWAY_CIRCUIT_WAIT_DURATION_IN_OPEN_STATE_S", 10.0),
        circuit_permitted_calls_in_half_open=_get_int("GATEWAY_CIRCUIT_PERMITTED_CALLS_IN_HALF_OPEN", 2),
        mock_fast_url=os.getenv("MOCK_FAST_URL", "http://localhost:8001/invoke"),
        mock_quality_url=os.getenv("MOCK_QUALITY_URL", "http://localhost:8002/invoke"),
        mock_unstable_url=os.getenv("MOCK_UNSTABLE_URL", "http://localhost:8003/invoke"),
    )
