from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status

from app.schemas import BackendResponse, GatewayRequest, TaskType


@dataclass(frozen=True)
class MockSettings:
    backend_id: str
    backend_type: str
    supported_tasks: List[TaskType]
    min_latency_ms: int
    max_latency_ms: int
    error_rate: float
    timeout_rate: float
    timeout_sleep_ms: int
    heuristic_quality_score: float
    random_seed: int


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def load_mock_settings() -> MockSettings:
    supported_tasks = [
        item.strip()
        for item in os.getenv("SUPPORTED_TASKS", "summary,semantic_explain").split(",")
        if item.strip()
    ]
    return MockSettings(
        backend_id=os.getenv("BACKEND_ID", "mock_backend"),
        backend_type=os.getenv("BACKEND_TYPE", "mock"),
        supported_tasks=supported_tasks,  # type: ignore[arg-type]
        min_latency_ms=_int_env("MIN_LATENCY_MS", 100),
        max_latency_ms=_int_env("MAX_LATENCY_MS", 300),
        error_rate=_float_env("ERROR_RATE", 0.0),
        timeout_rate=_float_env("TIMEOUT_RATE", 0.0),
        timeout_sleep_ms=_int_env("TIMEOUT_SLEEP_MS", 3000),
        heuristic_quality_score=_float_env("HEURISTIC_QUALITY_SCORE", 0.7),
        random_seed=_int_env("MOCK_RANDOM_SEED", 42),
    )


settings = load_mock_settings()
random_source = random.Random(f"{settings.backend_id}:{settings.random_seed}")
app = FastAPI(title=f"{settings.backend_id} service", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "backend_id": settings.backend_id,
        "supported_tasks": settings.supported_tasks,
    }


@app.post("/invoke", response_model=BackendResponse)
async def invoke(request: GatewayRequest) -> BackendResponse:
    if request.task_type not in settings.supported_tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{settings.backend_id} does not support task_type={request.task_type}",
        )

    started_at = perf_counter()
    if random_source.random() < settings.timeout_rate:
        await asyncio.sleep(settings.timeout_sleep_ms / 1000)

    if random_source.random() < settings.error_rate:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{settings.backend_id} injected failure",
        )

    latency_ms = random_source.randint(settings.min_latency_ms, settings.max_latency_ms)
    await asyncio.sleep(latency_ms / 1000)
    actual_latency_ms = round((perf_counter() - started_at) * 1000, 3)
    return BackendResponse(
        request_id=request.request_id or str(uuid4()),
        backend_id=settings.backend_id,
        task_type=request.task_type,
        output=_build_output(request),
        heuristic_quality_score=settings.heuristic_quality_score,
        latency_ms=actual_latency_ms,
        metadata={
            "backend_type": settings.backend_type,
            "configured_latency_ms": latency_ms,
        },
    )


def _build_output(request: GatewayRequest) -> str:
    return (
        f"{settings.backend_id} handled {request.task_type} for "
        f"user={request.user_id}: {request.input}"
    )
