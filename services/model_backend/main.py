from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="SemRoute Model Backend", version="0.7.0")


class ModelInvokeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "mock"
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "model_backend",
        "provider": os.getenv("MODEL_PROVIDER", "mock"),
    }


@app.post("/invoke")
async def invoke(request: ModelInvokeRequest) -> Dict[str, Any]:
    started_at = perf_counter()
    provider = os.getenv("MODEL_PROVIDER", "mock")
    answer = (
        "This is a deterministic mock model response. "
        "OpenAI-compatible real mode is reserved for a later adapter step."
    )
    estimated_prompt_tokens = max(1, int(len(request.prompt) / 4))
    estimated_completion_tokens = max(20, int(len(answer) / 4))
    return {
        "model": request.model if provider != "mock" else "mock_model",
        "provider": provider,
        "answer": answer,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "latency_ms": round((perf_counter() - started_at) * 1000, 3),
        "status": "success",
    }
