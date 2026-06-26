from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="SemRoute RAG Service", version="0.7.0")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    strategy: str = "hybrid"


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "rag_service"}


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> Dict[str, Any]:
    evidence = _demo_evidence(request.query, request.top_k)
    return {
        "query": request.query,
        "strategy": request.strategy,
        "evidence": evidence,
        "implementation_status": "phase_0_1_stub",
    }


def _demo_evidence(query: str, top_k: int) -> List[Dict[str, Any]]:
    candidates = [
        {
            "source_id": "doc_v07",
            "title": "SemRoute-Gateway_v0.7",
            "chunk_id": "c_agentic_router",
            "score": 0.83,
            "text": "Gateway builds TaskProfile and uses AgenticRouter to choose a workflow.",
        },
        {
            "source_id": "doc_architecture",
            "title": "architecture.md",
            "chunk_id": "c_trace_eval",
            "score": 0.78,
            "text": "Trace and eval provide replayable evidence for Agent/RAG behavior.",
        },
        {
            "source_id": "doc_tool_gateway",
            "title": "tool_gateway.md",
            "chunk_id": "c_permission_audit",
            "score": 0.72,
            "text": "Tool Gateway validates schema, permissions, and audit records.",
        },
    ]
    return candidates[:top_k]
