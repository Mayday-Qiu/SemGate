from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.schemas import AgentInvocationRequest, Citation


def initial_state(request: AgentInvocationRequest) -> Dict[str, Any]:
    return {
        "request": request,
        "status": "success",
        "tools": [],
        "tool_results": [],
        "trace_events": [],
        "errors": [],
        "evidence": [],
        "citations": [],
        "fallback_used": False,
        "retry_count": 0,
        "repair_count": 0,
        "should_retry": False,
        "should_repair": False,
    }


def evidence_to_citations(evidence: Sequence[Dict[str, Any]]) -> List[Citation]:
    citations: List[Citation] = []
    for item in evidence:
        try:
            citations.append(
                Citation(
                    source_id=str(item.get("source_id", "unknown_source")),
                    title=str(item.get("title", "unknown_title")),
                    chunk_id=str(item.get("chunk_id", "unknown_chunk")),
                    score=float(item.get("score", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return citations


def evidence_refs(evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs = []
    for item in evidence:
        source_id = str(item.get("source_id", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not source_id or not chunk_id:
            continue
        refs.append(
            {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "title": str(item.get("title", "")),
                "score": float(item.get("score", 0.0) or 0.0),
            }
        )
    return refs


def tool_succeeded(response: Dict[str, Any]) -> bool:
    return response.get("status") == "success"


def tool_result(response: Dict[str, Any]) -> Dict[str, Any]:
    result = response.get("result")
    return result if isinstance(result, dict) else {}


def tool_error(response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tool_name": response.get("tool_name"),
        "status": response.get("status"),
        "error_type": response.get("error_type"),
        "error": response.get("error"),
    }


def estimate_tokens(text: str) -> int:
    return max(80, int(len(text) / 4))


def merge_metadata(state: Dict[str, Any], **items: Any) -> Dict[str, Any]:
    metadata = state.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {**metadata, **items}
