from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.schemas import AgentInvocationRequest, Citation


def has_permissions(request: AgentInvocationRequest, required_scopes: Sequence[str]) -> bool:
    permissions = set(request.permissions)
    return "*" in permissions or set(required_scopes).issubset(permissions)


def permission_missing(request: AgentInvocationRequest, required_scopes: Sequence[str]) -> List[str]:
    permissions = set(request.permissions)
    if "*" in permissions:
        return []
    return sorted(set(required_scopes) - permissions)


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


def first_record(result: Dict[str, Any]) -> Dict[str, Any]:
    records = result.get("records")
    if isinstance(records, list) and records:
        record = records[0]
        if isinstance(record, dict):
            return record
    return {}


def estimate_tokens(text: str) -> int:
    return max(80, int(len(text) / 4))
