from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx

from services.tool_service.schemas import (
    BusinessQueryArgs,
    DocSearchArgs,
    EvidenceCheckArgs,
    InspectionArgs,
    MediaGenerationArgs,
    ServiceStatusArgs,
)


MOCK_REQUIREMENTS = [
    {
        "id": "REQ-001",
        "title": "Gateway trace replay",
        "status": "accepted",
        "summary": "Each request must produce a trace_id and a replayable trace file.",
    },
    {
        "id": "REQ-002",
        "title": "Tool governance",
        "status": "in_progress",
        "summary": "Tool calls must pass schema validation, permission check, and audit logging.",
    },
]

MOCK_APIS = [
    {
        "id": "API-001",
        "path": "/v1/invoke",
        "owner": "gateway",
        "summary": "Unified Agentic Gateway entrypoint.",
    },
    {
        "id": "API-002",
        "path": "/invoke",
        "owner": "tool_service",
        "summary": "Tool Gateway invocation endpoint.",
    },
]

MOCK_TICKETS = [
    {
        "id": "INC-001",
        "service": "gateway",
        "severity": "high",
        "summary": "Gateway returned intermittent timeout while calling downstream workflow.",
    },
    {
        "id": "INC-002",
        "service": "rag_service",
        "severity": "medium",
        "summary": "Document retrieval returned low confidence evidence for a project architecture question.",
    },
]

MOCK_SERVICE_STATUS = [
    {
        "service_name": "gateway",
        "status": "degraded",
        "latency_ms": 180,
        "error_rate": 0.03,
        "message": "Gateway is serving traffic but downstream workflow calls are slower than usual.",
    },
    {
        "service_name": "rag_service",
        "status": "ok",
        "latency_ms": 95,
        "error_rate": 0.01,
        "message": "RAG service is available.",
    },
    {
        "service_name": "tool_service",
        "status": "ok",
        "latency_ms": 70,
        "error_rate": 0.0,
        "message": "Tool service is available.",
    },
]


async def execute_tool(tool_name: str, args: Any) -> Dict[str, Any]:
    if tool_name == "doc_search_tool":
        return await doc_search(args)
    if tool_name == "evidence_check_tool":
        return await evidence_check(args)
    if tool_name == "business_query_tool":
        return business_query(args)
    if tool_name == "service_status_tool":
        return service_status(args)
    if tool_name == "inspection_tool":
        return inspection(args)
    if tool_name in {"image_generation_tool", "video_generation_tool"}:
        return media_generation_placeholder(tool_name, args)
    raise ValueError(f"Unknown tool {tool_name}")


async def doc_search(args: DocSearchArgs) -> Dict[str, Any]:
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "strategy": args.strategy,
    }
    async with httpx.AsyncClient(timeout=_rag_timeout_s()) as client:
        response = await client.post(_rag_url(), json=payload)
        response.raise_for_status()
    data = response.json()
    return {
        "query": args.query,
        "strategy": data.get("strategy", args.strategy),
        "evidence": data.get("evidence", []),
        "source": "rag_service",
    }


async def evidence_check(args: EvidenceCheckArgs) -> Dict[str, Any]:
    query = args.evidence_query or args.claim
    search_result = await doc_search(DocSearchArgs(query=query, top_k=args.top_k, strategy="hybrid"))
    evidence = search_result.get("evidence", [])
    support_score = max([float(item.get("score", 0.0)) for item in evidence] or [0.0])
    supported = support_score >= args.min_score
    return {
        "claim": args.claim,
        "evidence_query": query,
        "supported": supported,
        "support_score": round(support_score, 6),
        "min_score": args.min_score,
        "missing_items": [] if supported else ["No retrieved evidence reached the support threshold."],
        "evidence": evidence,
        "source": "rag_service",
    }


def business_query(args: BusinessQueryArgs) -> Dict[str, Any]:
    records_by_type = {
        "requirement": MOCK_REQUIREMENTS,
        "api": MOCK_APIS,
        "ticket": MOCK_TICKETS,
        "service_status": MOCK_SERVICE_STATUS,
    }
    records = _filter_records(records_by_type[args.entity_type], args.query)
    return {
        "entity_type": args.entity_type,
        "query": args.query,
        "records": records,
        "record_count": len(records),
    }


def service_status(args: ServiceStatusArgs) -> Dict[str, Any]:
    if args.simulate_failure == "timeout":
        return {
            "status": "timeout",
            "service_name": args.service_name,
            "message": "Simulated service status timeout.",
        }
    if args.simulate_failure == "failed":
        return {
            "status": "failed",
            "service_name": args.service_name,
            "message": "Simulated service status failure.",
        }
    records = _filter_records(MOCK_SERVICE_STATUS, args.service_name)
    if not records:
        return {
            "status": "unknown",
            "service_name": args.service_name,
            "message": "No mock service status record matched the query.",
        }
    return records[0]


def inspection(args: InspectionArgs) -> Dict[str, Any]:
    content = args.content.strip()
    findings: List[Dict[str, Any]] = []
    missing_items: List[str] = []
    suggested_next_tools: List[str] = []

    if args.inspection_type == "document_structure":
        required_markers = ["background", "goal", "solution", "acceptance"]
        missing_items = [marker for marker in required_markers if marker not in content]
        if missing_items:
            findings.append({"level": "medium", "message": "Document is missing expected structure markers."})
            suggested_next_tools.append("doc_search_tool")
    elif args.inspection_type == "policy_compliance":
        blocked_terms = ["泄露密钥", "绕过鉴权", "ignore safety"]
        hits = [term for term in blocked_terms if term.lower() in content.lower()]
        missing_items = hits
        if hits:
            findings.append({"level": "high", "message": "Content contains policy-risk wording."})
    elif args.inspection_type == "caption_quality":
        if len(content) < 30:
            missing_items.append("caption_too_short")
        if not any(word in content.lower() for word in ["scene", "style", "camera", "镜头", "场景"]):
            missing_items.append("missing_visual_detail")
        if missing_items:
            findings.append({"level": "medium", "message": "Caption lacks enough generation detail."})
    elif args.inspection_type == "task_feasibility":
        if "真实生成视频" in content or "train a model" in content.lower():
            findings.append({"level": "medium", "message": "Task may exceed the current Phase 2 capability boundary."})
            missing_items.append("requires_later_phase_capability")
    elif args.inspection_type == "evidence_consistency":
        if not args.context.get("evidence"):
            findings.append({"level": "high", "message": "No evidence was supplied for consistency inspection."})
            missing_items.append("evidence")
            suggested_next_tools.append("evidence_check_tool")

    passed = not missing_items
    risk_level = "low" if passed else max((item["level"] for item in findings), default="medium")
    return {
        "inspection_type": args.inspection_type,
        "passed": passed,
        "risk_level": risk_level,
        "findings": findings,
        "missing_items": missing_items,
        "suggested_next_tools": sorted(set(suggested_next_tools)),
    }


def media_generation_placeholder(tool_name: str, args: MediaGenerationArgs) -> Dict[str, Any]:
    return {
        "implementation_status": "placeholder",
        "tool_name": tool_name,
        "media_type": args.media_type,
        "prompt_preview": args.prompt[:120],
        "message": "Interface reserved for a future image/video generation backend.",
    }


def _filter_records(records: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    normalized = query.lower()
    matches = [
        record
        for record in records
        if normalized in " ".join(str(value).lower() for value in record.values())
    ]
    return matches or records[:1]


def _rag_url() -> str:
    return os.getenv("RAG_SERVICE_URL", "http://localhost:8020") + "/retrieve"


def _rag_timeout_s() -> float:
    try:
        return float(os.getenv("TOOL_RAG_TIMEOUT_S", "3.0"))
    except ValueError:
        return 3.0
