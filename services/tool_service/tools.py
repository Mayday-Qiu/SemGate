from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from services.tool_service.schemas import (
    DocSearchArgs,
    EvidenceCheckArgs,
    InspectionArgs,
    MediaGenerationArgs,
)


async def execute_tool(tool_name: str, args: Any) -> Dict[str, Any]:
    if tool_name == "doc_search_tool":
        return await doc_search(args)
    if tool_name == "evidence_check_tool":
        return await evidence_check(args)
    if tool_name in {"image_generation_tool", "video_generation_tool"}:
        return media_generation_placeholder(tool_name, args)
    if tool_name == "inspection_tool":
        return inspect_document(args)
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


def media_generation_placeholder(tool_name: str, args: MediaGenerationArgs) -> Dict[str, Any]:
    return {
        "implementation_status": "placeholder",
        "tool_name": tool_name,
        "media_type": args.media_type,
        "prompt_preview": args.prompt[:120],
        "message": "Interface reserved for a future image/video generation backend.",
    }


def inspect_document(args: InspectionArgs) -> Dict[str, Any]:
    text = args.document_text
    missing = [section for section in args.required_sections if section and section not in text]
    return {
        "passed": not missing,
        "missing_sections": missing,
        "warnings": [] if not missing else ["Document is missing required sections."],
        "inspection_type": args.check_type,
        "required_sections": args.required_sections,
    }


def _rag_url() -> str:
    return os.getenv("RAG_SERVICE_URL", "http://localhost:8020") + "/retrieve"


def _rag_timeout_s() -> float:
    try:
        return float(os.getenv("TOOL_RAG_TIMEOUT_S", "3.0"))
    except ValueError:
        return 3.0
