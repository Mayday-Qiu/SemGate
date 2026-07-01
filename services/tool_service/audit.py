from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.logging_utils import write_jsonl_record
from app.safe_logging import safe_metadata, safe_summary


def write_tool_audit(
    *,
    tool_name: str,
    user_id: str,
    consumer_id: str,
    permission_scope: Optional[str],
    input_summary: str,
    status: str,
    latency_ms: float,
    trace_id: Optional[str],
    request_id: Optional[str],
    error_type: Optional[str],
    error: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    log_path = Path(os.getenv("TOOL_AUDIT_LOG_PATH", "logs/tool_audit.jsonl"))
    write_jsonl_record(
        log_path,
        {
            "schema": "tool_audit_v1",
            "trace_id": trace_id,
            "request_id": request_id,
            "tool_name": tool_name,
            "user_id": user_id,
            "consumer_id": consumer_id,
            "permission_scope": permission_scope,
            "input_summary": safe_summary(input_summary),
            "status": status,
            "latency_ms": latency_ms,
            "error_type": error_type,
            "error": safe_summary(error),
            "metadata": safe_metadata(metadata or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
