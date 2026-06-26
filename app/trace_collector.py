from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.logging_utils import write_jsonl_record
from app.safe_logging import (
    DICT_ITEM_LIMIT,
    LIST_ITEM_LIMIT,
    METADATA_JSON_LIMIT,
    SUMMARY_CHAR_LIMIT,
    STRING_CHAR_LIMIT,
    safe_metadata,
    safe_summary,
)
from app.schemas import TraceEvent


class TraceCollector:
    def __init__(self, trace_id: str, request_id: str, log_path: Path, output_dir: Path) -> None:
        self.trace_id = trace_id
        self.request_id = request_id
        self._log_path = log_path
        self._output_dir = output_dir
        self._events: List[TraceEvent] = []

    @property
    def trace_path(self) -> Path:
        return self._output_dir / f"{self.trace_id}.json"

    @property
    def event_count(self) -> int:
        return len(self._events)

    def add(
        self,
        service: str,
        node: str,
        event_type: str,
        input_summary: str = "",
        output_summary: str = "",
        status: str = "success",
        latency_ms: float = 0.0,
        estimated_tokens: int = 0,
        estimated_cost: float = 0.0,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id,
            request_id=self.request_id,
            event_id=str(uuid4()),
            service=service,
            node=node,
            event_type=event_type,  # type: ignore[arg-type]
            input_summary=safe_summary(input_summary),
            output_summary=safe_summary(output_summary),
            status=status,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            error_type=error_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=safe_metadata(metadata or {}),
        )
        self._events.append(event)
        return event

    def extend(self, events: List[TraceEvent]) -> None:
        self._events.extend([self._sanitize_event(event) for event in events])

    def flush(self) -> None:
        self._events = [self._sanitize_event(event) for event in self._events]
        for event in self._events:
            write_jsonl_record(self._log_path, event.model_dump(mode="json"))

        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "agentic_trace_v1",
            "trace_detail_level": "full",
            "log_budget": {
                "summary_char_limit": SUMMARY_CHAR_LIMIT,
                "string_char_limit": STRING_CHAR_LIMIT,
                "list_item_limit": LIST_ITEM_LIMIT,
                "dict_item_limit": DICT_ITEM_LIMIT,
                "metadata_json_limit": METADATA_JSON_LIMIT,
                "redaction": "keys containing api_key, authorization, token, password, secret, credential, cookie",
            },
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "event_count": len(self._events),
            "events": [event.model_dump(mode="json") for event in self._events],
        }
        self.trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sanitize_event(self, event: TraceEvent) -> TraceEvent:
        return event.model_copy(
            update={
                "input_summary": safe_summary(event.input_summary),
                "output_summary": safe_summary(event.output_summary),
                "metadata": safe_metadata(event.metadata),
            }
        )
