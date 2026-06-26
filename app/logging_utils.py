from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app.schemas import RequestLog


def write_request_log(log_path: Path, record: RequestLog) -> None:
    payload = record.model_dump(mode="json")
    write_jsonl_record(log_path, payload)


def write_jsonl_record(log_path: Path, payload: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
