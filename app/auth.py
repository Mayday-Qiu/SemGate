from __future__ import annotations

import json
from secrets import compare_digest
from time import perf_counter
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import load_settings
from app.schemas import AuthenticatedConsumer


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> AuthenticatedConsumer:
    started_at = perf_counter()
    settings = load_settings()
    candidate = api_key
    if candidate is None and authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            candidate = token

    consumer = _find_consumer(candidate, _load_consumer_records(settings.consumers_path, settings.api_key))
    if consumer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return consumer.model_copy(update={"auth_latency_ms": _elapsed_ms(started_at)})


def _load_consumer_records(path: Any, fallback_api_key: str) -> List[Dict[str, Any]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records = payload.get("consumers", [])
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]

    return [
        {
            "consumer_id": "dev_consumer",
            "api_key_id": "fallback-dev-key",
            "api_key": fallback_api_key,
            "permissions": ["*"],
            "allowed_workflows": ["*"],
            "allowed_tools": ["*"],
            "rate_limit_key": "consumer:dev_consumer",
        }
    ]


def _find_consumer(candidate: Optional[str], records: List[Dict[str, Any]]) -> Optional[AuthenticatedConsumer]:
    if not candidate:
        return None

    for record in records:
        keys = _record_keys(record)
        if any(compare_digest(candidate, key) for key in keys):
            consumer_id = str(record.get("consumer_id") or record.get("name") or "unknown_consumer")
            return AuthenticatedConsumer(
                consumer_id=consumer_id,
                api_key_id=str(record.get("api_key_id") or consumer_id),
                permissions=_string_list(record.get("permissions", [])),
                allowed_workflows=_string_list(record.get("allowed_workflows", [])),
                allowed_tools=_string_list(record.get("allowed_tools", [])),
                rate_limit_key=str(record.get("rate_limit_key") or f"consumer:{consumer_id}"),
            )
    return None


def _record_keys(record: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    single_key = record.get("api_key") or record.get("credential")
    if isinstance(single_key, str) and single_key:
        keys.append(single_key)

    multiple_keys = record.get("api_keys") or record.get("credentials")
    if isinstance(multiple_keys, list):
        keys.extend(str(item) for item in multiple_keys if str(item))
    return keys


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
