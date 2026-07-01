from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


SUMMARY_CHAR_LIMIT = 300
STRING_CHAR_LIMIT = 800
LIST_ITEM_LIMIT = 20
DICT_ITEM_LIMIT = 50
METADATA_JSON_LIMIT = 12000
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass
class LogBudgetReport:
    redacted_keys: List[str] = field(default_factory=list)
    truncated_fields: List[str] = field(default_factory=list)
    dropped_fields: List[str] = field(default_factory=list)
    original_json_length: int = 0
    sanitized_json_length: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.redacted_keys or self.truncated_fields or self.dropped_fields)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "redacted_keys": sorted(set(self.redacted_keys)),
            "truncated_fields": sorted(set(self.truncated_fields)),
            "dropped_fields": sorted(set(self.dropped_fields)),
            "original_json_length": self.original_json_length,
            "sanitized_json_length": self.sanitized_json_length,
        }


def safe_summary(value: Any, limit: int = SUMMARY_CHAR_LIMIT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated chars={len(text) - limit}]"


def safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    report = LogBudgetReport()
    original_json = _json_dumps(metadata)
    report.original_json_length = len(original_json)
    sanitized = _sanitize_value(metadata, "$", report)
    if not isinstance(sanitized, dict):
        sanitized = {"value": sanitized}

    sanitized_json = _json_dumps(sanitized)
    if len(sanitized_json) > METADATA_JSON_LIMIT:
        report.truncated_fields.append("$")
        sanitized = {
            "_metadata_truncated": True,
            "preview": safe_summary(sanitized_json, METADATA_JSON_LIMIT),
        }
        sanitized_json = _json_dumps(sanitized)

    report.sanitized_json_length = len(sanitized_json)
    if report.changed:
        sanitized["_log_budget"] = report.to_metadata()
    return sanitized


def _sanitize_value(value: Any, path: str, report: LogBudgetReport) -> Any:
    if isinstance(value, dict):
        return _sanitize_dict(value, path, report)
    if isinstance(value, list):
        return _sanitize_list(value, path, report)
    if isinstance(value, tuple):
        return _sanitize_list(list(value), path, report)
    if isinstance(value, str):
        return _sanitize_string(value, path, report)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_string(str(value), path, report)


def _sanitize_dict(value: Dict[Any, Any], path: str, report: LogBudgetReport) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    items: List[Tuple[Any, Any]] = list(value.items())
    for index, (raw_key, raw_value) in enumerate(items):
        key = str(raw_key)
        key_path = f"{path}.{key}"
        if index >= DICT_ITEM_LIMIT:
            report.dropped_fields.append(f"{path}.*")
            sanitized["_dropped_fields_count"] = len(items) - DICT_ITEM_LIMIT
            break
        if _is_sensitive_key(key):
            report.redacted_keys.append(key_path)
            sanitized[key] = "[REDACTED]"
            continue
        sanitized[key] = _sanitize_value(raw_value, key_path, report)
    return sanitized


def _sanitize_list(value: List[Any], path: str, report: LogBudgetReport) -> List[Any]:
    if len(value) > LIST_ITEM_LIMIT:
        report.dropped_fields.append(f"{path}[]")
    items = value[:LIST_ITEM_LIMIT]
    sanitized = [_sanitize_value(item, f"{path}[{index}]", report) for index, item in enumerate(items)]
    if len(value) > LIST_ITEM_LIMIT:
        sanitized.append({"_dropped_items_count": len(value) - LIST_ITEM_LIMIT})
    return sanitized


def _sanitize_string(value: str, path: str, report: LogBudgetReport) -> str:
    if len(value) <= STRING_CHAR_LIMIT:
        return value
    report.truncated_fields.append(path)
    return f"{value[:STRING_CHAR_LIMIT]}...[truncated chars={len(value) - STRING_CHAR_LIMIT}]"


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
