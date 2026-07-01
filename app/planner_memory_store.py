from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class PlannerMemoryStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self) -> Dict[str, Any]:
        warnings: List[str] = []
        return {
            "route_rules": self._read_rules("route_rules.json", warnings),
            "contract_rules": self._read_rules("contract_rules.json", warnings),
            "warnings": warnings,
        }

    def _read_rules(self, filename: str, warnings: List[str]) -> List[Dict[str, Any]]:
        path = self._directory / filename
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{filename}: failed to read planner memory rules: {exc}")
            return []

        rules = raw.get("rules", raw) if isinstance(raw, dict) else raw
        if not isinstance(rules, list):
            warnings.append(f"{filename}: planner memory rules must be a list or {{\"rules\": [...]}}")
            return []

        valid_rules = []
        for index, rule in enumerate(rules):
            if isinstance(rule, dict):
                valid_rules.append(rule)
            else:
                warnings.append(f"{filename}[{index}]: planner memory rule must be an object")
        return valid_rules
