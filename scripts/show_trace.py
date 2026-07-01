from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a compact SemGateway trace.")
    parser.add_argument("trace", help="Trace id or trace JSON path.")
    parser.add_argument("--trace-dir", default="outputs/traces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = _trace_path(args.trace, Path(args.trace_dir))
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"trace_id={payload.get('trace_id')} events={payload.get('event_count')}")
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        node = event.get("node")
        status = event.get("status")
        summary = event.get("output_summary", "")
        print(f"- {node} [{status}] {summary}")
        if node in {"MemoryPlannerRead", "MemoryPlannerApply", "AgenticRouteDecision", "TaskContractBuild", "VerificationGate"}:
            _print_metadata(node, event.get("metadata", {}))


def _trace_path(value: str, trace_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = trace_dir / f"{value}.json"
    if candidate.exists():
        return candidate
    raise SystemExit(f"trace not found: {value}")


def _print_metadata(node: str, metadata: Dict[str, Any]) -> None:
    if node.startswith("MemoryPlanner"):
        print(f"  memory={json.dumps(metadata, ensure_ascii=False)}")
    elif node == "AgenticRouteDecision":
        print(f"  selected={metadata.get('selected_workflow')}")
    elif node == "TaskContractBuild":
        contract = metadata.get("task_contract", {})
        print(f"  contract_id={contract.get('contract_id')} criteria={len(contract.get('acceptance_criteria', []))}")
    elif node == "VerificationGate":
        verification = metadata.get("verification", {})
        print(f"  verification={verification.get('status')} failed={verification.get('failed_count')}")


if __name__ == "__main__":
    main()
