from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map lightweight task eval JSONL to demo Planner Memory rules.")
    parser.add_argument("--input", default="outputs/eval/task_eval_results.jsonl")
    parser.add_argument("--output-dir", default="outputs/planner_memory")
    parser.add_argument("--report", default="outputs/reports/planner_memory_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(Path(args.input))
    if not records:
        print(f"no eval records found at {args.input}; planner memory rules unchanged")
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_rules = _route_rules(records)
    contract_rules = _contract_rules(records)
    _write_rules(output_dir / "route_rules.json", route_rules)
    _write_rules(output_dir / "contract_rules.json", contract_rules)
    _write_report(Path(args.report), records, route_rules, contract_rules)
    print(f"mapped lightweight eval records to route_rules={len(route_rules)}, contract_rules={len(contract_rules)} in {output_dir}")


def _route_rules(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for record in records:
        if (
            record.get("task_type") == "knowledge_qa"
            and record.get("selected_workflow") == "large_knowledge_qa_workflow"
            and _hard_pass(record)
        ):
            return [
                {
                    "rule_id": "rr_deep_knowledge_001",
                    "status": "active",
                    "applies_when": {
                        "task_type": "knowledge_qa",
                        "capabilities_any": ["deep_rag", "multi_hop_retrieval", "evidence_synthesis"],
                    },
                    "routing_hint": {
                        "prefer_workflow": "large_knowledge_qa_workflow",
                        "score_boost": 0.12,
                    },
                    "reason": "Deep knowledge tasks should prefer the bounded deep research workflow.",
                }
            ]
    return []


def _contract_rules(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    has_document_boundary_failure = any(
        record.get("task_type") == "document_writing"
        and not _hard_pass(record)
        and (
            _failed_metric(record, "ToolComplianceMetric")
            or _failed_metric(record, "CitationSourceMetric")
            or _failed_metric(record, "ContractComplianceMetric")
            or _failed_metric(record, "TraceEventCoverageMetric")
        )
        for record in records
    )
    if not has_document_boundary_failure:
        return []
    return [
        {
            "rule_id": "cr_document_boundary_001",
            "status": "active",
            "applies_when": {
                "task_type": "document_writing",
                "capabilities_any": ["document_writing"],
            },
            "patch": {
                "add_required_trace_events": ["NeedKnowledge"],
                "add_forbidden_tools": ["doc_search_tool", "evidence_check_tool"],
                "add_acceptance_criteria": [
                    {
                        "criterion_id": "memory:document_direct_rag_boundary",
                        "type": "metadata_required",
                        "target": "direct_rag_access",
                        "required": True,
                        "description": "Document writing must report whether it used direct RAG.",
                    }
                ],
            },
            "reason": "Document writing must keep the direct RAG boundary visible in verification.",
        }
    ]


def _hard_pass(record: Dict[str, Any]) -> bool:
    failed_metrics = record.get("failed_metrics")
    if isinstance(failed_metrics, list) and failed_metrics:
        return False
    metric_results = record.get("metric_results")
    if isinstance(metric_results, dict):
        return all(bool(item.get("passed")) for item in metric_results.values() if isinstance(item, dict))
    return record.get("verification_status") in {"passed", "success"}


def _failed_metric(record: Dict[str, Any], metric_name: str) -> bool:
    failed = record.get("failed_metrics")
    if isinstance(failed, list):
        return metric_name in failed
    result = record.get("metric_results", {}).get(metric_name)
    return isinstance(result, dict) and not bool(result.get("passed"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _write_rules(path: Path, rules: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    records: List[Dict[str, Any]],
    route_rules: List[Dict[str, Any]],
    contract_rules: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [record for record in records if not _hard_pass(record)]
    lines = [
        "# Planner Memory Report",
        "",
        f"- eval_records: {len(records)}",
        f"- failed_records: {len(failed)}",
        f"- route_rules: {len(route_rules)}",
        f"- contract_rules: {len(contract_rules)}",
        "",
        "| case_id | task_type | selected_workflow | failed_metrics |",
        "|---|---|---|---|",
    ]
    for record in records:
        failed_metrics = ", ".join(str(item) for item in record.get("failed_metrics", [])) or "-"
        lines.append(
            f"| {record.get('case_id')} | {record.get('task_type')} | {record.get('selected_workflow')} | {failed_metrics} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
