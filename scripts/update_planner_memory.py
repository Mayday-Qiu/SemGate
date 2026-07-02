from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MIN_GROUP_RECORDS = 3
MIN_FAILURES = 2
MIN_FAILURE_RATIO = 0.67

FAILURE_FALLBACK = [
    ("PlannerSchemaValidityMetric", "schema_validation_failed", "planner_prompt"),
    ("PlanValidationPassMetric", "plan_validation_failed", "plan_validator"),
    ("TaskTypeCorrectnessMetric", "wrong_primary_task_type", "planner_prompt"),
    ("WorkflowPlanCorrectnessMetric", "workflow_plan_mismatch", "planner_context"),
    ("RouteCorrectnessMetric", "route_mismatch", "agentic_router"),
    ("ContractHintCoverageMetric", "missing_contract_hint", "task_contract_builder"),
    ("ContractComplianceMetric", "contract_compliance_failed", "task_contract_builder"),
    ("TraceEventCoverageMetric", "execution_trace_missing", "workflow_implementation"),
    ("ToolComplianceMetric", "tool_compliance_failed", "tool_service"),
    ("CitationSourceMetric", "citation_missing", "verification_gate"),
    ("VerificationPassMetric", "verification_failed", "verification_gate"),
    ("SchemaComplianceMetric", "output_schema_missing", "workflow_implementation"),
]
ROUTE_FAILURES = {"route_mismatch"}
ROUTE_TARGETS = {"agentic_router", "workflow_profile"}
CONTRACT_FAILURES = {"missing_contract_hint", "forbidden_tool_not_preserved", "contract_compliance_failed"}
CONTRACT_TARGETS = {"task_contract_builder"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map stable eval failures to Planner Memory candidates.")
    parser.add_argument("--input", default="outputs/eval/task_eval_results.jsonl")
    parser.add_argument("--output-dir", default="outputs/planner_memory")
    parser.add_argument("--report", default="outputs/reports/planner_memory_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(Path(args.input))
    if not records:
        print(f"no eval records found at {args.input}; planner memory candidates unchanged")
        return

    groups = _group_records(records)
    route_rules, contract_rules, group_summaries = _build_candidates(groups)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rules(output_dir / "route_rules.json", route_rules)
    _write_rules(output_dir / "contract_rules.json", contract_rules)
    _write_report(Path(args.report), records, group_summaries, route_rules, contract_rules)
    print(
        "mapped eval records to planner memory candidates "
        f"route={len(route_rules)}, contract={len(contract_rules)} in {output_dir}"
    )


def _build_candidates(
    groups: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    route_rules: List[Dict[str, Any]] = []
    contract_rules: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for group_key, records in sorted(groups.items()):
        summary = _group_summary(group_key, records)
        summaries.append(summary)
        stable = summary.get("stable_failure")
        if not stable:
            continue

        failure_type = stable["failure_type"]
        target = stable["feedback_target"]
        if failure_type in ROUTE_FAILURES and target in ROUTE_TARGETS:
            route_rule = _route_candidate(summary, records)
            if route_rule:
                route_rules.append(route_rule)
                summary.setdefault("candidate_rule_ids", []).append(route_rule["rule_id"])
        if failure_type in CONTRACT_FAILURES and target in CONTRACT_TARGETS:
            contract_rule = _contract_candidate(summary, records)
            if contract_rule:
                contract_rules.append(contract_rule)
                summary.setdefault("candidate_rule_ids", []).append(contract_rule["rule_id"])
    return route_rules, contract_rules, summaries


def _group_summary(group_key: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = [
        (_failure_type(record), _feedback_target(record))
        for record in records
        if _failure_type(record) != "none"
    ]
    counter = Counter(failures)
    top_pair, top_count = counter.most_common(1)[0] if counter else (("none", "none"), 0)
    ratio = top_count / len(records) if records else 0.0
    stable_failure = None
    if len(records) >= MIN_GROUP_RECORDS and top_count >= MIN_FAILURES and ratio >= MIN_FAILURE_RATIO:
        stable_failure = {
            "failure_type": top_pair[0],
            "feedback_target": top_pair[1],
            "count": top_count,
            "ratio": round(ratio, 4),
        }
    return {
        "group_id": _short_hash(group_key),
        "group_key": group_key,
        "task_type": _common(_task_type(record) for record in records),
        "expected_workflow": _common(_expected_workflow(record) for record in records),
        "semantic_features": _semantic_features(records[0]),
        "record_count": len(records),
        "failed_count": len(failures),
        "top_failure_type": top_pair[0],
        "top_feedback_target": top_pair[1],
        "top_failure_count": top_count,
        "top_failure_ratio": round(ratio, 4),
        "stable_failure": stable_failure,
        "case_ids": [str(record.get("case_id")) for record in records],
        "candidate_rule_ids": [],
    }


def _route_candidate(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    task_type = summary.get("task_type")
    workflow = summary.get("expected_workflow")
    if not task_type or not workflow:
        return None
    applies_when = _applies_when(task_type, summary.get("semantic_features", {}))
    return {
        "rule_id": f"candidate_route_{summary['group_id']}",
        "status": "candidate",
        "applies_when": applies_when,
        "routing_hint": {"prefer_workflow": workflow, "score_boost": 0.08},
        "reason": _candidate_reason(summary),
        "evidence": _evidence(summary, records),
    }


def _contract_candidate(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    task_type = summary.get("task_type")
    workflow = summary.get("expected_workflow")
    if not task_type or not workflow:
        return None
    patch: Dict[str, Any] = {}
    trace_events = _expected_trace_events(records)
    if trace_events:
        patch["add_required_trace_events"] = trace_events
    forbidden_tools = _expected_forbidden_tools(records)
    if not forbidden_tools and task_type == "document_writing":
        forbidden_tools = ["doc_search_tool", "evidence_check_tool"]
    if forbidden_tools:
        patch["add_forbidden_tools"] = forbidden_tools
    if not patch:
        return None

    applies_when = _applies_when(task_type, summary.get("semantic_features", {}))
    applies_when["selected_workflow"] = workflow
    return {
        "rule_id": f"candidate_contract_{summary['group_id']}",
        "status": "candidate",
        "applies_when": applies_when,
        "patch": patch,
        "reason": _candidate_reason(summary),
        "evidence": _evidence(summary, records),
    }


def _applies_when(task_type: str, features: Dict[str, Any]) -> Dict[str, Any]:
    applies: Dict[str, Any] = {"primary_task_type": task_type}
    if features:
        applies["semantic_features"] = _stable_features(features)
    return applies


def _expected_trace_events(records: List[Dict[str, Any]]) -> List[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for node in _as_list(_nested(record, "case.expected_plan.contract_hints.required_trace_events")):
            counter[str(node)] += 1
    threshold = max(1, len(records) // 2)
    return sorted(node for node, count in counter.items() if node and count >= threshold)


def _expected_forbidden_tools(records: List[Dict[str, Any]]) -> List[str]:
    counter: Counter[str] = Counter()
    for record in records:
        by_role = _nested(record, "case.expected_plan.forbidden_tools_by_step_role")
        if isinstance(by_role, dict):
            for tools in by_role.values():
                for tool in _as_list(tools):
                    counter[str(tool)] += 1
        for step in _as_list(_nested(record, "case.expected_plan.execution_plan")):
            if not isinstance(step, dict):
                continue
            for tool in _as_list(step.get("forbidden_tools")):
                counter[str(tool)] += 1
    threshold = max(1, len(records) // 2)
    return sorted(tool for tool, count in counter.items() if tool and count >= threshold)


def _candidate_reason(summary: Dict[str, Any]) -> str:
    stable = summary["stable_failure"]
    return (
        f"Stable eval group {summary['group_id']} had "
        f"{stable['failure_type']} targeting {stable['feedback_target']} "
        f"in {stable['count']}/{summary['record_count']} records."
    )


def _evidence(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stable = summary["stable_failure"]
    return {
        "group_id": summary["group_id"],
        "failure_type": stable["failure_type"],
        "feedback_target": stable["feedback_target"],
        "failure_count": stable["count"],
        "record_count": summary["record_count"],
        "failure_ratio": stable["ratio"],
        "case_ids": [str(record.get("case_id")) for record in records],
        "semantic_features": summary.get("semantic_features", {}),
    }


def _group_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)
    return dict(groups)


def _group_key(record: Dict[str, Any]) -> str:
    features = _semantic_features(record)
    return json.dumps(
        {
            "task_type": _task_type(record),
            "expected_workflow": _expected_workflow(record),
            "semantic_features": _stable_features(features),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _stable_features(features: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "deliverable_type",
        "requires_external_evidence",
        "requires_citations",
        "requires_media_asset",
        "evidence_need",
        "complexity",
    ]
    return {key: features[key] for key in keys if key in features}


def _failure_type(record: Dict[str, Any]) -> str:
    value = record.get("failure_type")
    if isinstance(value, str) and value:
        return value
    failed = set(record.get("failed_metrics") or [])
    if not failed:
        return "none"
    for metric_name, failure_type, _target in FAILURE_FALLBACK:
        if metric_name in failed:
            return failure_type
    return "unknown_failure"


def _feedback_target(record: Dict[str, Any]) -> str:
    value = record.get("feedback_target")
    if isinstance(value, str) and value:
        return value
    failed = set(record.get("failed_metrics") or [])
    for metric_name, _failure_type, target in FAILURE_FALLBACK:
        if metric_name in failed:
            return target
    return "none"


def _hard_pass(record: Dict[str, Any]) -> bool:
    return _failure_type(record) == "none"


def _task_type(record: Dict[str, Any]) -> str:
    return str(record.get("task_type") or _nested(record, "case.task_type") or "")


def _expected_workflow(record: Dict[str, Any]) -> str:
    return str(_nested(record, "case.expected_workflow") or "")


def _semantic_features(record: Dict[str, Any]) -> Dict[str, Any]:
    features = _nested(record, "case.expected_plan.semantic_features")
    if isinstance(features, dict) and features:
        return _stable_features(features)
    features = _nested(record, "validated_task_plan.semantic_features")
    return _stable_features(features) if isinstance(features, dict) else {}


def _common(values: Iterable[Any]) -> Optional[str]:
    clean = [str(value) for value in values if value not in (None, "")]
    if not clean:
        return None
    value, _count = Counter(clean).most_common(1)[0]
    return value


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
    summaries: List[Dict[str, Any]],
    route_rules: List[Dict[str, Any]],
    contract_rules: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [record for record in records if not _hard_pass(record)]
    lines = [
        "# Planner Memory Candidate Report",
        "",
        f"- eval_records: {len(records)}",
        f"- failed_records: {len(failed)}",
        f"- route_candidates: {len(route_rules)}",
        f"- contract_candidates: {len(contract_rules)}",
        f"- promotion: manual review required; generated rules are status=candidate",
        "",
        "| group | task_type | expected_workflow | records | top_failure | target | candidates |",
        "|---|---|---|---:|---|---|---|",
    ]
    for summary in summaries:
        candidates = ", ".join(summary.get("candidate_rule_ids", [])) or "-"
        lines.append(
            "| {group} | {task_type} | {workflow} | {failed}/{total} | {failure} | {target} | {candidates} |".format(
                group=summary["group_id"],
                task_type=summary.get("task_type") or "-",
                workflow=summary.get("expected_workflow") or "-",
                failed=summary["top_failure_count"],
                total=summary["record_count"],
                failure=summary["top_failure_type"],
                target=summary["top_feedback_target"],
                candidates=candidates,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nested(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    main()
