from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib import error as url_error
from urllib import request as url_request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.metrics.semgateway_metrics import evaluate_record, metric_results_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemGateway Phase 4 offline eval.")
    parser.add_argument("--cases", default="data/task_eval_cases.jsonl")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dev-key")
    parser.add_argument("--output", default="outputs/eval/task_eval_results.jsonl")
    parser.add_argument("--report", default="outputs/reports/deepeval_report.md")
    parser.add_argument("--tool-audit", default="logs/tool_audit.jsonl")
    parser.add_argument("--request-delay-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--no-fail", action="store_true", help="Do not exit non-zero on failed metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _read_jsonl(Path(args.cases))
    if not cases:
        raise SystemExit(f"no eval cases found: {args.cases}")

    records = []
    headers = {"x-api-key": args.api_key, "Content-Type": "application/json"}
    base_url = args.gateway_url.rstrip("/")
    for case in cases:
        record = _run_case(base_url, headers, case, Path(args.tool_audit), args.timeout_s)
        _attach_metrics(record)
        records.append(record)
        time.sleep(args.request_delay_s)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_report(Path(args.report), records)
    failed = [record for record in records if record.get("failed_metrics")]
    print(f"wrote eval records={len(records)} failed={len(failed)} to {output_path}")
    if failed and not args.no_fail:
        raise SystemExit(1)


def _run_case(
    base_url: str,
    headers: Dict[str, str],
    case: Dict[str, Any],
    tool_audit_path: Path,
    timeout_s: float,
) -> Dict[str, Any]:
    payload = case.get("payload") if isinstance(case.get("payload"), dict) else _payload_from_case(case)
    response_body: Dict[str, Any]
    http_status = None
    try:
        http_status, response_body = _post_json(f"{base_url}/v1/invoke", headers, payload, timeout_s)
    except Exception as exc:
        response_body = {"status": "failed", "error": str(exc)}

    trace_events = _load_trace_events(response_body)
    trace_id = response_body.get("trace_id", "")
    return {
        "schema": "semgateway_task_eval_result_v1",
        "case": _safe_case(case, payload),
        "case_id": case.get("case_id"),
        "task_type": payload.get("task_type") or case.get("task_type"),
        "selected_workflow": response_body.get("selected_workflow"),
        "status": response_body.get("status"),
        "verification_status": _nested(response_body, "verification.status"),
        "response_http_status": http_status,
        "response": response_body,
        "trace_events": trace_events,
        "trace_nodes": [event.get("node") for event in trace_events if isinstance(event, dict)],
        "tool_audit": _read_tool_audit(tool_audit_path, trace_id),
    }


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> tuple[int, Dict[str, Any]]:
    request = url_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with url_request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except url_error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body) if body else {}
        except json.JSONDecodeError:
            return exc.code, {"status": "failed", "error": body}


def _attach_metrics(record: Dict[str, Any]) -> None:
    results = evaluate_record(record)
    record["metric_results"] = metric_results_json(results)
    record["metric_scores"] = {name: result.score for name, result in results.items()}
    record["failed_metrics"] = [name for name, result in results.items() if not result.passed]
    record["deepeval_custom_metric_checked"] = _deepeval_smoke(record)


def _deepeval_smoke(record: Dict[str, Any]) -> bool:
    try:
        from deepeval.test_case import LLMTestCase
        from eval.metrics import (
            CitationSourceMetric,
            ContractComplianceMetric,
            RouteCorrectnessMetric,
            SchemaComplianceMetric,
            ToolComplianceMetric,
            TraceEventCoverageMetric,
            VerificationPassMetric,
        )
    except Exception:
        return False
    test_case = LLMTestCase(
        input=str(record.get("case_id", "")),
        actual_output=json.dumps(record, ensure_ascii=False),
        expected_output=str(_nested(record, "case.expected_workflow") or ""),
    )
    try:
        setattr(test_case, "additional_metadata", {"record": record})
    except Exception:
        pass
    for metric_cls in (
        RouteCorrectnessMetric,
        ContractComplianceMetric,
        TraceEventCoverageMetric,
        ToolComplianceMetric,
        VerificationPassMetric,
        CitationSourceMetric,
        SchemaComplianceMetric,
    ):
        metric_cls().measure(test_case)
    return True


def _payload_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": case.get("user_id", "u001"),
        "tenant_id": case.get("tenant_id", "tenant_demo"),
        "task_type": case.get("task_type", "knowledge_qa"),
        "input": case.get("input", ""),
        "priority": case.get("priority", "normal"),
        "metadata": case.get("metadata", {}),
    }


def _safe_case(case: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    safe = {key: value for key, value in case.items() if key not in {"input", "payload"}}
    safe["task_type"] = payload.get("task_type") or case.get("task_type")
    safe["input_sha256"] = hashlib.sha256(str(payload.get("input", "")).encode("utf-8")).hexdigest()
    safe["input_length"] = len(str(payload.get("input", "")))
    return safe


def _load_trace_events(response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace_path = _nested(response_body, "metadata.trace_path")
    if not trace_path:
        return []
    path = Path(str(trace_path))
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = payload.get("events")
    return events if isinstance(events, list) else []


def _read_tool_audit(path: Path, trace_id: str) -> List[Dict[str, Any]]:
    if not trace_id or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines()[-5000:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("trace_id") == trace_id:
            records.append(record)
    return records


def _write_report(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SemGateway DeepEval Report", ""]
    metric_names = sorted({name for record in records for name in record.get("metric_scores", {})})
    for metric in metric_names:
        scores = [float(record["metric_scores"][metric]) for record in records if metric in record.get("metric_scores", {})]
        passed = sum(1 for record in records if metric not in record.get("failed_metrics", []))
        average = round(sum(scores) / len(scores), 4) if scores else 0.0
        lines.append(f"- {metric}: avg={average}, passed={passed}/{len(records)}")
    lines.append("")
    lines.append("| case_id | workflow | status | failed_metrics |")
    lines.append("|---|---|---|---|")
    for record in records:
        failed = ", ".join(record.get("failed_metrics", [])) or "-"
        lines.append(
            f"| {record.get('case_id')} | {record.get('selected_workflow')} | {record.get('status')} | {failed} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _nested(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


if __name__ == "__main__":
    main()
