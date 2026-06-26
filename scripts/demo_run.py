from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import httpx


ROOT_DIR = Path(__file__).resolve().parents[1]
GATEWAY_LOG_PATH = ROOT_DIR / "logs" / "gateway.jsonl"
TOOL_AUDIT_LOG_PATH = ROOT_DIR / "logs" / "tool_audit.jsonl"

FORMAL_TOOLS = {
    "doc_search_tool",
    "business_query_tool",
    "service_status_tool",
    "inspection_tool",
    "evidence_check_tool",
}
PLACEHOLDER_TOOLS = {"image_generation_tool", "video_generation_tool"}

WORKFLOW_CASES: Dict[str, Dict[str, Any]] = {
    "enterprise_tech_qa": {
        "expected_workflow": "tech_qa_workflow",
        "expected_tools": {"doc_search_tool", "evidence_check_tool"},
        "expected_nodes": {
            "GatewayAuth",
            "TaskProfileBuild",
            "AgenticRouteDecision",
            "RetrieveEvidence",
            "EvidenceCheck",
            "GenerateAnswer",
            "CitationCheck",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "knowledge_qa",
            "input": "According to project docs, explain why SemRoute-Gateway needs AgenticRouter and citations.",
            "priority": "normal",
            "metadata": {
                "knowledge_base": "project_docs",
                "evidence_required": True,
                "latency_slo_ms": 5000,
                "cost_budget": "normal",
            },
        },
    },
    "doc_review": {
        "expected_workflow": "doc_review_workflow",
        "expected_tools": {"doc_search_tool", "inspection_tool", "evidence_check_tool"},
        "expected_nodes": {
            "GatewayAuth",
            "TaskProfileBuild",
            "AgenticRouteDecision",
            "RetrievePolicy",
            "RunInspection",
            "EvidenceCheck",
            "GenerateReview",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "doc_review",
            "input": "Please review this document draft and point out missing fields and risk items.",
            "priority": "normal",
            "metadata": {
                "knowledge_base": "project_docs",
                "evidence_required": True,
                "cost_budget": "normal",
            },
        },
    },
    "incident_status_analysis": {
        "expected_workflow": "incident_analysis_workflow",
        "expected_tools": {
            "service_status_tool",
            "business_query_tool",
            "doc_search_tool",
            "evidence_check_tool",
        },
        "expected_nodes": {
            "GatewayAuth",
            "TaskProfileBuild",
            "AgenticRouteDecision",
            "ServiceStatus",
            "BusinessQuery",
            "RetrieveRunbook",
            "EvidenceCheck",
            "FallbackCheck",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "knowledge_qa",
            "input": "Ticket INC-001 reports gateway service status errors. Query status and generate a fix plan.",
            "priority": "high",
            "metadata": {
                "knowledge_base": "project_docs",
                "latency_slo_ms": 5000,
                "cost_budget": "normal",
            },
        },
    },
    "media_generation": {
        "expected_workflow": "media_generation_workflow",
        "expected_tools": {"video_generation_tool"},
        "expected_nodes": {
            "GatewayAuth",
            "TaskProfileBuild",
            "AgenticRouteDecision",
            "ParseMediaRequest",
            "MediaGenerationStub",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "chat",
            "input": "Generate a short video scene for introducing the SemRoute-Gateway architecture.",
            "priority": "normal",
            "metadata": {
                "media_type": "video",
                "latency_slo_ms": 8000,
                "cost_budget": "normal",
            },
        },
    },
}

FALLBACK_CASE: Dict[str, Any] = {
    "expected_workflow": "incident_analysis_workflow",
    "payload": {
        "user_id": "u001",
        "tenant_id": "tenant_demo",
        "task_type": "incident_analysis",
        "input": "Ticket INC-001 reports gateway service status errors. Query status and generate a response plan.",
        "priority": "normal",
        "metadata": {
            "service_name": "gateway",
            "simulate_service_status_failure": "timeout",
        },
    },
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemRoute v0.7 demo and Phase 2 acceptance checks.")
    parser.add_argument("--all", action="store_true", help="Run the full demo and acceptance checks.")
    parser.add_argument("--case", choices=sorted(WORKFLOW_CASES), help="Run one workflow happy-path case only.")
    parser.add_argument("--gateway-url", default="http://localhost:8000/v1/invoke")
    parser.add_argument("--tool-url", default="http://localhost:8030")
    parser.add_argument("--api-key", default="dev-key")
    parser.add_argument("--request-delay-s", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headers = {"x-api-key": args.api_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=15.0) as client:
        if args.case:
            results = [run_workflow_case(client, args.gateway_url, headers, args.case, WORKFLOW_CASES[args.case], args.verbose)]
        else:
            results = run_full_demo(client, args, headers)

    print_results(results)
    if any(not result.passed for result in results):
        sys.exit(1)


def run_full_demo(client: httpx.Client, args: argparse.Namespace, headers: Dict[str, str]) -> List[CheckResult]:
    results: List[CheckResult] = []
    print("== SemRoute v0.7 demo / Phase 2 acceptance ==")

    results.extend(run_health_checks(client))
    results.append(run_tool_registry_check(client, args.tool_url))

    for case_name, case in WORKFLOW_CASES.items():
        results.append(run_workflow_case(client, args.gateway_url, headers, case_name, case, args.verbose))
        time.sleep(args.request_delay_s)

    results.append(run_tool_schema_error_check(client, args.tool_url, args.verbose))
    results.append(run_tool_permission_denied_check(client, args.tool_url, args.verbose))
    results.append(run_media_placeholder_tool_check(client, args.tool_url, args.verbose))
    results.append(run_fallback_check(client, args.gateway_url, headers, args.verbose))
    return results


def run_health_checks(client: httpx.Client) -> List[CheckResult]:
    endpoints = {
        "health:gateway": "http://localhost:8000/health",
        "health:agent_orchestrator": "http://localhost:8010/health",
        "health:rag_service": "http://localhost:8020/health",
        "health:tool_service": "http://localhost:8030/health",
        "health:model_backend": "http://localhost:8040/health",
    }
    results = []
    for name, url in endpoints.items():
        try:
            response = client.get(url)
            body = response.json()
        except Exception as exc:
            results.append(CheckResult(name, False, f"request failed: {exc}"))
            continue
        passed = response.status_code == 200 and body.get("status") == "ok"
        detail = f"http={response.status_code}, service={body.get('service')}"
        results.append(CheckResult(name, passed, detail))
    return results


def run_tool_registry_check(client: httpx.Client, tool_url: str) -> CheckResult:
    name = "tool registry"
    try:
        response = client.get(f"{tool_url}/tools")
        body = response.json()
    except Exception as exc:
        return CheckResult(name, False, f"request failed: {exc}")

    definitions = {item.get("tool_name"): item for item in body.get("tools", [])}
    errors = []
    missing_formal = sorted(FORMAL_TOOLS - set(definitions))
    missing_placeholder = sorted(PLACEHOLDER_TOOLS - set(definitions))
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if missing_formal:
        errors.append(f"missing formal tools={missing_formal}")
    if missing_placeholder:
        errors.append(f"missing placeholder tools={missing_placeholder}")

    for tool_name in sorted(FORMAL_TOOLS & set(definitions)):
        if definitions[tool_name].get("implementation_status") != "active":
            errors.append(f"{tool_name} is not active")
    for tool_name in sorted(PLACEHOLDER_TOOLS & set(definitions)):
        if definitions[tool_name].get("implementation_status") != "placeholder":
            errors.append(f"{tool_name} is not placeholder")

    detail = "formal=5, placeholders=2"
    return CheckResult(name, not errors, "; ".join(errors) if errors else detail)


def run_workflow_case(
    client: httpx.Client,
    gateway_url: str,
    headers: Dict[str, str],
    case_name: str,
    case: Dict[str, Any],
    verbose: bool,
) -> CheckResult:
    name = f"workflow:{case_name}"
    response = post_gateway_with_rate_retry(client, gateway_url, headers, case["payload"])
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult(name, False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    errors = []
    metadata = body.get("metadata", {})
    tools = set(metadata.get("tools", []))
    trace_id = body.get("trace_id")
    trace_path = metadata.get("trace_path")
    trace, trace_error = load_trace(trace_path)

    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("status") != "success":
        errors.append(f"status={body.get('status')}")
    if body.get("selected_workflow") != case["expected_workflow"]:
        errors.append(f"selected_workflow={body.get('selected_workflow')}")
    missing_tools = sorted(set(case["expected_tools"]) - tools)
    if missing_tools:
        errors.append(f"missing tools={missing_tools}")
    if trace_error:
        errors.append(trace_error)
    else:
        if "log_budget" not in trace:
            errors.append("missing trace log_budget")
        missing_nodes = sorted(set(case["expected_nodes"]) - trace_nodes(trace))
        if missing_nodes:
            errors.append(f"missing trace nodes={missing_nodes}")
        if trace.get("trace_id") != trace_id:
            errors.append("trace_id mismatch")
        if not has_resource_runtime_fields(trace):
            errors.append("missing resource runtime metric fields")
        if not has_trace_event_metadata_key(trace, node="InvokeSelectedWorkflow", metadata_key="runtime_metrics_finish"):
            errors.append("missing InvokeSelectedWorkflow runtime_metrics_finish")

    detail = f"workflow={body.get('selected_workflow')}, tools={sorted(tools)}, trace={trace_path}"
    return CheckResult(name, not errors, "; ".join(errors) if errors else detail)


def run_tool_schema_error_check(client: httpx.Client, tool_url: str, verbose: bool) -> CheckResult:
    trace_id = f"demo-schema-error-{uuid4()}"
    payload = {
        "tool_name": "inspection_tool",
        "arguments": {},
        "user_id": "u001",
        "consumer_id": "demo_consumer",
        "trace_id": trace_id,
        "request_id": trace_id,
        "permissions": ["tool:inspection:use"],
        "allowed_tools": ["inspection_tool"],
    }
    return run_tool_status_check(
        client=client,
        tool_url=tool_url,
        name="tool schema_error",
        payload=payload,
        expected_status="schema_error",
        trace_id=trace_id,
        verbose=verbose,
    )


def run_tool_permission_denied_check(client: httpx.Client, tool_url: str, verbose: bool) -> CheckResult:
    trace_id = f"demo-permission-denied-{uuid4()}"
    payload = {
        "tool_name": "inspection_tool",
        "arguments": {
            "content": "background goal solution acceptance",
            "inspection_type": "document_structure",
        },
        "user_id": "u001",
        "consumer_id": "demo_consumer",
        "trace_id": trace_id,
        "request_id": trace_id,
        "permissions": [],
        "allowed_tools": ["inspection_tool"],
    }
    return run_tool_status_check(
        client=client,
        tool_url=tool_url,
        name="tool permission_denied",
        payload=payload,
        expected_status="permission_denied",
        trace_id=trace_id,
        verbose=verbose,
    )


def run_media_placeholder_tool_check(client: httpx.Client, tool_url: str, verbose: bool) -> CheckResult:
    name = "tool media placeholder"
    trace_id = f"demo-media-placeholder-{uuid4()}"
    payload = {
        "tool_name": "video_generation_tool",
        "arguments": {
            "prompt": "Generate a 10 second architecture explainer video.",
            "media_type": "video",
        },
        "user_id": "u001",
        "consumer_id": "demo_consumer",
        "trace_id": trace_id,
        "request_id": trace_id,
        "permissions": ["tool:video_generation:use"],
        "allowed_tools": ["video_generation_tool"],
    }

    try:
        response = client.post(f"{tool_url}/invoke", json=payload)
    except Exception as exc:
        return CheckResult(name, False, f"request failed: {exc}")
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult(name, False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("status") != "success":
        errors.append(f"status={body.get('status')}, expected=success")
    implementation_status = body.get("result", {}).get("implementation_status")
    if implementation_status != "placeholder":
        errors.append(f"implementation_status={implementation_status}")
    if not wait_for_jsonl_record(TOOL_AUDIT_LOG_PATH, lambda record: record.get("trace_id") == trace_id):
        errors.append("tool audit record not found")

    detail = f"tool={body.get('tool_name')}, status={body.get('status')}, implementation_status={implementation_status}"
    return CheckResult(name, not errors, "; ".join(errors) if errors else detail)


def run_tool_status_check(
    *,
    client: httpx.Client,
    tool_url: str,
    name: str,
    payload: Dict[str, Any],
    expected_status: str,
    trace_id: str,
    verbose: bool,
) -> CheckResult:
    try:
        response = client.post(f"{tool_url}/invoke", json=payload)
        body, parse_error = parse_json_response(response)
    except Exception as exc:
        return CheckResult(name, False, f"request failed: {exc}")
    if parse_error:
        return CheckResult(name, False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("status") != expected_status:
        errors.append(f"status={body.get('status')}, expected={expected_status}")
    if not wait_for_jsonl_record(TOOL_AUDIT_LOG_PATH, lambda record: record.get("trace_id") == trace_id):
        errors.append("tool audit record not found")

    detail = f"tool={body.get('tool_name')}, status={body.get('status')}, trace={trace_id}"
    return CheckResult(name, not errors, "; ".join(errors) if errors else detail)


def run_fallback_check(
    client: httpx.Client,
    gateway_url: str,
    headers: Dict[str, str],
    verbose: bool,
) -> CheckResult:
    name = "incident fallback"
    response = post_gateway_with_rate_retry(client, gateway_url, headers, FALLBACK_CASE["payload"])
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult(name, False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    metadata = body.get("metadata", {})
    trace_id = body.get("trace_id")
    trace_path = metadata.get("trace_path")
    trace, trace_error = load_trace(trace_path)

    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("selected_workflow") != FALLBACK_CASE["expected_workflow"]:
        errors.append(f"selected_workflow={body.get('selected_workflow')}")
    if metadata.get("fallback_used") is not True:
        errors.append(f"response fallback_used={metadata.get('fallback_used')}")
    if trace_error:
        errors.append(trace_error)
    else:
        if not has_trace_event(trace, node="ServiceStatus", status="failed", error_type="timeout"):
            errors.append("missing failed ServiceStatus timeout trace event")
        if not has_trace_event(trace, node="FallbackCheck", event_type="fallback", status="success"):
            errors.append("missing FallbackCheck fallback trace event")
        if not has_trace_event(trace, node="TraceAggregation", metadata_key="fallback_used", metadata_value=True):
            errors.append("missing TraceAggregation fallback_used=true")

    if not wait_for_jsonl_record(
        GATEWAY_LOG_PATH,
        lambda record: record.get("trace_id") == trace_id and record.get("fallback_used") is True,
    ):
        errors.append("gateway summary fallback_used=true not found")
    if not wait_for_jsonl_record(
        TOOL_AUDIT_LOG_PATH,
        lambda record: record.get("trace_id") == trace_id
        and record.get("tool_name") == "service_status_tool"
        and record.get("status") == "timeout",
    ):
        errors.append("service_status_tool timeout audit not found")

    detail = f"trace={trace_path}, fallback_used={metadata.get('fallback_used')}"
    return CheckResult(name, not errors, "; ".join(errors) if errors else detail)


def post_gateway_with_rate_retry(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int = 3,
) -> httpx.Response:
    response: Optional[httpx.Response] = None
    for _ in range(max_retries + 1):
        response = client.post(url, headers=headers, json=payload)
        if response.status_code != 429:
            return response
        retry_after = response.headers.get("Retry-After")
        delay_s = 1.0
        if retry_after:
            try:
                delay_s = max(delay_s, float(retry_after))
            except ValueError:
                delay_s = 1.0
        time.sleep(delay_s)
    return response


def parse_json_response(response: httpx.Response) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        body = response.json()
    except ValueError:
        return {}, f"http={response.status_code}, non-json response={response.text[:200]}"
    if not isinstance(body, dict):
        return {}, "json response is not an object"
    return body, None


def load_trace(trace_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not trace_path:
        return {}, "missing trace_path"
    path = Path(trace_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return {}, f"trace file not found: {trace_path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, f"trace file parse failed: {exc}"


def trace_nodes(trace: Dict[str, Any]) -> set:
    return {event.get("node") for event in trace.get("events", [])}


def has_trace_event(
    trace: Dict[str, Any],
    *,
    node: str,
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    metadata_key: Optional[str] = None,
    metadata_value: Any = None,
) -> bool:
    for event in trace.get("events", []):
        if event.get("node") != node:
            continue
        if event_type is not None and event.get("event_type") != event_type:
            continue
        if status is not None and event.get("status") != status:
            continue
        if error_type is not None and event.get("error_type") != error_type:
            continue
        if metadata_key is not None and event.get("metadata", {}).get(metadata_key) != metadata_value:
            continue
        return True
    return False


def has_trace_event_metadata_key(trace: Dict[str, Any], *, node: str, metadata_key: str) -> bool:
    for event in trace.get("events", []):
        if event.get("node") == node and metadata_key in event.get("metadata", {}):
            return True
    return False


def has_resource_runtime_fields(trace: Dict[str, Any]) -> bool:
    required = {
        "ongoing_count",
        "recent_sample_count",
        "recent_avg_latency_ms",
        "recent_p95_latency_ms",
        "recent_error_rate",
    }
    for event in trace.get("events", []):
        if event.get("node") != "AgenticRouteDecision":
            continue
        resource_decision = event.get("metadata", {}).get("resource_decision", {})
        candidates = resource_decision.get("candidate_targets", [])
        if not candidates:
            continue
        return required.issubset(set(candidates[0]))
    return False


def wait_for_jsonl_record(
    path: Path,
    predicate: Callable[[Dict[str, Any]], bool],
    timeout_s: float = 3.0,
    interval_s: float = 0.2,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() <= deadline:
        if jsonl_contains(path, predicate):
            return True
        time.sleep(interval_s)
    return False


def jsonl_contains(path: Path, predicate: Callable[[Dict[str, Any]], bool], recent_limit: int = 1000) -> bool:
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text().splitlines()
    for line in reversed(lines[-recent_limit:]):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if predicate(record):
            return True
    return False


def print_results(results: Iterable[CheckResult]) -> None:
    result_list = list(results)
    for result in result_list:
        label = "PASS" if result.passed else "FAIL"
        print(f"[{label}] {result.name}: {result.detail}")

    passed = sum(1 for result in result_list if result.passed)
    failed = len(result_list) - passed
    print(f"Summary: passed={passed}, failed={failed}, total={len(result_list)}")


if __name__ == "__main__":
    main()
