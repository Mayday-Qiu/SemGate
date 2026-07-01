from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx

from app.schemas import (
    AcceptanceCriterion,
    AgentWorkflowResponse,
    AgenticGatewayRequest,
    AgenticMetrics,
    AuthenticatedConsumer,
    TaskContract,
    TaskProfile,
)
from app.agentic_router import AgenticRouter
from app.memory_planner import MemoryPlanner
from app.verification import VerificationGate
from app.workflow_profiles import load_workflow_profiles
from app.main import _finalize_verified_response
from app.task_contract import TaskContractBuilder


V10_TOOLS = {
    "doc_search_tool",
    "evidence_check_tool",
    "image_generation_tool",
    "video_generation_tool",
    "inspection_tool",
}

WORKFLOW_CASES: Dict[str, Dict[str, Any]] = {
    "knowledge_qa": {
        "expected_workflow": "knowledge_qa_workflow",
        "expected_tools": {"doc_search_tool", "evidence_check_tool"},
        "expected_nodes": {
            "GatewayAuth",
            "TaskProfileBuild",
            "AgenticRouteDecision",
            "TaskContractBuild",
            "ParseQuestion",
            "PlanRetrieval",
            "RetrieveEvidence",
            "EvidenceCheck",
            "GenerateAnswer",
            "CitationCheck",
            "VerificationGate",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "knowledge_qa",
            "input": "According to project docs, explain why SemGateway needs routing, evidence, and citations.",
            "priority": "normal",
            "metadata": {"knowledge_base": "project_docs", "evidence_required": True},
        },
    },
    "large_knowledge_qa": {
        "expected_workflow": "large_knowledge_qa_workflow",
        "expected_tools": {"doc_search_tool", "evidence_check_tool"},
        "expected_nodes": {
            "TaskContractBuild",
            "ParseDeepQuestion",
            "BuildResearchBrief",
            "SplitSubQuestions",
            "ParallelRetrieveEvidence",
            "EvidenceAggregate",
            "SynthesizeAnswer",
            "CitationCheck",
            "VerificationGate",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "knowledge_qa",
            "input": "Give a deep systematic multi-angle analysis of SemGateway evidence and routing design.",
            "priority": "normal",
            "metadata": {"knowledge_base": "project_docs", "evidence_required": True},
        },
    },
    "coding": {
        "expected_workflow": "coding_workflow",
        "expected_tools": set(),
        "expected_nodes": {
            "TaskContractBuild",
            "ParseCodingTask",
            "BuildCodeContext",
            "InvokeCodingModel",
            "StructurePatchOrAnswer",
            "CodingOutputCheck",
            "VerificationGate",
            "TraceAggregation",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "coding",
            "input": "Implement a small Python function and list the unit tests needed.",
            "priority": "normal",
            "metadata": {},
        },
    },
    "media_generation": {
        "expected_workflow": "media_generation_workflow",
        "expected_tools": {"video_generation_tool"},
        "expected_nodes": {
            "TaskContractBuild",
            "ParseMediaRequest",
            "PromptSafetyCheck",
            "BuildGenerationParams",
            "InvokeMediaBackend",
            "AssetMetadataBuild",
            "MediaOutputCheck",
            "VerificationGate",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "media_generation",
            "input": "Generate a short video scene for introducing the SemGateway architecture.",
            "priority": "normal",
            "metadata": {"media_type": "video"},
        },
    },
    "document_writing": {
        "expected_workflow": "document_writing_workflow",
        "expected_tools": {"inspection_tool"},
        "expected_nodes": {
            "TaskContractBuild",
            "ParseWritingTask",
            "BuildDocumentPlan",
            "NeedKnowledge",
            "A2AKnowledgeRequest",
            "NeedMedia",
            "ComposeDocument",
            "InspectionToolCheck",
            "DocumentVerification",
            "VerificationGate",
        },
        "payload": {
            "user_id": "u001",
            "tenant_id": "tenant_demo",
            "task_type": "document_writing",
            "input": "Write a technical document outline for the SemGateway v1.0 design.",
            "priority": "normal",
            "metadata": {},
        },
    },
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemGateway v1.0 Phase 1 acceptance checks.")
    parser.add_argument("--all", action="store_true", help="Run all checks.")
    parser.add_argument("--case", choices=sorted(WORKFLOW_CASES), help="Run one workflow case.")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--tool-url", default="http://localhost:8030")
    parser.add_argument("--api-key", default="dev-key")
    parser.add_argument("--request-delay-s", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headers = {"x-api-key": args.api_key, "Content-Type": "application/json"}
    base_url = gateway_base_url(args.gateway_url)

    with httpx.Client(timeout=20.0) as client:
        if args.case:
            case = WORKFLOW_CASES[args.case]
            results = [
                run_preview_case(client, base_url, headers, args.case, case, args.verbose),
                run_workflow_case(client, base_url, headers, args.case, case, args.verbose),
            ]
        else:
            results = run_full_demo(client, args, headers, base_url)

    print_results(results)
    if any(not result.passed for result in results):
        sys.exit(1)


def run_full_demo(
    client: httpx.Client,
    args: argparse.Namespace,
    headers: Dict[str, str],
    base_url: str,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    print("== SemGateway v1.0 Phase 1 acceptance ==")
    results.extend(run_health_checks(client, base_url))
    results.append(run_tool_registry_check(client, args.tool_url))
    results.append(run_preview_blocked_check(client, base_url, headers, args.verbose))
    results.append(run_local_verification_failure_check())
    results.append(run_raw_answer_preserved_check())
    results.append(run_tool_audit_verification_check())
    results.append(run_planner_memory_route_check())
    results.append(run_planner_memory_contract_patch_check())
    results.append(run_planner_memory_invalid_field_check())

    for case_name, case in WORKFLOW_CASES.items():
        results.append(run_preview_case(client, base_url, headers, case_name, case, args.verbose))
        results.append(run_workflow_case(client, base_url, headers, case_name, case, args.verbose))
        time.sleep(args.request_delay_s)

    results.append(run_tool_schema_error_check(client, args.tool_url, args.verbose))
    results.append(run_tool_permission_denied_check(client, args.tool_url, args.verbose))
    return results


def run_health_checks(client: httpx.Client, base_url: str) -> List[CheckResult]:
    endpoints = {
        "health:gateway": f"{base_url}/health",
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
        results.append(CheckResult(name, passed, f"http={response.status_code}, service={body.get('service')}"))
    return results


def run_tool_registry_check(client: httpx.Client, tool_url: str) -> CheckResult:
    try:
        response = client.get(f"{tool_url}/tools")
        body = response.json()
    except Exception as exc:
        return CheckResult("tool registry", False, f"request failed: {exc}")
    definitions = {item.get("tool_name"): item for item in body.get("tools", [])}
    missing = sorted(V10_TOOLS - set(definitions))
    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if missing:
        errors.append(f"missing tools={missing}")
    for tool_name in sorted({"image_generation_tool", "video_generation_tool"} & set(definitions)):
        if definitions[tool_name].get("implementation_status") != "placeholder":
            errors.append(f"{tool_name} is not placeholder")
    return CheckResult("tool registry", not errors, "; ".join(errors) if errors else "v1.0 tools registered")


def run_preview_case(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    case_name: str,
    case: Dict[str, Any],
    verbose: bool,
) -> CheckResult:
    response = client.post(f"{base_url}/v1/preview", headers=headers, json=case["payload"])
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult(f"preview:{case_name}", False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("preview_status") != "ready":
        errors.append(f"preview_status={body.get('preview_status')}")
    if body.get("selected_workflow") != case["expected_workflow"]:
        errors.append(f"selected_workflow={body.get('selected_workflow')}")
    if not body.get("task_contract", {}).get("acceptance_criteria"):
        errors.append("missing acceptance criteria")
    return CheckResult(
        f"preview:{case_name}",
        not errors,
        "; ".join(errors) if errors else f"workflow={body.get('selected_workflow')}",
    )


def run_preview_blocked_check(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    verbose: bool,
) -> CheckResult:
    payload = dict(WORKFLOW_CASES["knowledge_qa"]["payload"])
    payload["metadata"] = {"permissions": ["kb:project_docs:read"]}
    response = client.post(f"{base_url}/v1/preview", headers=headers, json=payload)
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult("preview:blocked-permission", False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))
    missing = set(body.get("missing_permissions", []))
    passed = (
        response.status_code == 200
        and body.get("preview_status") == "blocked"
        and {"tool:doc_search:use", "tool:evidence_check:use"}.issubset(missing)
    )
    return CheckResult(
        "preview:blocked-permission",
        passed,
        f"preview_status={body.get('preview_status')}, missing={sorted(missing)}",
    )


def run_workflow_case(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    case_name: str,
    case: Dict[str, Any],
    verbose: bool,
) -> CheckResult:
    response = post_with_rate_retry(client, f"{base_url}/v1/invoke", headers, case["payload"])
    body, parse_error = parse_json_response(response)
    if parse_error:
        return CheckResult(f"workflow:{case_name}", False, parse_error)
    if verbose:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    metadata = body.get("metadata", {})
    tools = set(metadata.get("tools", []))
    trace, trace_error = load_trace(metadata.get("trace_path"))
    errors = []
    if response.status_code != 200:
        errors.append(f"http={response.status_code}")
    if body.get("status") != "success":
        errors.append(f"status={body.get('status')}")
    if body.get("verification", {}).get("status") != "passed":
        errors.append(f"verification={body.get('verification', {}).get('status')}")
    if body.get("selected_workflow") != case["expected_workflow"]:
        errors.append(f"selected_workflow={body.get('selected_workflow')}")
    missing_tools = sorted(set(case["expected_tools"]) - tools)
    if missing_tools:
        errors.append(f"missing tools={missing_tools}")
    if case_name == "document_writing" and {"doc_search_tool", "evidence_check_tool"} & tools:
        errors.append(f"document workflow used forbidden tools={sorted(tools)}")
    if not body.get("contract_id"):
        errors.append("missing contract_id")
    if trace_error:
        errors.append(trace_error)
    else:
        missing_nodes = sorted(set(case["expected_nodes"]) - trace_nodes(trace))
        if missing_nodes:
            errors.append(f"missing trace nodes={missing_nodes}")

    return CheckResult(
        f"workflow:{case_name}",
        not errors,
        "; ".join(errors) if errors else f"workflow={body.get('selected_workflow')}, tools={sorted(tools)}",
    )


def run_local_verification_failure_check() -> CheckResult:
    contract = TaskContract(
        contract_id="contract_acceptance_failure",
        task_type="knowledge_qa",
        selected_workflow="knowledge_qa_workflow",
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="citation:min1",
                type="citation_required",
                target="citations",
                params={"min_count": 1},
            )
        ],
    )
    response = AgentWorkflowResponse(
        request_id="acceptance",
        trace_id="acceptance",
        selected_workflow="knowledge_qa_workflow",
        answer="unsupported answer",
        status="success",
        metrics=AgenticMetrics(latency_ms=1.0),
    )
    result = VerificationGate().verify(contract, response)
    passed = result.status == "failed" and result.failed_count == 1
    return CheckResult("verification:rejects-bad-answer", passed, result.failure_reason or result.status)


def run_raw_answer_preserved_check() -> CheckResult:
    response = AgentWorkflowResponse(
        request_id="acceptance",
        trace_id="acceptance",
        selected_workflow="knowledge_qa_workflow",
        answer="original workflow answer",
        status="success",
        metrics=AgenticMetrics(latency_ms=1.0),
    )
    status, answer, metadata = _finalize_verified_response(response, "failed", {})
    passed = (
        status == "verification_failed"
        and metadata.get("raw_answer") == "original workflow answer"
        and "failed Gateway verification" in answer
    )
    detail = f"status={status}, raw_answer={metadata.get('raw_answer')!r}"
    return CheckResult("verification:preserves-raw-answer", passed, detail)


def run_tool_audit_verification_check() -> CheckResult:
    contract = TaskContract(
        contract_id="contract_acceptance_tool_audit",
        task_type="knowledge_qa",
        selected_workflow="knowledge_qa_workflow",
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="tool_success:doc_search_tool",
                type="tool_success_required",
                target="doc_search_tool",
            )
        ],
    )
    response = AgentWorkflowResponse(
        request_id="acceptance",
        trace_id="audit-trace",
        selected_workflow="knowledge_qa_workflow",
        answer="answer",
        status="success",
        metrics=AgenticMetrics(latency_ms=1.0),
        tools=["doc_search_tool"],
        metadata={"tool_statuses": {"doc_search_tool": "success"}},
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_path = Path(temp_dir) / "tool_audit.jsonl"
        missing_result = VerificationGate().verify(contract, response, tool_audit_log_path=audit_path)
        audit_path.write_text(
            json.dumps(
                {
                    "schema": "tool_audit_v1",
                    "trace_id": "audit-trace",
                    "tool_name": "doc_search_tool",
                    "status": "success",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        passed_result = VerificationGate().verify(contract, response, tool_audit_log_path=audit_path)

    passed = missing_result.status == "failed" and passed_result.status == "passed"
    detail = f"missing_audit={missing_result.status}, with_audit={passed_result.status}"
    return CheckResult("verification:uses-tool-audit-log", passed, detail)


def run_planner_memory_route_check() -> CheckResult:
    store = load_workflow_profiles(ROOT_DIR / "configs/workflow_profiles.json")
    task_profile = TaskProfile(
        task_type="knowledge_qa",
        required_capabilities=["rag", "citation", "evidence_check", "deep_rag", "multi_hop_retrieval"],
        required_tools=["doc_search_tool", "evidence_check_tool"],
        evidence_required=True,
        permission_scope=["kb:project_docs:read", "tool:doc_search:use", "tool:evidence_check:use"],
    )
    result = MemoryPlanner().plan_route(
        enabled=True,
        task_profile=task_profile,
        workflow_profiles=store.all(),
        route_rules=[
            {
                "rule_id": "rr_acceptance_deep",
                "status": "active",
                "applies_when": {"task_type": "knowledge_qa", "capabilities_any": ["deep_rag"]},
                "routing_hint": {"prefer_workflow": "large_knowledge_qa_workflow", "score_boost": 0.12},
            }
        ],
    )
    route = AgenticRouter().select(
        trace_id="acceptance-memory-route",
        task_profile=task_profile,
        workflow_profiles=store.all(),
        user_permissions=["kb:project_docs:read", "tool:doc_search:use", "tool:evidence_check:use"],
        consumer=_acceptance_consumer(),
        route_hints=result.route_hints,
    )
    large_score = next(score for score in route.candidate_scores if score.workflow_id == "large_knowledge_qa_workflow")
    passed = result.matched_route_rules == ["rr_acceptance_deep"] and large_score.memory_route_boost == 0.12
    return CheckResult("planner_memory:route-boost", passed, f"boost={large_score.memory_route_boost}")


def run_planner_memory_contract_patch_check() -> CheckResult:
    store = load_workflow_profiles(ROOT_DIR / "configs/workflow_profiles.json")
    task_profile = TaskProfile(
        task_type="document_writing",
        required_capabilities=["document_writing", "a2a_knowledge_request", "citation"],
        required_tools=["inspection_tool"],
        permission_scope=["model:writing:use", "tool:inspection:use"],
    )
    base = MemoryPlanner().plan_route(enabled=True, task_profile=task_profile, workflow_profiles=store.all(), route_rules=[])
    result = MemoryPlanner().plan_contract(
        base=base,
        task_profile=task_profile,
        selected_workflow="document_writing_workflow",
        workflow_profiles=store.all(),
        contract_rules=[
            {
                "rule_id": "cr_acceptance_doc",
                "status": "active",
                "applies_when": {"task_type": "document_writing", "capabilities_any": ["document_writing"]},
                "patch": {
                    "add_required_tools": ["inspection_tool"],
                    "add_required_trace_events": ["NeedKnowledge"],
                    "add_acceptance_criteria": [
                        {
                            "criterion_id": "memory:acceptance_metadata",
                            "type": "metadata_required",
                            "target": "direct_rag_access",
                            "required": True,
                        }
                    ],
                },
            }
        ],
    )
    contract = TaskContractBuilder().build(
        AgenticGatewayRequest.model_validate(WORKFLOW_CASES["document_writing"]["payload"]),
        task_profile,
        "document_writing_workflow",
        contract_patches=result.contract_patches,
    )
    criterion_ids = {criterion.criterion_id for criterion in contract.acceptance_criteria}
    passed = (
        "memory:cr_acceptance_doc:trace_node:NeedKnowledge" in criterion_ids
        and "memory:cr_acceptance_doc:tool_success:inspection_tool" in criterion_ids
        and "memory:acceptance_metadata" in criterion_ids
    )
    return CheckResult("planner_memory:contract-patch", passed, f"criteria={sorted(criterion_ids)}")


def run_planner_memory_invalid_field_check() -> CheckResult:
    store = load_workflow_profiles(ROOT_DIR / "configs/workflow_profiles.json")
    task_profile = TaskProfile(task_type="knowledge_qa", required_capabilities=["rag"])
    result = MemoryPlanner().plan_route(
        enabled=True,
        task_profile=task_profile,
        workflow_profiles=store.all(),
        route_rules=[
            {
                "rule_id": "rr_bad_field",
                "status": "active",
                "applies_when": {"memory_feature": "made_up"},
                "routing_hint": {"prefer_workflow": "knowledge_qa_workflow", "score_boost": 0.1},
            }
        ],
    )
    passed = not result.route_hints and any("unsupported applies_when fields" in warning for warning in result.warnings)
    return CheckResult("planner_memory:rejects-feature-schema", passed, "; ".join(result.warnings))


def _acceptance_consumer() -> AuthenticatedConsumer:
    return AuthenticatedConsumer(
        consumer_id="acceptance_consumer",
        api_key_id="acceptance",
        permissions=["*"],
        allowed_workflows=["*"],
        allowed_tools=["*"],
        rate_limit_key="acceptance",
    )


def run_tool_schema_error_check(client: httpx.Client, tool_url: str, verbose: bool) -> CheckResult:
    trace_id = f"demo-schema-error-{uuid4()}"
    payload = {
        "tool_name": "doc_search_tool",
        "arguments": {},
        "user_id": "u001",
        "consumer_id": "demo_consumer",
        "trace_id": trace_id,
        "request_id": trace_id,
        "permissions": ["tool:doc_search:use"],
        "allowed_tools": ["doc_search_tool"],
    }
    return run_tool_status_check(client, tool_url, "tool schema_error", payload, "schema_error", verbose)


def run_tool_permission_denied_check(client: httpx.Client, tool_url: str, verbose: bool) -> CheckResult:
    trace_id = f"demo-permission-denied-{uuid4()}"
    payload = {
        "tool_name": "doc_search_tool",
        "arguments": {"query": "SemGateway", "top_k": 1, "strategy": "hybrid"},
        "user_id": "u001",
        "consumer_id": "demo_consumer",
        "trace_id": trace_id,
        "request_id": trace_id,
        "permissions": [],
        "allowed_tools": ["doc_search_tool"],
    }
    return run_tool_status_check(client, tool_url, "tool permission_denied", payload, "permission_denied", verbose)


def run_tool_status_check(
    client: httpx.Client,
    tool_url: str,
    name: str,
    payload: Dict[str, Any],
    expected_status: str,
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
    passed = response.status_code == 200 and body.get("status") == expected_status
    return CheckResult(name, passed, f"tool={body.get('tool_name')}, status={body.get('status')}")


def post_with_rate_retry(
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
        try:
            retry_after_s = float(response.headers.get("Retry-After", "1"))
        except ValueError:
            retry_after_s = 1.0
        time.sleep(max(1.0, retry_after_s))
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


def gateway_base_url(url: str) -> str:
    stripped = url.rstrip("/")
    for suffix in ("/v1/invoke", "/v1/preview"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)]
    return stripped


def print_results(results: List[CheckResult]) -> None:
    for result in results:
        prefix = "PASS" if result.passed else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")


if __name__ == "__main__":
    main()
