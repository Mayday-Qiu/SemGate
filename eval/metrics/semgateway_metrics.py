from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    reason: str
    details: Dict[str, Any]


def evaluate_record(record: Dict[str, Any]) -> Dict[str, MetricResult]:
    return {
        result.name: result
        for result in (
            planner_schema_validity(record),
            plan_validation_pass(record),
            task_type_correctness(record),
            workflow_plan_correctness(record),
            contract_hint_coverage(record),
            route_correctness(record),
            contract_compliance(record),
            trace_event_coverage(record),
            tool_compliance(record),
            verification_pass(record),
            citation_source(record),
            schema_compliance(record),
        )
    }


def metric_results_json(results: Dict[str, MetricResult]) -> Dict[str, Any]:
    return {name: asdict(result) for name, result in results.items()}


def route_correctness(record: Dict[str, Any]) -> MetricResult:
    case = _case(record)
    response = _response(record)
    expected = case.get("expected_workflow")
    allowed = set(_as_list(case.get("allowed_workflows_any")))
    selected = response.get("selected_workflow") or record.get("selected_workflow")
    if not expected and not allowed:
        return _result("RouteCorrectnessMetric", 1.0, True, "no route expectation", {})
    passed = selected == expected or (bool(allowed) and selected in allowed)
    route_decision = _route_decision(record)
    selected_excluded = any(item.get("workflow_id") == selected for item in _as_list(route_decision.get("excluded_candidates")))
    if selected_excluded:
        passed = False
    return _result(
        "RouteCorrectnessMetric",
        1.0 if passed else 0.0,
        passed,
        "selected expected workflow" if passed else "selected workflow mismatch",
        {"expected": expected, "allowed": sorted(allowed), "selected": selected, "selected_excluded": selected_excluded},
    )


def contract_compliance(record: Dict[str, Any]) -> MetricResult:
    case = _case(record)
    contract = _task_contract(record)
    checks: List[Tuple[str, bool]] = []
    if case.get("task_type"):
        checks.append(("task_type", contract.get("task_type") == case.get("task_type")))
    expected_workflow = case.get("expected_workflow")
    if expected_workflow:
        checks.append(("selected_workflow", contract.get("selected_workflow") == expected_workflow))
    for tool in _as_list(case.get("required_tools")):
        checks.append((f"required_tool:{tool}", tool in set(_as_list(contract.get("required_tools")))))
    forbidden = set(_as_list(case.get("forbidden_tools")))
    contract_forbidden = set(_as_list(_nested(contract, "forbidden_resources.tools")))
    for tool in forbidden:
        checks.append((f"forbidden_tool:{tool}", tool in contract_forbidden))
    criteria = _criteria(contract)
    for criterion in _as_list(case.get("required_criteria")):
        checks.append((f"criterion:{criterion}", criterion in criteria))
    return _checks_result("ContractComplianceMetric", checks, "contract satisfies expected constraints")


def trace_event_coverage(record: Dict[str, Any]) -> MetricResult:
    if record.get("mode") == "preview":
        return _result("TraceEventCoverageMetric", 1.0, True, "preview mode skips execution trace coverage", {})
    case = _case(record)
    contract = _task_contract(record)
    required = set(_as_list(case.get("expected_trace_nodes")))
    for criterion in _as_list(contract.get("acceptance_criteria")):
        if isinstance(criterion, dict) and criterion.get("type") == "trace_event_required":
            required.add(str(criterion.get("target", "")))
    required.discard("")
    nodes = set(_trace_nodes(record))
    if not required:
        return _result("TraceEventCoverageMetric", 1.0, True, "no trace expectation", {})
    missing = sorted(required - nodes)
    score = (len(required) - len(missing)) / len(required)
    return _result(
        "TraceEventCoverageMetric",
        score,
        not missing,
        "trace covered required nodes" if not missing else "missing trace nodes",
        {"missing": missing, "required": sorted(required)},
    )


def tool_compliance(record: Dict[str, Any]) -> MetricResult:
    if record.get("mode") == "preview":
        return _result("ToolComplianceMetric", 1.0, True, "preview mode skips tool execution compliance", {})
    case = _case(record)
    response = _response(record)
    contract = _task_contract(record)
    required = set(_as_list(case.get("required_tools"))) | set(_as_list(contract.get("required_tools")))
    forbidden = set(_as_list(case.get("forbidden_tools"))) | set(_as_list(_nested(contract, "forbidden_resources.tools")))
    direct_tools = set(_direct_tools(record))
    all_tools = set(_as_list(response.get("tools"))) | set(_as_list(_nested(response, "metadata.tools")))
    missing_success = sorted(tool for tool in required if not _tool_success(record, tool))
    forbidden_used = sorted(tool for tool in forbidden if tool in direct_tools or tool in all_tools)
    total = max(1, len(required) + len(forbidden))
    failed = len(missing_success) + len(forbidden_used)
    return _result(
        "ToolComplianceMetric",
        max(0.0, (total - failed) / total),
        failed == 0,
        "tools satisfy contract" if failed == 0 else "tool compliance failed",
        {"missing_success": missing_success, "forbidden_used": forbidden_used, "direct_tools": sorted(direct_tools)},
    )


def verification_pass(record: Dict[str, Any]) -> MetricResult:
    if record.get("mode") == "preview":
        return _result("VerificationPassMetric", 1.0, True, "preview mode skips output verification", {})
    case = _case(record)
    response = _response(record)
    expected = str(case.get("expected_status", "success"))
    status_value = str(response.get("status") or record.get("status") or "")
    verification_status = str(_nested(response, "verification.status") or record.get("verification_status") or "")
    if expected == "success":
        passed = status_value == "success" and verification_status == "passed"
    else:
        passed = status_value == expected
    return _result(
        "VerificationPassMetric",
        1.0 if passed else 0.0,
        passed,
        "verification matched expected status" if passed else "verification status mismatch",
        {"expected_status": expected, "status": status_value, "verification_status": verification_status},
    )


def citation_source(record: Dict[str, Any]) -> MetricResult:
    if record.get("mode") == "preview":
        return _result("CitationSourceMetric", 1.0, True, "preview mode skips citation source verification", {})
    case = _case(record)
    response = _response(record)
    contract = _task_contract(record)
    citation_required = bool(case.get("citation_required")) or _has_criterion(contract, "citation_required")
    if not citation_required:
        return _result("CitationSourceMetric", 1.0, True, "citation not required", {})
    citations = _citations(response)
    min_count = _citation_min_count(contract, int(case.get("min_citations", 1)))
    if len(citations) < min_count:
        return _result(
            "CitationSourceMetric",
            0.0,
            False,
            "missing required citations",
            {"citations": len(citations), "required": min_count},
        )
    refs = _document_citation_refs(record) if case.get("task_type") == "document_writing" else _evidence_refs(record)
    citation_keys = _ref_keys(citations)
    missing = sorted(citation_keys - refs)
    direct_rag = _nested(response, "metadata.agent_metadata.direct_rag_access")
    if case.get("task_type") == "document_writing" and direct_rag is True:
        missing = sorted(citation_keys)
    passed = not missing
    score = 1.0 if passed else max(0.0, (len(citation_keys) - len(missing)) / max(1, len(citation_keys)))
    return _result(
        "CitationSourceMetric",
        score,
        passed,
        "citations have traceable source" if passed else "citations are not traceable to evidence source",
        {"missing": missing, "citation_count": len(citations), "ref_count": len(refs)},
    )


def schema_compliance(record: Dict[str, Any]) -> MetricResult:
    if record.get("mode") == "preview":
        response = _response(record)
        required = {"request_id", "trace_id", "preview_status", "selected_workflow", "task_contract"}
        return _checks_result(
            "SchemaComplianceMetric",
            [(field, _has_path(response, field)) for field in sorted(required)],
            "preview response schema satisfies expected fields",
        )
    case = _case(record)
    response = _response(record)
    contract = _task_contract(record)
    required = set(_as_list(case.get("required_output_fields")))
    required.update(str(item) for item in _as_list(_nested(contract, "output_schema.required")))
    required.update({"request_id", "trace_id", "selected_workflow", "status", "metrics.latency_ms"})
    required.discard("")
    checks = [(field, _has_path(response, field)) for field in sorted(required)]
    return _checks_result("SchemaComplianceMetric", checks, "response schema satisfies expected fields")


def planner_schema_validity(record: Dict[str, Any]) -> MetricResult:
    raw_plan = _raw_task_plan(record)
    if not raw_plan:
        return _result("PlannerSchemaValidityMetric", 1.0, True, "no planner output expected", {})
    checks = [
        ("plan_version", raw_plan.get("plan_version") == "2.0"),
        ("primary_task_type", bool(raw_plan.get("primary_task_type"))),
        ("execution_plan", isinstance(raw_plan.get("execution_plan"), list) and bool(raw_plan.get("execution_plan"))),
    ]
    for index, step in enumerate(_as_list(raw_plan.get("execution_plan"))):
        if not isinstance(step, dict):
            checks.append((f"step:{index}:object", False))
            continue
        checks.append((f"step:{index}:step_role", bool(step.get("step_role"))))
        checks.append((f"step:{index}:workflow", bool(step.get("workflow"))))
    return _checks_result("PlannerSchemaValidityMetric", checks, "planner output has required structure")


def plan_validation_pass(record: Dict[str, Any]) -> MetricResult:
    validation = _plan_validation(record)
    if not validation:
        return _result("PlanValidationPassMetric", 1.0, True, "no plan validation expected", {})
    passed = validation.get("status") == "passed"
    return _result(
        "PlanValidationPassMetric",
        1.0 if passed else 0.0,
        passed,
        "plan validation passed" if passed else "plan validation failed",
        {"errors": validation.get("errors", [])},
    )


def task_type_correctness(record: Dict[str, Any]) -> MetricResult:
    expected = _nested(_expected_plan(record), "primary_task_type") or _case(record).get("task_type")
    plan = _validated_task_plan(record)
    if not expected or not plan:
        return _result("TaskTypeCorrectnessMetric", 1.0, True, "no task type expectation", {})
    actual = plan.get("primary_task_type")
    passed = actual == expected
    return _result(
        "TaskTypeCorrectnessMetric",
        1.0 if passed else 0.0,
        passed,
        "planner selected expected task type" if passed else "planner task type mismatch",
        {"expected": expected, "actual": actual},
    )


def workflow_plan_correctness(record: Dict[str, Any]) -> MetricResult:
    expected_plan = _expected_plan(record)
    required_secondary = _expected_secondary_tasks(expected_plan)
    required_roles = _expected_required_roles(expected_plan)
    role_workflows = _expected_role_workflows(expected_plan)
    required_dependencies = _expected_dependencies(expected_plan)
    if not required_secondary and not required_roles and not role_workflows and not required_dependencies:
        return _result("WorkflowPlanCorrectnessMetric", 1.0, True, "no workflow plan expectation", {})
    actual_plan = _validated_task_plan(record)
    actual_roles = _actual_role_steps(actual_plan)
    actual_secondary = _actual_secondary_tasks(actual_plan)
    actual_dependencies = _actual_dependencies(actual_plan)

    checks: List[Tuple[str, bool]] = []
    for task_type in sorted(required_secondary):
        checks.append((f"missing_required_secondary_task:{task_type}", task_type in actual_secondary))
    for role in sorted(required_roles):
        checks.append((f"missing_required_step_role:{role}", role in actual_roles))
    for role, allowed_workflows in sorted(role_workflows.items()):
        if role in required_roles or role in actual_roles:
            actual_workflow = str(actual_roles.get(role, {}).get("workflow", ""))
            checks.append((f"wrong_workflow_for_step_role:{role}", actual_workflow in allowed_workflows))
    for dependency in sorted(required_dependencies):
        checks.append((f"missing_required_dependency:{dependency[0]}->{dependency[1]}", dependency in actual_dependencies))
    return _checks_result("WorkflowPlanCorrectnessMetric", checks, "planner workflow roles match expected plan")


def contract_hint_coverage(record: Dict[str, Any]) -> MetricResult:
    expected_plan = _expected_plan(record)
    if not expected_plan:
        return _result("ContractHintCoverageMetric", 1.0, True, "no contract hint expectation", {})
    contract = _task_contract(record)
    required_nodes = set(_as_list(_nested(expected_plan, "contract_hints.required_trace_events")))
    required_fields = set(_as_list(_nested(expected_plan, "contract_hints.required_output_fields")))
    required_acceptance_types = set(_as_list(expected_plan.get("required_acceptance_types")))
    expected_forbidden = _expected_forbidden_tools(expected_plan)
    contract_nodes = {
        str(item.get("target"))
        for item in _as_list(contract.get("acceptance_criteria"))
        if isinstance(item, dict) and item.get("type") == "trace_event_required"
    }
    criteria_types = {
        str(item.get("type"))
        for item in _as_list(contract.get("acceptance_criteria"))
        if isinstance(item, dict) and item.get("type")
    }
    contract_fields = set(_as_list(_nested(contract, "output_schema.required")))
    contract_forbidden = set(_as_list(_nested(contract, "forbidden_resources.tools")))
    actual_roles = _actual_role_steps(_validated_task_plan(record))

    checks = [(f"missing_contract_hint:trace:{node}", node in contract_nodes) for node in sorted(required_nodes)]
    checks.extend((f"missing_contract_hint:field:{field}", field in contract_fields) for field in sorted(required_fields))
    checks.extend(
        (f"missing_contract_hint:acceptance_type:{criterion_type}", criterion_type in criteria_types)
        for criterion_type in sorted(required_acceptance_types)
    )
    for role, tools in sorted(expected_forbidden.items()):
        step_forbidden = set(_as_list(actual_roles.get(role, {}).get("forbidden_tools")))
        preserved = contract_forbidden | step_forbidden
        for tool in sorted(tools):
            checks.append((f"forbidden_tool_not_preserved:{role}:{tool}", tool in preserved))
    return _checks_result("ContractHintCoverageMetric", checks, "task contract covers expected planner hints")


try:
    from deepeval.metrics import BaseMetric
except Exception:  # pragma: no cover - optional dependency
    BaseMetric = object  # type: ignore[assignment]


class _DeepEvalMetric(BaseMetric):  # type: ignore[misc, valid-type]
    metric_name = "SemGatewayMetric"
    evaluator: Callable[[Dict[str, Any]], MetricResult] = staticmethod(lambda record: _result("SemGatewayMetric", 0.0, False, "missing evaluator", {}))

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: Any) -> float:
        result = self.evaluator(_record_from_test_case(test_case))
        self.score = result.score
        self.success = result.score >= self.threshold and result.passed
        self.reason = result.reason
        return self.score

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.metric_name


class RouteCorrectnessMetric(_DeepEvalMetric):
    metric_name = "RouteCorrectnessMetric"
    evaluator = staticmethod(route_correctness)


class PlannerSchemaValidityMetric(_DeepEvalMetric):
    metric_name = "PlannerSchemaValidityMetric"
    evaluator = staticmethod(planner_schema_validity)


class PlanValidationPassMetric(_DeepEvalMetric):
    metric_name = "PlanValidationPassMetric"
    evaluator = staticmethod(plan_validation_pass)


class TaskTypeCorrectnessMetric(_DeepEvalMetric):
    metric_name = "TaskTypeCorrectnessMetric"
    evaluator = staticmethod(task_type_correctness)


class WorkflowPlanCorrectnessMetric(_DeepEvalMetric):
    metric_name = "WorkflowPlanCorrectnessMetric"
    evaluator = staticmethod(workflow_plan_correctness)


class ContractHintCoverageMetric(_DeepEvalMetric):
    metric_name = "ContractHintCoverageMetric"
    evaluator = staticmethod(contract_hint_coverage)


class ContractComplianceMetric(_DeepEvalMetric):
    metric_name = "ContractComplianceMetric"
    evaluator = staticmethod(contract_compliance)


class TraceEventCoverageMetric(_DeepEvalMetric):
    metric_name = "TraceEventCoverageMetric"
    evaluator = staticmethod(trace_event_coverage)


class ToolComplianceMetric(_DeepEvalMetric):
    metric_name = "ToolComplianceMetric"
    evaluator = staticmethod(tool_compliance)


class VerificationPassMetric(_DeepEvalMetric):
    metric_name = "VerificationPassMetric"
    evaluator = staticmethod(verification_pass)


class CitationSourceMetric(_DeepEvalMetric):
    metric_name = "CitationSourceMetric"
    evaluator = staticmethod(citation_source)


class SchemaComplianceMetric(_DeepEvalMetric):
    metric_name = "SchemaComplianceMetric"
    evaluator = staticmethod(schema_compliance)


def _result(name: str, score: float, passed: bool, reason: str, details: Dict[str, Any]) -> MetricResult:
    return MetricResult(name=name, score=round(float(score), 4), passed=passed, reason=reason, details=details)


def _checks_result(name: str, checks: Sequence[Tuple[str, bool]], ok_reason: str) -> MetricResult:
    if not checks:
        return _result(name, 1.0, True, "no checks required", {})
    failed = [label for label, passed in checks if not passed]
    score = (len(checks) - len(failed)) / len(checks)
    return _result(name, score, not failed, ok_reason if not failed else "checks failed", {"failed": failed, "total": len(checks)})


def _case(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("case") if isinstance(record.get("case"), dict) else record


def _response(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("response") if isinstance(record.get("response"), dict) else record


def _route_decision(record: Dict[str, Any]) -> Dict[str, Any]:
    return _nested(_response(record), "metadata.route_decision") or record.get("route_decision") or {}


def _task_contract(record: Dict[str, Any]) -> Dict[str, Any]:
    return _nested(_response(record), "metadata.task_contract") or _response(record).get("task_contract") or record.get("task_contract") or {}


def _planner(record: Dict[str, Any]) -> Dict[str, Any]:
    return _nested(_response(record), "metadata.planner") or record.get("planner") or {}


def _raw_task_plan(record: Dict[str, Any]) -> Dict[str, Any]:
    return _nested(_response(record), "raw_task_plan") or _planner(record).get("raw_task_plan") or record.get("raw_task_plan") or {}


def _validated_task_plan(record: Dict[str, Any]) -> Dict[str, Any]:
    return (
        _nested(_response(record), "validated_task_plan")
        or _planner(record).get("validated_task_plan")
        or record.get("validated_task_plan")
        or {}
    )


def _plan_validation(record: Dict[str, Any]) -> Dict[str, Any]:
    return _nested(_response(record), "plan_validation") or _planner(record).get("plan_validation") or record.get("plan_validation") or {}


def _expected_plan(record: Dict[str, Any]) -> Dict[str, Any]:
    expected = _case(record).get("expected_plan")
    return expected if isinstance(expected, dict) else {}


def _role_workflows(plan: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for step in _as_list(plan.get("execution_plan")):
        if isinstance(step, dict) and step.get("step_role") and step.get("workflow"):
            result[str(step["step_role"])] = str(step["workflow"])
    return result


def _expected_secondary_tasks(plan: Dict[str, Any]) -> Set[str]:
    values = _as_list(plan.get("required_secondary_task_types"))
    if not values:
        values = _as_list(plan.get("secondary_task_types"))
    return {str(item) for item in values if str(item)}


def _expected_required_roles(plan: Dict[str, Any]) -> Set[str]:
    roles = {str(item) for item in _as_list(plan.get("required_step_roles")) if str(item)}
    if roles:
        return roles
    return set(_role_workflows(plan))


def _expected_role_workflows(plan: Dict[str, Any]) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    raw = plan.get("accepted_workflows_by_step_role")
    if isinstance(raw, dict):
        for role, workflows in raw.items():
            allowed = {str(item) for item in _as_list(workflows) if str(item)}
            if allowed:
                result[str(role)] = allowed
    for role, workflow in _role_workflows(plan).items():
        result.setdefault(role, {workflow})
    return result


def _expected_dependencies(plan: Dict[str, Any]) -> Set[Tuple[str, str]]:
    raw_dependencies = _as_list(plan.get("required_dependencies"))
    if raw_dependencies:
        return _dependency_pairs(raw_dependencies)
    return _actual_dependencies(plan)


def _actual_role_steps(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for step in _as_list(plan.get("execution_plan")):
        if isinstance(step, dict) and step.get("step_role"):
            result[str(step["step_role"])] = step
    return result


def _actual_secondary_tasks(plan: Dict[str, Any]) -> Set[str]:
    tasks = {str(item) for item in _as_list(plan.get("secondary_task_types")) if str(item)}
    for step in _as_list(plan.get("execution_plan")):
        if not isinstance(step, dict):
            continue
        if step.get("task_type"):
            tasks.add(str(step["task_type"]))
        workflow_task = _workflow_task_type(str(step.get("workflow", "")))
        if workflow_task:
            tasks.add(workflow_task)
    return tasks


def _workflow_task_type(workflow: str) -> Optional[str]:
    return {
        "knowledge_qa_workflow": "knowledge_qa",
        "large_knowledge_qa_workflow": "knowledge_qa",
        "coding_workflow": "coding",
        "media_generation_workflow": "media_generation",
        "document_writing_workflow": "document_writing",
    }.get(workflow)


def _actual_dependencies(plan: Dict[str, Any]) -> Set[Tuple[str, str]]:
    steps = [step for step in _as_list(plan.get("execution_plan")) if isinstance(step, dict)]
    id_to_role = {str(step.get("step_id")): str(step.get("step_role")) for step in steps if step.get("step_id") and step.get("step_role")}
    pairs = set()
    for step in steps:
        role = str(step.get("step_role", ""))
        if not role:
            continue
        for dependency_id in _as_list(step.get("depends_on")):
            dependency_role = id_to_role.get(str(dependency_id), str(dependency_id))
            if dependency_role:
                pairs.add((role, dependency_role))
    return pairs


def _dependency_pairs(raw_dependencies: List[Any]) -> Set[Tuple[str, str]]:
    pairs = set()
    for item in raw_dependencies:
        if not isinstance(item, dict):
            continue
        from_role = str(item.get("from_step_role", "") or item.get("step_role", "")).strip()
        depends_role = str(item.get("depends_on_step_role", "") or item.get("depends_on", "")).strip()
        if from_role and depends_role:
            pairs.add((from_role, depends_role))
    return pairs


def _expected_forbidden_tools(plan: Dict[str, Any]) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    raw = plan.get("forbidden_tools_by_step_role")
    if isinstance(raw, dict):
        for role, tools in raw.items():
            values = {str(item) for item in _as_list(tools) if str(item)}
            if values:
                result[str(role)] = values
    for step in _as_list(plan.get("execution_plan")):
        if not isinstance(step, dict) or not step.get("step_role"):
            continue
        tools = {str(item) for item in _as_list(step.get("forbidden_tools")) if str(item)}
        if tools:
            result.setdefault(str(step["step_role"]), set()).update(tools)
    return result


def _criteria(contract: Dict[str, Any]) -> Set[str]:
    result = set()
    for criterion in _as_list(contract.get("acceptance_criteria")):
        if not isinstance(criterion, dict):
            continue
        ctype = str(criterion.get("type", ""))
        target = str(criterion.get("target", ""))
        result.add(ctype)
        result.add(f"{ctype}:{target}")
    return result


def _has_criterion(contract: Dict[str, Any], ctype: str) -> bool:
    return any(isinstance(item, dict) and item.get("type") == ctype for item in _as_list(contract.get("acceptance_criteria")))


def _citation_min_count(contract: Dict[str, Any], default: int) -> int:
    result = default
    for criterion in _as_list(contract.get("acceptance_criteria")):
        if isinstance(criterion, dict) and criterion.get("type") == "citation_required":
            params = criterion.get("params") if isinstance(criterion.get("params"), dict) else {}
            result = max(result, int(params.get("min_count", 1)))
    return result


def _trace_nodes(record: Dict[str, Any]) -> List[str]:
    if isinstance(record.get("trace_nodes"), list):
        return [str(item) for item in record["trace_nodes"]]
    return [str(event.get("node")) for event in _trace_events(record) if isinstance(event, dict) and event.get("node")]


def _trace_events(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = record.get("trace_events")
    if isinstance(events, list):
        return [item for item in events if isinstance(item, dict)]
    return []


def _tool_success(record: Dict[str, Any], tool_name: str) -> bool:
    response = _response(record)
    statuses = _nested(response, "metadata.agent_metadata.tool_statuses") or _nested(response, "metadata.tool_statuses") or {}
    if isinstance(statuses, dict) and statuses.get(tool_name) == "success":
        return True
    trace_id = response.get("trace_id") or record.get("trace_id")
    for item in _as_list(record.get("tool_audit")):
        if not isinstance(item, dict):
            continue
        if item.get("trace_id") == trace_id and item.get("tool_name") == tool_name and item.get("status") == "success":
            return True
    return False


def _direct_tools(record: Dict[str, Any]) -> List[str]:
    response = _response(record)
    direct = _nested(response, "metadata.agent_metadata.direct_tools")
    if isinstance(direct, list):
        return [str(item) for item in direct]
    return _as_list(_nested(response, "metadata.tools") or response.get("tools"))


def _citations(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in _as_list(response.get("citations")) if isinstance(item, dict)]


def _evidence_refs(record: Dict[str, Any]) -> Set[Tuple[str, str]]:
    response = _response(record)
    refs = set()
    for path in ("metadata.agent_metadata.evidence_refs", "metadata.agent_metadata.citation_refs"):
        refs.update(_ref_keys(_nested(response, path)))
    for event in _trace_events(record):
        refs.update(_ref_keys(_nested(event, "metadata.evidence_refs")))
        refs.update(_ref_keys(_nested(event, "metadata.citation_refs")))
    return refs


def _document_citation_refs(record: Dict[str, Any]) -> Set[Tuple[str, str]]:
    response = _response(record)
    refs = _ref_keys(_nested(response, "metadata.agent_metadata.knowledge_context.citations"))
    refs.update(_ref_keys(_nested(response, "metadata.agent_metadata.citation_refs")))
    return refs


def _ref_keys(items: Any) -> Set[Tuple[str, str]]:
    keys = set()
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        if source_id and chunk_id:
            keys.add((source_id, chunk_id))
    return keys


def _has_path(payload: Dict[str, Any], path: str) -> bool:
    value = _nested(payload, path)
    return value not in (None, "", [], {})


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
    if isinstance(value, set):
        return list(value)
    return [value]


def _record_from_test_case(test_case: Any) -> Dict[str, Any]:
    metadata = getattr(test_case, "additional_metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("record"), dict):
        return metadata["record"]
    actual = getattr(test_case, "actual_output", None)
    if isinstance(actual, str):
        try:
            parsed = json.loads(actual)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}
