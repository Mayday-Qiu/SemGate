from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, get_args

from app.schemas import AcceptanceCriterion, CriterionType, TaskPlan, TaskProfile, WorkflowProfile
from services.tool_service.registry import get_tool


ALLOWED_APPLIES_KEYS = {
    "primary_task_type",
    "task_type",
    "selected_workflow",
    "semantic_features",
    "capabilities_any",
    "tools_any",
    "profile_reason_any",
}
ALLOWED_SEMANTIC_FEATURE_KEYS = {
    "deliverable_type",
    "requires_external_evidence",
    "requires_citations",
    "requires_media_asset",
    "evidence_need",
    "complexity",
}
ALLOWED_PATCH_KEYS = {
    "add_required_trace_events",
    "add_required_tools",
    "add_forbidden_tools",
    "add_acceptance_criteria",
}
MAX_ROUTE_BOOST = 0.20


@dataclass
class MemoryPlannerResult:
    enabled: bool
    matched_route_rules: List[str] = field(default_factory=list)
    matched_contract_rules: List[str] = field(default_factory=list)
    route_hints: List[Dict[str, Any]] = field(default_factory=list)
    contract_patches: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "matched_route_rules": self.matched_route_rules,
            "matched_contract_rules": self.matched_contract_rules,
            "route_hints": self.route_hints,
            "contract_patches_summary": [
                patch.get("summary", patch.get("rule_id", "contract_patch"))
                for patch in self.contract_patches
            ],
            "warnings": self.warnings,
        }


class MemoryPlanner:
    def plan_route(
        self,
        *,
        enabled: bool,
        task_profile: TaskProfile,
        workflow_profiles: Sequence[WorkflowProfile],
        route_rules: Sequence[Dict[str, Any]],
        warnings: Sequence[str] = (),
        task_plan: Optional[TaskPlan] = None,
    ) -> MemoryPlannerResult:
        result = MemoryPlannerResult(enabled=enabled, warnings=list(warnings))
        if not enabled:
            return result

        profiles_by_id = {profile.workflow_id: profile for profile in workflow_profiles}
        for rule in route_rules:
            rule_id = str(rule.get("rule_id", "route_rule"))
            if rule.get("status", "active") != "active":
                continue
            hint = rule.get("routing_hint")
            if not isinstance(hint, dict):
                result.warnings.append(f"{rule_id}: missing routing_hint")
                continue
            workflow_id = str(hint.get("prefer_workflow", ""))
            workflow_profile = profiles_by_id.get(workflow_id)
            if workflow_profile is None:
                result.warnings.append(f"{rule_id}: unknown workflow {workflow_id}")
                continue
            if not self._matches(
                rule,
                task_profile,
                workflow_profile,
                selected_workflow=None,
                warnings=result.warnings,
                task_plan=task_plan,
            ):
                continue
            boost = _safe_boost(hint.get("score_boost", 0.0))
            if boost <= 0:
                result.warnings.append(f"{rule_id}: score_boost must be positive")
                continue
            result.matched_route_rules.append(rule_id)
            result.route_hints.append({"rule_id": rule_id, "workflow_id": workflow_id, "score_boost": boost})
        return result

    def plan_contract(
        self,
        *,
        base: MemoryPlannerResult,
        task_profile: TaskProfile,
        selected_workflow: Optional[str],
        workflow_profiles: Sequence[WorkflowProfile],
        contract_rules: Sequence[Dict[str, Any]],
        task_plan: Optional[TaskPlan] = None,
    ) -> MemoryPlannerResult:
        if not base.enabled:
            return base

        profiles_by_id = {profile.workflow_id: profile for profile in workflow_profiles}
        workflow_profile = profiles_by_id.get(selected_workflow or "")
        for rule in contract_rules:
            rule_id = str(rule.get("rule_id", "contract_rule"))
            if rule.get("status", "active") != "active":
                continue
            if not self._matches(
                rule,
                task_profile,
                workflow_profile,
                selected_workflow,
                base.warnings,
                task_plan,
            ):
                continue
            patch = rule.get("patch")
            if not isinstance(patch, dict):
                base.warnings.append(f"{rule_id}: missing patch")
                continue
            if not self._valid_patch(rule_id, patch, base.warnings):
                continue
            base.matched_contract_rules.append(rule_id)
            base.contract_patches.append({"rule_id": rule_id, "summary": rule.get("reason", rule_id), **patch})
        return base

    def _matches(
        self,
        rule: Dict[str, Any],
        task_profile: TaskProfile,
        workflow_profile: Optional[WorkflowProfile],
        selected_workflow: Optional[str],
        warnings: List[str],
        task_plan: Optional[TaskPlan] = None,
    ) -> bool:
        rule_id = str(rule.get("rule_id", "memory_rule"))
        applies_when = rule.get("applies_when", {})
        if not isinstance(applies_when, dict):
            warnings.append(f"{rule_id}: applies_when must be an object")
            return False

        unknown_keys = sorted(set(applies_when) - ALLOWED_APPLIES_KEYS)
        if unknown_keys:
            warnings.append(f"{rule_id}: unsupported applies_when fields {unknown_keys}")
            return False

        if applies_when.get("task_type") and applies_when["task_type"] != task_profile.task_type:
            return False
        primary_task_type = _plan_primary_task_type(task_plan) or task_profile.task_type
        if applies_when.get("primary_task_type") and applies_when["primary_task_type"] != primary_task_type:
            return False
        if applies_when.get("selected_workflow") and applies_when["selected_workflow"] != selected_workflow:
            return False
        if not _semantic_features_match(rule_id, applies_when.get("semantic_features"), task_plan, warnings):
            return False

        task_capabilities = set(task_profile.required_capabilities)
        workflow_capabilities = set(workflow_profile.capabilities if workflow_profile else [])
        task_tools = set(task_profile.required_tools)
        workflow_tools = set(workflow_profile.allowed_tools if workflow_profile else [])
        reason_text = " ".join(task_profile.profile_reason).lower()

        return (
            _any_match(applies_when.get("capabilities_any"), task_capabilities | workflow_capabilities)
            and _any_match(applies_when.get("tools_any"), task_tools | workflow_tools)
            and _text_any_match(applies_when.get("profile_reason_any"), reason_text)
        )

    def _valid_patch(self, rule_id: str, patch: Dict[str, Any], warnings: List[str]) -> bool:
        unknown_keys = sorted(set(patch) - ALLOWED_PATCH_KEYS)
        if unknown_keys:
            warnings.append(f"{rule_id}: unsupported patch fields {unknown_keys}")
            return False

        for tool_name in _as_list(patch.get("add_required_tools")) + _as_list(patch.get("add_forbidden_tools")):
            if get_tool(str(tool_name)) is None:
                warnings.append(f"{rule_id}: unknown tool {tool_name}")
                return False

        allowed_criteria = set(get_args(CriterionType))
        for raw_criterion in _as_list(patch.get("add_acceptance_criteria")):
            if not isinstance(raw_criterion, dict):
                warnings.append(f"{rule_id}: acceptance criterion must be an object")
                return False
            if raw_criterion.get("type") not in allowed_criteria:
                warnings.append(f"{rule_id}: unsupported criterion type {raw_criterion.get('type')}")
                return False
            try:
                AcceptanceCriterion(
                    criterion_id=str(raw_criterion.get("criterion_id", f"memory:{rule_id}")),
                    type=raw_criterion["type"],
                    target=str(raw_criterion.get("target", "")),
                    required=bool(raw_criterion.get("required", True)),
                    params=dict(raw_criterion.get("params", {})),
                    description=str(raw_criterion.get("description", "")),
                )
            except Exception as exc:
                warnings.append(f"{rule_id}: invalid acceptance criterion: {exc}")
                return False
        return True


def _any_match(raw_values: Any, candidates: set[str]) -> bool:
    if raw_values is None:
        return True
    values = {str(item) for item in _as_list(raw_values)}
    return bool(values & candidates)


def _text_any_match(raw_values: Any, text: str) -> bool:
    if raw_values is None:
        return True
    values = [str(item).lower() for item in _as_list(raw_values)]
    return any(value in text for value in values)


def _semantic_features_match(
    rule_id: str,
    expected: Any,
    task_plan: Optional[TaskPlan],
    warnings: List[str],
) -> bool:
    if expected is None:
        return True
    if not isinstance(expected, dict):
        warnings.append(f"{rule_id}: semantic_features must be an object")
        return False
    unknown_keys = sorted(set(expected) - ALLOWED_SEMANTIC_FEATURE_KEYS)
    if unknown_keys:
        warnings.append(f"{rule_id}: unsupported semantic_features {unknown_keys}")
        return False
    if task_plan is None:
        return False
    actual = task_plan.semantic_features if isinstance(task_plan.semantic_features, dict) else {}
    return all(actual.get(key) == value for key, value in expected.items())


def _plan_primary_task_type(task_plan: Optional[TaskPlan]) -> Optional[str]:
    if task_plan is None:
        return None
    return str(task_plan.primary_task_type)


def _safe_boost(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(MAX_ROUTE_BOOST, value)), 6)


def _as_list(raw_value: Any) -> List[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    return [raw_value]
