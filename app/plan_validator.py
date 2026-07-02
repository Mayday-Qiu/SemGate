from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from pydantic import ValidationError

from app.planner_context import ALLOWED_A2A_PATHS
from app.schemas import PlanValidationResult, TaskPlan, WorkflowProfile
from services.tool_service.registry import get_tool


class PlanValidator:
    def validate(
        self,
        raw_plan: Dict[str, Any],
        *,
        workflow_profiles: Sequence[WorkflowProfile],
        user_permissions: Sequence[str],
    ) -> Tuple[Optional[TaskPlan], PlanValidationResult]:
        if raw_plan.get("_planner_error"):
            return None, PlanValidationResult(
                status="failed",
                errors=[str(raw_plan["_planner_error"])],
                repairable="_raw_content" in raw_plan,
            )
        normalized_plan, normalize_warnings = self._normalize(raw_plan)
        try:
            plan = TaskPlan.model_validate(normalized_plan)
        except ValidationError as exc:
            return None, PlanValidationResult(status="failed", errors=[exc.errors()[0]["msg"]], repairable=True)

        workflows = {profile.workflow_id: profile for profile in workflow_profiles}
        tools_by_workflow = {profile.workflow_id: set(profile.allowed_tools) for profile in workflow_profiles}
        errors = []
        warnings = list(normalize_warnings)
        step_ids = {step.step_id for step in plan.execution_plan}
        if not plan.execution_plan:
            errors.append("execution_plan must not be empty")
        if plan.plan_version != "2.0":
            errors.append("plan_version must be 2.0")

        user_scope = set(user_permissions)
        wildcard = "*" in user_scope
        for step in plan.execution_plan:
            profile = workflows.get(step.workflow)
            if profile is None:
                errors.append(f"unknown workflow: {step.workflow}")
                continue
            if step.task_type and step.task_type not in profile.supported_tasks:
                errors.append(f"workflow {step.workflow} does not support task_type {step.task_type}")
            missing_workflow_scope = sorted(set(profile.required_permission_scope) - user_scope)
            if missing_workflow_scope and not wildcard:
                errors.append(f"missing workflow permissions for {step.workflow}: {', '.join(missing_workflow_scope)}")
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"unknown dependency {dep} in step {step.step_id}")
            for tool_name in step.required_tools:
                tool = get_tool(tool_name)
                if tool is None:
                    errors.append(f"unknown tool: {tool_name}")
                    continue
                if tool_name not in tools_by_workflow.get(step.workflow, set()):
                    errors.append(f"tool {tool_name} is not allowed for workflow {step.workflow}")
                if tool.implementation_status == "placeholder":
                    warnings.append(f"tool {tool_name} is placeholder")
                if tool.permission_scope not in user_scope and not wildcard:
                    errors.append(f"missing tool permission for {tool_name}: {tool.permission_scope}")
            if step.workflow == "document_writing_workflow":
                forbidden = {"doc_search_tool", "evidence_check_tool"} & set(step.required_tools)
                if forbidden:
                    errors.append(f"document_writing_workflow must not directly require {', '.join(sorted(forbidden))}")

        self._validate_a2a(plan, errors)

        return plan, PlanValidationResult(
            status="failed" if errors else "passed",
            errors=errors,
            warnings=sorted(set(warnings)),
            repairable=bool(errors),
        )

    def _normalize(self, raw_plan: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
        plan = dict(raw_plan)
        warnings: list[str] = []
        if not isinstance(plan.get("semantic_features"), dict):
            plan["semantic_features"] = {}
            warnings.append("normalized semantic_features to object")
        if not isinstance(plan.get("contract_hints"), dict):
            plan["contract_hints"] = {}
            warnings.append("normalized contract_hints to object")
        hints = dict(plan.get("contract_hints", {}))
        if "acceptance_requirements" in hints:
            hints.pop("acceptance_requirements", None)
            warnings.append("dropped planner acceptance_requirements")
        for key in ("required_output_fields", "required_trace_events"):
            value = hints.get(key, [])
            if not isinstance(value, list):
                hints[key] = [str(value)] if value else []
                warnings.append(f"normalized contract_hints.{key} to list")
        plan["contract_hints"] = hints
        steps = plan.get("execution_plan", [])
        if isinstance(steps, list):
            normalized_steps = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                item = dict(step)
                item["step_role"] = self._normalize_step_role(
                    str(item.get("step_role", "")),
                    str(item.get("workflow", "")),
                    str(plan.get("primary_task_type", "")),
                )
                for key in ("required_tools", "forbidden_tools", "depends_on"):
                    value = item.get(key, [])
                    if not isinstance(value, list):
                        item[key] = [str(value)] if value else []
                        warnings.append(f"normalized step.{key} to list")
                normalized_steps.append(item)
            plan["execution_plan"] = normalized_steps
        return plan, warnings

    def _normalize_step_role(self, role: str, workflow: str, primary_task_type: str) -> str:
        if workflow in {"knowledge_qa_workflow", "large_knowledge_qa_workflow"}:
            return "primary_answer" if primary_task_type == "knowledge_qa" else "knowledge_context"
        if workflow == "media_generation_workflow":
            return "media_generation" if primary_task_type == "media_generation" else "media_asset"
        if workflow == "document_writing_workflow":
            return "final_compose"
        if workflow == "coding_workflow":
            return "code_draft"
        stable_roles = {"knowledge_context", "media_asset", "final_compose", "primary_answer", "code_draft", "media_generation"}
        if role in stable_roles:
            return role
        return role or "step"

    def _validate_a2a(self, plan: TaskPlan, errors: list[str]) -> None:
        workflows = {step.workflow for step in plan.execution_plan}
        if len(workflows) <= 1:
            return
        primary_workflows = {
            "document_writing": "document_writing_workflow",
            "coding": "coding_workflow",
            "media_generation": "media_generation_workflow",
        }
        final = primary_workflows.get(plan.primary_task_type)
        if final not in workflows:
            final = next((step.workflow for step in plan.execution_plan if step.step_role == "final_compose"), None)
        if final is None:
            final = next((step.workflow for step in reversed(plan.execution_plan) if step.required), "")
        allowed = {(item["from"], item["to"]) for item in ALLOWED_A2A_PATHS}
        for workflow in workflows:
            if workflow == final:
                continue
            if (final, workflow) not in allowed:
                errors.append(f"A2A path not allowed: {final} -> {workflow}")
