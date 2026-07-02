from __future__ import annotations

from typing import List

from app.schemas import AgenticGatewayRequest, PlannerPolicyDecision, TaskProfile


class PlannerPolicy:
    def decide(
        self,
        request: AgenticGatewayRequest,
        task_profile: TaskProfile,
        *,
        planner_enabled: bool,
    ) -> PlannerPolicyDecision:
        if not planner_enabled:
            return PlannerPolicyDecision(planner_required=False, bypass_reason="task planner disabled")
        signals = task_profile.rough_signals
        reasons: List[str] = []
        possible_task_types = signals.get("possible_task_types", [])
        if request.metadata.get("force_planner") is True:
            reasons.append("client_force_planner")
        if task_profile.task_type == "document_writing":
            reasons.append("document_writing_default_planner")
        if isinstance(possible_task_types, list) and len(possible_task_types) >= 2:
            reasons.append("multiple_possible_task_types")
        if signals.get("requires_a2a"):
            reasons.append("requires_a2a")
        if signals.get("requires_multi_step_execution"):
            reasons.append("requires_multi_step_execution")
        if signals.get("has_multi_deliverable"):
            reasons.append("multi_deliverable_request")
        if signals.get("requires_external_evidence") and task_profile.task_type != "knowledge_qa":
            reasons.append("external_evidence_required")
        if not reasons:
            return PlannerPolicyDecision(planner_required=False, bypass_reason="single workflow template is sufficient")
        return PlannerPolicyDecision(planner_required=True, reason_codes=sorted(set(reasons)))
