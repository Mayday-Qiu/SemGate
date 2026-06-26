from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from app.runtime_metrics import RuntimeMetricsSnapshot, RuntimeMetricsStore
from app.schemas import (
    AgenticIntentDecision,
    AgenticResourceDecision,
    AgenticRouteDecision,
    AuthenticatedConsumer,
    CandidateWorkflowScore,
    ResourceCandidateScore,
    TaskProfile,
    WorkflowProfile,
)


WORKFLOW_MATCH_WEIGHTS: Dict[str, float] = {
    "capability_match": 0.30,
    "tool_match": 0.15,
    "permission_match": 0.15,
    "consumer_workflow_access": 0.10,
    "consumer_tool_access": 0.10,
    "historical_success": 0.10,
    "citation_coverage": 0.05,
    "risk_fit": 0.05,
}

RESOURCE_TIE_THRESHOLD = 0.05


class AgenticRouter:
    def __init__(self, runtime_metrics: Optional[RuntimeMetricsStore] = None) -> None:
        self._intent_router = TaskIntentRouter()
        self._workflow_matcher = WorkflowMatcher()
        self._resource_balancer = ResourceBalancer(runtime_metrics)

    def select(
        self,
        trace_id: str,
        task_profile: TaskProfile,
        workflow_profiles: Sequence[WorkflowProfile],
        user_permissions: Sequence[str],
        consumer: AuthenticatedConsumer,
    ) -> AgenticRouteDecision:
        intent_decision = self._intent_router.describe(task_profile)
        candidate_scores: List[CandidateWorkflowScore] = []
        excluded_candidates: List[Dict[str, str]] = []

        for profile in workflow_profiles:
            score = self._workflow_matcher.score_profile(
                task_profile=task_profile,
                workflow_profile=profile,
                user_permissions=set(user_permissions),
                consumer=consumer,
            )
            candidate_scores.append(score)
            if not score.executable:
                excluded_candidates.append(
                    {
                        "workflow_id": profile.workflow_id,
                        "reason": score.exclusion_reason or "not_executable",
                    }
                )

        profiles_by_id = {profile.workflow_id: profile for profile in workflow_profiles}
        executable_scores = [score for score in candidate_scores if score.executable]
        resource_decision = self._resource_balancer.select(executable_scores, profiles_by_id, task_profile)
        resource_scores = {
            target.workflow_id: target.resource_score for target in resource_decision.candidate_targets
        }
        for score in candidate_scores:
            if score.workflow_id in resource_scores:
                score.resource_score = resource_scores[score.workflow_id]
        selected_workflow = resource_decision.selected_workflow
        selection_reason = resource_decision.selection_reason
        if selected_workflow is None:
            selection_reason = "no executable workflow matched task profile and consumer access"

        return AgenticRouteDecision(
            trace_id=trace_id,
            selected_workflow=selected_workflow,
            candidate_scores=sorted(candidate_scores, key=lambda item: item.score, reverse=True),
            excluded_candidates=excluded_candidates,
            selection_reason=selection_reason,
            score_breakdown={
                "workflow_match": WORKFLOW_MATCH_WEIGHTS,
                "resource_balance": {
                    "strategy": "match_first_resource_tiebreak",
                    "tie_threshold": RESOURCE_TIE_THRESHOLD,
                    "selection_rule": "choose best workflow match first; use resource score only inside the tie band",
                    "runtime_metrics": [
                        "ongoing_count",
                        "recent_avg_latency_ms",
                        "recent_p95_latency_ms",
                        "recent_error_rate",
                    ],
                    "resource_score_weights": {
                        "latency_score": 0.45,
                        "reliability_score": 0.25,
                        "runtime_load_score": 0.20,
                        "cost_score": 0.10,
                    },
                },
            },
            intent_decision=intent_decision,
            resource_decision=resource_decision,
        )


class TaskIntentRouter:
    def describe(self, task_profile: TaskProfile) -> AgenticIntentDecision:
        return AgenticIntentDecision(
            mode="rule_based",
            task_type_hint=task_profile.task_type_hint,
            inferred_task_type=task_profile.task_type,
            matched_reasons=task_profile.profile_reason,
            llm_intent_enabled=False,
        )


class WorkflowMatcher:
    def score_profile(
        self,
        task_profile: TaskProfile,
        workflow_profile: WorkflowProfile,
        user_permissions: Set[str],
        consumer: AuthenticatedConsumer,
    ) -> CandidateWorkflowScore:
        capability_match = self._match_ratio(task_profile.required_capabilities, workflow_profile.capabilities)
        tool_match = self._match_ratio(task_profile.required_tools, workflow_profile.allowed_tools)
        permission_match = self._permission_match(task_profile, workflow_profile, user_permissions)
        consumer_workflow_access = self._consumer_workflow_access(consumer, workflow_profile)
        consumer_tool_access = self._consumer_tool_access(consumer, task_profile.required_tools)
        historical_success = workflow_profile.success_rate
        citation_coverage = workflow_profile.citation_coverage if task_profile.evidence_required else 1.0
        latency_score = ResourceBalancer.latency_score(task_profile, workflow_profile)
        cost_score = ResourceBalancer.cost_score(task_profile.cost_budget, workflow_profile.cost_level)
        risk_fit = self._risk_fit(task_profile, workflow_profile)

        match_score = (
            WORKFLOW_MATCH_WEIGHTS["capability_match"] * capability_match
            + WORKFLOW_MATCH_WEIGHTS["tool_match"] * tool_match
            + WORKFLOW_MATCH_WEIGHTS["permission_match"] * permission_match
            + WORKFLOW_MATCH_WEIGHTS["consumer_workflow_access"] * consumer_workflow_access
            + WORKFLOW_MATCH_WEIGHTS["consumer_tool_access"] * consumer_tool_access
            + WORKFLOW_MATCH_WEIGHTS["historical_success"] * historical_success
            + WORKFLOW_MATCH_WEIGHTS["citation_coverage"] * citation_coverage
            + WORKFLOW_MATCH_WEIGHTS["risk_fit"] * risk_fit
        )
        resource_score = ResourceBalancer.resource_score(task_profile, workflow_profile)
        exclusion_reason = self._exclusion_reason(
            task_profile=task_profile,
            workflow_profile=workflow_profile,
            capability_match=capability_match,
            permission_match=permission_match,
            consumer_workflow_access=consumer_workflow_access,
            consumer_tool_access=consumer_tool_access,
        )
        executable = exclusion_reason is None
        return CandidateWorkflowScore(
            workflow_id=workflow_profile.workflow_id,
            score=round(match_score, 6),
            match_score=round(match_score, 6),
            resource_score=round(resource_score, 6),
            executable=executable,
            exclusion_reason=exclusion_reason,
            selection_reason=self._selection_reason(
                workflow_profile,
                capability_match,
                tool_match,
                citation_coverage,
                resource_score,
            ),
            capability_match=round(capability_match, 6),
            tool_match=round(tool_match, 6),
            permission_match=round(permission_match, 6),
            consumer_workflow_access=round(consumer_workflow_access, 6),
            consumer_tool_access=round(consumer_tool_access, 6),
            historical_success=round(historical_success, 6),
            citation_coverage=round(citation_coverage, 6),
            latency_score=round(latency_score, 6),
            cost_score=round(cost_score, 6),
            risk_fit=round(risk_fit, 6),
        )

    def _match_ratio(self, required: Sequence[str], provided: Sequence[str]) -> float:
        if not required:
            return 1.0
        provided_set = set(provided)
        matched = sum(1 for item in required if item in provided_set)
        return matched / len(required)

    def _permission_match(
        self,
        task_profile: TaskProfile,
        workflow_profile: WorkflowProfile,
        user_permissions: Set[str],
    ) -> float:
        if "*" in user_permissions:
            return 1.0
        required_permissions = set(task_profile.permission_scope) | set(workflow_profile.required_permission_scope)
        if not required_permissions:
            return 1.0
        return 1.0 if required_permissions.issubset(user_permissions) else 0.0

    def _consumer_workflow_access(self, consumer: AuthenticatedConsumer, workflow_profile: WorkflowProfile) -> float:
        allowed = set(consumer.allowed_workflows)
        if "*" in allowed or workflow_profile.workflow_id in allowed:
            return 1.0
        return 0.0

    def _consumer_tool_access(self, consumer: AuthenticatedConsumer, required_tools: Sequence[str]) -> float:
        if not required_tools:
            return 1.0
        allowed = set(consumer.allowed_tools)
        if "*" in allowed:
            return 1.0
        return 1.0 if set(required_tools).issubset(allowed) else 0.0

    def _risk_fit(self, task_profile: TaskProfile, workflow_profile: WorkflowProfile) -> float:
        if task_profile.risk_level != "high":
            return 1.0
        required_controls = {"evidence_check", "citation", "permission_check"}
        missing_count = len(required_controls - set(workflow_profile.capabilities))
        if missing_count == 0:
            return 1.0
        if missing_count == 1:
            return 0.4
        return 0.0

    def _exclusion_reason(
        self,
        task_profile: TaskProfile,
        workflow_profile: WorkflowProfile,
        capability_match: float,
        permission_match: float,
        consumer_workflow_access: float,
        consumer_tool_access: float,
    ) -> Optional[str]:
        if not workflow_profile.healthy:
            return "workflow_unhealthy"
        if task_profile.task_type not in workflow_profile.supported_tasks:
            return "unsupported_task_type"
        if consumer_workflow_access == 0.0:
            return "consumer_workflow_denied"
        if consumer_tool_access == 0.0:
            return "consumer_tool_denied"
        if permission_match == 0.0:
            return "permission_denied"
        if capability_match < 0.5:
            return "capability_match_below_threshold"
        if task_profile.evidence_required:
            capabilities = set(workflow_profile.capabilities)
            if "evidence_check" not in capabilities or "citation" not in capabilities:
                return "missing_evidence_controls"
        return None

    def _selection_reason(
        self,
        workflow_profile: WorkflowProfile,
        capability_match: float,
        tool_match: float,
        citation_coverage: float,
        resource_score: float,
    ) -> str:
        return (
            f"{workflow_profile.workflow_id} match capability={capability_match:.2f}, "
            f"tool={tool_match:.2f}, citation={citation_coverage:.2f}; "
            f"resource_score={resource_score:.2f}, metrics_source={workflow_profile.metrics_source}"
        )


class ResourceBalancer:
    def __init__(self, runtime_metrics: Optional[RuntimeMetricsStore] = None) -> None:
        self._runtime_metrics = runtime_metrics or RuntimeMetricsStore()

    def select(
        self,
        executable_scores: Sequence[CandidateWorkflowScore],
        profiles_by_id: Dict[str, WorkflowProfile],
        task_profile: TaskProfile,
    ) -> AgenticResourceDecision:
        candidate_targets = [
            self._candidate_target(score, profiles_by_id[score.workflow_id], task_profile)
            for score in executable_scores
            if score.workflow_id in profiles_by_id
        ]
        if not candidate_targets:
            return AgenticResourceDecision(
                strategy="match_first_resource_tiebreak",
                selected_target=None,
                selected_workflow=None,
                selection_reason="no executable resource target",
                candidate_targets=[],
                tie_threshold=RESOURCE_TIE_THRESHOLD,
            )

        best_match = max(candidate_targets, key=lambda item: item.match_score).match_score
        tie_band = [
            item
            for item in candidate_targets
            if best_match - item.match_score <= RESOURCE_TIE_THRESHOLD
        ]
        selected = sorted(tie_band, key=lambda item: (item.resource_score, -item.avg_latency_ms), reverse=True)[0]
        return AgenticResourceDecision(
            strategy="match_first_resource_tiebreak",
            selected_target=selected.target_id,
            selected_workflow=selected.workflow_id,
            selection_reason=(
                f"{selected.workflow_id} selected by workflow match first; "
                f"resource tiebreak score={selected.resource_score:.2f}"
            ),
            candidate_targets=sorted(candidate_targets, key=lambda item: item.resource_score, reverse=True),
            tie_threshold=RESOURCE_TIE_THRESHOLD,
        )

    def _candidate_target(
        self,
        score: CandidateWorkflowScore,
        workflow_profile: WorkflowProfile,
        task_profile: TaskProfile,
    ) -> ResourceCandidateScore:
        runtime_snapshot = self._runtime_metrics.snapshot(workflow_profile.workflow_id)
        resource_score = self.resource_score_with_runtime(score, workflow_profile, task_profile, runtime_snapshot)
        effective_avg_latency_ms = runtime_snapshot.recent_avg_latency_ms or workflow_profile.avg_latency_ms
        effective_p95_latency_ms = runtime_snapshot.recent_p95_latency_ms or workflow_profile.p95_latency_ms
        effective_error_rate = runtime_snapshot.recent_error_rate
        if effective_error_rate is None:
            effective_error_rate = workflow_profile.error_rate
        return ResourceCandidateScore(
            target_id=workflow_profile.workflow_id,
            workflow_id=workflow_profile.workflow_id,
            match_score=score.match_score,
            resource_score=round(resource_score, 6),
            avg_latency_ms=effective_avg_latency_ms,
            p95_latency_ms=effective_p95_latency_ms,
            error_rate=round(effective_error_rate, 6),
            cost_level=workflow_profile.cost_level,
            runtime_load_score=self.runtime_load_score(runtime_snapshot.ongoing_count),
            ongoing_count=runtime_snapshot.ongoing_count,
            recent_sample_count=runtime_snapshot.recent_sample_count,
            recent_avg_latency_ms=runtime_snapshot.recent_avg_latency_ms,
            recent_p95_latency_ms=runtime_snapshot.recent_p95_latency_ms,
            recent_error_rate=runtime_snapshot.recent_error_rate,
            metrics_source="runtime_window" if runtime_snapshot.has_recent_data or runtime_snapshot.ongoing_count > 0 else workflow_profile.metrics_source,
        )

    @staticmethod
    def resource_score(task_profile: TaskProfile, workflow_profile: WorkflowProfile) -> float:
        reliability_score = max(0.0, 1.0 - workflow_profile.error_rate)
        runtime_load_score = 1.0
        score = (
            0.50 * ResourceBalancer.latency_score(task_profile, workflow_profile)
            + 0.25 * reliability_score
            + 0.15 * ResourceBalancer.cost_score(task_profile.cost_budget, workflow_profile.cost_level)
            + 0.10 * runtime_load_score
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def resource_score_with_runtime(
        candidate_score: CandidateWorkflowScore,
        workflow_profile: WorkflowProfile,
        task_profile: TaskProfile,
        runtime_snapshot: RuntimeMetricsSnapshot,
    ) -> float:
        effective_avg_latency_ms = runtime_snapshot.recent_avg_latency_ms or workflow_profile.avg_latency_ms
        effective_error_rate = runtime_snapshot.recent_error_rate
        if effective_error_rate is None:
            effective_error_rate = workflow_profile.error_rate
        latency_score = max(0.0, 1.0 - (effective_avg_latency_ms / task_profile.latency_slo_ms))
        reliability_score = max(0.0, 1.0 - effective_error_rate)
        runtime_load_score = ResourceBalancer.runtime_load_score(runtime_snapshot.ongoing_count)
        score = (
            0.45 * latency_score
            + 0.25 * reliability_score
            + 0.20 * runtime_load_score
            + 0.10 * candidate_score.cost_score
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def runtime_load_score(ongoing_count: int) -> float:
        return round(1.0 / (1.0 + max(0, ongoing_count)), 6)

    @staticmethod
    def latency_score(task_profile: TaskProfile, workflow_profile: WorkflowProfile) -> float:
        return max(0.0, 1.0 - (workflow_profile.avg_latency_ms / task_profile.latency_slo_ms))

    @staticmethod
    def cost_score(cost_budget: str, cost_level: str) -> float:
        base_scores = {"low": 1.0, "normal": 0.7, "high": 0.4}
        score = base_scores.get(cost_level, 0.7)
        if cost_budget == "low" and cost_level == "high":
            return score * 0.5
        return score
