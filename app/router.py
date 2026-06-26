from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.profiles import ServiceProfileStore
from app.schemas import BackendInfo, ServiceProfile, TaskType


@dataclass(frozen=True)
class TaskRoutingConfig:
    quality_weight: float
    latency_weight: float
    reliability_weight: float
    task_slo_ms: float


@dataclass
class CandidateRoute:
    backend: BackendInfo
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend.backend_id,
            "score": self.score,
            "details": self.metadata,
        }


@dataclass
class RoutingDecision:
    backend: Optional[BackendInfo]
    policy: str
    reason: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ordered_candidates: List[CandidateRoute] = field(default_factory=list)


TASK_ROUTING_CONFIGS: Dict[str, TaskRoutingConfig] = {
    "summary": TaskRoutingConfig(
        quality_weight=0.30,
        latency_weight=0.50,
        reliability_weight=0.20,
        task_slo_ms=500.0,
    ),
    "semantic_explain": TaskRoutingConfig(
        quality_weight=0.45,
        latency_weight=0.30,
        reliability_weight=0.25,
        task_slo_ms=800.0,
    ),
    "rag_qa": TaskRoutingConfig(
        quality_weight=0.60,
        latency_weight=0.20,
        reliability_weight=0.20,
        task_slo_ms=1200.0,
    ),
    "tool_call": TaskRoutingConfig(
        quality_weight=0.25,
        latency_weight=0.20,
        reliability_weight=0.55,
        task_slo_ms=1000.0,
    ),
    "text_to_video": TaskRoutingConfig(
        quality_weight=0.75,
        latency_weight=0.10,
        reliability_weight=0.15,
        task_slo_ms=3000.0,
    ),
}


class Router:
    def __init__(self, profile_store: ServiceProfileStore) -> None:
        self._profile_store = profile_store
        self._round_robin_positions: Dict[str, int] = {}

    def select(
        self,
        policy: str,
        candidates: List[BackendInfo],
        task_type: TaskType,
    ) -> RoutingDecision:
        if not candidates:
            return RoutingDecision(
                backend=None,
                policy=policy,
                reason="no_candidate_backend",
            )

        if policy == "fixed":
            return self._select_fixed(policy, candidates)
        if policy == "round_robin":
            return self._select_round_robin(policy, candidates, task_type)
        if policy == "profile_aware":
            return self._select_profile_aware(policy, candidates, task_type)

        decision = self._select_fixed("fixed", candidates)
        decision.policy = policy
        decision.reason = "unknown_policy_fallback_to_fixed"
        decision.metadata["fallback_policy"] = "fixed"
        return decision

    def _select_fixed(self, policy: str, candidates: List[BackendInfo]) -> RoutingDecision:
        ordered_candidates = [CandidateRoute(backend=backend) for backend in candidates]
        return RoutingDecision(
            backend=candidates[0],
            policy=policy,
            reason="first_candidate",
            ordered_candidates=ordered_candidates,
        )

    def _select_round_robin(
        self,
        policy: str,
        candidates: List[BackendInfo],
        task_type: TaskType,
    ) -> RoutingDecision:
        position_key = str(task_type)
        current_position = self._round_robin_positions.get(position_key, 0)
        selected_index = current_position % len(candidates)
        self._round_robin_positions[position_key] = selected_index + 1
        ordered_backends = candidates[selected_index:] + candidates[:selected_index]
        ordered_candidates = [CandidateRoute(backend=backend) for backend in ordered_backends]
        return RoutingDecision(
            backend=candidates[selected_index],
            policy=policy,
            reason="task_scoped_round_robin",
            metadata={
                "round_robin_index": selected_index,
                "candidate_count": len(candidates),
            },
            ordered_candidates=ordered_candidates,
        )

    def _select_profile_aware(
        self,
        policy: str,
        candidates: List[BackendInfo],
        task_type: TaskType,
    ) -> RoutingDecision:
        scored_candidates: List[CandidateRoute] = []
        unscored_candidates: List[CandidateRoute] = []
        for backend in candidates:
            profile = self._profile_store.get(backend.backend_id, task_type)
            if profile is None:
                unscored_candidates.append(
                    CandidateRoute(
                        backend=backend,
                        metadata={"profile_status": "missing"},
                    )
                )
                continue
            score_metadata = self._score_profile(profile)
            scored_candidates.append(
                CandidateRoute(
                    backend=backend,
                    score=score_metadata["final_score"],
                    metadata=score_metadata,
                )
            )

        if not scored_candidates:
            decision = self._select_fixed("fixed", candidates)
            decision.policy = policy
            decision.reason = "missing_profiles_fallback_to_fixed"
            decision.metadata["fallback_policy"] = "fixed"
            return decision

        ordered_candidates = sorted(scored_candidates, key=lambda item: item.score or 0.0, reverse=True)
        ordered_candidates.extend(unscored_candidates)
        selected = ordered_candidates[0]
        return RoutingDecision(
            backend=selected.backend,
            policy=policy,
            reason="highest_profile_score",
            score=selected.score,
            metadata=selected.metadata,
            ordered_candidates=ordered_candidates,
        )

    def _score_profile(self, profile: ServiceProfile) -> Dict[str, Any]:
        config = TASK_ROUTING_CONFIGS[str(profile.task_type)]
        quality_score = profile.heuristic_quality_score
        latency_score = max(0.0, 1.0 - (profile.p95_latency_ms / config.task_slo_ms))
        reliability_score = max(0.0, 1.0 - profile.error_rate - profile.timeout_rate)
        final_score = (
            config.quality_weight * quality_score
            + config.latency_weight * latency_score
            + config.reliability_weight * reliability_score
        )
        return {
            "final_score": round(final_score, 6),
            "quality_score": round(quality_score, 6),
            "latency_score": round(latency_score, 6),
            "reliability_score": round(reliability_score, 6),
            "quality_weight": config.quality_weight,
            "latency_weight": config.latency_weight,
            "reliability_weight": config.reliability_weight,
            "task_slo_ms": config.task_slo_ms,
            "profile_backend_id": profile.backend_id,
            "profile_task_type": profile.task_type,
        }
