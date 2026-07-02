from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from app.schemas import AgenticGatewayRequest, TaskProfile, WorkflowProfile
from services.tool_service.registry import all_tools


ALLOWED_A2A_PATHS = [
    {"from": "document_writing_workflow", "to": "knowledge_qa_workflow"},
    {"from": "document_writing_workflow", "to": "large_knowledge_qa_workflow"},
    {"from": "document_writing_workflow", "to": "media_generation_workflow"},
]


class PlannerContextBuilder:
    def build(
        self,
        *,
        request: AgenticGatewayRequest,
        task_profile: TaskProfile,
        workflow_profiles: Sequence[WorkflowProfile],
        user_permissions: Sequence[str],
    ) -> Dict[str, Any]:
        candidate_profiles = self._candidate_profiles(task_profile, workflow_profiles)
        candidate_workflow_ids = {profile.workflow_id for profile in candidate_profiles}
        candidate_tool_names = {
            tool_name
            for profile in candidate_profiles
            for tool_name in profile.allowed_tools
        }
        return {
            "context_version": "1.0",
            "user_request": request.input,
            "task_profile": {
                "task_type": task_profile.task_type,
                "task_type_hint": task_profile.task_type_hint,
                "rough_signals": task_profile.rough_signals,
                "required_capabilities": task_profile.required_capabilities,
                "required_tools": task_profile.required_tools,
            },
            "available_workflows": [self._workflow(profile) for profile in candidate_profiles],
            "available_tools": [
                self._tool(tool)
                for tool in all_tools()
                if tool.tool_name in candidate_tool_names
            ],
            "allowed_a2a_paths": [
                path
                for path in ALLOWED_A2A_PATHS
                if path["from"] in candidate_workflow_ids and path["to"] in candidate_workflow_ids
            ],
            "permission_context": {"allowed_scopes": sorted(user_permissions)},
            "hard_rules": [
                "Only use workflows listed in available_workflows.",
                "Only use tools listed in available_tools.",
                "document_writing_workflow must not directly call doc_search_tool.",
                "document_writing_workflow must not directly call evidence_check_tool.",
                "document_writing_workflow must obtain citations through a knowledge workflow.",
                "coding_workflow must not write files or execute shell commands in v1.0.",
                "media_generation_workflow must not directly access RAG.",
                "TaskPlanner must not answer the user request. It must only return TaskPlan JSON.",
            ],
        }

    def summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context_version": context.get("context_version"),
            "workflow_count": len(context.get("available_workflows", [])),
            "tool_count": len(context.get("available_tools", [])),
            "workflow_ids": [item.get("workflow") for item in context.get("available_workflows", [])],
            "tool_names": [item.get("tool") for item in context.get("available_tools", [])],
            "allowed_a2a_paths": context.get("allowed_a2a_paths", []),
            "rough_signals": context.get("task_profile", {}).get("rough_signals", {}),
        }

    def _candidate_profiles(
        self,
        task_profile: TaskProfile,
        workflow_profiles: Sequence[WorkflowProfile],
    ) -> List[WorkflowProfile]:
        candidate_ids = self._candidate_workflow_ids(task_profile)
        return [
            profile
            for profile in workflow_profiles
            if profile.workflow_id in candidate_ids and profile.healthy
        ]

    def _candidate_workflow_ids(self, task_profile: TaskProfile) -> Set[str]:
        if task_profile.task_type == "document_writing":
            return {
                "document_writing_workflow",
                "knowledge_qa_workflow",
                "large_knowledge_qa_workflow",
                "media_generation_workflow",
            }
        if task_profile.task_type == "knowledge_qa":
            return {"knowledge_qa_workflow", "large_knowledge_qa_workflow"}
        if task_profile.task_type == "media_generation":
            return {"media_generation_workflow"}
        if task_profile.task_type == "coding":
            return {"coding_workflow"}
        return set()

    def _workflow(self, profile: WorkflowProfile) -> Dict[str, Any]:
        return {
            "workflow": profile.workflow_id,
            "task_types": profile.supported_tasks,
            "capabilities": profile.capabilities,
            "allowed_tools": profile.allowed_tools,
            "required_permission_scope": profile.required_permission_scope,
            "status": "healthy" if profile.healthy else "unhealthy",
            "latency_class": self._latency_class(profile.avg_latency_ms),
            "cost_class": profile.cost_level,
        }

    def _tool(self, tool: Any) -> Dict[str, Any]:
        properties = tool.input_schema.get("properties", {}) if isinstance(tool.input_schema, dict) else {}
        return {
            "tool": tool.tool_name,
            "permission_scope": tool.permission_scope,
            "implementation_status": tool.implementation_status,
            "input_fields": sorted(properties.keys()),
        }

    def _latency_class(self, avg_latency_ms: float) -> str:
        if avg_latency_ms <= 1800:
            return "low"
        if avg_latency_ms <= 4000:
            return "medium"
        return "high"
