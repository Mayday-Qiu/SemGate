from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.schemas import (
    AcceptanceCriterion,
    AgenticGatewayRequest,
    PreviewStatus,
    TaskContract,
    TaskProfile,
)


class TaskContractBuilder:
    def build(
        self,
        request: AgenticGatewayRequest,
        task_profile: TaskProfile,
        selected_workflow: Optional[str],
        contract_patches: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> TaskContract:
        acceptance_criteria = self._criteria(task_profile, selected_workflow)
        forbidden_resources = self._forbidden_resources(task_profile.task_type)
        contract_id = _contract_id(request, task_profile, selected_workflow)
        contract = TaskContract(
            contract_id=contract_id,
            task_type=task_profile.task_type,
            selected_workflow=selected_workflow,
            required_capabilities=list(task_profile.required_capabilities),
            required_tools=list(task_profile.required_tools),
            required_permissions=list(task_profile.permission_scope),
            allowed_resources={
                "workflows": [selected_workflow] if selected_workflow else [],
                "tools": list(task_profile.required_tools),
            },
            forbidden_resources=forbidden_resources,
            output_schema=self._output_schema(task_profile.task_type),
            acceptance_criteria=acceptance_criteria,
            latency_slo_ms=task_profile.latency_slo_ms,
            cost_budget=task_profile.cost_budget,
        )
        self._apply_contract_patches(contract, contract_patches or [])
        return contract

    def preview(
        self,
        contract: TaskContract,
        user_permissions: Sequence[str],
        selection_reason: str,
    ) -> Tuple[PreviewStatus, List[str], List[str]]:
        missing_permissions = self.missing_permissions(contract, user_permissions)
        if missing_permissions:
            return (
                "blocked",
                [f"missing permissions: {', '.join(missing_permissions)}"],
                ["grant missing scopes or choose a lower-privilege workflow"],
            )
        if contract.selected_workflow is None:
            return "blocked", [selection_reason], ["fix workflow access or task profile"]
        if not contract.acceptance_criteria:
            return "warning", ["no acceptance criteria generated"], ["add at least one acceptance criterion"]
        return "ready", [selection_reason], ["invoke /v1/invoke with the same payload"]

    def missing_permissions(self, contract: TaskContract, user_permissions: Sequence[str]) -> List[str]:
        permission_set = set(user_permissions)
        if "*" in permission_set:
            return []
        return sorted(set(contract.required_permissions) - permission_set)

    def _criteria(self, task_profile: TaskProfile, selected_workflow: Optional[str]) -> List[AcceptanceCriterion]:
        criteria = [
            AcceptanceCriterion(
                criterion_id="answer_present",
                type="schema_field_required",
                target="answer",
                description="Workflow must return a non-empty answer.",
            )
        ]
        for node in self._required_nodes(task_profile.task_type, selected_workflow):
            criteria.append(
                AcceptanceCriterion(
                    criterion_id=f"trace_node:{node}",
                    type="trace_event_required",
                    target=node,
                    description=f"Trace must include node {node}.",
                )
            )
        for tool_name in task_profile.required_tools:
            criteria.append(
                AcceptanceCriterion(
                    criterion_id=f"tool_success:{tool_name}",
                    type="tool_success_required",
                    target=tool_name,
                    description=f"Tool {tool_name} must complete successfully.",
                )
            )
        if task_profile.evidence_required:
            criteria.append(
                AcceptanceCriterion(
                    criterion_id="citation:min1",
                    type="citation_required",
                    target="citations",
                    params={"min_count": 1},
                    description="Evidence-required tasks must return at least one citation.",
                )
            )
        for tool_name in self._forbidden_resources(task_profile.task_type).get("tools", []):
            criteria.append(
                AcceptanceCriterion(
                    criterion_id=f"forbidden_tool:{tool_name}",
                    type="resource_forbidden",
                    target="tools",
                    params={"names": [tool_name]},
                    description=f"Workflow must not directly use {tool_name}.",
                )
            )
        return criteria

    def _required_nodes(self, task_type: str, selected_workflow: Optional[str]) -> List[str]:
        if selected_workflow == "large_knowledge_qa_workflow":
            return [
                "ParseDeepQuestion",
                "BuildResearchBrief",
                "SplitSubQuestions",
                "ParallelRetrieveEvidence",
                "EvidenceAggregate",
                "SynthesizeAnswer",
                "CitationCheck",
            ]
        return {
            "knowledge_qa": ["ParseQuestion", "PlanRetrieval", "RetrieveEvidence", "EvidenceCheck", "GenerateAnswer", "CitationCheck"],
            "coding": ["ParseCodingTask", "BuildCodeContext", "InvokeCodingModel", "StructurePatchOrAnswer", "CodingOutputCheck"],
            "media_generation": ["ParseMediaRequest", "PromptSafetyCheck", "BuildGenerationParams", "InvokeMediaBackend", "AssetMetadataBuild", "MediaOutputCheck"],
            "document_writing": ["ParseWritingTask", "BuildDocumentPlan", "NeedKnowledge", "NeedMedia", "ComposeDocument", "InspectionToolCheck", "DocumentVerification"],
        }.get(task_type, [])

    def _forbidden_resources(self, task_type: str) -> Dict[str, List[str]]:
        if task_type == "document_writing":
            return {"tools": ["doc_search_tool", "evidence_check_tool"]}
        return {"tools": []}

    def _output_schema(self, task_type: str) -> Dict[str, Any]:
        common = {"required": ["answer"], "properties": {"answer": "string"}}
        if task_type == "knowledge_qa":
            common["required"] = ["answer", "citations"]
            common["properties"]["citations"] = "list[Citation]"
        if task_type == "media_generation":
            common["properties"]["asset_metadata"] = "metadata.placeholder"
        return common

    def _apply_contract_patches(self, contract: TaskContract, patches: Sequence[Dict[str, Any]]) -> None:
        for patch in patches:
            for tool_name in _as_list(patch.get("add_required_tools")):
                tool = str(tool_name)
                _append_unique(contract.required_tools, tool)
                _append_unique(contract.allowed_resources.setdefault("tools", []), tool)
                self._append_criterion(
                    contract,
                    AcceptanceCriterion(
                        criterion_id=f"memory:{patch.get('rule_id', 'rule')}:tool_success:{tool}",
                        type="tool_success_required",
                        target=tool,
                        description=f"MemoryPlanner requires tool {tool} to complete successfully.",
                    ),
                )
            for tool_name in _as_list(patch.get("add_forbidden_tools")):
                tool = str(tool_name)
                _append_unique(contract.forbidden_resources.setdefault("tools", []), tool)
                self._append_criterion(
                    contract,
                    AcceptanceCriterion(
                        criterion_id=f"memory:{patch.get('rule_id', 'rule')}:forbidden_tool:{tool}",
                        type="resource_forbidden",
                        target="tools",
                        params={"names": [tool]},
                        description=f"MemoryPlanner requires workflow not to directly use {tool}.",
                    ),
                )
            for node in _as_list(patch.get("add_required_trace_events")):
                node_name = str(node)
                self._append_criterion(
                    contract,
                    AcceptanceCriterion(
                        criterion_id=f"memory:{patch.get('rule_id', 'rule')}:trace_node:{node_name}",
                        type="trace_event_required",
                        target=node_name,
                        description=f"MemoryPlanner requires trace node {node_name}.",
                    ),
                )
            for raw_criterion in _as_list(patch.get("add_acceptance_criteria")):
                if not isinstance(raw_criterion, dict):
                    continue
                criterion = AcceptanceCriterion(
                    criterion_id=str(raw_criterion.get("criterion_id", f"memory:{patch.get('rule_id', 'rule')}:{raw_criterion.get('type', 'criterion')}:{raw_criterion.get('target', 'target')}")),
                    type=raw_criterion["type"],
                    target=str(raw_criterion.get("target", "")),
                    required=bool(raw_criterion.get("required", True)),
                    params=dict(raw_criterion.get("params", {})),
                    description=str(raw_criterion.get("description", "Added by MemoryPlanner.")),
                )
                self._append_criterion(contract, criterion)

    def _append_criterion(self, contract: TaskContract, criterion: AcceptanceCriterion) -> None:
        if any(existing.criterion_id == criterion.criterion_id for existing in contract.acceptance_criteria):
            return
        contract.acceptance_criteria.append(criterion)


def _contract_id(
    request: AgenticGatewayRequest,
    task_profile: TaskProfile,
    selected_workflow: Optional[str],
) -> str:
    raw = "|".join(
        [
            request.request_id or "",
            request.user_id,
            request.tenant_id,
            task_profile.task_type,
            selected_workflow or "",
            request.input,
        ]
    )
    return f"contract_{sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _as_list(raw_value: Any) -> List[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    return [raw_value]
