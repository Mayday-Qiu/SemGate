from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.schemas import (
    AgentWorkflowResponse,
    CriterionResult,
    TaskContract,
    TraceEvent,
    VerificationResult,
)


class VerificationGate:
    def verify(
        self,
        contract: TaskContract,
        response: AgentWorkflowResponse,
        tool_audit_log_path: Optional[Path] = None,
    ) -> VerificationResult:
        results = [self._check(criterion, response, tool_audit_log_path) for criterion in contract.acceptance_criteria]
        failed = [result for result in results if result.status == "failed"]
        passed = [result for result in results if result.status == "passed"]
        return VerificationResult(
            status="failed" if failed else "passed",
            passed_count=len(passed),
            failed_count=len(failed),
            criteria_results=results,
            failure_reason=failed[0].reason if failed else None,
        )

    def _check(
        self,
        criterion: Any,
        response: AgentWorkflowResponse,
        tool_audit_log_path: Optional[Path],
    ) -> CriterionResult:
        ok, reason = self._evaluate(criterion, response, tool_audit_log_path)
        if ok:
            status = "passed"
        elif criterion.required:
            status = "failed"
        else:
            status = "skipped"
        return CriterionResult(
            criterion_id=criterion.criterion_id,
            type=criterion.type,
            target=criterion.target,
            status=status,
            reason=reason,
        )

    def _evaluate(
        self,
        criterion: Any,
        response: AgentWorkflowResponse,
        tool_audit_log_path: Optional[Path],
    ) -> tuple[bool, str]:
        if criterion.type == "schema_field_required":
            value = getattr(response, criterion.target, None)
            if value:
                return True, "field present"
            return False, f"missing or empty field: {criterion.target}"

        if criterion.type == "trace_event_required":
            if any(event.node == criterion.target for event in response.trace_events):
                return True, "trace node present"
            return False, f"missing trace node: {criterion.target}"

        if criterion.type == "tool_success_required":
            if tool_audit_log_path is not None:
                if _tool_audit_has_success(tool_audit_log_path, response.trace_id, criterion.target):
                    return True, "tool success found in audit log"
                return False, f"missing tool audit success: trace_id={response.trace_id}, tool={criterion.target}"
            if self._tool_succeeded(criterion.target, response):
                return True, "tool succeeded"
            return False, f"tool did not succeed: {criterion.target}"

        if criterion.type == "citation_required":
            min_count = int(criterion.params.get("min_count", 1))
            if len(response.citations) >= min_count:
                return True, "citation count satisfied"
            return False, f"citations={len(response.citations)}, required>={min_count}"

        if criterion.type == "citation_source_required":
            ok, reason = _citations_have_required_source(response, str(criterion.params.get("source", "")))
            return ok, reason

        if criterion.type == "resource_forbidden":
            forbidden = set(str(item) for item in criterion.params.get("names", []))
            used = set(response.tools)
            blocked = sorted(forbidden & used)
            if not blocked:
                return True, "forbidden resources not used"
            return False, f"forbidden resources used: {blocked}"

        if criterion.type == "metadata_required":
            if criterion.target in response.metadata and response.metadata.get(criterion.target) is not None:
                return True, "metadata present"
            return False, f"missing metadata: {criterion.target}"

        return False, f"unknown criterion type: {criterion.type}"

    def _tool_succeeded(self, tool_name: str, response: AgentWorkflowResponse) -> bool:
        tool_statuses = response.metadata.get("tool_statuses")
        if isinstance(tool_statuses, dict) and tool_statuses.get(tool_name) == "success":
            return True
        return tool_name in response.tools and _has_success_tool_event(response.trace_events, tool_name)


def _has_success_tool_event(events: Sequence[TraceEvent], tool_name: str) -> bool:
    for event in events:
        if event.event_type != "tool_call" or event.status != "success":
            continue
        tool_response = event.metadata.get("tool_response")
        if isinstance(tool_response, dict) and tool_response.get("tool_name") == tool_name:
            return True
        if tool_name in event.output_summary:
            return True
    return False


def _tool_audit_has_success(path: Path, trace_id: str, tool_name: str, recent_limit: int = 2000) -> bool:
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in reversed(lines[-recent_limit:]):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("trace_id") == trace_id
            and record.get("tool_name") == tool_name
            and record.get("status") == "success"
        ):
            return True
    return False


def _citations_have_required_source(response: AgentWorkflowResponse, source: str) -> tuple[bool, str]:
    if not response.citations:
        return False, "missing citations"
    citation_keys = {(item.source_id, item.chunk_id) for item in response.citations}
    if source == "a2a_knowledge":
        if response.metadata.get("direct_rag_access") is True:
            return False, "document workflow used direct RAG"
        context = response.metadata.get("knowledge_context")
        refs = _ref_keys(context.get("citations", [])) if isinstance(context, dict) else set()
        if citation_keys and citation_keys.issubset(refs):
            return True, "citations sourced from A2A knowledge context"
        return False, "citations not found in A2A knowledge context"

    refs = set()
    for key in ("citation_refs", "evidence_refs"):
        refs.update(_ref_keys(response.metadata.get(key, [])))
    for event in response.trace_events:
        refs.update(_ref_keys(event.metadata.get("citation_refs", [])))
        refs.update(_ref_keys(event.metadata.get("evidence_refs", [])))
    if citation_keys and citation_keys.issubset(refs):
        return True, "citations sourced from retrieved evidence"
    return False, "citations not found in evidence refs"


def _ref_keys(items: Any) -> set[tuple[str, str]]:
    if not isinstance(items, list):
        return set()
    keys = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        if source_id and chunk_id:
            keys.add((source_id, chunk_id))
    return keys
