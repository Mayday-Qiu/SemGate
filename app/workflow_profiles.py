from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from app.schemas import WorkflowProfile
from services.tool_service.registry import get_argument_model, get_tool


class WorkflowProfileStore:
    def __init__(self, profiles: List[WorkflowProfile]) -> None:
        self._profiles = profiles
        self._by_id = {profile.workflow_id: profile for profile in profiles}

    def all(self) -> List[WorkflowProfile]:
        return list(self._profiles)

    def get(self, workflow_id: str) -> Optional[WorkflowProfile]:
        return self._by_id.get(workflow_id)


def load_workflow_profiles(path: Path) -> WorkflowProfileStore:
    if not path.exists():
        return WorkflowProfileStore([])

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_profiles = raw.get("profiles", raw if isinstance(raw, list) else [])
    profiles = [WorkflowProfile.model_validate(item) for item in raw_profiles]
    return WorkflowProfileStore(profiles)


def validate_workflow_tool_definitions(
    store: WorkflowProfileStore,
    required_tools: Sequence[str] = (),
) -> None:
    errors = []
    tool_contexts = []
    for profile in store.all():
        for tool_name in profile.allowed_tools:
            tool_contexts.append((profile.workflow_id, tool_name))
    for tool_name in sorted(set(required_tools)):
        tool_contexts.append(("task_profile", tool_name))

    for context, tool_name in tool_contexts:
        definition = get_tool(tool_name)
        argument_model = get_argument_model(tool_name)
        if definition is None:
            errors.append(f"{context}: unknown tool {tool_name}")
            continue
        if argument_model is None:
            errors.append(f"{context}: missing argument model for {tool_name}")
        if not isinstance(definition.input_schema, dict) or not definition.input_schema:
            errors.append(f"{context}: empty input schema for {tool_name}")
    if errors:
        raise ValueError("Invalid workflow tool definitions: " + "; ".join(errors))
