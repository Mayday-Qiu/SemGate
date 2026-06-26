from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.schemas import WorkflowProfile


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
