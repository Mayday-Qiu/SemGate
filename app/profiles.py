from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.schemas import ServiceProfile, TaskType


class ServiceProfileStore:
    def __init__(self, profiles: List[ServiceProfile]) -> None:
        self._profiles: Dict[Tuple[str, str], ServiceProfile] = {
            (profile.backend_id, profile.task_type): profile for profile in profiles
        }

    def get(self, backend_id: str, task_type: TaskType) -> Optional[ServiceProfile]:
        return self._profiles.get((backend_id, task_type))

    def all(self) -> List[ServiceProfile]:
        return list(self._profiles.values())


def load_service_profiles(path: Path) -> ServiceProfileStore:
    if not path.exists():
        return ServiceProfileStore([])

    with path.open("r", encoding="utf-8") as file:
        raw_payload = json.load(file)

    if isinstance(raw_payload, dict):
        raw_profiles = raw_payload.get("profiles", [])
    else:
        raw_profiles = raw_payload

    profiles = [ServiceProfile.model_validate(item) for item in raw_profiles]
    return ServiceProfileStore(profiles)
