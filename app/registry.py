from __future__ import annotations

from typing import List

from app.config import Settings
from app.schemas import BackendInfo, TaskType


class BackendRegistry:
    def __init__(self, backends: List[BackendInfo]) -> None:
        self._backends = {backend.backend_id: backend for backend in backends}

    def all(self) -> List[BackendInfo]:
        return list(self._backends.values())

    def healthy_for_task(self, task_type: TaskType) -> List[BackendInfo]:
        return [
            backend
            for backend in self._backends.values()
            if backend.healthy and task_type in backend.supported_tasks
        ]

def load_default_registry(settings: Settings) -> BackendRegistry:
    return BackendRegistry(
        [
            BackendInfo(
                backend_id="mock_fast",
                backend_type="mock",
                url=settings.mock_fast_url,
                supported_tasks=["summary", "semantic_explain"],
            ),
            BackendInfo(
                backend_id="mock_quality",
                backend_type="mock",
                url=settings.mock_quality_url,
                supported_tasks=["summary", "semantic_explain", "rag_qa", "text_to_video"],
            ),
            BackendInfo(
                backend_id="mock_unstable",
                backend_type="mock",
                url=settings.mock_unstable_url,
                supported_tasks=["summary", "semantic_explain", "rag_qa", "tool_call", "text_to_video"],
            ),
        ]
    )
