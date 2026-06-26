from __future__ import annotations

from typing import Dict, Optional, Type

from pydantic import BaseModel

from services.tool_service.schemas import (
    BusinessQueryArgs,
    DocSearchArgs,
    EvidenceCheckArgs,
    InspectionArgs,
    MediaGenerationArgs,
    ServiceStatusArgs,
    ToolDefinition,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: Dict[str, ToolDefinition] = {}
        self._argument_models: Dict[str, Type[BaseModel]] = {}
        self._register(
            ToolDefinition(
                tool_name="doc_search_tool",
                description="Search project documentation through RAG service.",
                input_schema=DocSearchArgs.model_json_schema(),
                permission_scope="tool:doc_search:use",
                timeout_ms=3000,
            ),
            DocSearchArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="business_query_tool",
                description="Query mock business entities such as requirements, APIs, tickets, and service status.",
                input_schema=BusinessQueryArgs.model_json_schema(),
                permission_scope="tool:business:read",
                timeout_ms=3000,
            ),
            BusinessQueryArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="service_status_tool",
                description="Query mock service health, latency, and current incident status.",
                input_schema=ServiceStatusArgs.model_json_schema(),
                permission_scope="tool:service_status:read",
                timeout_ms=3000,
            ),
            ServiceStatusArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="inspection_tool",
                description="Run structured inspections for documents, policy compliance, caption quality, task feasibility, or evidence consistency.",
                input_schema=InspectionArgs.model_json_schema(),
                permission_scope="tool:inspection:use",
                timeout_ms=3000,
            ),
            InspectionArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="evidence_check_tool",
                description="Check whether a claim is supported by evidence retrieved from RAG service.",
                input_schema=EvidenceCheckArgs.model_json_schema(),
                permission_scope="tool:evidence_check:use",
                timeout_ms=3000,
            ),
            EvidenceCheckArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="image_generation_tool",
                description="Placeholder interface for future text-to-image generation backend.",
                input_schema=MediaGenerationArgs.model_json_schema(),
                permission_scope="tool:image_generation:use",
                timeout_ms=10000,
                implementation_status="placeholder",
            ),
            MediaGenerationArgs,
        )
        self._register(
            ToolDefinition(
                tool_name="video_generation_tool",
                description="Placeholder interface for future text-to-video generation backend.",
                input_schema=MediaGenerationArgs.model_json_schema(),
                permission_scope="tool:video_generation:use",
                timeout_ms=10000,
                implementation_status="placeholder",
            ),
            MediaGenerationArgs,
        )

    def all(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(tool_name)

    def argument_model(self, tool_name: str) -> Optional[Type[BaseModel]]:
        return self._argument_models.get(tool_name)

    def _register(self, definition: ToolDefinition, argument_model: Type[BaseModel]) -> None:
        self._definitions[definition.tool_name] = definition
        self._argument_models[definition.tool_name] = argument_model


registry = ToolRegistry()
