from __future__ import annotations

from typing import Dict, Optional, Type

from pydantic import BaseModel

from services.tool_service.schemas import (
    DocSearchArgs,
    EvidenceCheckArgs,
    InspectionArgs,
    MediaGenerationArgs,
    ToolDefinition,
)


TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {}
ARGUMENT_MODELS: Dict[str, Type[BaseModel]] = {}


def register(definition: ToolDefinition, argument_model: Type[BaseModel]) -> None:
    TOOL_DEFINITIONS[definition.tool_name] = definition
    ARGUMENT_MODELS[definition.tool_name] = argument_model


def all_tools() -> list[ToolDefinition]:
    return list(TOOL_DEFINITIONS.values())


def get_tool(tool_name: str) -> Optional[ToolDefinition]:
    return TOOL_DEFINITIONS.get(tool_name)


def get_argument_model(tool_name: str) -> Optional[Type[BaseModel]]:
    return ARGUMENT_MODELS.get(tool_name)


register(
    ToolDefinition(
        tool_name="doc_search_tool",
        description="Search project documentation through RAG service.",
        input_schema=DocSearchArgs.model_json_schema(),
        permission_scope="tool:doc_search:use",
        timeout_ms=3000,
    ),
    DocSearchArgs,
)
register(
    ToolDefinition(
        tool_name="evidence_check_tool",
        description="Check whether a claim is supported by evidence retrieved from RAG service.",
        input_schema=EvidenceCheckArgs.model_json_schema(),
        permission_scope="tool:evidence_check:use",
        timeout_ms=3000,
    ),
    EvidenceCheckArgs,
)
register(
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
register(
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
register(
    ToolDefinition(
        tool_name="inspection_tool",
        description="Check document structure and required sections.",
        input_schema=InspectionArgs.model_json_schema(),
        permission_scope="tool:inspection:use",
        timeout_ms=2000,
    ),
    InspectionArgs,
)
