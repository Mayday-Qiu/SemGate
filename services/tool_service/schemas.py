from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ToolStatus = Literal["success", "permission_denied", "schema_error", "timeout", "failed"]
InspectionType = Literal[
    "document_structure",
    "policy_compliance",
    "caption_quality",
    "task_feasibility",
    "evidence_consistency",
]


class ToolInvokeRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "unknown_user"
    consumer_id: str = "unknown_consumer"
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)


class ToolInvokeResponse(BaseModel):
    tool_name: str
    status: ToolStatus
    result: Dict[str, Any] = Field(default_factory=dict)
    required_permission: Optional[str] = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    error_type: Optional[str] = None
    error: Optional[str] = None


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=0, ge=0)
    backoff_ms: int = Field(default=0, ge=0)


class ToolDefinition(BaseModel):
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    permission_scope: str
    timeout_ms: int = Field(default=3000, ge=1)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    implementation_status: Literal["active", "placeholder"] = "active"


class DocSearchArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    strategy: Literal["vector", "keyword", "hybrid"] = "hybrid"


class EvidenceCheckArgs(BaseModel):
    claim: str = Field(min_length=1)
    evidence_query: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class BusinessQueryArgs(BaseModel):
    entity_type: Literal["requirement", "api", "ticket", "service_status"]
    query: str = Field(min_length=1)


class ServiceStatusArgs(BaseModel):
    service_name: str = Field(min_length=1)
    simulate_failure: Optional[Literal["timeout", "failed"]] = None


class InspectionArgs(BaseModel):
    inspection_type: InspectionType
    content: str = Field(min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)


class MediaGenerationArgs(BaseModel):
    prompt: str = Field(min_length=1)
    media_type: Literal["image", "video"]
