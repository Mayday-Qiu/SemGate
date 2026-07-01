from __future__ import annotations

from typing import Any, Dict, List, Set

from app.schemas import AgenticGatewayRequest, AgenticTaskType, CostBudget, TaskProfile


DEMO_FULL_PERMISSIONS = {
    "kb:project_docs:read",
    "tool:doc_search:use",
    "tool:evidence_check:use",
    "tool:image_generation:use",
    "tool:video_generation:use",
    "tool:inspection:use",
    "model:coding:use",
    "model:writing:use",
}

LIMITED_PERMISSIONS = {
    "kb:project_docs:read",
    "tool:doc_search:use",
}


class TaskProfileBuilder:
    def build(self, request: AgenticGatewayRequest) -> TaskProfile:
        inferred_task_type, reasons = self._infer_task_type(request)
        metadata = request.metadata
        profile_seed = self._profile_seed(inferred_task_type, request)

        evidence_required = bool(metadata.get("evidence_required", profile_seed["evidence_required"]))
        latency_slo_ms = self._metadata_float(metadata, "latency_slo_ms", 5000.0)
        cost_budget = self._metadata_cost_budget(metadata.get("cost_budget", "normal"))

        if request.task_type != inferred_task_type:
            reasons.append(
                f"task_type_hint={request.task_type} overridden_by_inference={inferred_task_type}"
            )

        return TaskProfile(
            task_type=inferred_task_type,
            required_capabilities=profile_seed["required_capabilities"],
            required_tools=profile_seed["required_tools"],
            evidence_required=evidence_required,
            risk_level=profile_seed["risk_level"],
            latency_slo_ms=latency_slo_ms,
            cost_budget=cost_budget,
            permission_scope=profile_seed["permission_scope"],
            priority=request.priority,
            task_type_hint=request.task_type,
            profile_reason=reasons,
        )

    def resolve_user_permissions(self, request: AgenticGatewayRequest) -> List[str]:
        metadata_permissions = request.metadata.get("permissions")
        if isinstance(metadata_permissions, list):
            return [str(item) for item in metadata_permissions]
        if request.user_id in {"limited_user", "u_limited", "u_denied"}:
            return sorted(LIMITED_PERMISSIONS)
        return sorted(DEMO_FULL_PERMISSIONS)

    def known_required_tools(self) -> List[str]:
        sample_requests = [
            AgenticGatewayRequest(user_id="_startup", tenant_id="_startup", task_type="knowledge_qa", input="_startup"),
            AgenticGatewayRequest(user_id="_startup", tenant_id="_startup", task_type="coding", input="_startup"),
            AgenticGatewayRequest(
                user_id="_startup",
                tenant_id="_startup",
                task_type="media_generation",
                input="_startup",
                metadata={"media_type": "image"},
            ),
            AgenticGatewayRequest(
                user_id="_startup",
                tenant_id="_startup",
                task_type="media_generation",
                input="_startup",
                metadata={"media_type": "video"},
            ),
            AgenticGatewayRequest(user_id="_startup", tenant_id="_startup", task_type="document_writing", input="_startup"),
        ]
        tools = set()
        for request in sample_requests:
            tools.update(self._profile_seed(request.task_type, request)["required_tools"])
        return sorted(tools)

    def _infer_task_type(self, request: AgenticGatewayRequest) -> tuple[AgenticTaskType, List[str]]:
        text = f"{request.input} {request.metadata}".lower()
        reasons: List[str] = []

        knowledge_keywords = {
            "knowledge base",
            "citation",
            "source",
            "evidence",
            "project docs",
            "according to docs",
            "知识库",
            "引用",
            "出处",
            "证据",
            "根据文档",
        }
        coding_keywords = {
            "code",
            "coding",
            "debug",
            "unit test",
            "pull request",
            "python",
            "typescript",
            "代码",
            "调试",
            "单元测试",
            "修复 bug",
            "接口实现",
        }
        document_writing_keywords = {
            "write document",
            "technical document",
            "proposal",
            "report",
            "draft",
            "文档写作",
            "技术文档",
            "方案",
            "报告",
            "论文",
            "撰写",
        }
        media_generation_keywords = {
            "generate image",
            "generate video",
            "image generation",
            "video generation",
            "text to image",
            "text to video",
            "poster",
            "storyboard",
            "thumbnail",
            "生成图片",
            "生成图像",
            "生成视频",
            "文生图",
            "文生视频",
            "海报",
            "分镜",
            "短视频",
            "封面图",
        }
        media_generation_verbs = {"generate", "create", "make", "生成", "制作", "设计"}
        media_generation_nouns = {
            "image",
            "video",
            "poster",
            "storyboard",
            "thumbnail",
            "图片",
            "图像",
            "视频",
            "海报",
            "分镜",
            "短视频",
            "封面图",
        }

        if self._contains_any(text, media_generation_keywords) or (
            self._contains_any(text, media_generation_verbs)
            and self._contains_any(text, media_generation_nouns)
        ):
            reasons.append("matched media-generation keywords")
            return "media_generation", reasons
        if self._contains_any(text, coding_keywords):
            reasons.append("matched coding keywords")
            return "coding", reasons
        if self._contains_any(text, document_writing_keywords):
            reasons.append("matched document-writing keywords")
            return "document_writing", reasons
        if self._contains_any(text, knowledge_keywords):
            reasons.append("matched knowledge/citation keywords")
            return "knowledge_qa", reasons
        if request.task_type in {"knowledge_qa", "coding", "media_generation", "document_writing"}:
            reasons.append("used client task_type hint because no stronger rule matched")
            return request.task_type, reasons

        reasons.append("defaulted to knowledge_qa")
        return "knowledge_qa", reasons

    def _profile_seed(self, task_type: AgenticTaskType, request: AgenticGatewayRequest) -> Dict[str, Any]:
        seeds: Dict[str, Dict[str, Any]] = {
            "knowledge_qa": {
                "required_capabilities": ["rag", "citation", "evidence_check"],
                "required_tools": ["doc_search_tool", "evidence_check_tool"],
                "evidence_required": True,
                "risk_level": "medium",
                "permission_scope": [
                    "kb:project_docs:read",
                    "tool:doc_search:use",
                    "tool:evidence_check:use",
                ],
            },
            "coding": {
                "required_capabilities": ["code_generation", "test_planning", "permission_check"],
                "required_tools": [],
                "evidence_required": False,
                "risk_level": "high",
                "permission_scope": ["model:coding:use"],
            },
            "media_generation": {
                "required_capabilities": ["media_generation", "prompt_safety", "asset_metadata"],
                "required_tools": [],
                "evidence_required": False,
                "risk_level": "medium",
                "permission_scope": [],
            },
            "document_writing": {
                "required_capabilities": [
                    "document_writing",
                    "a2a_knowledge_request",
                    "a2a_media_request",
                    "citation",
                    "permission_check",
                ],
                "required_tools": ["inspection_tool"],
                "evidence_required": False,
                "risk_level": "medium",
                "permission_scope": ["model:writing:use", "tool:inspection:use"],
            },
        }
        seed = dict(seeds[task_type])
        if task_type == "knowledge_qa" and self._deep_rag(request):
            seed["required_capabilities"] = list(seed["required_capabilities"]) + [
                "deep_rag",
                "multi_hop_retrieval",
                "evidence_synthesis",
            ]
        if task_type == "media_generation":
            media_type = self._media_type(request)
            seed["required_capabilities"] = list(seed["required_capabilities"])
            seed["required_tools"] = list(seed["required_tools"])
            seed["permission_scope"] = list(seed["permission_scope"])
            if media_type == "video":
                seed["required_capabilities"].append("video_generation")
                seed["required_tools"].append("video_generation_tool")
                seed["permission_scope"].append("tool:video_generation:use")
            else:
                seed["required_capabilities"].append("image_generation")
                seed["required_tools"].append("image_generation_tool")
                seed["permission_scope"].append("tool:image_generation:use")
        return seed

    def _deep_rag(self, request: AgenticGatewayRequest) -> bool:
        text = f"{request.input} {request.metadata}".lower()
        return self._contains_any(
            text,
            {
                "deep analysis",
                "comprehensive",
                "systematic",
                "multi-angle",
                "深度",
                "全面",
                "系统性",
                "多角度",
                "论文级",
            },
        )

    def _media_type(self, request: AgenticGatewayRequest) -> str:
        metadata_media_type = request.metadata.get("media_type")
        if metadata_media_type in {"image", "video"}:
            return str(metadata_media_type)
        text = f"{request.input} {request.metadata}".lower()
        video_keywords = {"video", "storyboard", "短视频", "视频", "文生视频", "分镜"}
        if self._contains_any(text, video_keywords):
            return "video"
        return "image"

    def _metadata_float(self, metadata: Dict[str, Any], name: str, default: float) -> float:
        value = metadata.get(name, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed <= 0:
            return default
        return parsed

    def _metadata_cost_budget(self, value: Any) -> CostBudget:
        if value in {"low", "normal", "high"}:
            return value
        return "normal"

    def _contains_any(self, text: str, keywords: Set[str]) -> bool:
        return any(keyword in text for keyword in keywords)
