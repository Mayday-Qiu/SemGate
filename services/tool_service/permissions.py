from __future__ import annotations

from services.tool_service.schemas import ToolInvokeRequest


def is_allowed(request: ToolInvokeRequest, tool_name: str, permission_scope: str) -> bool:
    permissions = set(request.permissions)
    if "*" not in permissions and permission_scope not in permissions:
        return False

    allowed_tools = set(request.allowed_tools)
    if allowed_tools and "*" not in allowed_tools and tool_name not in allowed_tools:
        return False

    return True
