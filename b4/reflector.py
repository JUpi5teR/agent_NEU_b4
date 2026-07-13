from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Reflection:
    trigger: str
    problem: str
    diagnosis: str
    suggestion: str
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "problem": self.problem,
            "diagnosis": self.diagnosis,
            "suggestion": self.suggestion,
            "retryable": self.retryable,
        }


class Reflector:
    """Diagnose B4-visible failures without executing tools itself."""

    def reflect(
        self,
        messages: list[dict[str, Any]],
        goal_analysis: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
    ) -> Reflection | None:
        tool_reflection = self.reflect_latest_tool(messages)
        if tool_reflection is not None:
            return tool_reflection
        if plan is not None and plan.get("status") == "failed":
            return Reflection(
                trigger="plan_execution_failed",
                problem="plan status is failed",
                diagnosis="plan did not complete within the allowed attempts",
                suggestion="review failed step observations and create a smaller next step",
                retryable=False,
            )
        return None

    def reflect_latest_tool(self, messages: list[dict[str, Any]]) -> Reflection | None:
        latest_tool = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
        if not latest_tool or latest_tool.get("status") != "error":
            return None
        problem = _tool_problem(latest_tool)
        diagnosis, suggestion, retryable = _diagnose(problem)
        return Reflection(
            trigger="tool_execution_failed",
            problem=problem,
            diagnosis=diagnosis,
            suggestion=suggestion,
            retryable=retryable,
        )


def _tool_problem(tool_message: dict[str, Any]) -> str:
    content = tool_message.get("content")
    if not isinstance(content, str):
        return "tool returned an invalid error payload"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content[:200]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return str(payload)[:200]


def _diagnose(problem: str) -> tuple[str, str, bool]:
    lowered = problem.lower()
    if any(marker in lowered for marker in ("not found", "no such file", "path", "file")):
        return "path_or_file_error", "retry with an existing path or search for the file first", True
    if any(marker in lowered for marker in ("parameter", "required", "type", "invalid", "must be")):
        return "invalid_tool_arguments", "fix the tool arguments before retrying", True
    if any(marker in lowered for marker in ("permission", "denied")):
        return "permission_error", "report the permission issue instead of retrying blindly", False
    return "tool_error", "summarize the failure and avoid repeating the same call", False
