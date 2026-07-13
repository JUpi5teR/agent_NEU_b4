from __future__ import annotations

from copy import deepcopy
from typing import Any

from common.schemas import validate_ai_message

from .complexity import latest_user_text
from .errors import PlanExecutionError

DEFAULT_MAX_ATTEMPTS = 3
STEP_STATUS_PENDING = "pending"
STEP_STATUS_SUCCESS = "success"
STEP_STATUS_FAILED = "failed"
_STEP_STATUSES = {STEP_STATUS_PENDING, STEP_STATUS_SUCCESS, STEP_STATUS_FAILED}


class PlanExecutor:
    """Create and locally replan B4 plans; B1 remains responsible for executing tool calls."""

    def create_plan(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any] | None = None,
        fixture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if fixture and isinstance(fixture.get("plan"), dict):
            return _normalize_plan(deepcopy(fixture["plan"]), _goal_text(messages, goal_analysis))
        goal = _goal_text(messages, goal_analysis)
        steps = _build_steps(goal, tools_schema, goal_analysis or {})
        return _normalize_plan(
            {
                "goal": goal,
                "steps": steps,
                "status": STEP_STATUS_PENDING,
                "retry_policy": {"max_attempts_per_step": DEFAULT_MAX_ATTEMPTS},
            },
            goal,
        )

    def execute_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_plan(deepcopy(plan), plan.get("goal", "complete user request"))
        normalized["status"] = "ready"
        return normalized

    def reflect_and_replan(self, plan: dict[str, Any], observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        updated = _normalize_plan(deepcopy(plan), plan.get("goal", "complete user request"))
        updated["observations"] = observations or []
        if observations:
            updated["steps"].append(
                {
                    "id": len(updated["steps"]) + 1,
                    "type": "reflect",
                    "content": "Review failed observations and adjust the next action.",
                    "max_attempts": 1,
                    "status": STEP_STATUS_PENDING,
                }
            )
        return _normalize_plan(updated, updated["goal"])

    def replan(
        self,
        plan: dict[str, Any],
        failed_step_id: int,
        diagnosis: dict[str, Any] | str,
        tools_schema: list[dict[str, Any]] | None = None,
        goal_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = _normalize_plan(deepcopy(plan), plan.get("goal", "complete user request"))
        failed_index = _find_step_index(normalized["steps"], failed_step_id)
        diagnosis_name = _diagnosis_name(diagnosis)
        preserved_steps = deepcopy(normalized["steps"][:failed_index])
        discarded_steps = deepcopy(normalized["steps"][failed_index:])
        for step in preserved_steps:
            if step.get("status") == STEP_STATUS_PENDING:
                step["status"] = STEP_STATUS_SUCCESS
        failed_step = deepcopy(normalized["steps"][failed_index])
        failed_step["status"] = STEP_STATUS_FAILED
        failed_step["result"] = _diagnosis_suggestion(diagnosis)

        if diagnosis_name == "permission_error":
            next_steps = preserved_steps + [failed_step]
            status = STEP_STATUS_FAILED
        else:
            recovery_steps = _build_recovery_steps(
                normalized["goal"],
                failed_step,
                diagnosis_name,
                tools_schema or [],
                goal_analysis or {},
            )
            next_steps = preserved_steps + recovery_steps
            status = STEP_STATUS_PENDING

        next_steps = _renumber_steps(next_steps)
        replanned = _normalize_plan(
            {
                "goal": normalized["goal"],
                "steps": next_steps,
                "status": status,
                "retry_policy": normalized.get("retry_policy", {}),
                "replan": {
                    "failed_step_id": failed_step_id,
                    "diagnosis": diagnosis_name,
                    "suggestion": _diagnosis_suggestion(diagnosis),
                    "preserved_step_ids": [step.get("id") for step in preserved_steps],
                    "discarded_step_ids": [step.get("id") for step in discarded_steps],
                },
            },
            normalized["goal"],
        )
        return replanned

    def plan_to_ai_message(self, plan: dict[str, Any], route_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized = _normalize_plan(deepcopy(plan), plan.get("goal", "complete user request"))
        tool_steps = [
            step
            for step in normalized["steps"]
            if step.get("type") == "tool_call" and step.get("status") != STEP_STATUS_SUCCESS
        ]
        if tool_steps:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{int(step.get('id', index)):03d}",
                        "name": step["tool"],
                        "args": step.get("args", {}),
                    }
                    for index, step in enumerate(tool_steps, 1)
                ],
                "type": "plan_execute",
                "plan": normalized,
                "metadata": route_metadata or {},
            }
        else:
            message = {
                "role": "assistant",
                "content": "Plan created. No external tool is required for the first step.",
                "tool_calls": [],
                "type": "plan_execute",
                "plan": normalized,
                "metadata": route_metadata or {},
            }
        validate_ai_message(message)
        return message


def _goal_text(messages: list[dict[str, Any]], goal_analysis: dict[str, Any] | None) -> str:
    if goal_analysis and isinstance(goal_analysis.get("goal"), str) and goal_analysis["goal"]:
        return goal_analysis["goal"]
    return latest_user_text(messages) or "complete user request"


def _build_steps(goal: str, tools_schema: list[dict[str, Any]], goal_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    selected_tools = _select_tools(goal, tools_schema, goal_analysis)
    steps: list[dict[str, Any]] = []
    for index, tool in enumerate(selected_tools, 1):
        steps.append(
            {
                "id": index,
                "type": "tool_call",
                "tool": tool["name"],
                "args": tool["args"],
                "max_attempts": DEFAULT_MAX_ATTEMPTS,
                "status": STEP_STATUS_PENDING,
            }
        )
    steps.append(_make_synthesize_step(len(steps) + 1))
    return steps


def _normalize_plan(plan: dict[str, Any], fallback_goal: str) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanExecutionError("plan must be an object")
    goal = plan.get("goal") if isinstance(plan.get("goal"), str) and plan.get("goal") else fallback_goal
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanExecutionError("plan steps must be a non-empty array")
    normalized_steps = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise PlanExecutionError(f"plan step {index} must be an object")
        step_type = step.get("type")
        if step_type not in {"tool_call", "synthesize", "reflect"}:
            raise PlanExecutionError(f"invalid plan step type: {step_type}")
        normalized = dict(step)
        normalized["id"] = int(step.get("id", index))
        normalized["max_attempts"] = int(step.get("max_attempts", DEFAULT_MAX_ATTEMPTS if step_type == "tool_call" else 1))
        status = step.get("status", STEP_STATUS_PENDING)
        normalized["status"] = status if status in _STEP_STATUSES else STEP_STATUS_PENDING
        if step_type == "tool_call":
            if not isinstance(step.get("tool"), str) or not step["tool"]:
                raise PlanExecutionError(f"tool_call step {index} missing tool")
            args = step.get("args", {})
            if not isinstance(args, dict):
                raise PlanExecutionError(f"tool_call step {index} args must be an object")
            normalized["args"] = args
        normalized_steps.append(normalized)
    retry_policy = plan.get("retry_policy") if isinstance(plan.get("retry_policy"), dict) else {}
    retry_policy.setdefault("max_attempts_per_step", DEFAULT_MAX_ATTEMPTS)
    normalized_plan = {
        "goal": goal,
        "steps": normalized_steps,
        "status": plan.get("status", STEP_STATUS_PENDING),
        "retry_policy": retry_policy,
    }
    if isinstance(plan.get("observations"), list):
        normalized_plan["observations"] = plan["observations"]
    if isinstance(plan.get("replan"), dict):
        normalized_plan["replan"] = plan["replan"]
    if isinstance(plan.get("source"), str):
        normalized_plan["source"] = plan["source"]
    return normalized_plan


def _select_tools(goal: str, tools_schema: list[dict[str, Any]], goal_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    names = _tool_names(tools_schema)
    lowered = goal.lower()
    intent = goal_analysis.get("intent")
    selected: list[dict[str, Any]] = []
    if intent == "search" and "local_file_search" in names:
        selected.append({"name": "local_file_search", "args": {"query": goal, "top_k": 5}})
    if intent == "calculate" and "calculator" in names:
        selected.append({"name": "calculator", "args": {"expression": "1 + 1"}})
    if "file_reader" in names and any(marker in lowered or marker in goal for marker in ("read", "file", "docs/", "\u9605\u8bfb", "\u6587\u4ef6")):
        selected.append({"name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}})
    if "table_analyzer" in names and any(marker in lowered for marker in ("csv", "table", "spreadsheet")):
        selected.append({"name": "table_analyzer", "args": {"path": "sales.csv", "describe": True}})
    if not selected and goal_analysis.get("needs_tool") and "file_reader" in names:
        selected.append({"name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}})
    return _dedupe_tools(selected)


def _build_recovery_steps(
    goal: str,
    failed_step: dict[str, Any],
    diagnosis: str,
    tools_schema: list[dict[str, Any]],
    goal_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    names = _tool_names(tools_schema)
    steps: list[dict[str, Any]] = []
    if diagnosis == "path_or_file_error" and "local_file_search" in names:
        steps.append(
            {
                "id": 1,
                "type": "tool_call",
                "tool": "local_file_search",
                "args": {"query": goal, "top_k": 5},
                "max_attempts": DEFAULT_MAX_ATTEMPTS,
                "status": STEP_STATUS_PENDING,
                "replan_from": failed_step.get("id"),
            }
        )
    elif diagnosis == "invalid_tool_arguments" and failed_step.get("type") == "tool_call":
        tool = str(failed_step.get("tool"))
        if tool in names or not names:
            steps.append(
                {
                    "id": 1,
                    "type": "tool_call",
                    "tool": tool,
                    "args": _repair_args(tool, failed_step.get("args", {}), goal, goal_analysis),
                    "max_attempts": DEFAULT_MAX_ATTEMPTS,
                    "status": STEP_STATUS_PENDING,
                    "replan_from": failed_step.get("id"),
                }
            )
    elif diagnosis == "tool_error" and failed_step.get("type") == "tool_call":
        steps.append(
            {
                "id": 1,
                "type": "reflect",
                "content": "Summarize the tool failure and continue with the available observations.",
                "max_attempts": 1,
                "status": STEP_STATUS_PENDING,
                "replan_from": failed_step.get("id"),
            }
        )
    if not steps:
        steps.append(
            {
                "id": 1,
                "type": "reflect",
                "content": "Ask for clarification or report why the failed step cannot be recovered automatically.",
                "max_attempts": 1,
                "status": STEP_STATUS_PENDING,
                "replan_from": failed_step.get("id"),
            }
        )
    if steps[-1].get("type") != "synthesize":
        steps.append(_make_synthesize_step(len(steps) + 1))
    return steps


def _repair_args(tool: str, args: Any, goal: str, goal_analysis: dict[str, Any]) -> dict[str, Any]:
    current = dict(args) if isinstance(args, dict) else {}
    if tool == "file_reader":
        path = current.get("path") if isinstance(current.get("path"), str) and current.get("path") else "docs/agent_intro.txt"
        return {"path": path, "max_chars": int(current.get("max_chars", 2000) or 2000)}
    if tool == "local_file_search":
        query = current.get("query") if isinstance(current.get("query"), str) and current.get("query") else goal
        return {"query": query, "top_k": int(current.get("top_k", 5) or 5)}
    if tool == "calculator":
        expression = current.get("expression") if isinstance(current.get("expression"), str) and current.get("expression") else "1 + 1"
        return {"expression": expression}
    if tool == "table_analyzer":
        path = current.get("path") if isinstance(current.get("path"), str) and current.get("path") else "sales.csv"
        return {"path": path, "describe": bool(current.get("describe", True))}
    return current


def _make_synthesize_step(step_id: int) -> dict[str, Any]:
    return {
        "id": step_id,
        "type": "synthesize",
        "content": "Synthesize available observations and answer the user goal.",
        "max_attempts": 1,
        "status": STEP_STATUS_PENDING,
    }


def _find_step_index(steps: list[dict[str, Any]], failed_step_id: int) -> int:
    for index, step in enumerate(steps):
        if int(step.get("id", index + 1)) == int(failed_step_id):
            return index
    for index, step in enumerate(steps):
        if step.get("status") == STEP_STATUS_FAILED:
            return index
    raise PlanExecutionError(f"failed step not found: {failed_step_id}")


def _diagnosis_name(diagnosis: dict[str, Any] | str) -> str:
    if isinstance(diagnosis, dict):
        value = diagnosis.get("diagnosis") or diagnosis.get("problem") or "tool_error"
    else:
        value = diagnosis
    return str(value or "tool_error")


def _diagnosis_suggestion(diagnosis: dict[str, Any] | str) -> str:
    if isinstance(diagnosis, dict):
        suggestion = diagnosis.get("suggestion") or diagnosis.get("problem") or diagnosis.get("diagnosis")
        return str(suggestion or "review failed step")
    return str(diagnosis or "review failed step")


def _renumber_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        updated = deepcopy(step)
        updated["id"] = index
        renumbered.append(updated)
    return renumbered


def _dedupe_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tool in tools:
        key = (tool["name"], repr(sorted(tool["args"].items())))
        if key not in seen:
            seen.add(key)
            result.append(tool)
    return result


def _tool_names(tools_schema: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.add(name)
    return names
