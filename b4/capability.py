from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .complexity import COMPLEXITY_COMPLEX, COMPLEXITY_MULTI_STEP, latest_user_text


CAPABILITY_CALCULATE = "calculate"
CAPABILITY_FILE_READ = "file_read"
CAPABILITY_FILE_WRITE = "file_write"
CAPABILITY_INTERNET_SEARCH = "internet_search"
CAPABILITY_LOCAL_SEARCH = "local_search"
CAPABILITY_TABLE_ANALYSIS = "table_analysis"
CAPABILITY_WEATHER = "weather"

ACTION_CAPABILITIES = {
    CAPABILITY_CALCULATE,
    CAPABILITY_FILE_READ,
    CAPABILITY_FILE_WRITE,
    CAPABILITY_INTERNET_SEARCH,
    CAPABILITY_LOCAL_SEARCH,
    CAPABILITY_TABLE_ANALYSIS,
    CAPABILITY_WEATHER,
}


@dataclass(frozen=True)
class CapabilityNeed:
    name: str
    required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "required": self.required, "reason": self.reason}


@dataclass(frozen=True)
class CapabilityMatch:
    needs: tuple[CapabilityNeed, ...]
    available_tools: dict[str, list[str]]
    missing_required: tuple[CapabilityNeed, ...]
    needs_plan: bool
    needs_deep_reasoning: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs": [need.to_dict() for need in self.needs],
            "available_tools": self.available_tools,
            "missing_required": [need.to_dict() for need in self.missing_required],
            "needs_plan": self.needs_plan,
            "needs_deep_reasoning": self.needs_deep_reasoning,
        }

    @property
    def unsupported(self) -> bool:
        return bool(self.missing_required)

    @property
    def supported_action_needs(self) -> list[CapabilityNeed]:
        return [
            need
            for need in self.needs
            if need.name in ACTION_CAPABILITIES and need.name in self.available_tools
        ]


def match_capabilities(
    messages: list[dict[str, Any]],
    tools_schema: list[dict[str, Any]],
    goal_analysis: dict[str, Any] | None = None,
    complexity: dict[str, Any] | None = None,
) -> CapabilityMatch:
    """Extract required capabilities from the task and match them to tools.

    This keeps capability decisions centralized instead of scattering one-off
    route patches for weather, web search, file writing, and path-sensitive tools.
    """

    text = latest_user_text(messages)
    available = tool_capabilities_from_schema(tools_schema)
    needs = _dedupe_needs(_detect_capability_needs(text, goal_analysis or {}))
    missing = tuple(
        need for need in needs if need.required and need.name in ACTION_CAPABILITIES and need.name not in available
    )
    complexity_value = str((complexity or {}).get("complexity") or "")
    return CapabilityMatch(
        needs=tuple(needs),
        available_tools=available,
        missing_required=missing,
        needs_plan=_needs_plan(text, goal_analysis or {}, complexity_value),
        needs_deep_reasoning=_needs_deep_reasoning(text, goal_analysis or {}, complexity_value),
    )


def tool_capabilities_from_schema(tools_schema: list[dict[str, Any]]) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        capabilities = _capabilities_for_tool(function)
        for capability in capabilities:
            available.setdefault(capability, [])
            if name not in available[capability]:
                available[capability].append(name)
    return available


def unsupported_capability_problem(match: CapabilityMatch) -> dict[str, Any] | None:
    if not match.missing_required:
        return None
    primary = match.missing_required[0]
    return {
        "type": "unsupported_capability",
        "capability": primary.name,
        "message": _unsupported_message(primary.name),
        "missing_required": [need.to_dict() for need in match.missing_required],
        "available_tools": sorted(_all_available_tool_names(match.available_tools)),
        "capability_match": match.to_dict(),
    }


def unsupported_capability_content(problem: dict[str, Any]) -> str:
    capability = problem.get("capability")
    message = problem.get("message") or "B4 cannot fulfill this request with the current skills."
    tools = problem.get("available_tools")
    tool_text = ", ".join(tools) if isinstance(tools, list) and tools else "none"
    if capability == CAPABILITY_WEATHER:
        return (
            "B4 cannot answer this weather request with the current skills. "
            "No weather or weather_forecast tool is available, so using local_file_search would be unreliable. "
            f"Available tools: {tool_text}. Please add a weather skill or pass weather data as context."
        )
    if capability == CAPABILITY_INTERNET_SEARCH:
        return (
            "B4 cannot perform internet search with the current skills. "
            "The current local search skill only searches local files, not the web. "
            f"Available tools: {tool_text}. Please add a web_search/internet_search skill or pass search results as context."
        )
    if capability == CAPABILITY_FILE_WRITE:
        return (
            "B4 cannot create or save files in the current folder with the current skills. "
            "No general file/code/project writer skill is available. "
            f"Available tools: {tool_text}. B4 can draft code in content, but another module must write files."
        )
    return f"{message} Available tools: {tool_text}."


def validate_tool_calls(
    messages: list[dict[str, Any]],
    ai_message: dict[str, Any],
    tools_schema: list[dict[str, Any]],
    capability_match: CapabilityMatch | None = None,
) -> dict[str, Any] | None:
    calls = ai_message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    schemas = _tool_schema_by_name(tools_schema)
    match = capability_match or match_capabilities(messages, tools_schema)
    for call in calls:
        if not isinstance(call, dict):
            return {"type": "invalid_tool_call", "message": "tool call must be an object", "call": call}
        name = call.get("name")
        if not isinstance(name, str) or name not in schemas:
            return {
                "type": "unknown_tool",
                "tool": name,
                "message": f"B4 tried to call an unavailable tool: {name}",
                "available_tools": sorted(schemas),
            }
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        missing_args = _missing_required_args(schemas[name], args)
        if missing_args:
            return {
                "type": "missing_required_args",
                "tool": name,
                "missing_args": missing_args,
                "message": f"B4 tried to call {name} without required args: {', '.join(missing_args)}",
            }
        path_problem = _path_grounding_problem(messages, name, args)
        if path_problem is not None:
            return path_problem
        tool_caps = set(_capabilities_for_tool(schemas[name]))
        missing_required_caps = {need.name for need in match.missing_required}
        if missing_required_caps and not tool_caps.intersection(missing_required_caps):
            return {
                "type": "capability_mismatch",
                "tool": name,
                "tool_capabilities": sorted(tool_caps),
                "missing_required_capabilities": sorted(missing_required_caps),
                "message": f"B4 tried to use {name}, but it does not satisfy the missing required capability.",
            }
    return None


def tool_validation_content(problem: dict[str, Any]) -> str:
    problem_type = problem.get("type")
    if problem_type == "missing_file_path":
        return (
            f"B4 cannot call {problem.get('tool')} because no file path was provided. "
            "Please provide the file path or pass the required file context."
        )
    if problem_type == "ungrounded_file_path":
        return (
            f"B4 refused to call {problem.get('tool')} with an unverified file path: {problem.get('path')}. "
            "The file path was not provided in the user request or prior module feedback. "
            "Please provide the correct file path or have the upstream module pass the required file context."
        )
    return str(problem.get("message") or "B4 refused an invalid tool call.")


def _detect_capability_needs(text: str, goal_analysis: dict[str, Any]) -> list[CapabilityNeed]:
    lowered = text.lower()
    needs: list[CapabilityNeed] = []
    if _contains_any(lowered, text, ("weather", "forecast", "temperature", "天气", "气温", "降雨", "下雨", "明天会下雨")):
        needs.append(CapabilityNeed(CAPABILITY_WEATHER, True, "task asks for weather or forecast data"))
    if _contains_any(
        lowered,
        text,
        ("web search", "internet search", "online search", "search online", "网上搜索", "互联网搜索", "网络搜索", "去网上搜索"),
    ):
        needs.append(CapabilityNeed(CAPABILITY_INTERNET_SEARCH, True, "task asks for internet search"))
    elif _contains_any(lowered, text, ("search", "find", "lookup", "搜索", "查找", "检索")):
        needs.append(CapabilityNeed(CAPABILITY_LOCAL_SEARCH, True, "task asks for local search"))
    if _looks_like_file_write(lowered, text):
        needs.append(CapabilityNeed(CAPABILITY_FILE_WRITE, True, "task asks to create or save files"))
    if _looks_like_file_read(lowered, text):
        needs.append(CapabilityNeed(CAPABILITY_FILE_READ, True, "task references reading a local file"))
    if _looks_like_table_analysis(lowered, text):
        needs.append(CapabilityNeed(CAPABILITY_TABLE_ANALYSIS, True, "task asks for table analysis"))
    if _looks_like_calculation(lowered, text) or str(goal_analysis.get("intent") or "") == "calculate":
        needs.append(CapabilityNeed(CAPABILITY_CALCULATE, True, "task asks for calculation"))
    return needs


def _capabilities_for_tool(function: dict[str, Any]) -> list[str]:
    explicit = function.get("x-capabilities") or function.get("capabilities")
    if isinstance(explicit, list):
        return [item for item in explicit if isinstance(item, str) and item]
    name = str(function.get("name") or "").lower()
    description = str(function.get("description") or "").lower()
    text = f"{name} {description}"
    capabilities: list[str] = []
    if "calculator" in text or "calculate" in text or "arithmetic" in text:
        capabilities.append(CAPABILITY_CALCULATE)
    if "file_reader" in text or "read a local" in text or "read local" in text:
        capabilities.append(CAPABILITY_FILE_READ)
    if "local_file_search" in text or ("search local" in text and "file" in text):
        capabilities.append(CAPABILITY_LOCAL_SEARCH)
    if "table_analyzer" in text or "csv" in text or "tsv" in text or "spreadsheet" in text:
        capabilities.append(CAPABILITY_TABLE_ANALYSIS)
    if ("write" in text or "save" in text) and any(marker in text for marker in ("file_writer", "code_writer", "project_writer")):
        capabilities.append(CAPABILITY_FILE_WRITE)
    if "web_search" in text or "internet_search" in text or "search the web" in text:
        capabilities.append(CAPABILITY_INTERNET_SEARCH)
    if "weather" in text or "forecast" in text:
        capabilities.append(CAPABILITY_WEATHER)
    return capabilities


def _needs_plan(text: str, goal_analysis: dict[str, Any], complexity_value: str) -> bool:
    lowered = text.lower()
    if complexity_value == COMPLEXITY_MULTI_STEP:
        return True
    if _contains_any(lowered, text, ("先", "然后", "最后", "步骤", "完整", "系统", "项目", "first", "then", "finally")):
        return True
    return bool(goal_analysis.get("needs_tool")) and len(_detect_capability_needs(text, goal_analysis)) > 1


def _needs_deep_reasoning(text: str, goal_analysis: dict[str, Any], complexity_value: str) -> bool:
    lowered = text.lower()
    if _looks_like_direct_answer_request(lowered, text, goal_analysis):
        return False
    if complexity_value == COMPLEXITY_COMPLEX:
        return True
    if str(goal_analysis.get("intent") or "") in {"analyze", "compare"}:
        return True
    return _contains_any(lowered, text, ("证明", "推导", "为什么", "性质", "公式", "reason", "prove", "derive"))


def _looks_like_direct_answer_request(lowered: str, text: str, goal_analysis: dict[str, Any]) -> bool:
    if str(goal_analysis.get("intent") or "") == "direct":
        return True
    return _contains_any(
        lowered,
        text,
        (
            "simple answer",
            "answer briefly",
            "direct answer",
            "answer directly",
            "do not analyze",
            "without analysis",
            "简单回答",
            "直接回答",
            "简短回答",
            "不要分析",
            "不用分析",
            "无需分析",
        ),
    )

def _looks_like_file_write(lowered: str, text: str) -> bool:
    return _contains_any(
        lowered,
        text,
        (
            "write file",
            "save file",
            "current folder",
            "current directory",
            "生成文件",
            "保存到当前",
            "放在当前文件夹",
            "放在当前目录",
            "写入当前",
            "生成到当前",
        ),
    )


def _looks_like_file_read(lowered: str, text: str) -> bool:
    return _contains_any(lowered, text, ("docs/", ".txt", ".md", "read file", "阅读", "读取", "文件"))


def _looks_like_table_analysis(lowered: str, text: str) -> bool:
    return _contains_any(lowered, text, (".csv", ".tsv", "table", "spreadsheet", "表格"))


def _looks_like_calculation(lowered: str, text: str) -> bool:
    if re.search(r"\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?", text):
        return True
    return _contains_any(lowered, text, ("calculate", "compute", "sum", "计算", "求和"))


def _dedupe_needs(needs: list[CapabilityNeed]) -> list[CapabilityNeed]:
    result: list[CapabilityNeed] = []
    seen: set[str] = set()
    for need in needs:
        if need.name in seen:
            continue
        seen.add(need.name)
        result.append(need)
    return result


def _contains_any(lowered: str, original: str, markers: tuple[str, ...]) -> bool:
    return any(marker.lower() in lowered if marker.isascii() else marker in original for marker in markers)


def _all_available_tool_names(available_tools: dict[str, list[str]]) -> set[str]:
    names: set[str] = set()
    for tools in available_tools.values():
        names.update(tools)
    return names


def _tool_schema_by_name(tools_schema: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and isinstance(function, dict):
            result[name] = function
    return result


def _missing_required_args(function: dict[str, Any], args: dict[str, Any]) -> list[str]:
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    return [item for item in required if isinstance(item, str) and item not in args]


def _path_grounding_problem(messages: list[dict[str, Any]], tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    if tool_name not in {"file_reader", "table_analyzer"}:
        return None
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        return {"type": "missing_file_path", "tool": tool_name, "path": path}
    if not _path_is_grounded(path, messages):
        user_text = latest_user_text(messages)
        return {"type": "ungrounded_file_path", "tool": tool_name, "path": path, "user_text": user_text}
    return None


def _path_is_grounded(path: str, messages: list[dict[str, Any]]) -> bool:
    normalized_path = path.replace("\\", "/").strip()
    basename = normalized_path.rsplit("/", 1)[-1]
    for text in _grounding_texts(messages):
        normalized_text = text.replace("\\", "/")
        if normalized_path in normalized_text:
            return True
        if basename and "." in basename and basename in normalized_text:
            return True
    return False


def _grounding_texts(messages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            texts.append(message["content"])
            continue
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            texts.append(message["content"])
            continue
        if not isinstance(payload, dict) or payload.get("status") != "success":
            continue
        input_payload = payload.get("input")
        if isinstance(input_payload, dict) and isinstance(input_payload.get("path"), str):
            texts.append(input_payload["path"])
        texts.append(message["content"])
    return texts


def _unsupported_message(capability: str) -> str:
    return f"Required capability is not available: {capability}"
