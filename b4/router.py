from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io_utils import read_json
from common.path_utils import resolve_from_file

from .complexity import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_MULTI_STEP,
    COMPLEXITY_SIMPLE,
    ComplexityJudge,
    latest_user_text,
)

ROUTE_DIRECT_ANSWER = "direct_answer"
ROUTE_DEEP_THINK = "deep_think"
ROUTE_TOOL_CALL = "tool_call"
ROUTE_PLAN_EXECUTE = "plan_execute"

_ALLOWED_ROUTES = {ROUTE_DIRECT_ANSWER, ROUTE_DEEP_THINK, ROUTE_TOOL_CALL, ROUTE_PLAN_EXECUTE}
_TOOL_INTENTS = {"execute", "search", "calculate"}
_REASONING_INTENTS = {"analyze", "compare", "reason", "evaluate"}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    complexity: str
    reason: str
    source: str = "rules"
    goal_analysis: dict[str, Any] | None = None
    route_chain: tuple[str, ...] | None = None
    tool_recall: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "route": self.route,
            "route_chain": list(self.route_chain or (self.route,)),
            "complexity": self.complexity,
            "reason": self.reason,
            "source": self.source,
        }
        if self.goal_analysis is not None:
            record["goal_analysis"] = self.goal_analysis
        if self.tool_recall:
            record["tool_recall"] = self.tool_recall
        return record


class Router:
    """Route B4 requests without taking over the B1 agent loop."""

    def __init__(self, judge: ComplexityJudge | None = None) -> None:
        self._judge = judge or ComplexityJudge()

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        fixture: dict[str, Any] | None = None,
        goal_analysis: dict[str, Any] | None = None,
    ) -> RouteDecision:
        if fixture is not None:
            return _decision_from_fixture(fixture, goal_analysis)
        analysis = goal_analysis or {}
        complexity = self._judge.judge(messages, analysis)
        user_input = latest_user_text(messages)
        if _latest_message_is_tool(messages):
            if messages[-1].get("status") == "success" and _needs_deep_think(
                user_input, complexity.complexity, analysis
            ):
                return RouteDecision(
                    ROUTE_DEEP_THINK,
                    complexity.complexity,
                    "latest tool result should be reasoned over before answering",
                    goal_analysis=goal_analysis,
                    route_chain=(ROUTE_DEEP_THINK, ROUTE_DIRECT_ANSWER),
                )
            return RouteDecision(
                ROUTE_DIRECT_ANSWER,
                complexity.complexity,
                "latest tool result is ready",
                goal_analysis=goal_analysis,
                route_chain=(ROUTE_DIRECT_ANSWER,),
            )
        if complexity.complexity == COMPLEXITY_MULTI_STEP:
            return RouteDecision(
                ROUTE_PLAN_EXECUTE,
                complexity.complexity,
                complexity.reason,
                goal_analysis=goal_analysis,
                route_chain=(ROUTE_PLAN_EXECUTE,),
            )
        if _analysis_needs_tool(analysis, tools_schema) or _looks_like_tool_task(user_input, tools_schema):
            route_chain = _tool_route_chain(user_input, complexity.complexity, analysis)
            return RouteDecision(
                route_chain[0],
                complexity.complexity,
                "goal analysis or user input matches available tool capability",
                goal_analysis=goal_analysis,
                route_chain=route_chain,
            )
        if complexity.complexity == COMPLEXITY_COMPLEX:
            return RouteDecision(
                ROUTE_DEEP_THINK,
                complexity.complexity,
                complexity.reason,
                goal_analysis=goal_analysis,
                route_chain=(ROUTE_DEEP_THINK, ROUTE_DIRECT_ANSWER),
            )
        return RouteDecision(
            ROUTE_DIRECT_ANSWER,
            COMPLEXITY_SIMPLE,
            complexity.reason,
            goal_analysis=goal_analysis,
            route_chain=(ROUTE_DIRECT_ANSWER,),
        )


def is_router_enabled(config: dict[str, Any], fixture: dict[str, Any] | None = None) -> bool:
    routing = config.get("routing", {}) if isinstance(config.get("routing"), dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    return bool(fixture) or bool(routing.get("enabled")) or bool(runtime.get("routing_enabled"))


def load_router_fixture(config_path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    path_value = os.environ.get("B4_ROUTER_FIXTURE")
    routing = config.get("routing", {}) if isinstance(config.get("routing"), dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    path_value = path_value or routing.get("fixture_path") or runtime.get("router_fixture_path")
    if not path_value:
        return None
    fixture_path = resolve_from_file(str(path_value), config_path)
    fixture = read_json(fixture_path)
    if not isinstance(fixture, dict):
        raise ValueError("B4 router fixture must be a JSON object")
    return fixture


def _decision_from_fixture(fixture: dict[str, Any], goal_analysis: dict[str, Any] | None = None) -> RouteDecision:
    fixture_chain = fixture.get("route_chain")
    route = fixture.get("route")
    if not route and fixture_chain:
        route = _normalize_route_chain(fixture_chain, ROUTE_DIRECT_ANSWER)[0]
    complexity = fixture.get("complexity", COMPLEXITY_SIMPLE)
    allowed_complexities = {COMPLEXITY_SIMPLE, COMPLEXITY_COMPLEX, COMPLEXITY_MULTI_STEP}
    if route not in _ALLOWED_ROUTES:
        raise ValueError(f"invalid fixture route: {route}")
    if complexity not in allowed_complexities:
        raise ValueError(f"invalid fixture complexity: {complexity}")
    route_chain = _normalize_route_chain(fixture_chain, str(route))
    tool_recall = fixture.get("tool_recall") if isinstance(fixture.get("tool_recall"), list) else None
    return RouteDecision(str(route), str(complexity), "fixture route", "fixture", goal_analysis, route_chain, tool_recall)


def _normalize_route_chain(value: Any, fallback_route: str) -> tuple[str, ...]:
    if value is None:
        chain = [fallback_route]
    elif isinstance(value, str):
        chain = [value]
    elif isinstance(value, (list, tuple)):
        chain = [item for item in value if isinstance(item, str) and item]
    else:
        raise ValueError("route_chain must be a string or array of strings")
    if not chain:
        chain = [fallback_route]
    invalid = [route for route in chain if route not in _ALLOWED_ROUTES]
    if invalid:
        raise ValueError(f"invalid fixture route_chain item: {invalid[0]}")
    return tuple(_dedupe_consecutive(chain))


def _latest_message_is_tool(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[-1].get("role") == "tool"


def _tool_names(tools_schema: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.add(name)
    return names


def _analysis_needs_tool(goal_analysis: dict[str, Any], tools_schema: list[dict[str, Any]]) -> bool:
    if not goal_analysis.get("needs_tool"):
        return False
    intent = goal_analysis.get("intent")
    names = _tool_names(tools_schema)
    if intent == "search":
        return "local_file_search" in names
    if intent == "calculate":
        return "calculator" in names
    if intent == "execute":
        return bool(names)
    return bool(names) and intent in _TOOL_INTENTS


def _looks_like_tool_task(user_input: str, tools_schema: list[dict[str, Any]]) -> bool:
    lowered = user_input.lower()
    names = _tool_names(tools_schema)
    if "file_reader" in names and any(
        marker in lowered or marker in user_input for marker in ("read", "file", "docs/", "\u9605\u8bfb", "\u6587\u4ef6")
    ):
        return True
    if "local_file_search" in names and any(
        marker in lowered or marker in user_input for marker in ("search", "find", "\u641c\u7d22", "\u67e5\u627e")
    ):
        return True
    if "calculator" in names and any(marker in lowered for marker in ("calculate", "sum", "+", "-", "*", "/")):
        return True
    if "table_analyzer" in names and any(marker in lowered for marker in ("csv", "table", "spreadsheet")):
        return True
    return False


def _tool_route_chain(user_input: str, complexity: str, goal_analysis: dict[str, Any]) -> tuple[str, ...]:
    chain = [ROUTE_TOOL_CALL]
    if _needs_deep_think(user_input, complexity, goal_analysis):
        chain.append(ROUTE_DEEP_THINK)
    chain.append(ROUTE_DIRECT_ANSWER)
    return tuple(_dedupe_consecutive(chain))


def _needs_deep_think(user_input: str, complexity: str, goal_analysis: dict[str, Any]) -> bool:
    lowered = user_input.lower()
    intent = goal_analysis.get("intent")
    if complexity == COMPLEXITY_COMPLEX or intent in _REASONING_INTENTS:
        return True
    return any(
        marker in lowered or marker in user_input
        for marker in (
            "analyze",
            "analysis",
            "compare",
            "why",
            "reason",
            "evaluate",
            "tradeoff",
            "\u5206\u6790",
            "\u6bd4\u8f83",
            "\u4e3a\u4ec0\u4e48",
            "\u63a8\u7406",
        )
    )


def _dedupe_consecutive(routes: list[str]) -> list[str]:
    result: list[str] = []
    for route in routes:
        if not result or result[-1] != route:
            result.append(route)
    return result
