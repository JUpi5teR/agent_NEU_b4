from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.schemas import normalize_tool_call

from .complexity import COMPLEXITY_COMPLEX, COMPLEXITY_MULTI_STEP, COMPLEXITY_SIMPLE, latest_user_text
from .deep_think_tot import DeepThinkToT, ThoughtStep, ToTResult
from .goal_analyzer import normalize_goal_analysis
from .qwen_client import extract_json_object, qwen_generate_completion, qwen_generate_json
from .router import ROUTE_DEEP_THINK, ROUTE_DIRECT_ANSWER, ROUTE_PLAN_EXECUTE, ROUTE_TOOL_CALL, RouteDecision

_ALLOWED_COMPLEXITIES = {COMPLEXITY_SIMPLE, COMPLEXITY_COMPLEX, COMPLEXITY_MULTI_STEP}
_ALLOWED_ROUTES = {ROUTE_DIRECT_ANSWER, ROUTE_DEEP_THINK, ROUTE_TOOL_CALL, ROUTE_PLAN_EXECUTE}
_ALLOWED_DIAGNOSES = {"path_or_file_error", "invalid_tool_arguments", "permission_error", "tool_error"}


class QwenB4MetaEngine:
    """Qwen-backed structured decisions for B4 metadata and routing."""

    def __init__(self, config_path: Path, config: dict[str, Any]) -> None:
        self.config_path = config_path
        self.config = config

    def analyze_goal_with_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_input = latest_user_text(messages)
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Analyze the user's goal for B4 routing. Use intent=direct for greetings or requests that explicitly ask for a simple/direct answer without analysis.",
            {"user_input": user_input, "latest_messages": _compact_messages(messages)},
            {
                "goal": "string",
                "intent": "execute|analyze|compare|search|calculate|explain|direct",
                "keywords": ["string"],
                "constraints": {"must_include": ["string"], "must_exclude": ["string"]},
                "needs_tool": "boolean",
            },
            max_new_tokens=512,
        )
        return normalize_goal_analysis(data, user_input or "complete user request", source="qwen")

    def judge_complexity(self, messages: list[dict[str, Any]], goal_analysis: dict[str, Any]) -> dict[str, Any]:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Classify task complexity for B4. Greetings and explicit simple/direct-answer requests are simple even if they contain comparison wording.",
            {"messages": _compact_messages(messages), "goal_analysis": goal_analysis},
            {"complexity": "simple|complex|multi_step", "reason": "string"},
            max_new_tokens=256,
        )
        return {
            "complexity": _choice(data.get("complexity"), _ALLOWED_COMPLEXITIES, COMPLEXITY_SIMPLE),
            "reason": _string(data.get("reason"), "Qwen classified task complexity"),
            "source": "qwen",
        }

    def decide_route(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        complexity: dict[str, Any],
    ) -> RouteDecision:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            (
                "Choose the B4 route chain. Use tool_call when a tool is needed, "
                "deep_think for reasoning, direct_answer for final answer, and plan_execute for multi-step plans. "
                "Use direct_answer for greetings and explicit simple/direct-answer requests. "
                "After a successful tool result, verify every qualifier in the latest user request before choosing direct_answer. "
                "For local corpus questions with a search_root, use a focused local_file_search for another passage when the current evidence identifies an answer but does not establish its location, date, relationship, or other qualifier. "
                "Build search queries from distinctive named entities and relation or action terms; omit broad category and location words that also match distractors. "
                "A corroborating search must use the discovered entity plus a missing relation and must not repeat an earlier query. "
                "If evidence says yesterday or another relative date but the selected chunk has no record date, call file_reader on the supplied primary_file or reference with max_chars at most 8000 to recover the dated session header. "
                "Do not request an identical successful tool call again. Use keyword mode only when a primary_file transcript is supplied; otherwise prefer hybrid mode."
            ),
            {
                "messages": _compact_messages(messages),
                "latest_message_role": messages[-1].get("role") if messages else None,
                "tools": _tool_summaries(tools_schema),
                "goal_analysis": goal_analysis,
                "complexity": complexity,
            },
            {
                "route_chain": ["direct_answer|tool_call|deep_think|plan_execute"],
                "complexity": "simple|complex|multi_step",
                "reason": "string",
                "tool_recall": [{"name": "string", "args": {}, "reason": "string"}],
            },
            max_new_tokens=512,
        )
        route_chain = [route for route in _string_list(data.get("route_chain"), limit=6) if route in _ALLOWED_ROUTES]
        if not route_chain:
            route_chain = [ROUTE_PLAN_EXECUTE if complexity.get("complexity") == COMPLEXITY_MULTI_STEP else ROUTE_DIRECT_ANSWER]
        tool_recall = _normalize_tool_recall(data.get("tool_recall"), tools_schema)
        if tool_recall and goal_analysis.get("needs_tool") and not _latest_message_is_tool(messages):
            route_chain = [ROUTE_TOOL_CALL] + [route for route in route_chain if route != ROUTE_TOOL_CALL]
            if ROUTE_DIRECT_ANSWER not in route_chain:
                route_chain.append(ROUTE_DIRECT_ANSWER)
        return RouteDecision(
            route_chain[0],
            _choice(data.get("complexity"), _ALLOWED_COMPLEXITIES, str(complexity.get("complexity", COMPLEXITY_SIMPLE))),
            _string(data.get("reason"), "Qwen selected route chain"),
            "qwen",
            goal_analysis,
            tuple(_dedupe_consecutive(route_chain)),
            tool_recall,
        )

    def reflect(
        self,
        messages: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        latest_tool = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
        has_tool_error = bool(latest_tool and latest_tool.get("status") == "error")
        has_plan_error = bool(plan and plan.get("status") == "failed")
        if not has_tool_error and not has_plan_error:
            return None
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Diagnose the latest B4-visible failure and decide whether retry/replan is safe.",
            {
                "latest_tool_message": latest_tool,
                "goal_analysis": goal_analysis,
                "plan": plan,
            },
            {
                "trigger": "tool_execution_failed|plan_execution_failed",
                "problem": "string",
                "diagnosis": "path_or_file_error|invalid_tool_arguments|permission_error|tool_error",
                "suggestion": "string",
                "retryable": "boolean",
            },
            max_new_tokens=512,
        )
        diagnosis = _choice(data.get("diagnosis"), _ALLOWED_DIAGNOSES, "tool_error")
        return {
            "trigger": _string(data.get("trigger"), "tool_execution_failed" if has_tool_error else "plan_execution_failed"),
            "problem": _string(data.get("problem"), _tool_problem(latest_tool) if latest_tool else "plan failed"),
            "diagnosis": diagnosis,
            "suggestion": _string(data.get("suggestion"), "review the failure and choose a safer next step"),
            "retryable": bool(data.get("retryable", diagnosis not in {"permission_error", "tool_error"})),
            "source": "qwen",
        }

    def select_tool_call(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        route_decision: RouteDecision,
    ) -> dict[str, Any]:
        from_recall = _first_recalled_tool(route_decision.tool_recall, tools_schema)
        if from_recall is not None:
            return from_recall
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Select exactly one tool call for the next B3 execution.",
            {
                "messages": _compact_messages(messages),
                "goal_analysis": goal_analysis,
                "route": route_decision.to_dict(),
                "tools": tools_schema,
            },
            {"id": "string", "name": "string", "args": {}},
            max_new_tokens=512,
        )
        return _normalize_tool_call(data, tools_schema)

    def select_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        route_decision: RouteDecision,
    ) -> list[dict[str, Any]]:
        recalled = _recalled_tools(route_decision.tool_recall, tools_schema)
        if recalled:
            return recalled
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Select one or more independent tool calls for the next B3 execution. Use multiple calls only when they can run in the same assistant turn.",
            {
                "messages": _compact_messages(messages),
                "goal_analysis": goal_analysis,
                "route": route_decision.to_dict(),
                "tools": tools_schema,
            },
            {"tool_calls": [{"id": "string", "name": "string", "args": {}}]},
            max_new_tokens=768,
        )
        raw_calls = data.get("tool_calls") if isinstance(data.get("tool_calls"), list) else [data]
        calls = [_normalize_tool_call(item, tools_schema) for item in raw_calls if isinstance(item, dict)]
        return _renumber_tool_calls(calls[:4]) or [self.select_tool_call(messages, tools_schema, goal_analysis, route_decision)]

    def create_plan(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Create a short executable B4 plan. Tool steps must use available tool names and valid argument objects.",
            {
                "messages": _compact_messages(messages),
                "goal_analysis": goal_analysis,
                "tools": tools_schema,
            },
            {
                "goal": "string",
                "steps": [
                    {"id": "integer", "type": "tool_call|synthesize|reflect", "tool": "string", "args": {}, "content": "string"}
                ],
                "status": "pending",
            },
            max_new_tokens=1024,
        )
        return _normalize_plan_payload(data, tools_schema, goal_analysis)

    def replan(
        self,
        plan: dict[str, Any],
        failed_step_id: int,
        reflection: dict[str, Any],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Locally replan only the failed step and later steps. Preserve successful prefix steps.",
            {
                "plan": plan,
                "failed_step_id": failed_step_id,
                "reflection": reflection,
                "tools": tools_schema,
                "goal_analysis": goal_analysis,
            },
            {
                "goal": "string",
                "steps": [
                    {"id": "integer", "type": "tool_call|synthesize|reflect", "tool": "string", "args": {}, "content": "string", "status": "pending|success|failed"}
                ],
                "status": "pending|failed",
                "replan": {
                    "failed_step_id": "integer",
                    "diagnosis": "string",
                    "suggestion": "string",
                    "preserved_step_ids": ["integer"],
                    "discarded_step_ids": ["integer"],
                },
            },
            max_new_tokens=1024,
        )
        normalized = _normalize_plan_payload(data, tools_schema, goal_analysis)
        replan = data.get("replan") if isinstance(data.get("replan"), dict) else {}
        normalized["replan"] = {
            "failed_step_id": int(replan.get("failed_step_id", failed_step_id) or failed_step_id),
            "diagnosis": _string(replan.get("diagnosis"), str(reflection.get("diagnosis", "tool_error"))),
            "suggestion": _string(replan.get("suggestion"), str(reflection.get("suggestion", "review failed step"))),
            "preserved_step_ids": _int_list(replan.get("preserved_step_ids")),
            "discarded_step_ids": _int_list(replan.get("discarded_step_ids")) or [failed_step_id],
            "source": "qwen",
        }
        return normalized

    def direct_answer(self, messages: list[dict[str, Any]], goal_analysis: dict[str, Any], reflection: dict[str, Any] | None = None) -> str:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are the B4 final answer writer. Start with the direct answer, using standard English relationship and kinship terminology, "
                    "then give the minimum supporting evidence needed to verify every qualifier in the latest user request. "
                    "Answer only the latest user request. Treat earlier messages as context only, and do not repeat an earlier answer unless the latest user explicitly asks for it. "
                    "A JSON object with a content field is preferred, but plain text is acceptable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Write the final assistant answer. Ground it in the successful tool results, include concise supporting evidence for every qualifier, and resolve relative dates from the containing record date. Do not treat search snippets or intermediate calculator values as final evidence when another tool result is needed. Preserve the user language and any JSON/format constraints.",
                        "payload": {
                            "latest_user_request": latest_user_text(messages),
                            "messages": _answer_focus_messages(messages, goal_analysis, max_chars=4000),
                            "goal_analysis": goal_analysis,
                            "reflection": reflection,
                            "output_rules": [
                                "Answer latest_user_request only.",
                                "Use earlier messages only when latest_user_request is an explicit follow-up.",
                                "Do not repeat an earlier assistant answer unless latest_user_request asks for it.",
                                "Verify names, locations, dates, and relationships against tool evidence.",
                                "Include a short quote or faithful paraphrase of the supporting evidence.",
                                "Resolve yesterday, last week, and similar relative dates from the record or session date.",
                                "Never use Day N as a calendar date; if the source lacks a dated header, more evidence is required.",
                                "Normalize informal relationship words to their standard English noun while preserving the source wording in evidence.",
                            ],
                        },
                        "preferred_json_schema": {"content": "string"},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        completion = qwen_generate_completion(
            self.config_path,
            self.config,
            prompt_messages,
            max_new_tokens=768,
        )
        usage_records = self.config.setdefault("_b4_usage_records", [])
        if isinstance(usage_records, list):
            model_config = self.config.get("model", {}) if isinstance(self.config.get("model"), dict) else {}
            tool_calling = self.config.get("tool_calling", {}) if isinstance(self.config.get("tool_calling"), dict) else {}
            usage_records.append(
                {
                    "task": "Write the final assistant answer. Ground it in the successful tool results, include concise supporting evidence for every qualifier, and resolve relative dates from the containing record date. Do not treat search snippets or intermediate calculator values as final evidence when another tool result is needed. Preserve the user language and any JSON/format constraints.",
                    "usage": completion.get("usage"),
                    "model": model_config.get("model_name_or_path"),
                    "model_profile": self.config.get("selected_model_profile"),
                    "tool_calling_mode": tool_calling.get("mode"),
                }
            )
        raw_text = completion["text"].strip()
        if not raw_text:
            return "I could not produce a final answer from the available context."
        try:
            data = extract_json_object(raw_text)
            content = _string(data.get("content"), "")
            return content or raw_text
        except Exception:
            return raw_text

    def deep_think(self, messages: list[dict[str, Any]], goal_analysis: dict[str, Any] | None = None) -> ToTResult:
        goal = str((goal_analysis or {}).get("goal") or latest_user_text(messages) or "complete reasoning task")
        context = _latest_tool_context(messages)
        engine = DeepThinkToT(
            max_steps=3,
            max_node_retries=1,
            max_global_backtracks=4,
            step_generator=self._generate_thought_step,
            reflection_checker=self._reflect_thought_step,
        )
        return engine.run(goal, context)

    def _generate_thought_step(
        self,
        goal: str,
        context: str | None,
        previous_steps: list[ThoughtStep],
        step_id: int,
        retry_count: int,
    ) -> dict[str, str]:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Generate one concise DeepThink ToT reasoning step.",
            {
                "goal": goal,
                "context": context,
                "previous_steps": [step.to_dict() for step in previous_steps],
                "step_id": step_id,
                "retry_count": retry_count,
            },
            {"thought": "string", "reasoning": "string", "conclusion": "string"},
            max_new_tokens=512,
        )
        return {
            "thought": _string(data.get("thought"), f"Reasoning step {step_id}"),
            "reasoning": _string(data.get("reasoning"), "Based on the available context, continue the reasoning chain."),
            "conclusion": _string(data.get("conclusion"), "The reasoning remains in scope."),
        }

    def _reflect_thought_step(self, goal: str, step: ThoughtStep, previous_steps: list[ThoughtStep]) -> dict[str, Any]:
        data = qwen_generate_json(
            self.config_path,
            self.config,
            "Reflect on whether one DeepThink step is coherent, useful, and within scope.",
            {
                "goal": goal,
                "step": step.to_dict(),
                "previous_steps": [item.to_dict() for item in previous_steps],
            },
            {"passed": "boolean", "check": "string", "reason": "string"},
            max_new_tokens=256,
        )
        return {
            "passed": bool(data.get("passed", True)),
            "check": _string(data.get("check"), "qwen_reflection"),
            "reason": _string(data.get("reason"), "Qwen reflection completed."),
            "source": "qwen",
        }


def _compact_messages(messages: list[dict[str, Any]], max_chars: int = 1600) -> list[dict[str, Any]]:
    compact = []
    for message in messages[-6:]:
        item = {key: message.get(key) for key in ("role", "content", "status", "name", "tool_call_id") if key in message}
        if isinstance(item.get("content"), str) and len(item["content"]) > max_chars:
            item["content"] = item["content"][:max_chars] + "...[truncated]"
        if "tool_calls" in message:
            item["tool_calls"] = message["tool_calls"]
        if "plan" in message:
            item["plan"] = message["plan"]
        compact.append(item)
    return compact


def _answer_focus_messages(messages: list[dict[str, Any]], goal_analysis: dict[str, Any], max_chars: int = 1600) -> list[dict[str, Any]]:
    if messages and messages[-1].get("role") == "tool":
        return _compact_messages(messages, max_chars=max_chars)
    target = latest_user_text(messages)
    intent = str(goal_analysis.get("intent") or "")
    direct_markers = ("\u76f4\u63a5\u56de\u7b54", "\u53ea\u56de\u7b54", "\u7b80\u77ed\u56de\u7b54", "\u4e00\u53e5\u8bdd")
    if target and (intent == "direct" or target.startswith(direct_markers)):
        content = target if len(target) <= max_chars else target[:max_chars] + "...[truncated]"
        return [{"role": "user", "content": content}]
    return _compact_messages(messages, max_chars=max_chars)


def _tool_summaries(tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        summaries.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
            }
        )
    return summaries


def _tool_names(tools_schema: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("name")) for item in _tool_summaries(tools_schema) if item.get("name")}


def _normalize_tool_recall(value: Any, tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = _tool_names(tools_schema)
    result = []
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name not in names:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        result.append({"name": name, "args": args, "reason": _string(item.get("reason"), "Qwen recalled this tool")})
    return result[:3]


def _first_recalled_tool(tool_recall: list[dict[str, Any]] | None, tools_schema: list[dict[str, Any]]) -> dict[str, Any] | None:
    tools = _recalled_tools(tool_recall, tools_schema)
    return tools[0] if tools else None


def _recalled_tools(tool_recall: list[dict[str, Any]] | None, tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tool_recall:
        return []
    names = _tool_names(tools_schema)
    calls = []
    for item in tool_recall:
        if item.get("name") in names:
            calls.append(normalize_tool_call({"name": item["name"], "args": item.get("args", {})}, len(calls)))
    return _renumber_tool_calls(calls[:4])


def _renumber_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": f"call_{index:03d}", "name": call["name"], "args": call.get("args", {})}
        for index, call in enumerate(calls, 1)
    ]


def _normalize_tool_call(data: dict[str, Any], tools_schema: list[dict[str, Any]]) -> dict[str, Any]:
    names = _tool_names(tools_schema)
    name = data.get("name")
    if name not in names:
        name = next(iter(names), "file_reader")
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    return normalize_tool_call({"id": _string(data.get("id"), "call_001"), "name": name, "args": args})


def _normalize_plan_payload(data: dict[str, Any], tools_schema: list[dict[str, Any]], goal_analysis: dict[str, Any]) -> dict[str, Any]:
    names = _tool_names(tools_schema)
    steps = []
    raw_steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    for item in raw_steps[:6]:
        if not isinstance(item, dict):
            continue
        step_type = item.get("type")
        if step_type == "tool_call":
            tool = item.get("tool")
            if tool not in names:
                continue
            steps.append(
                {
                    "id": len(steps) + 1,
                    "type": "tool_call",
                    "tool": tool,
                    "args": item.get("args") if isinstance(item.get("args"), dict) else {},
                    "max_attempts": 3,
                    "status": _choice(item.get("status"), {"pending", "success", "failed"}, "pending"),
                }
            )
        elif step_type in {"synthesize", "reflect"}:
            steps.append(
                {
                    "id": len(steps) + 1,
                    "type": step_type,
                    "content": _string(item.get("content"), "Synthesize available observations and answer the user goal."),
                    "max_attempts": 1,
                    "status": _choice(item.get("status"), {"pending", "success", "failed"}, "pending"),
                }
            )
    if not steps or steps[-1].get("type") != "synthesize":
        steps.append(
            {
                "id": len(steps) + 1,
                "type": "synthesize",
                "content": "Synthesize available observations and answer the user goal.",
                "max_attempts": 1,
                "status": "pending",
            }
        )
    return {
        "goal": _string(data.get("goal"), str(goal_analysis.get("goal") or "complete user request")),
        "steps": steps,
        "status": _choice(data.get("status"), {"pending", "failed"}, "pending"),
        "retry_policy": {"max_attempts_per_step": 3},
        "source": "qwen",
    }


def _latest_tool_context(messages: list[dict[str, Any]]) -> str | None:
    latest_tool = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
    if latest_tool is None or not isinstance(latest_tool.get("content"), str):
        return None
    return latest_tool["content"]


def _latest_message_is_tool(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[-1].get("role") == "tool"


def _tool_problem(tool_message: dict[str, Any] | None) -> str:
    if not tool_message or not isinstance(tool_message.get("content"), str):
        return "tool returned an invalid error payload"
    try:
        payload = json.loads(tool_message["content"])
    except json.JSONDecodeError:
        return tool_message["content"][:200]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return str(payload)[:200]


def _string(value: Any, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        if len(result) >= limit:
            break
    return result


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _choice(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return default


def _dedupe_consecutive(routes: list[str]) -> list[str]:
    result = []
    for route in routes:
        if not result or result[-1] != route:
            result.append(route)
    return result
