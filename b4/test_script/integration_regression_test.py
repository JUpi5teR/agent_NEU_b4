from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from b4.router import ROUTE_DIRECT_ANSWER, ROUTE_PLAN_EXECUTE, ROUTE_TOOL_CALL, RouteDecision
from b4.service import (
    _advance_route_from_state,
    _first_unread_search_result_path,
    _focus_search_tool_calls,
)


def _tool_message(skill_name: str, input_payload: dict, output: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_001",
        "name": skill_name,
        "status": "success",
        "content": json.dumps(
            {
                "skill_name": skill_name,
                "status": "success",
                "input": input_payload,
                "output": output,
                "error": None,
            }
        ),
    }


def _decision(route: str, recall: list[dict] | None = None) -> RouteDecision:
    return RouteDecision(
        route,
        "simple",
        "test decision",
        "test",
        {"intent": "search"},
        (route,),
        recall or [],
    )


def test_direct_continuation_converges() -> None:
    messages = [_tool_message("calculator", {"expression": "(7.5) / (1/2)"}, {"result": 15.0})]
    previous = {
        "status": "waiting_for_tool",
        "current_route": ROUTE_DIRECT_ANSWER,
        "next_route": None,
        "route_chain": [ROUTE_TOOL_CALL, ROUTE_DIRECT_ANSWER],
    }
    result = _advance_route_from_state(messages, previous, _decision(ROUTE_PLAN_EXECUTE), {"intent": "calculate"}, {"complexity": "multi_step"})
    assert result.route == ROUTE_DIRECT_ANSWER
    assert result.source == "route_chain_state"


def test_new_evidence_recall_overrides_direct_answer() -> None:
    messages = [_tool_message("file_reader", {"path": "docs/first.txt"}, {"content": "first fact"})]
    previous = {
        "status": "waiting_for_tool",
        "current_route": ROUTE_TOOL_CALL,
        "next_route": ROUTE_DIRECT_ANSWER,
        "route_chain": [ROUTE_TOOL_CALL, ROUTE_DIRECT_ANSWER],
    }
    recall = [{"name": "local_file_search", "args": {"query": "second fact", "root_dir": "docs"}, "reason": "qualifier missing"}]
    result = _advance_route_from_state(messages, previous, _decision(ROUTE_DIRECT_ANSWER, recall), {"intent": "search"}, {"complexity": "simple"})
    assert result.route == ROUTE_TOOL_CALL
    assert result.source == "evidence_sufficiency"


def test_duplicate_recall_does_not_loop() -> None:
    args = {"path": "docs/first.txt", "max_chars": 8000}
    messages = [_tool_message("file_reader", args, {"content": "complete fact"})]
    previous = {
        "status": "waiting_for_tool",
        "current_route": ROUTE_TOOL_CALL,
        "next_route": ROUTE_DIRECT_ANSWER,
        "route_chain": [ROUTE_TOOL_CALL, ROUTE_DIRECT_ANSWER],
    }
    recall = [{"name": "file_reader", "args": args, "reason": "duplicate"}]
    result = _advance_route_from_state(messages, previous, _decision(ROUTE_DIRECT_ANSWER, recall), {"intent": "search"}, {"complexity": "simple"})
    assert result.route == ROUTE_DIRECT_ANSWER


def test_oversized_transcript_reader_becomes_focused_search() -> None:
    user = {
        "role": "user",
        "content": (
            "Who gave Maria's family money when she was younger and her family was going through tough times?\n\n"
            "Available local execution assets (paths are relative to the configured data root):\n"
            "- primary_file: b1_advanced/materialized/locomo/conv-41/conversation.txt\n"
            "- search_root: b1_advanced/materialized/locomo/conv-41"
        ),
    }
    ai_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_001", "name": "file_reader", "args": {"path": "b1_advanced/materialized/locomo/conv-41/conversation.txt", "max_chars": 100000}}],
    }
    tools = [
        {"type": "function", "function": {"name": "file_reader", "parameters": {}}},
        {"type": "function", "function": {"name": "local_file_search", "parameters": {}}},
    ]
    result = _focus_search_tool_calls(ai_message, [user], tools, {"intent": "search"})
    call = result["tool_calls"][0]
    assert call["name"] == "local_file_search"
    assert call["args"] == {
        "query": "gave money younger through tough",
        "root_dir": "b1_advanced/materialized/locomo/conv-41",
        "file_types": ["txt", "md"],
        "top_k": 10,
        "mode": "keyword",
    }


def test_search_result_prefers_focused_chunk() -> None:
    user = {
        "role": "user",
        "content": (
            "When did Maria donate her car?\n\n"
            "Available local execution assets (paths are relative to the configured data root):\n"
            "- primary_file: corpus/conversation.txt\n"
            "- search_root: corpus"
        ),
    }
    payload = {
        "input": {"query": "donate car"},
        "output": {
            "results": [
                {"path": "corpus/conversation.txt", "snippet": "I donated my old car yesterday"},
                {"path": "corpus/session_2.txt", "snippet": "I donated my old car yesterday"},
                {"path": "corpus/session_11.txt", "snippet": "my car broke down"},
            ]
        },
    }
    assert _first_unread_search_result_path([user], payload) == "corpus/session_2.txt"


def main() -> int:
    test_direct_continuation_converges()
    test_new_evidence_recall_overrides_direct_answer()
    test_duplicate_recall_does_not_loop()
    test_oversized_transcript_reader_becomes_focused_search()
    test_search_result_prefers_focused_chunk()
    print(json.dumps({"status": "success", "tests": 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
