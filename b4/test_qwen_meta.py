from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from b4.deep_think_tot import ThoughtStep, ToTResult
from b4.router import ROUTE_DEEP_THINK, ROUTE_DIRECT_ANSWER, ROUTE_PLAN_EXECUTE, ROUTE_TOOL_CALL
from b4.service import generate_ai_message
from common.io_utils import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model.yaml"
DEFAULT_TOOLS_SCHEMA = ROOT / "data" / "messages" / "tools_schema_basic.json"
DEFAULT_OUTDIR = ROOT / "outputs" / "B4" / "v2_qwen_meta"


def _messages(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _model_config(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "model_qwen_meta_fake.yaml"
    path.write_text(
        "runtime:\n"
        "  default_mode: prompt_json\n"
        "model:\n"
        "  backend: qwen_fake\n",
        encoding="utf-8",
    )
    return path


class FakeQwenB4MetaEngine:
    def __init__(self, config_path: Path, config: dict[str, Any]) -> None:
        self.config_path = config_path
        self.config = config

    def analyze_goal_with_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        text = _latest_user_text(messages)
        return {
            "goal": text,
            "intent": "analyze",
            "keywords": ["llm_goal"],
            "constraints": {"must_include": [], "must_exclude": []},
            "needs_tool": False,
            "source": "qwen",
            "confidence": "high",
            "reason": "fake qwen goal fallback",
        }

    def judge_complexity(self, messages: list[dict[str, Any]], goal_analysis: dict[str, Any]) -> dict[str, Any]:
        text = _latest_user_text(messages).lower()
        if "plan" in text:
            complexity = "multi_step"
        elif "deep" in text:
            complexity = "complex"
        else:
            complexity = "simple"
        return {"complexity": complexity, "reason": "fake qwen complexity", "source": "qwen"}

    def decide_route(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        complexity: dict[str, Any],
    ) -> Any:
        from b4.router import RouteDecision

        text = _latest_user_text(messages).lower()
        if "plan" in text:
            return RouteDecision(ROUTE_PLAN_EXECUTE, "multi_step", "fake qwen plan", "qwen", goal_analysis, (ROUTE_PLAN_EXECUTE,))
        if "deep" in text:
            return RouteDecision(
                ROUTE_DEEP_THINK,
                "complex",
                "fake qwen deep think",
                "qwen",
                goal_analysis,
                (ROUTE_DEEP_THINK, ROUTE_DIRECT_ANSWER),
            )
        if "read" in text:
            return RouteDecision(
                ROUTE_TOOL_CALL,
                "simple",
                "fake qwen tool recall",
                "qwen",
                goal_analysis,
                (ROUTE_TOOL_CALL, ROUTE_DIRECT_ANSWER),
                [{"name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 5000}, "reason": "read request"}],
            )
        return RouteDecision(ROUTE_DIRECT_ANSWER, "simple", "fake qwen direct", "qwen", goal_analysis, (ROUTE_DIRECT_ANSWER,))

    def reflect(
        self,
        messages: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        latest_tool = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
        if latest_tool and latest_tool.get("status") == "error":
            return {
                "trigger": "tool_execution_failed",
                "problem": "missing file",
                "diagnosis": "path_or_file_error",
                "suggestion": "search for the file first",
                "retryable": True,
                "source": "qwen",
            }
        return None

    def select_tool_call(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        route_decision: Any,
    ) -> dict[str, Any]:
        return {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 5000}}

    def create_plan(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "goal": goal_analysis["goal"],
            "steps": [
                {
                    "id": 1,
                    "type": "tool_call",
                    "tool": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 5000},
                    "status": "pending",
                },
                {"id": 2, "type": "synthesize", "content": "answer from observations", "status": "pending"},
            ],
            "status": "pending",
            "retry_policy": {"max_attempts_per_step": 3},
            "source": "qwen",
        }

    def replan(
        self,
        plan: dict[str, Any],
        failed_step_id: int,
        reflection: dict[str, Any],
        tools_schema: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "goal": plan["goal"],
            "steps": [
                {
                    "id": 1,
                    "type": "tool_call",
                    "tool": "local_file_search",
                    "args": {"query": plan["goal"], "top_k": 5},
                    "status": "pending",
                },
                {"id": 2, "type": "synthesize", "content": "answer after recovery", "status": "pending"},
            ],
            "status": "pending",
            "retry_policy": {"max_attempts_per_step": 3},
            "replan": {
                "failed_step_id": failed_step_id,
                "diagnosis": reflection["diagnosis"],
                "suggestion": reflection["suggestion"],
                "preserved_step_ids": [],
                "discarded_step_ids": [failed_step_id],
                "source": "qwen",
            },
            "source": "qwen",
        }

    def direct_answer(
        self,
        messages: list[dict[str, Any]],
        goal_analysis: dict[str, Any],
        reflection: dict[str, Any] | None = None,
    ) -> str:
        return "fake qwen direct answer"

    def deep_think(self, messages: list[dict[str, Any]], goal_analysis: dict[str, Any] | None = None) -> ToTResult:
        step = ThoughtStep(
            id=1,
            thought="fake qwen thought",
            reasoning="Based on the goal, reason within scope.",
            conclusion="fake qwen deep conclusion",
            status="success",
        )
        return ToTResult(
            goal=str((goal_analysis or {}).get("goal") or "deep goal"),
            steps=[step],
            final_answer="fake qwen deep final answer",
            trace=[{"action": "reflection", "source": "qwen"}, {"action": "final_answer", "step_count": 1}],
        )


class GoalFailureFakeQwenB4MetaEngine(FakeQwenB4MetaEngine):
    def analyze_goal_with_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("forced goal analysis failure")


def test_fake_qwen_tool_recall(outdir: Path) -> None:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    config = _model_config(outdir)
    case_dir = outdir / "fake_tool"
    with patch("b4.service.QwenB4MetaEngine", FakeQwenB4MetaEngine):
        result = generate_ai_message(
            str(config),
            _messages("read docs/agent_intro.txt"),
            tools_schema,
            "prompt_json",
            str(case_dir),
            "tool",
        )
    message = result["ai_message"]
    metadata = message["metadata"]
    assert result["status"] == "success"
    assert message["tool_calls"][0]["name"] == "file_reader"
    assert metadata["qwen_meta_enabled"] is True
    assert metadata["goal_analysis"]["source"] == "rules"
    assert metadata["goal_analysis"]["strategy"] == "rule_first"
    assert metadata["route_source"] == "qwen"
    assert metadata["tool_recall"][0]["name"] == "file_reader"
    assert (case_dir / "tool_ai_message.json").exists()
    assert (case_dir / "tool_raw_model_output.json").exists()


def test_fake_qwen_plan_direct_and_deep(outdir: Path) -> None:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    config = _model_config(outdir)
    with patch("b4.service.QwenB4MetaEngine", FakeQwenB4MetaEngine):
        plan_result = generate_ai_message(str(config), _messages("plan read docs/agent_intro.txt then answer"), tools_schema, "prompt_json")
        direct_result = generate_ai_message(str(config), _messages("answer directly"), tools_schema, "prompt_json")
        deep_result = generate_ai_message(str(config), _messages("deep analyze this"), tools_schema, "prompt_json")
    assert plan_result["ai_message"]["type"] == ROUTE_PLAN_EXECUTE
    assert plan_result["ai_message"]["plan"]["source"] == "qwen"
    assert direct_result["ai_message"]["content"] == "fake qwen direct answer"
    assert "DeepThink reasoning process:" in deep_result["ai_message"]["content"]
    assert "fake qwen thought" in deep_result["ai_message"]["content"]
    assert deep_result["ai_message"]["metadata"]["deep_think_tot"]["steps"][0]["thought"] == "fake qwen thought"


def test_low_confidence_goal_uses_llm_fallback(outdir: Path) -> None:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    config = _model_config(outdir)
    with patch("b4.service.QwenB4MetaEngine", FakeQwenB4MetaEngine):
        result = generate_ai_message(
            str(config),
            _messages("handle this ambiguous request"),
            tools_schema,
            "prompt_json",
        )
    goal_analysis = result["ai_message"]["metadata"]["goal_analysis"]
    assert goal_analysis["source"] == "qwen"
    assert goal_analysis["strategy"] == "llm_fallback"
    assert goal_analysis["keywords"] == ["llm_goal"]
    assert goal_analysis["rule_attempt"]["confidence"] == "low"


def test_low_confidence_goal_llm_failure_uses_minimal_fallback(outdir: Path) -> None:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    config = _model_config(outdir)
    with patch("b4.service.QwenB4MetaEngine", GoalFailureFakeQwenB4MetaEngine):
        result = generate_ai_message(
            str(config),
            _messages("handle this ambiguous request"),
            tools_schema,
            "prompt_json",
        )
    goal_analysis = result["ai_message"]["metadata"]["goal_analysis"]
    errors = result["ai_message"]["metadata"]["qwen_meta_errors"]
    assert goal_analysis["source"] == "minimal_fallback"
    assert goal_analysis["keywords"] == []
    assert goal_analysis["constraints"] == {"must_include": [], "must_exclude": []}
    assert any(error["stage"] == "qwen_goal_analysis" for error in errors)


def test_fake_qwen_reflect_and_replan(outdir: Path) -> None:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    config = _model_config(outdir)
    messages = _messages("plan read docs/agent_intro.txt then answer")
    with patch("b4.service.QwenB4MetaEngine", FakeQwenB4MetaEngine):
        first = generate_ai_message(str(config), messages, tools_schema, "prompt_json")
        failed_messages = messages + [
            first["ai_message"],
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "name": "file_reader",
                "status": "error",
                "content": json.dumps({"status": "error", "error": {"message": "no such file"}}, ensure_ascii=False),
            },
        ]
        result = generate_ai_message(str(config), failed_messages, tools_schema, "prompt_json", str(outdir / "fake_replan"), "replan")
    message = result["ai_message"]
    assert result["status"] == "success"
    assert message["type"] == ROUTE_PLAN_EXECUTE
    assert message["tool_calls"][0]["name"] == "local_file_search"
    assert message["metadata"]["reflection"]["source"] == "qwen"
    assert message["plan"]["replan"]["source"] == "qwen"


class DirectIntentFakeQwenB4MetaEngine(FakeQwenB4MetaEngine):
    def analyze_goal_with_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        text = _latest_user_text(messages)
        return {
            "goal": text,
            "intent": "analyze",
            "keywords": ["llm_goal"],
            "constraints": {"must_include": [], "must_exclude": []},
            "needs_tool": False,
            "source": "qwen",
            "confidence": "high",
            "reason": "fake qwen misclassified direct request",
        }


def test_direct_answer_constraints_prevent_deep_think(outdir: Path) -> None:
    config = _model_config(outdir / "direct_constraints")
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    cases = [
        "你好啊",
        "你觉得天空大还是地球大，简单回答我，不要分析",
        "你觉得天空大还是地球大，我要真正的数据，证明给我，简单回答",
    ]
    with patch("b4.service.QwenB4MetaEngine", DirectIntentFakeQwenB4MetaEngine):
        for index, query in enumerate(cases, 1):
            result = generate_ai_message(
                str(config),
                _messages(query),
                tools_schema,
                "prompt_json",
                str(outdir / "direct_constraints"),
                f"case_{index}",
            )
            metadata = result["ai_message"].get("metadata", {})
            assert result["status"] == "success"
            assert metadata.get("route") == ROUTE_DIRECT_ANSWER
            assert metadata.get("route_chain") == [ROUTE_DIRECT_ANSWER]
            assert metadata.get("goal_analysis", {}).get("intent") == "direct"


def test_complexity_uses_goal_analysis() -> None:
    from b4.complexity import COMPLEXITY_COMPLEX, COMPLEXITY_SIMPLE, ComplexityJudge

    judge = ComplexityJudge()
    direct = {"intent": "direct", "keywords": [], "needs_tool": False}
    calculate = {"intent": "calculate", "keywords": [], "needs_tool": True}
    reasoning = {"intent": "compare", "keywords": ["proof"], "needs_tool": False}

    assert judge.judge("answer directly, do not analyze", direct).complexity == COMPLEXITY_SIMPLE
    assert judge.judge("calculate 25+37*2", calculate).complexity == COMPLEXITY_SIMPLE
    assert judge.judge("prove matrix multiplication is not an abelian group", reasoning).complexity == COMPLEXITY_COMPLEX


def test_qwen_simple_complexity_is_rule_scored_for_proof(outdir: Path) -> None:
    config = _model_config(outdir / "complexity_score")
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    with patch("b4.service.QwenB4MetaEngine", FakeQwenB4MetaEngine):
        result = generate_ai_message(
            str(config),
            _messages("prove this statement"),
            tools_schema,
            "prompt_json",
            str(outdir / "complexity_score"),
            "proof",
        )
    metadata = result["ai_message"].get("metadata", {})
    assert result["status"] == "success"
    assert metadata.get("complexity_analysis", {}).get("complexity") == "complex"
    assert metadata.get("route") == ROUTE_DEEP_THINK


def run_real_qwen_smoke(model_config: Path, outdir: Path) -> dict[str, Any]:
    tools_schema = read_json(DEFAULT_TOOLS_SCHEMA)
    case_dir = outdir / "qwen_real"
    case_dir.mkdir(parents=True, exist_ok=True)
    result = generate_ai_message(
        str(model_config),
        _messages("Read the local file docs/agent_intro.txt and return a tool call if file content is needed."),
        tools_schema,
        "prompt_json",
        str(case_dir),
        "real_tool",
    )
    if result.get("status") != "success":
        raise AssertionError(f"qwen meta smoke failed: {result.get('error')}")
    message = result["ai_message"]
    metadata = message.get("metadata", {})
    if metadata.get("goal_analysis", {}).get("source") != "rules":
        raise AssertionError(f"rule fast path did not handle clear tool goal: {metadata.get('goal_analysis')}")
    if metadata.get("route_source") != "qwen":
        raise AssertionError(f"route decision did not come from Qwen: {metadata.get('route_source')}")
    raw = read_json(case_dir / "real_tool_raw_model_output.json")
    if raw.get("backend") != "qwen_meta":
        raise AssertionError(f"unexpected backend: {raw.get('backend')}")
    llm_goal_result = generate_ai_message(
        str(model_config),
        _messages("Handle this ambiguous request with the most appropriate response."),
        tools_schema,
        "prompt_json",
        str(case_dir),
        "real_llm_goal",
    )
    if llm_goal_result.get("status") != "success":
        raise AssertionError(f"qwen low-confidence goal smoke failed: {llm_goal_result.get('error')}")
    llm_goal_metadata = llm_goal_result["ai_message"].get("metadata", {})
    if llm_goal_metadata.get("goal_analysis", {}).get("source") != "qwen":
        raise AssertionError(f"low-confidence goal did not fall back to Qwen: {llm_goal_metadata.get('goal_analysis')}")
    summary = {
        "status": "success",
        "stage": "qwen_meta_real_smoke",
        "artifact_dir": str(case_dir),
        "route": metadata.get("route"),
        "route_chain": metadata.get("route_chain"),
        "has_tool_calls": bool(message.get("tool_calls")),
        "rule_goal_source": metadata.get("goal_analysis", {}).get("source"),
        "llm_goal_source": llm_goal_metadata.get("goal_analysis", {}).get("source"),
    }
    write_json(summary, case_dir / "real_qwen_meta_summary.json")
    return summary


def run_all(outdir: Path, run_real: bool, model_config: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    test_fake_qwen_tool_recall(outdir)
    test_fake_qwen_plan_direct_and_deep(outdir)
    test_low_confidence_goal_uses_llm_fallback(outdir)
    test_low_confidence_goal_llm_failure_uses_minimal_fallback(outdir)
    test_fake_qwen_reflect_and_replan(outdir)
    test_direct_answer_constraints_prevent_deep_think(outdir)
    test_complexity_uses_goal_analysis()
    test_qwen_simple_complexity_is_rule_scored_for_proof(outdir)
    summary: dict[str, Any] = {
        "status": "success",
        "stage": "b4_qwen_meta",
        "tests": 8,
        "artifact_dir": str(outdir),
        "real_qwen": "skipped",
    }
    if run_real:
        summary["real_qwen"] = run_real_qwen_smoke(model_config, outdir)
    write_json(summary, outdir / "test_summary.json")
    return summary


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="B4 Qwen meta integration tests.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--run-real", action="store_true", help="Load the configured local Qwen model and run a smoke test.")
    args = parser.parse_args()
    summary = run_all(Path(args.outdir).resolve(), args.run_real, Path(args.model_config).resolve())
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
