from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_COMPLEX = "complex"
COMPLEXITY_MULTI_STEP = "multi_step"


@dataclass(frozen=True)
class ComplexityDecision:
    complexity: str
    reason: str
    source: str = "rules"

    def to_dict(self) -> dict[str, str]:
        return {
            "complexity": self.complexity,
            "reason": self.reason,
            "source": self.source,
        }


class ComplexityJudge:
    """Small rule-first complexity classifier for B4 routing."""

    _MULTI_STEP_MARKERS = (
        "first",
        "then",
        "finally",
        "step",
        "steps",
        "multi-step",
        "read and",
        "search and",
        "calculate and",
        "\u5148",
        "\u7136\u540e",
        "\u6700\u540e",
        "\u6b65\u9aa4",
        "\u591a\u6b65",
        "\u8bfb\u53d6",
        "\u8ba1\u7b97",
        "\u8f93\u51fa",
    )
    _COMPLEX_MARKERS = (
        "analyze",
        "analysis",
        "compare",
        "why",
        "evaluate",
        "reason",
        "prove",
        "proof",
        "derive",
        "tradeoff",
        "pros and cons",
        "\u5206\u6790",
        "\u6bd4\u8f83",
        "\u4e3a\u4ec0\u4e48",
        "\u533a\u522b",
        "\u8bc4\u4f30",
        "\u539f\u56e0",
        "\u63a8\u7406",
    )
    _SIMPLE_MARKERS = (
        "what is",
        "who is",
        "define",
        "\u4ec0\u4e48\u662f",
        "\u662f\u4ec0\u4e48",
        "\u8c01\u662f",
    )

    def judge(
        self,
        messages_or_text: list[dict[str, Any]] | str,
        goal_analysis: dict[str, Any] | None = None,
    ) -> ComplexityDecision:
        text = latest_user_text(messages_or_text).strip()
        lowered = text.lower()
        analysis = goal_analysis or {}
        intent = str(analysis.get("intent") or "")
        if not text:
            return ComplexityDecision(COMPLEXITY_SIMPLE, "empty input falls back to simple")
        if intent == "direct" or _looks_like_direct_answer_request(lowered, text):
            return ComplexityDecision(COMPLEXITY_SIMPLE, "direct-answer goal matched")
        multi_hits = [marker for marker in self._MULTI_STEP_MARKERS if marker in lowered or marker in text]
        if len(multi_hits) >= 2:
            return ComplexityDecision(COMPLEXITY_MULTI_STEP, f"multi-step markers: {', '.join(multi_hits[:3])}")
        if _goal_needs_deep_reasoning(intent, lowered, text, analysis):
            return ComplexityDecision(COMPLEXITY_COMPLEX, "goal analysis indicates reasoning depth")
        if intent in {"search", "calculate", "execute"}:
            return ComplexityDecision(COMPLEXITY_SIMPLE, "tool/action goal is simple unless multi-step")
        if _contains_reasoning_marker(lowered, text):
            return ComplexityDecision(COMPLEXITY_COMPLEX, "reasoning-depth marker matched")
        if any(marker in lowered or marker in text for marker in self._COMPLEX_MARKERS):
            return ComplexityDecision(COMPLEXITY_COMPLEX, "complex reasoning marker matched")
        if any(marker in lowered or marker in text for marker in self._SIMPLE_MARKERS):
            return ComplexityDecision(COMPLEXITY_SIMPLE, "simple question marker matched")
        if len(text) > 120 or text.count("?") + text.count("\uff1f") > 1:
            return ComplexityDecision(COMPLEXITY_COMPLEX, "long or compound question")
        return ComplexityDecision(COMPLEXITY_SIMPLE, "default simple fallback")


def latest_user_text(messages_or_text: list[dict[str, Any]] | str) -> str:
    if isinstance(messages_or_text, str):
        return messages_or_text
    for message in reversed(messages_or_text):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _looks_like_direct_answer_request(lowered: str, text: str) -> bool:
    stripped = text.strip()
    if stripped in {"你好", "你好啊", "您好", "hello", "hi", "hey"}:
        return True
    return any(
        (marker in lowered) if marker.isascii() else (marker in text)
        for marker in (
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
        )
    )


def _goal_needs_deep_reasoning(intent: str, lowered: str, text: str, goal_analysis: dict[str, Any]) -> bool:
    if intent not in {"analyze", "compare"}:
        return False
    if _contains_reasoning_marker(lowered, text):
        return True
    keywords = " ".join(str(item) for item in goal_analysis.get("keywords", []) if isinstance(item, str))
    return _contains_reasoning_marker(keywords.lower(), keywords)


def _contains_reasoning_marker(lowered: str, text: str) -> bool:
    return any(
        (marker in lowered) if marker.isascii() else (marker in text)
        for marker in (
            "analyze",
            "analysis",
            "why",
            "reason",
            "evaluate",
            "prove",
            "derive",
            "proof",
            "tradeoff",
            "分析",
            "为什么",
            "评估",
            "原因",
            "推理",
            "证明",
            "推导",
            "公式",
            "性质",
        )
    )
