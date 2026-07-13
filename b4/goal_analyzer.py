from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .complexity import latest_user_text

INTENT_EXECUTE = "execute"
INTENT_ANALYZE = "analyze"
INTENT_COMPARE = "compare"
INTENT_SEARCH = "search"
INTENT_CALCULATE = "calculate"
INTENT_EXPLAIN = "explain"
INTENT_DIRECT = "direct"
SOURCE_RULES = "rules"
SOURCE_LLM = "qwen"
SOURCE_MINIMAL_FALLBACK = "minimal_fallback"
CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"

_ALLOWED_INTENTS = {
    INTENT_EXECUTE,
    INTENT_ANALYZE,
    INTENT_COMPARE,
    INTENT_SEARCH,
    INTENT_CALCULATE,
    INTENT_EXPLAIN,
    INTENT_DIRECT,
}
GoalLlmAnalyzer = Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class GoalAnalysis:
    goal: str
    intent: str
    keywords: list[str]
    constraints: dict[str, list[str]]
    needs_tool: bool
    source: str = SOURCE_RULES
    confidence: str = CONFIDENCE_LOW
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "intent": self.intent,
            "keywords": self.keywords,
            "constraints": self.constraints,
            "needs_tool": self.needs_tool,
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class GoalAnalyzer:
    """Rule-based goal analyzer used before B4 routing."""

    def analyze(self, messages_or_text: list[dict[str, Any]] | str) -> GoalAnalysis:
        user_input = latest_user_text(messages_or_text).strip()
        intent, reason, matched = _detect_intent(user_input)
        keywords = _extract_keywords(user_input)
        constraints = _extract_constraints(user_input)
        needs_tool = _needs_tool(user_input, intent)
        confidence = CONFIDENCE_HIGH if matched or needs_tool or any(constraints.values()) else CONFIDENCE_LOW
        return GoalAnalysis(
            goal=_clean_goal(user_input),
            intent=intent,
            keywords=keywords,
            constraints=constraints,
            needs_tool=needs_tool,
            source=SOURCE_RULES,
            confidence=confidence,
            reason=reason,
        )


def analyze_goal(
    messages_or_text: list[dict[str, Any]] | str,
    *,
    llm_analyzer: GoalLlmAnalyzer | None = None,
    rule_analyzer: GoalAnalyzer | None = None,
) -> dict[str, Any]:
    """Analyze a user goal through one rule-first interface.

    Rules handle high-confidence keyword/tool patterns quickly. Ambiguous
    low-confidence input falls back to the provided LLM analyzer.
    """

    analyzer = rule_analyzer or GoalAnalyzer()
    rule_record = analyzer.analyze(messages_or_text).to_dict()
    if _rule_is_confident(rule_record) or llm_analyzer is None:
        rule_record["strategy"] = "rule_first"
        return rule_record
    llm_record = llm_analyzer(_as_messages(messages_or_text))
    normalized = normalize_goal_analysis(llm_record, fallback_goal=rule_record["goal"], source=SOURCE_LLM)
    normalized["strategy"] = "llm_fallback"
    normalized["rule_attempt"] = {
        "intent": rule_record.get("intent"),
        "confidence": rule_record.get("confidence"),
        "reason": rule_record.get("reason"),
    }
    return normalized


def normalize_goal_analysis(data: dict[str, Any], fallback_goal: str, source: str = SOURCE_LLM) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("goal analysis must be an object")
    constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    goal = _string(data.get("goal"), fallback_goal or "complete user request")
    if _looks_corrupt_text(goal, fallback_goal):
        goal = fallback_goal or "complete user request"
    return {
        "goal": goal,
        "intent": _choice(data.get("intent"), _ALLOWED_INTENTS, INTENT_EXPLAIN),
        "keywords": _filter_corrupt_strings(_string_list(data.get("keywords"), limit=8), fallback_goal),
        "constraints": {
            "must_include": _filter_corrupt_strings(_string_list(constraints.get("must_include"), limit=5), fallback_goal),
            "must_exclude": _filter_corrupt_strings(_string_list(constraints.get("must_exclude"), limit=5), fallback_goal),
        },
        "needs_tool": bool(data.get("needs_tool", False)),
        "source": _string(data.get("source"), source),
        "confidence": _choice(data.get("confidence"), {CONFIDENCE_HIGH, CONFIDENCE_LOW}, CONFIDENCE_HIGH),
        "reason": _string(data.get("reason"), "LLM goal analysis completed"),
    }


def minimal_goal_analysis(messages_or_text: list[dict[str, Any]] | str) -> dict[str, Any]:
    user_text = latest_user_text(messages_or_text)
    return {
        "goal": user_text or "complete user request",
        "intent": INTENT_EXPLAIN,
        "keywords": [],
        "constraints": {"must_include": [], "must_exclude": []},
        "needs_tool": False,
        "source": SOURCE_MINIMAL_FALLBACK,
        "confidence": CONFIDENCE_LOW,
        "reason": "Goal analysis failed; returned minimal no-keyword fallback.",
        "strategy": "minimal_fallback",
    }


def _rule_is_confident(record: dict[str, Any]) -> bool:
    return record.get("confidence") == CONFIDENCE_HIGH


def _clean_goal(text: str) -> str:
    text = " ".join(text.split())
    return text or "complete user request"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _looks_corrupt_text(text: str, fallback: str = "") -> bool:
    if not isinstance(text, str) or not text:
        return False
    if _contains_cjk(fallback) and "???" in text:
        return True
    question_marks = text.count("?")
    if question_marks < 3:
        return False
    non_space = len([char for char in text if not char.isspace()])
    if non_space and question_marks / non_space >= 0.35:
        return True
    return _contains_cjk(fallback) and not _contains_cjk(text)


def _filter_corrupt_strings(values: list[str], fallback: str = "") -> list[str]:
    return [value for value in values if not _looks_corrupt_text(value, fallback)]


def _detect_intent(text: str) -> tuple[str, str, bool]:
    lowered = text.lower()
    if _looks_like_direct_answer_request(lowered, text):
        return INTENT_DIRECT, "rule matched direct-answer marker", True
    has_math_expression = re.search(r"\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?", text) is not None
    has_sum_word = re.search(r"\bsum\b", lowered) is not None
    if has_math_expression or has_sum_word or _contains_any(lowered, text, ("calculate", "compute", "\u8ba1\u7b97", "\u6c42\u548c")):
        return INTENT_CALCULATE, "rule matched calculation marker", True
    if _contains_any(lowered, text, ("search", "find", "lookup", "\u641c\u7d22", "\u67e5\u627e", "\u68c0\u7d22")):
        return INTENT_SEARCH, "rule matched search marker", True
    if _contains_any(lowered, text, ("compare", "versus", " vs ", "difference", "\u5bf9\u6bd4", "\u6bd4\u8f83", "\u5dee\u5f02", "\u533a\u522b")):
        return INTENT_COMPARE, "rule matched compare marker", True
    if _contains_any(lowered, text, ("analyze", "analysis", "why", "reason", "evaluate", "\u5206\u6790", "\u4e3a\u4ec0\u4e48", "\u8bc4\u4f30")):
        return INTENT_ANALYZE, "rule matched analysis marker", True
    if _contains_any(lowered, text, ("read", "write", "convert", "run", "execute", "\u8bfb\u53d6", "\u5199\u5165", "\u8f6c\u6362", "\u6267\u884c")):
        return INTENT_EXECUTE, "rule matched execution marker", True
    if _contains_any(lowered, text, ("what is", "who is", "define", "\u4ec0\u4e48\u662f", "\u662f\u4ec0\u4e48", "\u8c01\u662f")):
        return INTENT_EXPLAIN, "rule matched simple explanation marker", True
    return INTENT_EXPLAIN, "no high-confidence rule marker matched", False


def _looks_like_direct_answer_request(lowered: str, text: str) -> bool:
    stripped = text.strip()
    if stripped in {"你好", "你好啊", "您好", "hello", "hi", "hey"}:
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

def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|[\u4e00-\u9fff]{2,}", text)
    stop_words = {
        "what",
        "is",
        "the",
        "and",
        "then",
        "finally",
        "please",
        "\u4ec0\u4e48",
        "\u7136\u540e",
        "\u6700\u540e",
        "\u8bf7",
        "\u5e2e\u6211",
    }
    result: list[str] = []
    for word in words:
        normalized = word.strip(" ,.;:!?()[]{}\"'")
        if not normalized or normalized.lower() in stop_words or normalized in stop_words:
            continue
        if normalized not in result:
            result.append(normalized)
        if len(result) == 8:
            break
    return result


def _extract_constraints(text: str) -> dict[str, list[str]]:
    return {
        "must_include": _extract_after_markers(
            text,
            ("must include", "include", "\u5fc5\u987b\u5305\u542b", "\u9700\u8981\u5305\u542b", "\u5305\u542b"),
        ),
        "must_exclude": _extract_after_markers(
            text,
            ("must exclude", "do not include", "don't include", "exclude", "\u4e0d\u8981\u5305\u542b", "\u6392\u9664", "\u4e0d\u5305\u542b"),
        ),
    }


def _extract_after_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    values: list[str] = []
    for marker in markers:
        marker_lower = marker.lower()
        source = lowered if marker.isascii() else text
        index = source.find(marker_lower if marker.isascii() else marker)
        if index == -1:
            continue
        start = index + len(marker)
        tail = text[start:]
        tail = re.split(r"[,.!?:;\n\u3002\uff0c\uff01\uff1f\uff1b]", tail, maxsplit=1)[0].strip(" :\uff1a")
        if tail and tail not in values:
            values.append(tail[:80])
    return values[:3]


def _needs_tool(text: str, intent: str) -> bool:
    lowered = text.lower()
    if intent in {INTENT_SEARCH, INTENT_CALCULATE}:
        return True
    return _contains_any(
        lowered,
        text,
        (
            "docs/",
            ".txt",
            ".md",
            ".csv",
            "file",
            "read",
            "search",
            "\u6587\u4ef6",
            "\u8bfb\u53d6",
            "\u641c\u7d22",
            "\u8868\u683c",
        ),
    )


def _contains_any(lowered: str, original: str, markers: tuple[str, ...]) -> bool:
    return any((marker in lowered) if marker.isascii() else (marker in original) for marker in markers)


def _as_messages(messages_or_text: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(messages_or_text, str):
        return [{"role": "user", "content": messages_or_text}]
    return messages_or_text


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


def _choice(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return default
