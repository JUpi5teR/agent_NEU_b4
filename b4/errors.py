from __future__ import annotations

from typing import Any


class B4Error(Exception):
    """Base error with a stable code for B4 artifacts."""

    code = "B4Error"
    stage = "unknown"

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        if stage is not None:
            self.stage = stage

    def to_record(self, raw_text: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "type": type(self).__name__,
            "code": self.code,
            "stage": self.stage,
            "message": str(self),
        }
        if raw_text is not None:
            record["raw_text_preview"] = raw_text[:500]
        return record


class OutputParserError(B4Error):
    code = "OutputParserError"
    stage = "output_parser"


class InvalidJsonError(OutputParserError):
    code = "InvalidJsonError"


class UnknownKeyError(OutputParserError):
    code = "UnknownKeyError"


class InvalidToolCallError(OutputParserError):
    code = "InvalidToolCallError"


class MixedContentAndToolCallsError(OutputParserError):
    code = "MixedContentAndToolCallsError"


class EmptyAIMessageError(OutputParserError):
    code = "EmptyAIMessageError"


class RouteDecisionError(B4Error):
    code = "RouteDecisionError"
    stage = "route_decision"


class PlanExecutionError(B4Error):
    code = "PlanExecutionError"
    stage = "plan_execute"


def error_record(exc: Exception, *, stage: str, raw_text: str | None = None) -> dict[str, Any]:
    if isinstance(exc, B4Error):
        return exc.to_record(raw_text)
    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "code": type(exc).__name__,
        "stage": stage,
        "message": str(exc),
    }
    if raw_text is not None:
        record["raw_text_preview"] = raw_text[:500]
    return record
