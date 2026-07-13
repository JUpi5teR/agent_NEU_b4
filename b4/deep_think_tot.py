from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

StepGenerator = Callable[[str, str | None, list["ThoughtStep"], int, int], "ThoughtStep | dict[str, str]"]
ReflectionChecker = Callable[[str, "ThoughtStep", list["ThoughtStep"]], dict[str, Any]]

_CONNECTORS = (
    "therefore",
    "so",
    "next",
    "then",
    "based on",
    "according to",
    "because",
    "\u56e0\u6b64",
    "\u6240\u4ee5",
    "\u63a5\u7740",
    "\u7136\u540e",
    "\u57fa\u4e8e",
    "\u6839\u636e",
)
_BOUNDARY_MARKERS = (
    "unrelated",
    "out of scope",
    "irrelevant",
    "\u4e0e\u95ee\u9898\u65e0\u5173",
    "\u8d85\u51fa\u8303\u56f4",
    "\u4e0d\u76f8\u5173",
)


@dataclass
class ThoughtStep:
    id: int
    thought: str
    reasoning: str
    parent_id: int | None = None
    retry_count: int = 0
    status: str = "pending"
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "parent_id": self.parent_id,
            "retry_count": self.retry_count,
            "status": self.status,
            "conclusion": self.conclusion,
        }


@dataclass
class ToTResult:
    goal: str
    steps: list[ThoughtStep]
    final_answer: str
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "trace": self.trace,
        }

    def to_process_text(self) -> str:
        if not self.steps:
            return self.final_answer
        lines = ["DeepThink reasoning process:", f"Goal: {self.goal}"]
        for step in self.steps:
            lines.append(f"{step.id}. Thought: {step.thought}")
            lines.append(f"   Reasoning: {step.reasoning}")
            if step.conclusion:
                lines.append(f"   Conclusion: {step.conclusion}")
        lines.append(f"Final answer: {self.final_answer}")
        return "\n".join(lines)


class DeepThinkToT:
    """Single-chain backtracking ToT engine for B4 deep_think routes."""

    def __init__(
        self,
        max_steps: int = 5,
        max_node_retries: int = 2,
        max_global_backtracks: int = 10,
        step_generator: StepGenerator | None = None,
        reflection_checker: ReflectionChecker | None = None,
    ) -> None:
        self.max_steps = max(1, max_steps)
        self.max_node_retries = max(0, max_node_retries)
        self.max_global_backtracks = max(1, max_global_backtracks)
        self._step_generator = step_generator
        self._reflection_checker = reflection_checker

    def run(self, goal: str, context: str | None = None) -> ToTResult:
        clean_goal = goal.strip() or "complete reasoning task"
        clean_context = context.strip() if isinstance(context, str) and context.strip() else None
        steps: list[ThoughtStep] = []
        trace: list[dict[str, Any]] = []
        retry_counts: dict[int, int] = {}
        global_backtracks = 0
        next_step_id = 1

        while len(steps) < self.max_steps:
            retry_count = retry_counts.get(next_step_id, 0)
            parent_id = steps[-1].id if steps else None
            step = self._generate_step(clean_goal, clean_context, steps, next_step_id, parent_id, retry_count)
            trace.append(
                {
                    "action": "regenerate" if retry_count else "generate",
                    "step_id": step.id,
                    "retry_count": retry_count,
                    "thought": step.thought,
                }
            )
            reflection = self._reflect_step(clean_goal, step, steps)
            trace.append({"action": "reflection", "step_id": step.id, **reflection})
            if reflection["passed"]:
                step.status = "success"
                steps.append(step)
                if self._is_complete(steps):
                    break
                next_step_id += 1
                continue

            step.status = "failed"
            global_backtracks += 1
            trace.append(
                {
                    "action": "backtrack",
                    "step_id": step.id,
                    "reason": reflection["reason"],
                    "global_backtracks": global_backtracks,
                }
            )
            if global_backtracks >= self.max_global_backtracks:
                break
            retry_counts[next_step_id] = retry_count + 1
            if retry_counts[next_step_id] <= self.max_node_retries:
                continue
            retry_counts[next_step_id] = 0
            if steps:
                removed = steps.pop()
                next_step_id = removed.id
                trace.append(
                    {
                        "action": "backtrack",
                        "step_id": removed.id,
                        "target_step_id": next_step_id,
                        "reason": "node retry limit exceeded",
                    }
                )
            else:
                break

        final_answer = self._build_final_answer(clean_goal, clean_context, steps)
        trace.append({"action": "final_answer", "step_count": len(steps), "answer": final_answer})
        return ToTResult(clean_goal, steps, final_answer, trace)

    def _reflect_step(self, goal: str, step: ThoughtStep, previous_steps: list[ThoughtStep]) -> dict[str, Any]:
        if self._reflection_checker is None:
            return self.reflect_step(goal, step, previous_steps)
        reflection = self._reflection_checker(goal, step, previous_steps)
        if not isinstance(reflection, dict):
            return {"passed": False, "check": "reflection_checker", "reason": "reflection checker returned invalid data"}
        return {
            "passed": bool(reflection.get("passed", False)),
            "check": str(reflection.get("check") or "reflection_checker"),
            "reason": str(reflection.get("reason") or "reflection checker completed"),
            **{key: value for key, value in reflection.items() if key not in {"passed", "check", "reason"}},
        }

    def reflect_step(self, goal: str, step: ThoughtStep, previous_steps: list[ThoughtStep]) -> dict[str, Any]:
        combined = " ".join([step.thought, step.reasoning, step.conclusion]).strip()
        lowered = combined.lower()
        if not step.conclusion.strip() or len(step.conclusion.strip()) < 5:
            return {"passed": False, "check": "substance", "reason": "conclusion is too short"}
        if any(marker in lowered or marker in combined for marker in _BOUNDARY_MARKERS):
            return {"passed": False, "check": "boundary", "reason": "step appears outside the task boundary"}
        if previous_steps and not any(marker in lowered or marker in combined for marker in _CONNECTORS):
            return {"passed": False, "check": "coherence", "reason": "step is not connected to previous reasoning"}
        return {"passed": True, "check": "rule_reflection", "reason": "step is coherent and in scope"}

    def _generate_step(
        self,
        goal: str,
        context: str | None,
        previous_steps: list[ThoughtStep],
        step_id: int,
        parent_id: int | None,
        retry_count: int,
    ) -> ThoughtStep:
        if self._step_generator is None:
            data = _default_step_data(goal, context, previous_steps, step_id)
        else:
            data = self._step_generator(goal, context, previous_steps, step_id, retry_count)
        if isinstance(data, ThoughtStep):
            step = data
            step.id = step_id
            step.parent_id = parent_id
            step.retry_count = retry_count
            return step
        return ThoughtStep(
            id=step_id,
            thought=str(data.get("thought", "")),
            reasoning=str(data.get("reasoning", "")),
            parent_id=parent_id,
            retry_count=retry_count,
            conclusion=str(data.get("conclusion", "")),
        )

    def _is_complete(self, steps: list[ThoughtStep]) -> bool:
        if len(steps) >= self.max_steps:
            return True
        return len(steps) >= 4 or any("final" in step.conclusion.lower() for step in steps[-1:])

    def _build_final_answer(self, goal: str, context: str | None, steps: list[ThoughtStep]) -> str:
        if not steps:
            return "DeepThink could not build a reliable reasoning chain from the available input."
        context_note = " using the latest tool context" if context else ""
        return f"DeepThink conclusion{context_note}: {steps[-1].conclusion}"


def _default_step_data(goal: str, context: str | None, previous_steps: list[ThoughtStep], step_id: int) -> dict[str, str]:
    context_hint = " with the provided context" if context else ""
    templates = [
        {
            "thought": "Clarify the reasoning target.",
            "reasoning": f"The task asks to reason about: {goal}{context_hint}.",
            "conclusion": "The scope and expected answer are now explicit.",
        },
        {
            "thought": "Identify constraints and usable evidence.",
            "reasoning": "Based on the clarified scope, keep only evidence that supports the user goal.",
            "conclusion": "The reasoning should use relevant evidence and avoid unsupported assumptions.",
        },
        {
            "thought": "Compare possible reasoning paths.",
            "reasoning": "Then compare candidate explanations and discard paths that do not follow from the evidence.",
            "conclusion": "The strongest path is the one directly grounded in the available evidence.",
        },
        {
            "thought": "Form the final conclusion.",
            "reasoning": "Therefore combine the supported points into one concise answer.",
            "conclusion": "Final answer should be concise, scoped, and supported by the reasoning chain.",
        },
        {
            "thought": "Verify the final answer boundary.",
            "reasoning": "According to the original goal, remove claims that are not needed for the answer.",
            "conclusion": "Final answer remains within scope and avoids unrelated claims.",
        },
    ]
    return templates[min(step_id, len(templates)) - 1]
