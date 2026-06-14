from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from algoagent.schema import ProblemSpec


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    explanation: str
    time_complexity: str
    space_complexity: str
    code: str


class ModelClient(Protocol):
    def generate_solution(self, problem: ProblemSpec, feedback: str | None, attempt: int) -> ModelResponse:
        ...

    def answer_followup(
        self,
        problem: ProblemSpec,
        verified_code: str,
        explanation: str,
        complexity: str,
        history: list[tuple[str, str]],
        question: str,
    ) -> str:
        ...


class RuleBasedModel:
    """Small deterministic model for unit tests and smoke demos."""

    def generate_solution(self, problem: ProblemSpec, feedback: str | None, attempt: int) -> ModelResponse:
        if problem.id == "sum_two":
            code = "a,b=map(int,input().split())\nprint(a+b)\n"
            return ModelResponse(
                raw_text=_format("读取两个整数并输出它们的和。", "O(1)", "O(1)", code),
                explanation="读取两个整数并输出它们的和。",
                time_complexity="O(1)",
                space_complexity="O(1)",
                code=code,
            )
        if problem.id == "needs_repair" and attempt == 1:
            code = "a,b=map(int,input().split())\nprint(a-b)\n"
        else:
            code = "a,b=map(int,input().split())\nprint(a+b)\n"
        return ModelResponse(
            raw_text=_format("根据反馈修正运算符并输出正确答案。", "O(1)", "O(1)", code),
            explanation="根据反馈修正运算符并输出正确答案。",
            time_complexity="O(1)",
            space_complexity="O(1)",
            code=code,
        )

    def answer_followup(
        self,
        problem: ProblemSpec,
        verified_code: str,
        explanation: str,
        complexity: str,
        history: list[tuple[str, str]],
        question: str,
    ) -> str:
        return f"基于已验证代码回答：{question}"


def _format(explanation: str, time: str, space: str, code: str) -> str:
    return (
        f"Solution Explanation:\n{explanation}\n\n"
        f"Time Complexity: {time}\n"
        f"Space Complexity: {space}\n"
        f"```python\n{code}```"
    )

