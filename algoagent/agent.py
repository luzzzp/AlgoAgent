from __future__ import annotations

from algoagent.executor import PythonExecutor
from algoagent.model_client import ModelClient
from algoagent.schema import (
    AgentResult,
    AgentStatus,
    AgentTrace,
    AttemptRecord,
    ExecutionReport,
    ProblemSpec,
    TestSuite,
)


class PythonAlgoAgent:
    def __init__(
        self,
        model: ModelClient,
        executor: PythonExecutor | None = None,
        max_repair_turns: int = 3,
    ):
        self.model = model
        self.executor = executor or PythonExecutor()
        self.max_repair_turns = max_repair_turns

    def solve(self, problem: ProblemSpec, tests: TestSuite) -> AgentResult:
        traces = [AgentTrace(0, "problem", f"time_limit={problem.time_limit_sec:g}s")]
        records: list[AttemptRecord] = []
        feedback: str | None = None
        last_diag = ""
        if not tests.visible_tests:
            return self._failed(problem, records, traces, "MISSING_VISIBLE_TESTS", "No visible tests.")

        for turn in range(1, self.max_repair_turns + 2):
            response = self.model.generate_solution(problem, feedback, turn)
            visible_result = self.executor.evaluate(
                response.code,
                tests.visible_tests,
                default_timeout_sec=problem.time_limit_sec,
                suite_name="visible",
                entry_point=problem.entry_point if problem.io_mode == "callable" else "",
            )
            record = AttemptRecord(turn=turn, code=response.code, visible_result=visible_result)
            records.append(record)
            traces.append(AgentTrace(turn, "visible_tests", _summarize_execution(visible_result)))
            if not visible_result.all_passed:
                feedback = self._visible_feedback(visible_result)
                last_diag = feedback
                traces.append(AgentTrace(turn, "repair_feedback", feedback))
                continue

            if not tests.eval_tests:
                return AgentResult(
                    problem_id=problem.id,
                    status=AgentStatus.SAMPLE_SOLVED,
                    attempts=turn,
                    code=response.code,
                    explanation=response.explanation,
                    failure_reason=None,
                    diagnostic_summary="Visible tests passed; no internal eval tests configured.",
                    attempt_records=records,
                    traces=traces,
                )

            eval_result = self.executor.evaluate(
                response.code,
                tests.eval_tests,
                default_timeout_sec=problem.time_limit_sec,
                suite_name="eval",
                entry_point=problem.entry_point if problem.io_mode == "callable" else "",
            )
            records[-1] = AttemptRecord(
                turn=turn,
                code=response.code,
                visible_result=visible_result,
                eval_result=eval_result,
            )
            traces.append(AgentTrace(turn, "internal_eval", _summarize_execution(eval_result)))
            if eval_result.all_passed:
                return AgentResult(
                    problem_id=problem.id,
                    status=AgentStatus.VERIFIED_SOLVED,
                    attempts=turn,
                    code=response.code,
                    explanation=response.explanation,
                    failure_reason=None,
                    diagnostic_summary="Visible and internal eval tests passed.",
                    attempt_records=records,
                    traces=traces,
                )
            last_diag = "Internal verification failed; hidden test cases are withheld."
            return self._failed(problem, records, traces, "INTERNAL_EVAL_FAILED", last_diag)

        return self._failed(problem, records, traces, _failure_reason(records), last_diag)

    def answer_followup(self, result: AgentResult, problem: ProblemSpec, question: str) -> str:
        if result.status != AgentStatus.VERIFIED_SOLVED or not result.code:
            return "当前题解尚未通过内部验证，只能说明失败原因，不能解释未验证代码。"
        return self.model.answer_followup(
            problem,
            verified_code=_numbered(result.code),
            explanation=result.explanation or "",
            complexity="",
            history=[],
            question=question,
        )

    def _visible_feedback(self, report: ExecutionReport) -> str:
        if not report.syntax_valid:
            return f"Syntax error:\n{report.syntax_error}"
        failed = next(run for run in report.runs if not run.passed)
        timeout = "\nFailure type: TIMEOUT" if failed.timed_out else ""
        return (
            f"Visible test failed: {failed.test_id}{timeout}\n"
            f"Expected:\n{failed.expected}\n"
            f"Actual:\n{failed.actual}\n"
            f"Stderr:\n{failed.stderr}"
        )

    def _failed(
        self,
        problem: ProblemSpec,
        records: list[AttemptRecord],
        traces: list[AgentTrace],
        reason: str,
        diagnostic: str,
    ) -> AgentResult:
        return AgentResult(
            problem_id=problem.id,
            status=AgentStatus.FAILED,
            attempts=len(records),
            code=None,
            explanation=None,
            failure_reason=reason,
            diagnostic_summary=diagnostic,
            attempt_records=records,
            traces=traces,
        )


def _summarize_execution(report: ExecutionReport) -> str:
    if not report.syntax_valid:
        return f"syntax_error={report.syntax_error[:200]}"
    total = len(report.runs)
    passed = sum(1 for run in report.runs if run.passed)
    timed_out = sum(1 for run in report.runs if run.timed_out)
    return f"passed={passed}/{total}; timed_out={timed_out}"


def _failure_reason(records: list[AttemptRecord]) -> str:
    if not records:
        return "NO_CANDIDATE"
    last = records[-1]
    report = last.visible_result
    if report is None:
        return "NO_VISIBLE_RESULT"
    if not report.syntax_valid:
        return "SYNTAX_ERROR"
    if any(run.timed_out for run in report.runs):
        return "VISIBLE_TEST_TIMEOUT"
    return "VISIBLE_TEST_FAILED"


def _numbered(code: str) -> str:
    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(code.splitlines(), start=1))
