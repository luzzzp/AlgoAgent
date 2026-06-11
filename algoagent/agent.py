from __future__ import annotations

from algoagent.complexity import ComplexityFeasibilityChecker
from algoagent.executor import CppExecutor
from algoagent.model_client import ModelClient
from algoagent.schema import (
    AgentResult,
    AgentStatus,
    AgentTrace,
    AttemptRecord,
    CandidateResult,
    CheckStatus,
    ComplexityEstimate,
    ProblemSpec,
    ResourceVerdict,
    TestSuite,
)


class AlgoAgent:
    def __init__(
        self,
        model: ModelClient,
        executor: CppExecutor | None = None,
        checker: ComplexityFeasibilityChecker | None = None,
        max_repair_turns: int = 3,
        explain_on_success: bool = True,
        capture_attempt_code: bool = False,
    ):
        self.model = model
        self.executor = executor or CppExecutor()
        self.checker = checker or ComplexityFeasibilityChecker()
        self.max_repair_turns = max_repair_turns
        self.explain_on_success = explain_on_success
        self.capture_attempt_code = capture_attempt_code

    def solve(self, problem: ProblemSpec, tests: TestSuite) -> AgentResult:
        traces = [
            AgentTrace(
                0,
                "problem",
                (
                    f"language={problem.language}; time_limit={problem.time_limit_sec:g}s; "
                    f"memory_limit={problem.memory_limit_mb}MB"
                ),
            )
        ]
        records: list[AttemptRecord] = []
        feedback: str | None = None
        last_diagnostic = ""
        last_complexity: ComplexityEstimate | None = None
        last_verdict = _unknown_verdict(problem)

        if not tests.repair_tests:
            return self._failure(
                problem,
                records,
                traces,
                last_verdict,
                "MISSING_REPAIR_TESTS",
                "No repair tests are configured.",
            )
        if not tests.eval_tests:
            return self._failure(
                problem,
                records,
                traces,
                last_verdict,
                "MISSING_EVAL_TESTS",
                "No held-out evaluation tests are configured.",
            )

        for attempt in range(1, self.max_repair_turns + 2):
            response = self.model.generate_solution(problem, feedback, attempt)
            traces.append(AgentTrace(attempt, "generate", response.draft_reasoning))

            complexity, verdict = self.checker.check(problem, response.complexity, response.code)
            last_complexity = complexity
            last_verdict = verdict
            traces.append(AgentTrace(attempt, "resource_check", _format_verdict(verdict)))
            if verdict.failed:
                records.append(
                    AttemptRecord(
                        attempt,
                        complexity,
                        verdict,
                        None,
                        self._captured_code(response.code),
                    )
                )
                last_diagnostic = "\n".join(verdict.reasons)
                feedback = last_diagnostic
                continue

            repair_result = self.executor.evaluate(
                response.code,
                tests.repair_tests,
                default_timeout_sec=problem.time_limit_sec,
                suite_name="repair",
            )
            records.append(
                AttemptRecord(
                    attempt,
                    complexity,
                    verdict,
                    repair_result,
                    self._captured_code(response.code),
                )
            )
            if not repair_result.all_passed:
                feedback = self._repair_feedback(repair_result, tests)
                last_diagnostic = feedback
                traces.append(AgentTrace(attempt, "repair_tests", feedback))
                continue

            traces.append(AgentTrace(attempt, "repair_tests", "All repair tests passed."))
            held_out_result = self.executor.evaluate(
                response.code,
                tests.eval_tests,
                default_timeout_sec=problem.time_limit_sec,
                suite_name="eval",
            )
            if not held_out_result.all_passed:
                summary = _held_out_summary(held_out_result)
                traces.append(AgentTrace(attempt, "held_out_tests", summary))
                return AgentResult(
                    problem_id=problem.id,
                    status=AgentStatus.FAILED,
                    attempts=attempt,
                    code=None,
                    explanation=None,
                    complexity=complexity,
                    resource_verdict=verdict,
                    failure_reason="HELD_OUT_TEST_FAILED",
                    diagnostic_summary=summary,
                    attempt_records=records,
                    held_out_result=held_out_result,
                    traces=traces,
                )

            traces.append(AgentTrace(attempt, "held_out_tests", "All held-out tests passed."))
            explanation = (
                self.model.explain_solution(problem, response.code, complexity).strip()
                if self.explain_on_success
                else ""
            )
            return AgentResult(
                problem_id=problem.id,
                status=AgentStatus.SOLVED,
                attempts=attempt,
                code=response.code,
                explanation=explanation,
                complexity=complexity,
                resource_verdict=verdict,
                failure_reason=None,
                diagnostic_summary=None,
                attempt_records=records,
                held_out_result=held_out_result,
                traces=traces,
            )

        reason = _failure_reason(records)
        return self._failure(
            problem,
            records,
            traces,
            last_verdict,
            reason,
            last_diagnostic or "Maximum repair turns reached.",
            last_complexity,
        )

    def _captured_code(self, code: str) -> str | None:
        return code if self.capture_attempt_code else None

    def _repair_feedback(self, result: CandidateResult, tests: TestSuite) -> str:
        if not result.compiled:
            return f"Compilation failed:\n{result.compile_error}"
        failed = next(test for test in result.tests if not test.passed)
        source = next(case for case in tests.repair_tests if case.id == failed.test_id)
        timeout = "\nFailure type: TIMEOUT" if failed.timed_out else ""
        return (
            f"Repair test failed: {failed.test_id}{timeout}\n"
            f"Input:\n{source.stdin}\n"
            f"Expected:\n{failed.expected}\n"
            f"Actual:\n{failed.actual}\n"
            f"Stderr:\n{failed.stderr}"
        )

    def _failure(
        self,
        problem: ProblemSpec,
        records: list[AttemptRecord],
        traces: list[AgentTrace],
        verdict: ResourceVerdict,
        reason: str,
        diagnostic: str,
        complexity: ComplexityEstimate | None = None,
    ) -> AgentResult:
        return AgentResult(
            problem_id=problem.id,
            status=AgentStatus.FAILED,
            attempts=len(records),
            code=None,
            explanation=None,
            complexity=complexity,
            resource_verdict=verdict,
            failure_reason=reason,
            diagnostic_summary=diagnostic,
            attempt_records=records,
            held_out_result=None,
            traces=traces,
        )


def _unknown_verdict(problem: ProblemSpec) -> ResourceVerdict:
    return ResourceVerdict(
        time_status=CheckStatus.UNKNOWN,
        memory_status=CheckStatus.UNKNOWN,
        estimated_operations=None,
        operation_budget=int(problem.time_limit_sec * 100_000_000),
        estimated_memory_bytes=None,
        memory_budget_bytes=problem.memory_limit_mb * 1024 * 1024,
        reasons=["No candidate was evaluated."],
    )


def _format_verdict(verdict: ResourceVerdict) -> str:
    return (
        f"time={verdict.time_status.value} ({verdict.estimated_operations}/{verdict.operation_budget}); "
        f"memory={verdict.memory_status.value} "
        f"({verdict.estimated_memory_bytes}/{verdict.memory_budget_bytes}); "
        f"reasons={verdict.reasons}"
    )


def _held_out_summary(result: CandidateResult) -> str:
    if not result.compiled:
        return "Held-out evaluation failed because the final candidate did not compile."
    failed = sum(1 for test in result.tests if not test.passed)
    timed_out = sum(1 for test in result.tests if test.timed_out)
    return (
        f"Held-out evaluation failed: {failed}/{len(result.tests)} tests failed; "
        f"{timed_out} timed out. Test inputs and expected outputs are withheld."
    )


def _failure_reason(records: list[AttemptRecord]) -> str:
    if not records:
        return "NO_CANDIDATE"
    last = records[-1]
    if last.resource_verdict.time_status == CheckStatus.FAIL:
        return "THEORETICAL_TLE"
    if last.resource_verdict.memory_status == CheckStatus.FAIL:
        return "THEORETICAL_MLE"
    if last.repair_result is None:
        return "RESOURCE_CHECK_FAILED"
    if not last.repair_result.compiled:
        return "COMPILE_FAILED"
    if any(test.timed_out for test in last.repair_result.tests):
        return "REPAIR_TEST_TIMEOUT"
    return "REPAIR_TEST_FAILED"
