from __future__ import annotations

import unittest

from algoagent.agent import AlgoAgent
from algoagent.model_client import ModelResponse
from algoagent.schema import (
    AgentStatus,
    ComplexityEstimate,
    ProblemSpec,
    TestCase,
    TestSuite,
    load_problem,
)


PASS_ONE = """#include <bits/stdc++.h>
using namespace std;
int main(){cout << 1 << "\\n";}
"""

PASS_TWO = """#include <bits/stdc++.h>
using namespace std;
int main(){cout << 2 << "\\n";}
"""


class FixedModel:
    def __init__(self, code: str):
        self.code = code
        self.generate_calls = 0
        self.explain_calls = 0

    def generate_solution(self, problem, feedback, attempt):
        self.generate_calls += 1
        return ModelResponse("fixed", self.code, ComplexityEstimate("O(1)", "O(1)"))

    def explain_solution(self, problem, verified_code, complexity):
        self.explain_calls += 1
        return "已验证解法。"


class ReplanningModel(FixedModel):
    def generate_solution(self, problem, feedback, attempt):
        self.generate_calls += 1
        complexity = ComplexityEstimate("O(n^2)", "O(1)") if attempt == 1 else ComplexityEstimate("O(n)", "O(1)")
        return ModelResponse("replan", self.code, complexity)


class AgentReliabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ProblemSpec(
            id="no_oracle",
            title="No Oracle",
            statement="Print one integer.",
            input_format="No input.",
            output_format="One integer.",
            constraints=[],
        )

    def test_loader_generates_test_ids_and_prompt_excludes_oracle(self) -> None:
        bundle = load_problem("data/problems/sample/two_sum.json")
        self.assertEqual(bundle.tests.repair_tests[0].id, "repair-0001")
        self.assertEqual(bundle.tests.eval_tests[0].id, "eval-0001")
        prompt = bundle.spec.prompt().lower()
        self.assertNotIn("difficulty", prompt)
        self.assertNotIn("hash-table", prompt)
        self.assertNotIn("reference_solution", prompt)
        self.assertIn("time limit: 2 seconds", prompt)
        self.assertIn("memory limit: 256 mb", prompt)

    def test_held_out_failure_stops_without_feedback_repair(self) -> None:
        model = FixedModel(PASS_ONE)
        tests = TestSuite(
            repair_tests=[TestCase(stdin="", expected_stdout="1\n", id="repair-0001")],
            eval_tests=[TestCase(stdin="", expected_stdout="2\n", id="eval-0001")],
        )
        result = AlgoAgent(model, max_repair_turns=3).solve(self.problem, tests)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_reason, "HELD_OUT_TEST_FAILED")
        self.assertIsNone(result.code)
        self.assertIsNone(result.explanation)
        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(model.explain_calls, 0)
        self.assertNotIn("eval-0001", result.diagnostic_summary)

    def test_verified_success_contains_explanation_and_code(self) -> None:
        model = FixedModel(PASS_ONE)
        tests = TestSuite(
            repair_tests=[TestCase(stdin="", expected_stdout="1\n", id="repair-0001")],
            eval_tests=[TestCase(stdin="", expected_stdout="1\n", id="eval-0001")],
        )
        result = AlgoAgent(model).solve(self.problem, tests)
        self.assertEqual(result.status, AgentStatus.SOLVED)
        self.assertEqual(result.code, PASS_ONE)
        self.assertEqual(result.explanation, "已验证解法。")
        self.assertEqual(model.explain_calls, 1)

    def test_failed_repair_does_not_return_unverified_code(self) -> None:
        model = FixedModel(PASS_TWO)
        tests = TestSuite(
            repair_tests=[TestCase(stdin="", expected_stdout="1\n", id="repair-0001")],
            eval_tests=[TestCase(stdin="", expected_stdout="1\n", id="eval-0001")],
        )
        result = AlgoAgent(model, max_repair_turns=1).solve(self.problem, tests)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNone(result.code)
        self.assertIsNone(result.explanation)
        self.assertEqual(model.generate_calls, 2)
        self.assertIsNone(result.attempt_records[-1].generated_code)

    def test_failed_attempt_code_is_captured_only_when_requested(self) -> None:
        model = FixedModel(PASS_TWO)
        tests = TestSuite(
            repair_tests=[TestCase(stdin="", expected_stdout="1\n", id="repair-0001")],
            eval_tests=[TestCase(stdin="", expected_stdout="1\n", id="eval-0001")],
        )
        result = AlgoAgent(model, max_repair_turns=0, capture_attempt_code=True).solve(self.problem, tests)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNone(result.code)
        self.assertEqual(result.attempt_records[-1].generated_code, PASS_TWO)

    def test_theoretical_tle_triggers_replanning_before_execution(self) -> None:
        problem = ProblemSpec(
            id="replan",
            title="Replan",
            statement="Print one.",
            input_format="n",
            output_format="one",
            constraints=["1 <= n <= 100000"],
            time_limit_sec=1,
        )
        model = ReplanningModel(PASS_ONE)
        tests = TestSuite(
            repair_tests=[TestCase(stdin="", expected_stdout="1\n", id="repair-0001")],
            eval_tests=[TestCase(stdin="", expected_stdout="1\n", id="eval-0001")],
        )
        result = AlgoAgent(model, max_repair_turns=2).solve(problem, tests)
        self.assertEqual(result.status, AgentStatus.SOLVED)
        self.assertEqual(result.attempts, 2)
        self.assertIsNone(result.attempt_records[0].repair_result)
        self.assertIsNotNone(result.attempt_records[1].repair_result)


if __name__ == "__main__":
    unittest.main()
