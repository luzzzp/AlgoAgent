from __future__ import annotations

import unittest

from algoagent.agent import PythonAlgoAgent
from algoagent.model_client import RuleBasedModel
from algoagent.schema import AgentStatus, ProblemSpec, TestCase, TestSuite


class AgentTest(unittest.TestCase):
    def test_agent_repairs_visible_failure(self) -> None:
        agent = PythonAlgoAgent(RuleBasedModel(), max_repair_turns=1)
        problem = ProblemSpec(id="needs_repair", title="Sum", statement="Sum two numbers.")
        tests = TestSuite(
            visible_tests=[TestCase("2 3\n", "5\n")],
            eval_tests=[TestCase("10 20\n", "30\n")],
        )

        result = agent.solve(problem, tests)

        self.assertEqual(result.status, AgentStatus.VERIFIED_SOLVED)
        self.assertEqual(result.attempts, 2)

    def test_eval_failure_does_not_leak_hidden_case(self) -> None:
        agent = PythonAlgoAgent(RuleBasedModel(), max_repair_turns=0)
        problem = ProblemSpec(id="sum_two", title="Sum", statement="Sum two numbers.")
        tests = TestSuite(
            visible_tests=[TestCase("2 3\n", "5\n")],
            eval_tests=[TestCase("10 20\n", "31\n")],
        )

        result = agent.solve(problem, tests)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_reason, "INTERNAL_EVAL_FAILED")
        self.assertNotIn("10 20", result.diagnostic_summary)

    def test_followup_requires_verified_solution(self) -> None:
        agent = PythonAlgoAgent(RuleBasedModel(), max_repair_turns=0)
        problem = ProblemSpec(id="sum_two", title="Sum", statement="Sum two numbers.")
        tests = TestSuite(visible_tests=[TestCase("2 3\n", "6\n")])

        result = agent.solve(problem, tests)
        answer = agent.answer_followup(result, problem, "第二行是什么意思？")

        self.assertIn("尚未通过内部验证", answer)


if __name__ == "__main__":
    unittest.main()

