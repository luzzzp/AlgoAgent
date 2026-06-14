from __future__ import annotations

import unittest

from algoagent.executor import PythonExecutor
from algoagent.reward import compute_reward, score_completion
from algoagent.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite


class RewardTest(unittest.TestCase):
    def test_correct_code_gets_high_reward(self) -> None:
        bundle = _bundle()
        text = _completion("a,b=map(int,input().split())\nprint(a+b)\n")

        reward = score_completion(text, bundle, PythonExecutor())

        self.assertGreater(reward.total, 0.8)
        self.assertGreater(reward.explanation, 0.0)

    def test_wrong_code_gates_explanation_reward(self) -> None:
        bundle = _bundle()
        text = _completion("a,b=map(int,input().split())\nprint(a-b)\n")

        reward = score_completion(text, bundle, PythonExecutor())

        self.assertLess(reward.total, 0.5)
        self.assertLessEqual(reward.explanation, 0.02)

    def test_timeout_is_penalized(self) -> None:
        bundle = _bundle(timeout=0.2)
        text = _completion("while True:\n    pass\n")

        reward = score_completion(text, bundle, PythonExecutor())

        self.assertIn("timeout", reward.reasons)
        self.assertLess(reward.total, 0.0)


def _bundle(timeout: float = 2.0) -> ProblemBundle:
    tests = TestSuite(
        visible_tests=[TestCase("2 3\n", "5\n", timeout_sec=timeout)],
        reward_tests=[TestCase("10 20\n", "30\n", timeout_sec=timeout)],
        eval_tests=[TestCase("1 1\n", "2\n", timeout_sec=timeout)],
    )
    return ProblemBundle(ProblemSpec(id="sum_two", title="Sum", statement="Sum two numbers."), tests)


def _completion(code: str) -> str:
    return (
        "Solution Explanation:\n使用输入中的两个整数并输出它们的和，算法只需要常数时间。\n\n"
        "Time Complexity: O(1)\n"
        "Space Complexity: O(1)\n"
        f"```python\n{code}```"
    )


if __name__ == "__main__":
    unittest.main()

