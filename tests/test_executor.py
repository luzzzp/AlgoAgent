from __future__ import annotations

import unittest

from algoagent.executor import PythonExecutor
from algoagent.schema import TestCase


class PythonExecutorTest(unittest.TestCase):
    def test_accepts_correct_program(self) -> None:
        result = PythonExecutor().evaluate(
            "a,b=map(int,input().split())\nprint(a+b)\n",
            [TestCase("2 3\n", "5\n", id="visible-1")],
        )

        self.assertTrue(result.syntax_valid)
        self.assertTrue(result.all_passed)

    def test_wrong_answer(self) -> None:
        result = PythonExecutor().evaluate(
            "print(1)\n",
            [TestCase("", "2\n", id="visible-1")],
        )

        self.assertTrue(result.syntax_valid)
        self.assertFalse(result.all_passed)
        self.assertEqual(result.runs[0].actual, "1")

    def test_runtime_error(self) -> None:
        result = PythonExecutor().evaluate(
            "raise RuntimeError('boom')\n",
            [TestCase("", "", id="visible-1")],
        )

        self.assertTrue(result.syntax_valid)
        self.assertFalse(result.all_passed)
        self.assertNotEqual(result.runs[0].returncode, 0)

    def test_timeout(self) -> None:
        result = PythonExecutor().evaluate(
            "while True:\n    pass\n",
            [TestCase("", "", id="visible-1", timeout_sec=0.2)],
        )

        self.assertTrue(result.syntax_valid)
        self.assertTrue(result.runs[0].timed_out)

    def test_output_cap(self) -> None:
        result = PythonExecutor(max_stdout_bytes=10).evaluate(
            "print('x'*100)\n",
            [TestCase("", "x\n", id="visible-1")],
        )

        self.assertTrue(result.runs[0].output_truncated)
        self.assertFalse(result.runs[0].passed)


if __name__ == "__main__":
    unittest.main()

