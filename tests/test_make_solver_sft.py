from __future__ import annotations

import importlib.util
import unittest

from algoagent.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
)


SPEC = importlib.util.spec_from_file_location("make_solver_sft", "scripts/make_solver_sft.py")
assert SPEC and SPEC.loader
make_solver_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_solver_sft)


class MakeSolverSftTest(unittest.TestCase):
    def test_answer_has_chinese_explanation_and_python_code(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(id="p", title="P", statement="Solve it."),
            tests=TestSuite(visible_tests=[TestCase("1\n", "1\n")]),
            oracle=OracleMetadata(
                expected_complexity="Time: O(1); Space: O(1)",
                solutions=[OracleSolution("python3", "print(input())", True)],
            ),
        )

        answer = make_solver_sft._answer(bundle)

        self.assertIn("Solution Explanation:", answer)
        self.assertRegex(answer, r"[\u4e00-\u9fff]")
        self.assertIn("Time Complexity: O(1)", answer)
        self.assertIn("```python", answer)


if __name__ == "__main__":
    unittest.main()

