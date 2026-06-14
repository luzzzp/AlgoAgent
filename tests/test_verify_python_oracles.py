from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoagent.executor import PythonExecutor
from algoagent.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
)


SCRIPT = Path("scripts/verify_python_oracles.py").resolve()
SPEC = importlib.util.spec_from_file_location("verify_python_oracles", SCRIPT)
assert SPEC and SPEC.loader
verify_python_oracles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_python_oracles)


class VerifyPythonOracleTest(unittest.TestCase):
    def test_failed_solution_returns_failure_detail(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(id="bad", title="Bad", statement="Sum."),
            tests=TestSuite(visible_tests=[TestCase("2 3\n", "5\n")]),
            oracle=OracleMetadata(
                solutions=[OracleSolution("python3", "a,b=map(int,input().split())\nprint(a-b)\n")]
            ),
        )

        updated, ok, has_solution, detail = verify_python_oracles.verify_bundle(
            bundle,
            PythonExecutor(),
            max_solutions=1,
        )

        self.assertFalse(ok)
        self.assertTrue(has_solution)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["problem_id"], "bad")
        self.assertEqual(detail["attempts"][0]["first_failed"]["actual"], "-1")
        self.assertFalse(updated.oracle.solutions[0].verified)

    def test_callable_solution_can_be_verified(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(
                id="callable",
                title="Callable",
                statement="Return sum.",
                io_mode="callable",
                entry_point="add",
            ),
            tests=TestSuite(visible_tests=[TestCase("[2, 3]", "5")]),
            oracle=OracleMetadata(solutions=[OracleSolution("python3", "def add(a, b):\n    return a + b\n")]),
        )

        updated, ok, has_solution, detail = verify_python_oracles.verify_bundle(
            bundle,
            PythonExecutor(),
            max_solutions=1,
        )

        self.assertTrue(ok)
        self.assertTrue(has_solution)
        self.assertIsNone(detail)
        self.assertTrue(updated.oracle.solutions[0].verified)


if __name__ == "__main__":
    unittest.main()
