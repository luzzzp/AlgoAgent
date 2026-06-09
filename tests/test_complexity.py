from __future__ import annotations

import unittest

from algoagent.complexity import ComplexityFeasibilityChecker
from algoagent.schema import CheckStatus, ComplexityEstimate, ProblemSpec


def _problem(max_n: int, time_limit: float = 2.0, memory_limit: int = 256) -> ProblemSpec:
    return ProblemSpec(
        id="complexity_test",
        title="Complexity Test",
        statement="Test resource feasibility.",
        input_format="n",
        output_format="answer",
        constraints=[f"1 <= N <= {max_n}"],
        time_limit_sec=time_limit,
        memory_limit_mb=memory_limit,
    )


class ComplexityFeasibilityCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = ComplexityFeasibilityChecker()

    def test_n_squared_at_one_second_boundary_passes(self) -> None:
        estimate, verdict = self.checker.check(
            _problem(10_000, time_limit=1.0),
            ComplexityEstimate("O(N^2)", "O(1)"),
            "int main(){return 0;}",
        )
        self.assertEqual(estimate.estimated_operations, 100_000_000)
        self.assertEqual(verdict.time_status, CheckStatus.PASS)

    def test_large_n_squared_fails_theoretical_time_limit(self) -> None:
        _, verdict = self.checker.check(
            _problem(100_000, time_limit=2.0),
            ComplexityEstimate("O(N^2)", "O(1)"),
            "int main(){return 0;}",
        )
        self.assertEqual(verdict.time_status, CheckStatus.FAIL)
        self.assertIn("THEORETICAL_TLE", verdict.reasons[0])

    def test_large_static_matrix_fails_memory_limit(self) -> None:
        _, verdict = self.checker.check(
            _problem(10_000, memory_limit=256),
            ComplexityEstimate("O(N)", "O(N^2)"),
            "int dp[10000][10000]; int main(){return 0;}",
        )
        self.assertEqual(verdict.memory_status, CheckStatus.FAIL)
        self.assertTrue(any("THEORETICAL_MLE" in reason for reason in verdict.reasons))

    def test_unparseable_complexity_is_unknown(self) -> None:
        _, verdict = self.checker.check(
            _problem(100),
            ComplexityEstimate("unknown", "unknown"),
            "int main(){return 0;}",
        )
        self.assertEqual(verdict.time_status, CheckStatus.UNKNOWN)
        self.assertEqual(verdict.memory_status, CheckStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

