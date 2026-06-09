from __future__ import annotations

import unittest

from algoagent.agent import AlgoAgent
from algoagent.evaluation import evaluate_problems
from algoagent.model_client import RuleBasedModel
from algoagent.schema import load_problems


class EvaluationTest(unittest.TestCase):
    def test_sample_problems_are_verified_by_rule_model(self) -> None:
        problems = load_problems("data/problems/sample")
        report = evaluate_problems(AlgoAgent(RuleBasedModel(), max_repair_turns=3), problems)
        summary = report["summary"]
        self.assertEqual(summary["num_problems"], 3)
        self.assertEqual(summary["verified_success_rate"], 1.0)
        self.assertEqual(summary["final_compile_rate"], 1.0)
        self.assertEqual(summary["repair_test_pass_rate"], 1.0)
        self.assertEqual(summary["held_out_test_pass_rate"], 1.0)
        self.assertEqual(summary["explanation_success_rate"], 1.0)
        self.assertGreater(summary["avg_repair_turns"], 0.0)


if __name__ == "__main__":
    unittest.main()

