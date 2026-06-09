from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/evaluate_hf_model.py").resolve()
SPEC = importlib.util.spec_from_file_location("evaluate_hf_model", SCRIPT)
assert SPEC and SPEC.loader
evaluate_hf_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_hf_model)


class EvaluateHfModelTest(unittest.TestCase):
    def test_summarize_serialized_results_for_resume_report(self) -> None:
        summary = evaluate_hf_model._summarize_serialized(
            [
                {
                    "problem_id": "ok",
                    "status": "SOLVED",
                    "attempts": 1,
                    "explanation": "",
                    "resource_verdict": {"time_status": "PASS", "memory_status": "PASS"},
                    "attempt_records": [
                        {
                            "repair_result": {
                                "compiled": True,
                                "tests": [{"passed": True}, {"passed": True}],
                            }
                        }
                    ],
                    "held_out_result": {"compiled": True, "tests": [{"passed": True}]},
                },
                {
                    "problem_id": "bad",
                    "status": "FAILED",
                    "attempts": 2,
                    "failure_reason": "REPAIR_TEST_FAILED",
                    "resource_verdict": {"time_status": "UNKNOWN", "memory_status": "PASS"},
                    "attempt_records": [
                        {
                            "repair_result": {
                                "compiled": True,
                                "tests": [{"passed": False}, {"passed": True}],
                            }
                        }
                    ],
                    "held_out_result": None,
                },
            ]
        )

        self.assertEqual(summary["num_problems"], 2)
        self.assertEqual(summary["initial_compile_rate"], 1.0)
        self.assertEqual(summary["verified_success_rate"], 0.5)
        self.assertEqual(summary["failure_breakdown"], {"REPAIR_TEST_FAILED": 1})


if __name__ == "__main__":
    unittest.main()
