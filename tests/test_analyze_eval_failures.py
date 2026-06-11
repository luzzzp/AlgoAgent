from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path("scripts/analyze_eval_failures.py").resolve()
SPEC = importlib.util.spec_from_file_location("analyze_eval_failures", SCRIPT)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


class AnalyzeEvalFailuresTest(unittest.TestCase):
    def test_analyzes_failure_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps(_report()), encoding="utf-8")

            analysis = analyzer.analyze_reports([report], max_examples_per_category=2)

        self.assertEqual(analysis["total_problems"], 4)
        self.assertEqual(analysis["solved"], 1)
        self.assertEqual(analysis["failure_reason_counts"]["REPAIR_TEST_FAILED"], 2)
        self.assertEqual(analysis["diagnostic_category_counts"]["COMPILE_ERROR"], 1)
        self.assertEqual(analysis["diagnostic_category_counts"]["WRONG_ANSWER"], 1)
        self.assertEqual(analysis["diagnostic_category_counts"]["TIMEOUT"], 1)

    def test_renders_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps(_report()), encoding="utf-8")
            analysis = analyzer.analyze_reports([report], max_examples_per_category=1)

        markdown = analyzer.render_markdown(analysis)

        self.assertIn("# AlgoAgent Failure Analysis", markdown)
        self.assertIn("## Diagnostic Categories", markdown)
        self.assertIn("WRONG_ANSWER", markdown)


def _report() -> dict:
    return {
        "summary": {
            "num_problems": 4,
            "verified_success_rate": 0.25,
            "final_compile_rate": 0.75,
            "repair_test_pass_rate": 0.25,
            "held_out_test_pass_rate": 0.25,
        },
        "problems": [
            {
                "problem_id": "ok",
                "status": "SOLVED",
                "attempts": 1,
                "failure_reason": None,
                "attempt_records": [
                    {"repair_result": {"compiled": True, "tests": [{"passed": True}]}}
                ],
                "held_out_result": {"compiled": True, "tests": [{"passed": True}]},
            },
            {
                "problem_id": "compile",
                "status": "FAILED",
                "attempts": 1,
                "failure_reason": "COMPILE_FAILED",
                "attempt_records": [
                    {"repair_result": {"compiled": False, "compile_error": "error", "tests": []}}
                ],
                "held_out_result": None,
            },
            {
                "problem_id": "wa",
                "status": "FAILED",
                "attempts": 1,
                "failure_reason": "REPAIR_TEST_FAILED",
                "attempt_records": [
                    {
                        "repair_result": {
                            "compiled": True,
                            "tests": [
                                {
                                    "test_id": "repair-1",
                                    "passed": False,
                                    "expected": "2",
                                    "actual": "3",
                                    "timed_out": False,
                                }
                            ],
                        }
                    }
                ],
                "held_out_result": None,
            },
            {
                "problem_id": "timeout",
                "status": "FAILED",
                "attempts": 1,
                "failure_reason": "REPAIR_TEST_FAILED",
                "attempt_records": [
                    {
                        "repair_result": {
                            "compiled": True,
                            "tests": [
                                {
                                    "test_id": "repair-1",
                                    "passed": False,
                                    "expected": "",
                                    "actual": "",
                                    "timed_out": True,
                                }
                            ],
                        }
                    }
                ],
                "held_out_result": None,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
