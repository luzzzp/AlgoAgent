from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algoagent.schema import load_problems


SCRIPT = Path("scripts/build_dpo_from_failures.py").resolve()
SPEC = importlib.util.spec_from_file_location("build_dpo_from_failures", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BuildDpoFromFailuresTest(unittest.TestCase):
    def test_builds_dpo_pair_from_failed_generated_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problems = root / "problems"
            problems.mkdir()
            (problems / "sum.json").write_text(json.dumps(_problem()), encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps(_report()), encoding="utf-8")

            bundles = {bundle.spec.id: bundle for bundle in load_problems(problems)}
            records, stats = builder.build_dpo_records([report], bundles)

        self.assertEqual(stats["written"], 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["problem_id"], "sum")
        self.assertIn("cout << 3", records[0]["rejected"])
        self.assertIn("cout << 2", records[0]["chosen"])

    def test_skips_when_failed_code_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problems = root / "problems"
            problems.mkdir()
            (problems / "sum.json").write_text(json.dumps(_problem()), encoding="utf-8")
            report = root / "report.json"
            payload = _report()
            del payload["problems"][0]["attempt_records"][0]["generated_code"]
            report.write_text(json.dumps(payload), encoding="utf-8")

            bundles = {bundle.spec.id: bundle for bundle in load_problems(problems)}
            records, stats = builder.build_dpo_records([report], bundles)

        self.assertEqual(records, [])
        self.assertEqual(stats["skipped_no_failed_code"], 1)


def _problem() -> dict:
    return {
        "problem": {
            "id": "sum",
            "title": "Sum",
            "statement": "Print 2.",
            "input_format": "",
            "output_format": "",
            "constraints": [],
        },
        "tests": {
            "repair_tests": [{"stdin": "", "expected_stdout": "2\n"}],
            "eval_tests": [{"stdin": "", "expected_stdout": "2\n"}],
        },
        "oracle": {
            "expected_complexity": "O(1)",
            "reference_solution": "",
            "solutions": [
                {
                    "language": "cpp17",
                    "verified": True,
                    "code": "#include <bits/stdc++.h>\nusing namespace std;\nint main(){cout << 2 << '\\n';}\n",
                }
            ],
        },
    }


def _report() -> dict:
    return {
        "problems": [
            {
                "problem_id": "sum",
                "status": "FAILED",
                "failure_reason": "REPAIR_TEST_FAILED",
                "diagnostic_summary": "wrong answer",
                "attempt_records": [
                    {
                        "complexity": {"time_complexity": "O(1)", "space_complexity": "O(1)"},
                        "generated_code": "#include <bits/stdc++.h>\nusing namespace std;\nint main(){cout << 3 << '\\n';}\n",
                        "repair_result": {
                            "compiled": True,
                            "tests": [{"passed": False, "expected": "2", "actual": "3"}],
                        },
                    }
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
