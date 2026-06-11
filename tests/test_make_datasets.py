from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algoagent.schema import load_problem


SCRIPT = Path("scripts/make_datasets.py").resolve()
SPEC = importlib.util.spec_from_file_location("make_datasets", SCRIPT)
assert SPEC and SPEC.loader
make_datasets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_datasets)


class MakeDatasetsTest(unittest.TestCase):
    def test_sft_answer_matches_runtime_response_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem.json"
            path.write_text(json.dumps(_problem("Time: O(n); Space: O(1)")), encoding="utf-8")
            bundle = load_problem(path)

        answer = make_datasets._format_answer(bundle)

        self.assertIn("Solution Explanation:", answer)
        self.assertIn("Time Complexity: O(n)", answer)
        self.assertIn("Space Complexity: O(1)", answer)
        self.assertIn("```cpp", answer)
        self.assertNotIn("Algorithm tags:", answer)

    def test_single_complexity_is_used_as_time_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem.json"
            path.write_text(json.dumps(_problem("O(n log n)")), encoding="utf-8")
            bundle = load_problem(path)

        answer = make_datasets._format_answer(bundle)

        self.assertIn("Time Complexity: O(n log n)", answer)
        self.assertIn("Space Complexity: unknown", answer)


def _problem(expected_complexity: str) -> dict:
    return {
        "problem": {
            "id": "sum",
            "title": "Sum",
            "statement": "Given n numbers, output their sum.",
            "input_format": "",
            "output_format": "",
            "constraints": ["1 <= n <= 100"],
        },
        "tests": {
            "repair_tests": [{"stdin": "1\n2\n", "expected_stdout": "2\n"}],
            "eval_tests": [{"stdin": "2\n2 3\n", "expected_stdout": "5\n"}],
        },
        "oracle": {
            "expected_complexity": expected_complexity,
            "reference_solution": "",
            "solutions": [
                {
                    "language": "cpp17",
                    "verified": True,
                    "code": "#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}\n",
                }
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
