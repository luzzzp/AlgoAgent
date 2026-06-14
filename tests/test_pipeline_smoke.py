from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PipelineSmokeTest(unittest.TestCase):
    def test_make_solver_sft_cli_on_mock_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problems = root / "problems"
            out = root / "processed"
            problems.mkdir()
            (problems / "sum_two.json").write_text(json.dumps(_problem()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_solver_sft.py",
                    "--problems",
                    str(problems),
                    "--out-dir",
                    str(out),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = (out / "solver_sft.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0])
            self.assertIn("```python", payload["output"])


def _problem() -> dict:
    return {
        "problem": {
            "id": "sum_two",
            "title": "Sum Two",
            "statement": "Read two integers and print their sum.",
            "language": "python3",
        },
        "tests": {
            "visible_tests": [{"stdin": "2 3\n", "expected_stdout": "5\n"}],
            "reward_tests": [{"stdin": "10 20\n", "expected_stdout": "30\n"}],
            "eval_tests": [{"stdin": "1 1\n", "expected_stdout": "2\n"}],
        },
        "oracle": {
            "expected_complexity": "Time: O(1); Space: O(1)",
            "solutions": [
                {
                    "language": "python3",
                    "verified": True,
                    "code": "a,b=map(int,input().split())\nprint(a+b)\n",
                }
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()

