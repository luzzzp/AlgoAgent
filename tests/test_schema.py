from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from algoagent.schema import load_problem, normalize_output


class SchemaTest(unittest.TestCase):
    def test_loads_visible_reward_eval_tests(self) -> None:
        payload = {
            "problem": {"id": "p", "title": "P", "statement": "Solve it."},
            "tests": {
                "visible_tests": [{"stdin": "1\n", "expected_stdout": "1\n"}],
                "reward_tests": [{"stdin": "2\n", "expected_stdout": "2\n"}],
                "eval_tests": [{"stdin": "3\n", "expected_stdout": "3\n"}],
            },
            "oracle": {"solutions": [{"language": "python3", "code": "print(input())", "verified": True}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bundle = load_problem(path)

        self.assertEqual(bundle.spec.language, "python3")
        self.assertEqual(bundle.tests.visible_tests[0].id, "visible-0001")
        self.assertEqual(bundle.tests.reward_tests[0].id, "reward-0001")
        self.assertEqual(bundle.tests.eval_tests[0].id, "eval-0001")
        self.assertIn("Language: python3", bundle.spec.prompt(bundle.tests.visible_tests))

    def test_normalize_output_ignores_outer_line_whitespace(self) -> None:
        self.assertEqual(normalize_output("        0\n        3\n"), "0\n3")


if __name__ == "__main__":
    unittest.main()
