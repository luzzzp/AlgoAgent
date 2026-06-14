from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/convert_taco_python.py").resolve()
SPEC = importlib.util.spec_from_file_location("convert_taco_python", SCRIPT)
assert SPEC and SPEC.loader
convert_taco_python = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_taco_python)


class ConvertTacoPythonTest(unittest.TestCase):
    def test_skipped_record_contains_reason_and_previews(self) -> None:
        row = {
            "title": "Bad IO",
            "statement": "Solve it.\nSecond line.",
            "input_output": {"inputs": ["1"], "outputs": []},
            "solutions": ["print(1)"],
        }

        record = convert_taco_python._skipped_record(row, 7, "missing_or_mismatched_tests")

        self.assertEqual(record["index"], 7)
        self.assertEqual(record["reason"], "missing_or_mismatched_tests")
        self.assertEqual(record["title"], "Bad IO")
        self.assertIn("input_output", record["keys"])
        self.assertIn("\\n", record["statement_preview"])


if __name__ == "__main__":
    unittest.main()

